"""Regression gate: run the public set and compare against a stored baseline.

Every change must keep Hit Rate@10 at 1.000, introduce no new miss, and not
lower MRR. This prints the numbers that decide that, plus the rank histogram --
the only view that shows whether a ranking change actually moved targets up.

Usage:
    python3 tools/regress.py                      # compare against the baseline
    python3 tools/regress.py --save-baseline      # accept current run as baseline
    python3 tools/regress.py --label rerank_on    # keep the raw run under a name
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev  # noqa: E402
from starter.agent import Agent  # noqa: E402

BASELINE = ROOT / "artifacts" / "baseline.json"
RUNS = ROOT / "artifacts" / "runs"
WATCH = "public_0020"


def run() -> dict:
    catalog_ids, categories, products = ev.catalog_index(ROOT / "data" / "catalog.jsonl")
    samples = ev.load_jsonl(ROOT / "data" / "public_set.jsonl")
    started = time.perf_counter()
    result = evaluate_quietly(samples, catalog_ids, categories, products)
    result["wall_clock_seconds"] = round(time.perf_counter() - started, 1)
    return result


def evaluate_quietly(samples, catalog_ids, categories, products) -> dict:
    agent = Agent(str(ROOT / "data" / "catalog.jsonl"))
    return ev.evaluate(agent, samples, catalog_ids, categories, products)


def rank_histogram(sessions: list[dict]) -> dict[str, int]:
    counts = collections.Counter(
        str(item["best_rank"]) if item["best_rank"] is not None else "miss"
        for item in sessions
    )
    return {key: counts[key] for key in sorted(counts, key=lambda x: (x == "miss", x.zfill(3)))}


def watched(sessions: list[dict]) -> dict:
    for item in sessions:
        if item["sample_id"] == WATCH:
            return {"rank": item["best_rank"], "turn": item["first_hit_turn"], "hit": item["hit"]}
    return {}


def digest(result: dict) -> dict:
    sessions = result["sessions"]
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "score": result["recommended_technical_score"],
        "rank1": sum(item["best_rank"] == 1 for item in sessions),
        "misses": sorted(item["sample_id"] for item in sessions if not item["hit"]),
        "rank_histogram": rank_histogram(sessions),
        "scenario_metrics": result["scenario_metrics"],
        "tokens": result["reported_token_usage"]["total_tokens"],
        "wall_clock_seconds": result.get("wall_clock_seconds"),
        WATCH: watched(sessions),
    }


def delta(current: float | None, previous: float | None, places: int = 6) -> str:
    if current is None or previous is None:
        return ""
    difference = current - previous
    if abs(difference) < 10 ** -places:
        return "  (=)"
    return f"  ({difference:+.{places}f})"


def report(now: dict, before: dict | None) -> bool:
    print(f"\n{'':<22}{'current':>12}{'  baseline' if before else ''}")
    for key, places in (("hit_rate_at_10", 6), ("mrr", 6), ("mttc", 3), ("score", 6)):
        line = f"{key:<22}{now[key]:>12.6f}"
        if before:
            line += f"{delta(now[key], before.get(key), places)}"
        print(line)
    print(f"{'rank-1 sessions':<22}{now['rank1']:>12}", end="")
    if before:
        print(f"  ({now['rank1'] - before.get('rank1', 0):+d})")
    else:
        print()
    print(f"{'tokens':<22}{now['tokens']:>12}")
    print(f"{'wall clock (s)':<22}{now['wall_clock_seconds']:>12}")

    print("\nrank histogram:")
    print("  " + "  ".join(f"{key}:{value}" for key, value in now["rank_histogram"].items()))

    print("\nby scenario (hit / mrr / mttc):")
    for name, metrics in sorted(now["scenario_metrics"].items()):
        print(f"  {name:<16}{metrics['hit_rate_at_10']:>7.3f}{metrics['mrr']:>9.3f}{metrics['mttc']:>8.3f}")

    print(f"\n{WATCH}: {now[WATCH] or 'not in dataset'}")
    if now["misses"]:
        print(f"misses ({len(now['misses'])}): {', '.join(now['misses'])}")

    if before is None:
        print("\nno baseline stored -- run with --save-baseline to create one")
        return True

    failures = []
    if now["hit_rate_at_10"] < before["hit_rate_at_10"]:
        failures.append(f"hit rate dropped {before['hit_rate_at_10']} -> {now['hit_rate_at_10']}")
    new_misses = set(now["misses"]) - set(before["misses"])
    if new_misses:
        failures.append(f"new misses: {', '.join(sorted(new_misses))}")
    if now["mrr"] < before["mrr"] - 1e-9:
        failures.append(f"MRR dropped {before['mrr']:.6f} -> {now['mrr']:.6f}")

    print()
    if failures:
        for item in failures:
            print(f"FAIL  {item}")
        return False
    print("PASS  hit rate held, no new miss, MRR not lower")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-baseline", action="store_true", help="accept this run as the baseline")
    parser.add_argument("--label", default="", help="also store the raw run under artifacts/runs/<label>.json")
    args = parser.parse_args()

    result = run()
    now = digest(result)
    before = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    passed = report(now, before)

    if args.label:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nraw run written to artifacts/runs/{args.label}.json")
    if args.save_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(now, indent=2) + "\n")
        print(f"baseline written to {BASELINE.relative_to(ROOT)}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
