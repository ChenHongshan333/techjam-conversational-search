"""Sweep how many opening turns are worth withholding.

Withholding the opening turns lets a session convert on accumulated evidence
rather than on turn 1. Two turns is the shipped setting; this re-validates that
choice and shows how the score degrades either side of it.

Usage:
    python3 tools/suppression_sweep.py
    python3 tools/suppression_sweep.py --turns 1 2 3
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
DATASET = ROOT / "data" / "public_set.jsonl"
MANAGED = (
    "TECHJAM_SUPPRESSION", "TECHJAM_SUPPRESSION_TURNS",
    "TECHJAM_SUPPRESSION_MAX_TURNS",
)
WATCH = "public_0020"


def run(name: str, config: dict[str, str], samples, catalog_ids, categories, products) -> dict:
    for key in MANAGED:
        os.environ.pop(key, None)
    os.environ.update(config)
    import shopping_agent.config
    import starter.agent
    importlib.reload(shopping_agent.config)
    importlib.reload(starter.agent)
    agent = starter.agent.Agent(str(CATALOG))

    started = time.perf_counter()
    result = ev.evaluate(agent, samples, catalog_ids, categories, products)
    sessions = result["sessions"]
    suppressed = sum(
        d.get("suppressed_turns", 0) for d in agent.diagnostics.values()
    )
    ranks = collections.Counter(
        str(s["best_rank"]) if s["best_rank"] is not None else "miss" for s in sessions
    )
    watch = next((s for s in sessions if s["sample_id"] == WATCH), None)
    row = {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "score": result["recommended_technical_score"],
        "rank1": sum(s["best_rank"] == 1 for s in sessions),
        "misses": sorted(s["sample_id"] for s in sessions if not s["hit"]),
        "rank_histogram": {k: ranks[k] for k in sorted(ranks, key=lambda x: (x == "miss", x.zfill(3)))},
        "suppressed_turns_total": suppressed,
        "watch": {"rank": watch["best_rank"], "turn": watch["first_hit_turn"]} if watch else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(f"  {name:<26} hit={row['hit_rate_at_10']:.3f} mrr={row['mrr']:.5f} "
          f"mttc={row['mttc']:.3f} score={row['score']:.6f} rank1={row['rank1']:>3} "
          f"suppressed={suppressed:>3}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--label", default="suppression_sweep")
    args = parser.parse_args()

    samples = ev.load_jsonl(DATASET)
    catalog_ids, categories, products = ev.catalog_index(CATALOG)

    plan: dict[str, dict[str, str]] = {"off": {"TECHJAM_SUPPRESSION": "0"}}
    for turns in args.turns:
        plan[f"turns{turns}"] = {
            "TECHJAM_SUPPRESSION": "1",
            "TECHJAM_SUPPRESSION_TURNS": str(turns),
            "TECHJAM_SUPPRESSION_MAX_TURNS": str(max(turns, 2)),
        }

    rows = {name: run(name, config, samples, catalog_ids, categories, products)
            for name, config in plan.items()}

    base = rows["off"]
    print(f"\n{'config':<26}{'hit':>7}{'MRR':>10}{'MTTC':>8}{'score':>10}{'dScore':>10}{'rank1':>7}")
    for name, row in sorted(rows.items(), key=lambda kv: -kv[1]["score"]):
        flag = "" if row["hit_rate_at_10"] >= 1.0 else "  <-- HIT RATE LOST"
        print(f"{name:<26}{row['hit_rate_at_10']:>7.3f}{row['mrr']:>10.5f}{row['mttc']:>8.3f}"
              f"{row['score']:>10.6f}{row['score'] - base['score']:>+10.6f}{row['rank1']:>7}{flag}")

    best = max(rows.items(), key=lambda kv: (kv[1]["hit_rate_at_10"] >= 1.0, kv[1]["score"]))
    print(f"\nbest: {best[0]}  score={best[1]['score']:.6f}  {WATCH}={best[1]['watch']}")
    print(f"histogram: " + "  ".join(f"{k}:{v}" for k, v in best[1]["rank_histogram"].items()))

    out = ROOT / "artifacts" / "runs" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
