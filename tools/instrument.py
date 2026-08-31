"""Phase A: record per-turn retrieval signals alongside ground truth.

Answers one question: at turn 1, is there an observable signal that separates
the sessions that end up in the rank tail from the ones that correctly hit rank
1 immediately? If nothing separates them, suppressing early recommendations
would be blind and would cost turns for nothing.

Writes a sidecar file; results.json and the baseline are untouched.

Usage:
    python3 tools/instrument.py                    # run and write the sidecar
    python3 tools/instrument.py --analyze-only     # re-analyze the sidecar
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev  # noqa: E402
from starter.agent import Agent  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
DATASET = ROOT / "data" / "public_set.jsonl"
SIDECAR = ROOT / "artifacts" / "instrumentation.json"
SIGNALS = (
    "fused_candidate_count", "fused_top1_score", "fused_top2_score",
    "fused_margin", "fused_relative_margin", "fused_pool_size",
)


def collect(limit: int = 0) -> list[dict]:
    """Replay the evaluator protocol exactly, recording signals per turn."""
    samples = ev.load_jsonl(DATASET)
    if limit:
        samples = samples[:limit]
    catalog_ids, categories, products = ev.catalog_index(CATALOG)
    agent = Agent(str(CATALOG))
    sessions: list[dict] = []

    for sample in samples:
        session_id = f"instr_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = ev.materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = ev.initial_message(
            effective, ev.coarse_category(categories.get(target, [])), disclosed
        )
        turns: list[dict] = []
        hit_turn = best_rank = None

        for turn in range(1, ev.MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, ev.TOP_K)
            diagnostics = agent.get_diagnostics(session_id)
            pool = agent.last_pool.get(session_id, [])
            emitted = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
            record = {
                "turn": turn,
                "constraints": len(diagnostics.get("active_constraints") or []),
                "ask_attribute": response.get("ask_attribute"),
                "emitted": len(emitted),
                # Rank in the full fused pool, not just the ten emitted ids.
                "target_rank_in_pool": pool.index(target) + 1 if target in pool else None,
                "target_rank_emitted": emitted.index(target) + 1 if target in emitted else None,
                **{key: diagnostics.get(key) for key in SIGNALS},
            }
            turns.append(record)

            if override_applied and target in emitted:
                best_rank, hit_turn = emitted.index(target) + 1, turn
                break
            if turn == ev.MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = ev.customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "turns": turns,
        })
        if len(sessions) % 40 == 0:
            print(f"  {len(sessions)}/{len(samples)}", flush=True)
    return sessions


def describe(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:<26} (none)"
    values = sorted(values)
    return (f"{name:<26} n={len(values):>3}  median={statistics.median(values):>10.4f}  "
            f"mean={statistics.fmean(values):>10.4f}  "
            f"p10={values[int(len(values) * 0.1)]:>10.4f}  "
            f"p90={values[int(len(values) * 0.9)]:>10.4f}")


def analyze(sessions: list[dict]) -> None:
    # The tail is every session that converts at rank 5 or worse -- exactly the
    # 30 sessions holding the bulk of the lost MRR.
    tail = [s for s in sessions if s["best_rank"] and s["best_rank"] >= 5]
    clean = [s for s in sessions if s["first_hit_turn"] == 1 and s["best_rank"] == 1]
    print(f"\ntail (best_rank >= 5): {len(tail)}    clean (turn 1, rank 1): {len(clean)}")

    print("\n--- turn-1 signals ---")
    for key in SIGNALS:
        tail_values = [s["turns"][0][key] for s in tail if s["turns"][0].get(key) is not None]
        clean_values = [s["turns"][0][key] for s in clean if s["turns"][0].get(key) is not None]
        print(describe(f"{key} [tail]", tail_values))
        print(describe(f"{key} [clean]", clean_values))
        if tail_values and clean_values:
            tm, cm = statistics.median(tail_values), statistics.median(clean_values)
            ratio = tm / cm if cm else float("inf")
            print(f"{'':<26} median ratio tail/clean = {ratio:.3f}")
        print()

    print("--- separation check: can a threshold split them? ---")
    for key in ("fused_relative_margin", "fused_margin", "fused_candidate_count"):
        tail_values = [s["turns"][0][key] for s in tail if s["turns"][0].get(key) is not None]
        clean_values = [s["turns"][0][key] for s in clean if s["turns"][0].get(key) is not None]
        if not tail_values or not clean_values:
            continue
        best = None
        for candidate in sorted(set(tail_values + clean_values)):
            # Suppress when the signal is below the threshold.
            caught = sum(v <= candidate for v in tail_values) / len(tail_values)
            harmed = sum(v <= candidate for v in clean_values) / len(clean_values)
            if best is None or caught - harmed > best[1]:
                best = (candidate, caught - harmed, caught, harmed)
        print(f"{key:<26} best threshold={best[0]:.4f}  "
              f"catches {best[2]:.0%} of tail, hits {best[3]:.0%} of clean  "
              f"(separation {best[1]:+.0%})")

    print("\n--- where does the target actually sit at turn 1? ---")
    for label, group in (("tail", tail), ("clean", clean)):
        ranks = [s["turns"][0]["target_rank_in_pool"] for s in group
                 if s["turns"][0]["target_rank_in_pool"]]
        missing = sum(1 for s in group if s["turns"][0]["target_rank_in_pool"] is None)
        if ranks:
            print(f"  {label:<6} median pool rank {statistics.median(ranks):>6.0f}  "
                  f"in top 10: {sum(r <= 10 for r in ranks)}/{len(group)}  "
                  f"not in pool: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        sessions = json.loads(SIDECAR.read_text())["sessions"]
    else:
        sessions = collect(args.limit)
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        SIDECAR.write_text(json.dumps({"sessions": sessions}, indent=2) + "\n")
        print(f"sidecar written to {SIDECAR.relative_to(ROOT)}")
    analyze(sessions)


if __name__ == "__main__":
    main()
