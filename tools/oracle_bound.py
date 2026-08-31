"""Oracle upper bound: how well does retrieval do when the customer hides nothing?

Runs the agent's retriever ONCE per session on progressively more generous
queries, so you can see where the score is actually lost:

  category_only  what a Browsing session gives you on turn 1 (no constraints)
  turn1_buying   what a Buying session gives you on turn 1 (first hard constraint)
  hard_only      both hard constraints
  full_oracle    every constraint the simulator would ever reveal

full_oracle is the retrieval ceiling: no dialogue policy can beat it, because
there is nothing left for the customer to say. The gap between turn1_buying and
full_oracle is what better questioning is worth.

Usage:  python3 tools/oracle_bound.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev  # noqa: E402
from starter.agent import Agent  # noqa: E402

# Single-shot probe: turn 1 measures pure retrieval, before any clarification
# policy or exhaustion logic can alter the ranking.
PROBE_TURN = 1
CATALOG = ROOT / "data" / "catalog.jsonl"
DATASET = ROOT / "data" / "public_set.jsonl"
TOP_K = 10


def build_dialogues(card: dict, category: str) -> dict[str, list[str]]:
    """Express each variant in the protocol the parser accepts.

    The retriever is only ever fed through message parsing, so a bare constraint
    string would produce empty state and measure nothing. Each variant is a real
    conversation; the ranking is read after the last message is ingested.
    """
    hard = [str(v) for v in card.get("hard_constraints", [])]
    soft = [str(v) for v in card.get("soft_preferences", [])]
    buying = (
        f"I'm looking for {category}. A key requirement is: {hard[0]}."
        if hard else f"I'm looking for {category}, but I'm still exploring."
    )
    rest_hard = hard[1:]
    everything = [value for value in dict.fromkeys(hard + soft) if value not in hard[:1]]
    return {
        "category_only": [f"I'm looking for {category}, but I'm still exploring."],
        "turn1_buying": [buying],
        "hard_only": [buying] + (
            ["For that, what matters is: " + "; ".join(rest_hard) + "."] if rest_hard else []
        ),
        "full_oracle": [buying] + (
            ["For that, what matters is: " + "; ".join(everything) + "."] if everything else []
        ),
    }


def rank_of(
    agent: Agent,
    messages: list[str],
    target: str,
    catalog_ids: set[str],
    session: str,
) -> int | None:
    # Each variant needs its own session: state accumulates across calls, so a
    # shared id would let an earlier variant leak into a later one.
    agent.reset(session, {})
    response = None
    for turn, message in enumerate(messages, start=1):
        try:
            response = agent.respond(session, message, turn, TOP_K)
        except Exception as exc:  # a broken retriever should be loud, not silently zero
            print(f"  !! respond() raised: {exc!r}", file=sys.stderr)
            return None
    ranked = ev.normalize_recommendations((response or {}).get("recommendations"), catalog_ids)
    return ranked.index(target) + 1 if target in ranked else None


def summarize(ranks: list[int | None]) -> dict:
    hits = [r for r in ranks if r is not None]
    return {
        "n": len(ranks),
        "hit_rate": len(hits) / len(ranks) if ranks else 0.0,
        "mrr": statistics.fmean([0.0 if r is None else 1.0 / r for r in ranks]) if ranks else 0.0,
        "rank1": sum(1 for r in hits if r == 1) / len(ranks) if ranks else 0.0,
    }


def main() -> None:
    print(f"loading catalog from {CATALOG} ...")
    catalog_ids, categories, products = ev.catalog_index(CATALOG)
    samples = ev.load_jsonl(DATASET)
    print(f"catalog={len(catalog_ids)} products, dataset={len(samples)} sessions")

    # How much does the free turn-1 category string actually narrow the field?
    pool = Counter(ev.coarse_category(cats) for cats in categories.values())
    target_pools = [
        pool[ev.coarse_category(categories.get(str(s["ground_truth"]["parent_asin"]), []))]
        for s in samples
    ]
    target_pools.sort()
    print("\n=== category filter power ===")
    print(f"distinct coarse categories in catalog : {len(pool)}")
    print("candidates sharing the target's category:")
    print(f"  median {statistics.median(target_pools):.0f} | mean {statistics.fmean(target_pools):.0f} "
          f"| p90 {target_pools[int(len(target_pools) * 0.9)]} | max {target_pools[-1]}")
    print(f"  sessions where it cuts 50k -> under 500 candidates: "
          f"{sum(1 for c in target_pools if c < 500) / len(target_pools):.1%}")

    print("\nbuilding agent index ...")
    agent = Agent(str(CATALOG))

    variants = ["category_only", "turn1_buying", "hard_only", "full_oracle"]
    ranks: dict[str, list[int | None]] = defaultdict(list)
    by_scenario: dict[tuple[str, str], list[int | None]] = defaultdict(list)

    print(f"scoring {len(samples)} sessions x {len(variants)} query variants ...")
    for n, sample in enumerate(samples):
        target = str(sample["ground_truth"]["parent_asin"])
        card, _ = ev.materialize_hidden_fields(sample, products)
        category = ev.coarse_category(categories.get(target, []))
        dialogues = build_dialogues(card, category)
        for variant in variants:
            rank = rank_of(agent, dialogues[variant], target, catalog_ids, f"oracle_{n}_{variant}")
            ranks[variant].append(rank)
            by_scenario[(variant, str(sample["scenario_type"]))].append(rank)

    print("\n=== retrieval quality by how much the customer disclosed ===")
    print(f"{'state given to retriever':<16} {'hit@10':>7} {'MRR':>7} {'rank1':>7}")
    for variant in variants:
        s = summarize(ranks[variant])
        print(f"{variant:<16} {s['hit_rate']:>7.3f} {s['mrr']:>7.3f} {s['rank1']:>7.3f}")

    scenarios = sorted({str(s["scenario_type"]) for s in samples})
    print("\n=== hit@10 by scenario ===")
    print(f"{'query':<16}" + "".join(f"{sc:>17}" for sc in scenarios))
    for variant in variants:
        row = "".join(f"{summarize(by_scenario[(variant, sc)])['hit_rate']:>17.3f}" for sc in scenarios)
        print(f"{variant:<16}{row}")

    ceiling = summarize(ranks["full_oracle"])
    turn1 = summarize(ranks["turn1_buying"])
    print("\n=== read this ===")
    print(f"retrieval ceiling (full_oracle hit@10) : {ceiling['hit_rate']:.3f}")
    print(f"dialogue headroom (full - turn1)       : {ceiling['hit_rate'] - turn1['hit_rate']:+.3f}")
    if ceiling["hit_rate"] < 0.60:
        print("-> RETRIEVAL is the bottleneck: even with every constraint the retriever misses.")
        print("   Work on matching/ranking before touching the ask policy.")
    elif ceiling["hit_rate"] - turn1["hit_rate"] > 0.25:
        print("-> DIALOGUE is the bottleneck: the retriever finds it once it knows enough.")
        print("   Work on ask_attribute policy and carrying state across turns.")
    else:
        print("-> Mixed. Both levers pay; start with whichever scenario column is weakest.")


if __name__ == "__main__":
    main()
