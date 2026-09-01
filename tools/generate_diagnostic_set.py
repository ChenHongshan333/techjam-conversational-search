"""Generate a deterministic 100-session catalog-derived diagnostic holdout.

The target products are excluded from the public 200-session set. Selection is
stratified by product family and by how many catalog products remain after all
four simulator-visible constraints are intersected. The output deliberately
contains both identifiable cases and ambiguous stress cases.

This is a local robustness suite, not an estimate of the private leaderboard.

Usage:
    .venv/bin/python tools/generate_diagnostic_set.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import intent_card, load_jsonl  # noqa: E402
from shopping_agent.retrieval.catalog import CatalogIndex  # noqa: E402
from shopping_agent.text import normalize  # noqa: E402


FAMILY_SCENARIOS = {
    "clothing": {"buying": 16, "browsing": 16, "intent_override": 6, "boundary": 2},
    "shoes": {"buying": 14, "browsing": 14, "intent_override": 5, "boundary": 2},
    "jewelry": {"buying": 10, "browsing": 10, "intent_override": 4, "boundary": 1},
}
AMBIGUITY_QUOTAS = {
    "clothing": {"unique": 16, "narrow": 16, "ambiguous": 8},
    "shoes": {"unique": 14, "narrow": 13, "ambiguous": 8},
    "jewelry": {"unique": 23, "narrow": 1, "ambiguous": 1},
}
DIFFICULTY = {
    "buying": "easy",
    "browsing": "medium",
    "intent_override": "hard",
    "boundary": "medium",
}


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    family: str
    leaf_category: str
    ambiguity: str
    full_evidence_ambiguity: int
    first_constraint_frequency: int


def stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256("\0".join((seed, *values)).encode()).hexdigest()


def product_family(categories: list[object]) -> str:
    normalized = [str(value).casefold() for value in categories][1:]
    if "jewelry" in normalized:
        return "jewelry"
    if "shoes" in normalized or "boot shop" in normalized:
        return "shoes"
    return "clothing"


def ambiguity_bucket(count: int) -> str | None:
    if count == 1:
        return "unique"
    if 2 <= count <= 10:
        return "narrow"
    if 11 <= count <= 100:
        return "ambiguous"
    return None


def choose_diverse(
    candidates: list[Candidate],
    count: int,
    seed: str,
    already_selected: set[str],
) -> list[Candidate]:
    ordered = sorted(
        (item for item in candidates if item.parent_asin not in already_selected),
        key=lambda item: stable_key(seed, item.parent_asin),
    )
    for leaf_cap in (2, 3, 4, 6, 10, count):
        selected: list[Candidate] = []
        leaf_counts: dict[str, int] = defaultdict(int)
        for item in ordered:
            if leaf_counts[item.leaf_category] >= leaf_cap:
                continue
            selected.append(item)
            leaf_counts[item.leaf_category] += 1
            if len(selected) == count:
                return selected
    raise RuntimeError(f"Only found {len(selected)} of {count} requested candidates")


def build_candidates(
    catalog_path: Path,
    public_targets: set[str],
) -> list[Candidate]:
    products = {
        str(product["parent_asin"]): product
        for product in (
            json.loads(line) for line in catalog_path.open(encoding="utf-8") if line.strip()
        )
    }
    index = CatalogIndex(catalog_path)
    candidates: list[Candidate] = []
    for parent_asin, product in products.items():
        if parent_asin in public_targets:
            continue
        card = intent_card(product)
        values = [
            *card.get("hard_constraints", []),
            *card.get("soft_preferences", []),
        ]
        if len(values) < 4:
            continue
        constraint_sets = [
            index.constraint_index.get(normalize(str(value)), set()) for value in values
        ]
        if not all(parent_asin in identifiers for identifiers in constraint_sets):
            continue
        full_intersection = set(constraint_sets[0])
        for identifiers in constraint_sets[1:]:
            full_intersection.intersection_update(identifiers)
        ambiguity = ambiguity_bucket(len(full_intersection))
        if ambiguity is None:
            continue
        categories = list(product.get("categories") or [])
        candidates.append(Candidate(
            parent_asin=parent_asin,
            family=product_family(categories),
            leaf_category=str(categories[-1]) if categories else "unknown",
            ambiguity=ambiguity,
            full_evidence_ambiguity=len(full_intersection),
            first_constraint_frequency=len(constraint_sets[0]),
        ))
    return candidates


def generate(catalog_path: Path, public_path: Path, seed: str) -> list[dict]:
    public_samples = load_jsonl(public_path)
    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public_samples
    }
    profiles_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for sample in public_samples:
        profiles_by_scenario[str(sample["scenario_type"])].append(
            copy.deepcopy(sample["user_profile"])
        )

    pool = build_candidates(catalog_path, public_targets)
    selected_by_family: dict[str, list[Candidate]] = {}
    selected_ids: set[str] = set()
    for family, quotas in AMBIGUITY_QUOTAS.items():
        family_selected: list[Candidate] = []
        for ambiguity, count in quotas.items():
            chosen = choose_diverse(
                [
                    item for item in pool
                    if item.family == family and item.ambiguity == ambiguity
                ],
                count,
                f"{seed}:{family}:{ambiguity}",
                selected_ids,
            )
            family_selected.extend(chosen)
            selected_ids.update(item.parent_asin for item in chosen)
        selected_by_family[family] = sorted(
            family_selected,
            key=lambda item: stable_key(seed, family, item.parent_asin),
        )

    rows: list[dict] = []
    for family, scenario_counts in FAMILY_SCENARIOS.items():
        scenarios = [
            scenario
            for scenario, count in scenario_counts.items()
            for _ in range(count)
        ]
        scenarios.sort(key=lambda scenario: stable_key(seed, family, scenario, str(len(rows))))
        # Rotate equal scenario labels deterministically instead of grouping them.
        scenarios = sorted(
            enumerate(scenarios),
            key=lambda pair: stable_key(seed, family, pair[1], str(pair[0])),
        )
        scenario_order = [scenario for _, scenario in scenarios]
        for candidate, scenario in zip(selected_by_family[family], scenario_order):
            profiles = profiles_by_scenario[scenario]
            profile_index = int(stable_key(seed, candidate.parent_asin)[:8], 16) % len(profiles)
            rows.append({
                "category_bucket": family,
                "difficulty_bucket": DIFFICULTY[scenario],
                "ground_truth": {"parent_asin": candidate.parent_asin},
                "sample_id": "",
                "scenario_type": scenario,
                "user_profile": copy.deepcopy(profiles[profile_index]),
                "diagnostic_design": {
                    "source": "catalog-derived unseen target",
                    "leaf_category": candidate.leaf_category,
                    "evidence_ambiguity": candidate.ambiguity,
                    "full_evidence_candidate_count": candidate.full_evidence_ambiguity,
                    "first_constraint_candidate_count": candidate.first_constraint_frequency,
                },
            })

    rows.sort(key=lambda row: stable_key(seed, row["ground_truth"]["parent_asin"]))
    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"diagnostic_{index:04d}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--public", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "diagnostic_set_100.jsonl"
    )
    parser.add_argument("--seed", default="techjam-diagnostic-v1")
    args = parser.parse_args()

    rows = generate(args.catalog, args.public, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
