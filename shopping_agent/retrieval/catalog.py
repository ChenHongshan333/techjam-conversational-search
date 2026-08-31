from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..models import SessionState
from ..text import (
    COLORS,
    MATERIALS,
    clean_constraint,
    flatten_text,
    flatten_values,
    normalize,
    terms,
)


# Dual-track routing. Buying has hard constraints to satisfy, so exact evidence
# is weighted up and loose category/BM25 signal down; browsing is open-ended, so
# category fit and the broad lexical routes carry more. Multipliers are applied
# to the shared scoring terms, blended by `scale` -- at scale 0 every track
# reduces to the single-track weighting, which is the pre-routing behaviour.
# Dual-track routing. Buying commits to hard constraints, so exact and
# intersection evidence is weighted up and loose category/BM25 signal down.
# Browsing carries a softer version of the same shape: by the turn the agent
# first emits, a browsing session has disclosed the same kind of catalog
# constraints a buying session has, so the two tracks differ in degree rather
# than in kind -- an inverted browsing profile was measured four ways and every
# one scored below this.
#
# A third track handles overrides. Those sessions relax to neutral because their
# requirements were just rewritten, so the evidence is fresher and thinner than
# the constraint count suggests; without that guard the stronger emphasis costs
# intent_override 0.024 MRR.
INTENT_EMPHASIS: dict[str, tuple[float, float, float, float, float]] = {
    # exact, exact_coverage, intersection, route_bonus, category_coverage
    "buying":    (1.50, 1.50, 1.50, 0.70, 0.60),
    "browsing":  (1.30, 1.30, 1.30, 0.80, 0.75),
    "uncertain": (1.00, 1.00, 1.00, 1.00, 1.00),
}
INTENT_ROUTE_EMPHASIS: dict[str, dict[str, float]] = {
    "buying": {"exact": 1.5, "intersection": 1.5, "strict_bm25": 1.3, "broad_bm25": 0.6},
    "browsing": {"exact": 1.3, "intersection": 1.3, "broad_bm25": 0.75},
    "uncertain": {},
}


def blend(multiplier: float, scale: float) -> float:
    """Interpolate a multiplier towards 1.0; scale 0 disables routing entirely."""
    return 1.0 + scale * (multiplier - 1.0)


MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.IGNORECASE)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.IGNORECASE)


@dataclass
class Product:
    parent_asin: str
    title: str
    categories: str
    category_values: list[str]
    identity_text: str
    attribute_text: str
    search_text: str
    atomic_values: set[str]
    average_rating: float
    rating_number: int


class CatalogIndex:
    """In-memory structured index plus SQLite FTS5 candidate retrieval."""

    def __init__(
        self,
        catalog_path: str | Path,
        slot_decay: float = 1.0,
        intent_routing_scale: float = 0.0,
        retracted_weight: float = 0.0,
        constraint_lock: bool = False,
        lock_tracks: tuple[str, ...] = ("buying",),
        dynamic_truncation: bool = False,
        truncation_strong_evidence: int = 50,
        truncation_floor: int = 100,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.slot_decay = slot_decay
        self.intent_routing_scale = intent_routing_scale
        self.retracted_weight = retracted_weight
        self.constraint_lock = constraint_lock
        self.lock_tracks = lock_tracks
        self.dynamic_truncation = dynamic_truncation
        self.truncation_strong_evidence = truncation_strong_evidence
        self.truncation_floor = truncation_floor
        self.products: dict[str, Product] = {}
        self.constraint_index: dict[str, set[str]] = defaultdict(set)
        self.connection = sqlite3.connect(":memory:")
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = flatten_text(product.get("title"))
                categories = flatten_text(product.get("categories"))
                features = flatten_text(product.get("features"))
                details = flatten_text(product.get("details"))
                store = flatten_text(product.get("store"))
                description = flatten_text(product.get("description"))
                identity_text = " ".join((title, categories, store)).strip()
                attribute_text = " ".join((features, details, description)).strip()
                search_text = " ".join((identity_text, attribute_text)).strip()

                values = [
                    *flatten_values(product.get("features")),
                    *flatten_values(product.get("details")),
                ]
                material = MATERIAL_RE.search(search_text)
                color = COLOR_RE.search(search_text)
                if material:
                    values.insert(0, material.group(1).casefold())
                if color:
                    values.insert(1, f"color: {color.group(1).casefold()}")
                if product.get("price") not in (None, ""):
                    values.append(f"budget around ${product['price']}")
                atomic_values = {
                    normalize(clean_constraint(value))
                    for value in values
                    if clean_constraint(value)
                }

                self.products[parent_asin] = Product(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    category_values=[str(value) for value in product.get("categories") or []],
                    identity_text=identity_text,
                    attribute_text=attribute_text,
                    search_text=search_text.casefold(),
                    atomic_values=atomic_values,
                    average_rating=float(product.get("average_rating") or 0.0),
                    rating_number=int(product.get("rating_number") or 0),
                )
                for value in atomic_values:
                    self.constraint_index[value].add(parent_asin)

                batch.append((parent_asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def resolve_constraint_payload(self, payload: str) -> list[str]:
        """Recover up to two catalog values from the simulator's ambiguous `;` join.

        Product feature strings themselves frequently contain semicolons. Splitting on
        every semicolon turns one fabric-composition feature into several contradictory
        requirements. The public/private simulator joins at most two catalog values, so
        prefer a split where both complete sides are known catalog values.
        """
        cleaned = clean_constraint(payload)
        normalized = normalize(cleaned)
        if normalized in self.constraint_index:
            return [cleaned]

        candidates: list[tuple[tuple[float, int], list[str]]] = []
        for match in re.finditer(r";\s*", cleaned):
            left = clean_constraint(cleaned[:match.start()])
            right = clean_constraint(cleaned[match.end():])
            left_key = normalize(left)
            right_key = normalize(right)
            if left_key not in self.constraint_index or right_key not in self.constraint_index:
                continue
            left_frequency = len(self.constraint_index[left_key])
            right_frequency = len(self.constraint_index[right_key])
            rarity = -math.log1p(left_frequency) - math.log1p(right_frequency)
            candidates.append(((rarity, len(left) + len(right)), [left, right]))

        if not candidates:
            return []
        # A rare pair is less likely to be an accidental split inside a feature.
        return max(candidates, key=lambda item: item[0])[1]

    def _fts(
        self,
        query_terms: list[str],
        operator: str,
        limit: int,
        weights: tuple[float, float, float, float, float, float, float],
    ) -> list[str]:
        if not query_terms:
            return []
        expression = f" {operator} ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in query_terms)
        weight_sql = ", ".join(str(float(value)) for value in weights)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weight_sql}) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _truncation_scale(self, evidence: int, intent_mode: str):
        """Return a function scaling each route's base cut to evidence strength."""
        if not self.dynamic_truncation:
            return lambda base: base
        if evidence == 0:
            factor = 1.5
        elif evidence >= self.truncation_strong_evidence:
            factor = 0.5
        else:
            factor = 1.0
        if intent_mode == "buying":
            factor *= 0.85
        elif intent_mode == "browsing":
            factor *= 1.15
        return lambda base: max(self.truncation_floor, int(base * factor))

    def retrieve(self, state: SessionState, limit: int = 10) -> list[str]:
        ranked, _ = self.retrieve_with_diagnostics(state, limit=limit)
        return ranked

    def retrieve_with_diagnostics(
        self,
        state: SessionState,
        limit: int = 10,
    ) -> tuple[list[str], dict]:
        active = state.active_constraints
        active_values = [normalize(item.value) for item in active]
        indexed = [
            (item, self.constraint_index[value])
            for item, value in zip(active, active_values)
            if value in self.constraint_index
        ]
        indexed_sets = [identifiers for _, identifiers in indexed]

        # A retracted preference is weak evidence, not counter-evidence: the
        # customer stopped requiring it, they did not say the product lacks it.
        # It contributes to the exact-match signal at a reduced weight but never
        # to the intersection or the coverage denominator, so it can only break
        # ties -- it cannot make a product look like it satisfies a live slot.
        retracted = []
        if self.retracted_weight > 0.0:
            retracted = [
                (item, self.constraint_index[normalize(item.value)])
                for item in state.superseded_constraints
                if normalize(item.value) in self.constraint_index
            ]

        # Slot decay: a constraint stated later is worth more than an earlier
        # one, so a customer who changes direction is followed rather than
        # averaged against their opening statement. decay == 1.0 weights every
        # slot equally, which is the pre-decay behaviour.
        latest_turn = max((item.turn for item, _ in indexed + retracted), default=0)
        exact_counts: Counter[str] = Counter()
        for item, identifiers in indexed:
            weight = self.slot_decay ** max(0, latest_turn - item.turn)
            for parent_asin in identifiers:
                exact_counts[parent_asin] += weight
        for item, identifiers in retracted:
            weight = self.retracted_weight * self.slot_decay ** max(0, latest_turn - item.turn)
            for parent_asin in identifiers:
                exact_counts[parent_asin] += weight

        intersection: set[str] = set()
        if indexed_sets:
            ordered_sets = sorted(indexed_sets, key=len)
            intersection = set(ordered_sets[0])
            for identifiers in ordered_sets[1:]:
                intersection.intersection_update(identifiers)

        constraint_text = " ".join(item.value for item in state.active_constraints)
        query_terms = terms(" ".join((state.category or "", constraint_text)), limit=48)
        identity_terms = terms(state.category or "", limit=16) + terms(constraint_text, limit=24)
        identity_terms = list(dict.fromkeys(identity_terms))
        attribute_terms = terms(constraint_text, limit=40)
        strict_terms = query_terms[:12]

        balanced_weights = (0.0, 7.0, 6.0, 4.0, 3.0, 1.5, 1.0)
        identity_weights = (0.0, 9.0, 7.0, 0.5, 0.5, 3.0, 0.25)
        attribute_weights = (0.0, 1.0, 1.0, 8.0, 7.0, 0.5, 3.0)
        # Custom dynamic truncation: cut each route to the strength of the
        # evidence rather than to a constant. A large exact-match intersection
        # means the constraints already identify the product, so a deep tail only
        # adds noise; no intersection at all means the tail is the only place the
        # target can be, so widen. Buying tightens further, browsing loosens --
        # the same precision/recall split the tracks apply to weighting.
        depth = self._truncation_scale(len(intersection), state.intent_mode)
        strict_ranked = self._fts(strict_terms, "AND", depth(300), balanced_weights)
        identity_ranked = self._fts(identity_terms, "OR", depth(600), identity_weights)
        attribute_ranked = self._fts(attribute_terms, "OR", depth(600), attribute_weights)
        broad_ranked = self._fts(query_terms, "OR", depth(800), balanced_weights)

        category_terms = set(terms(state.category or "", limit=12))
        def category_coverage(parent_asin: str) -> float:
            if not category_terms:
                return 0.0
            product = self.products[parent_asin]
            return sum(term in product.identity_text.casefold() for term in category_terms) / len(category_terms)

        exact_ranked = sorted(
            exact_counts,
            key=lambda parent_asin: (
                exact_counts[parent_asin],
                parent_asin in intersection,
                category_coverage(parent_asin),
                parent_asin,
            ),
            reverse=True,
        )[:3000]
        intersection_ranked = sorted(
            intersection,
            key=lambda parent_asin: (category_coverage(parent_asin), parent_asin),
            reverse=True,
        )[:3000]

        routes: list[tuple[str, float, list[str]]] = [
            ("exact", 2.0, exact_ranked),
            ("intersection", 2.0, intersection_ranked),
            ("strict_bm25", 2.5, strict_ranked),
            ("identity_bm25", 1.5, identity_ranked),
            ("attribute_bm25", 1.5, attribute_ranked),
            ("broad_bm25", 0.5, broad_ranked),
        ]
        candidates: set[str] = set()
        for _, _, ranking in routes:
            candidates.update(ranking)
        if not candidates:
            return [], {
                "exact_candidate_count": 0,
                "intersection_candidate_count": 0,
                "bm25_and_candidate_count": 0,
                "bm25_or_candidate_count": 0,
                "fused_candidate_count": 0,
                "fused_top1_score": 0.0,
                "fused_top2_score": 0.0,
                "fused_margin": 0.0,
                "fused_relative_margin": 0.0,
                "query_terms": query_terms,
                "retrieval_routes": {},
            }

        # Keep the proven public-set scorer as one strong route while moving all
        # cross-route combination to rank space. This avoids a sudden relevance
        # regression and gives dense/reranker routes a clean place to plug in.
        route_bonus: Counter[str] = Counter()
        for rank, parent_asin in enumerate(strict_ranked, start=1):
            route_bonus[parent_asin] += 60.0 / (60.0 + rank)
        for rank, parent_asin in enumerate(broad_ranked, start=1):
            route_bonus[parent_asin] += 60.0 / (60.0 + rank)
        profile_terms = set(terms(" ".join(state.profile_preferences), limit=12))
        avoidance_terms = set(terms(" ".join(state.profile_avoidances), limit=12))
        scale = self.intent_routing_scale
        # An override rewrites the requirements mid-session, so the evidence is
        # fresher and thinner than the intent label suggests. Those sessions fall
        # back to the neutral track rather than inheriting buying's aggressive
        # hard-constraint emphasis.
        track = state.intent_mode if state.intent_mode in INTENT_EMPHASIS else "uncertain"
        if state.override_seen:
            track = "uncertain"
        w_exact, w_cover, w_inter, w_route, w_category = (
            blend(m, scale) for m in INTENT_EMPHASIS[track]
        )

        def legacy_score(parent_asin: str) -> tuple[float, str]:
            product = self.products[parent_asin]
            exact = exact_counts[parent_asin]
            exact_coverage = exact / max(1, len(indexed_sets))
            profile_coverage = (
                sum(term in product.search_text for term in profile_terms) / len(profile_terms)
                if profile_terms else 0.0
            )
            avoidance_coverage = (
                sum(term in product.search_text for term in avoidance_terms) / len(avoidance_terms)
                if avoidance_terms else 0.0
            )
            quality_prior = 0.20 * math.log1p(product.rating_number) + 0.03 * product.average_rating
            value = (
                8.0 * w_exact * exact
                + 5.0 * w_cover * exact_coverage
                + (4.0 * w_inter if parent_asin in intersection else 0.0)
                + 2.0 * w_route * route_bonus[parent_asin]
                + 1.5 * w_category * category_coverage(parent_asin)
                + 0.15 * profile_coverage
                - 0.25 * avoidance_coverage
                + quality_prior
            )
            return value, parent_asin

        legacy_ranked = sorted(candidates, key=legacy_score, reverse=True)
        routes.insert(0, ("legacy", 100.0, legacy_ranked))
        route_emphasis = INTENT_ROUTE_EMPHASIS[track]
        routes = [
            (name, weight * blend(route_emphasis.get(name, 1.0), scale), ranking)
            for name, weight, ranking in routes
        ]

        rrf_scores: Counter[str] = Counter()
        for _, weight, ranking in routes:
            for rank, parent_asin in enumerate(ranking, start=1):
                rrf_scores[parent_asin] += weight / (60.0 + rank)

        def score(parent_asin: str) -> tuple[float, float, str]:
            return (
                rrf_scores[parent_asin],
                category_coverage(parent_asin),
                parent_asin,
            )

        ranked = sorted(candidates, key=score, reverse=True)

        # High-precision filter track. On the buying track a disclosed constraint
        # is a hard requirement, so products satisfying every one of them are
        # locked above products satisfying only some. Strict precedence rather
        # than deletion: the non-matching tail stays below, so exhaustion
        # rotation and facet exploration keep something to work with, and a
        # target outside the intersection is demoted rather than lost.
        locked = False
        if self.constraint_lock and intersection and track in self.lock_tracks:
            satisfying = [item for item in ranked if item in intersection]
            if satisfying:
                ranked = satisfying + [item for item in ranked if item not in intersection]
                locked = True

        # Separation between the best and second-best fused candidate. A wide
        # margin means one product dominates the evidence; a narrow one means the
        # disclosed constraints do not yet distinguish the leaders.
        top1 = rrf_scores[ranked[0]] if ranked else 0.0
        top2 = rrf_scores[ranked[1]] if len(ranked) > 1 else 0.0
        return ranked[:limit], {
            "constraint_lock_active": locked,
            "constraint_lock_size": len(intersection) if locked else 0,
            "exact_candidate_count": len(exact_counts),
            "intersection_candidate_count": len(intersection),
            "bm25_and_candidate_count": len(strict_ranked),
            "bm25_or_candidate_count": len(broad_ranked),
            "fused_candidate_count": len(candidates),
            "fused_top1_score": top1,
            "fused_top2_score": top2,
            "fused_margin": top1 - top2,
            "fused_relative_margin": (top1 - top2) / top1 if top1 else 0.0,
            "query_terms": query_terms,
            "retrieval_routes": {name: len(ranking) for name, _, ranking in routes},
        }
