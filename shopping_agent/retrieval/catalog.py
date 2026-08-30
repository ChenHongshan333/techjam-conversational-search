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

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
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

    def retrieve(self, state: SessionState, limit: int = 10) -> list[str]:
        ranked, _ = self.retrieve_with_diagnostics(state, limit=limit)
        return ranked

    def retrieve_with_diagnostics(
        self,
        state: SessionState,
        limit: int = 10,
    ) -> tuple[list[str], dict]:
        active_values = [normalize(item.value) for item in state.active_constraints]
        indexed_sets = [self.constraint_index[value] for value in active_values if value in self.constraint_index]

        exact_counts: Counter[str] = Counter()
        for identifiers in indexed_sets:
            exact_counts.update(identifiers)

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
        strict_ranked = self._fts(strict_terms, "AND", 300, balanced_weights)
        identity_ranked = self._fts(identity_terms, "OR", 600, identity_weights)
        attribute_ranked = self._fts(attribute_terms, "OR", 600, attribute_weights)
        broad_ranked = self._fts(query_terms, "OR", 800, balanced_weights)

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
        profile_terms = set(terms(" ".join(state.user_profile.get("preference_tags") or []), limit=12))
        def legacy_score(parent_asin: str) -> tuple[float, str]:
            product = self.products[parent_asin]
            exact = exact_counts[parent_asin]
            exact_coverage = exact / max(1, len(indexed_sets))
            profile_coverage = (
                sum(term in product.search_text for term in profile_terms) / len(profile_terms)
                if profile_terms else 0.0
            )
            quality_prior = 0.20 * math.log1p(product.rating_number) + 0.03 * product.average_rating
            value = (
                8.0 * exact
                + 5.0 * exact_coverage
                + (4.0 if parent_asin in intersection else 0.0)
                + 2.0 * route_bonus[parent_asin]
                + 1.5 * category_coverage(parent_asin)
                + 0.15 * profile_coverage
                + quality_prior
            )
            return value, parent_asin

        legacy_ranked = sorted(candidates, key=legacy_score, reverse=True)
        routes.insert(0, ("legacy", 100.0, legacy_ranked))

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
        return ranked[:limit], {
            "exact_candidate_count": len(exact_counts),
            "intersection_candidate_count": len(intersection),
            "bm25_and_candidate_count": len(strict_ranked),
            "bm25_or_candidate_count": len(broad_ranked),
            "fused_candidate_count": len(candidates),
            "query_terms": query_terms,
            "retrieval_routes": {name: len(ranking) for name, _, ranking in routes},
        }
