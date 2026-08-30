from __future__ import annotations

import json
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from shopping_agent.agent import ShoppingAgent

from .session_runner import ReplaySession


class DashboardService:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.dataset_path = Path(dataset_path)
        self.samples = load_jsonl(self.dataset_path)
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in self.samples}
        self.agent = ShoppingAgent(self.catalog_path)
        self.catalog_ids = set(self.agent.catalog.products)
        self.categories = {
            parent_asin: product.category_values
            for parent_asin, product in self.agent.catalog.products.items()
        }
        self.target_products = self._load_target_products()
        self.product_views = {
            parent_asin: {
                "parent_asin": product.parent_asin,
                "title": product.title,
            }
            for parent_asin, product in self.agent.catalog.products.items()
        }
        self.product_views.update(self.target_products)
        self.sessions: dict[str, ReplaySession] = {}

    def _load_target_products(self) -> dict[str, dict]:
        target_ids = {
            str(sample["ground_truth"]["parent_asin"])
            for sample in self.samples
        }
        products: dict[str, dict] = {}
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                if parent_asin in target_ids:
                    products[parent_asin] = product
        missing = target_ids - set(products)
        if missing:
            raise ValueError(f"Catalog is missing {len(missing)} public target products")
        return products

    def list_test_cases(
        self,
        scenario: str | None = None,
        difficulty: str | None = None,
        query: str | None = None,
    ) -> list[dict]:
        query_value = (query or "").casefold().strip()
        rows: list[dict] = []
        for sample in self.samples:
            if scenario and sample["scenario_type"] != scenario:
                continue
            if difficulty and sample.get("difficulty_bucket") != difficulty:
                continue
            target = str(sample["ground_truth"]["parent_asin"])
            title = str(self.target_products[target].get("title") or target)
            if query_value and query_value not in f"{sample['sample_id']} {title}".casefold():
                continue
            rows.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "difficulty_bucket": sample.get("difficulty_bucket"),
                "category_bucket": sample.get("category_bucket"),
                "target_title": title,
            })
        return rows

    def create_session(self, sample_id: str) -> ReplaySession:
        sample = self.samples_by_id.get(sample_id)
        if sample is None:
            raise KeyError(f"Unknown test case: {sample_id}")
        session = ReplaySession(
            sample=sample,
            agent=self.agent,
            catalog_ids=self.catalog_ids,
            categories=self.categories,
            products=self.product_views,
        )
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ReplaySession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown replay session: {session_id}")
        return session
