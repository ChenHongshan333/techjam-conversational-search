from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dashboard.service import DashboardService
from dashboard.session_runner import ReplaySession
from shopping_agent import ShoppingAgent


PRODUCTS = [
    {
        "parent_asin": "TARGET",
        "title": "Black Leather Winter Boot",
        "features": ["leather", "Warm fleece lining"],
        "details": {"Color": "Black", "Department": "Womens"},
        "description": ["Outdoor winter boot"],
        "categories": ["Clothing", "Shoes", "Boots"],
        "store": "Example",
        "price": 89.0,
        "average_rating": 4.8,
        "rating_number": 100,
    },
    {
        "parent_asin": "OTHER",
        "title": "Blue Fabric Running Shoe",
        "features": ["fabric", "Rubber sole"],
        "details": {"Color": "Blue", "Department": "Womens"},
        "description": ["Lightweight running shoe"],
        "categories": ["Clothing", "Shoes", "Athletic"],
        "store": "Example",
        "price": 49.0,
        "average_rating": 4.0,
        "rating_number": 10,
    },
]


def write_fixture(directory: str, scenario: str = "browsing") -> tuple[Path, Path]:
    root = Path(directory)
    catalog_path = root / "catalog.jsonl"
    dataset_path = root / "public_set.jsonl"
    catalog_path.write_text(
        "".join(json.dumps(product) + "\n" for product in PRODUCTS),
        encoding="utf-8",
    )
    sample = {
        "sample_id": "public_test",
        "scenario_type": scenario,
        "difficulty_bucket": "easy",
        "category_bucket": "shoes",
        "user_profile": {
            "purchase_frequency": "3-4 prior purchases",
            "average_prior_rating": 5.0,
            "rating_style": "usually positive",
            "preference_tags": ["comfort"],
            "summary": "Prior purchases emphasize comfort.",
        },
        "ground_truth": {"parent_asin": "TARGET"},
    }
    dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return catalog_path, dataset_path


class DashboardServiceTest(unittest.TestCase):
    def test_lists_cases_and_runs_a_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, dataset_path = write_fixture(directory)
            service = DashboardService(catalog_path, dataset_path)
            cases = service.list_test_cases(scenario="browsing")
            self.assertEqual(cases[0]["sample_id"], "public_test")
            self.assertEqual(service.product_views["TARGET"]["features"][0], "leather")

            session = service.create_session("public_test")
            self.assertEqual(session.snapshot()["user_profile"]["rating_style"], "usually positive")
            event = session.step()
            self.assertEqual(event["turn"], 1)
            self.assertEqual(event["ask_attribute"], "other")
            self.assertIn("fused_candidate_count", event["diagnostics"])
            # The shipped policy shows only rank 1 while gathering evidence.
            self.assertTrue(event["diagnostics"]["gathering_turn"])
            self.assertTrue(event["diagnostics"]["recommendations_narrowed"])
            self.assertFalse(event["diagnostics"]["recommendations_suppressed"])
            self.assertEqual(len(event["recommendations"]), 1)
            self.assertEqual(event["intent"]["label"], "Exploring options")
            self.assertIn("profile", event["ranking_explanation"]["profile_note"].casefold())
            self.assertTrue(event["recommendations"][0]["explanation"]["reasons"])


class ReplaySessionTest(unittest.TestCase):
    def test_intent_override_target_is_not_scored_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path, _ = write_fixture(directory, scenario="intent_override")
            agent = ShoppingAgent(catalog_path)
            # This test is about when a hit may be scored, not about which turns
            # emit a list, so the emission policy is held out of the way.
            agent.settings = replace(agent.settings, suppression_enabled=False)
            sample = {
                "sample_id": "override_test",
                "scenario_type": "intent_override",
                "difficulty_bucket": "hard",
                "user_profile": {"preference_tags": []},
                "ground_truth": {"parent_asin": "TARGET"},
                "intent_card": {
                    "target_category": "Black Leather Winter Boot",
                    "hard_constraints": ["leather"],
                    "soft_preferences": ["Warm fleece lining"],
                },
                "behavior": {
                    "scenario_type": "intent_override",
                    "override": {
                        "turn": 3,
                        "old_value": "Warm fleece lining",
                        "new_value": "leather",
                        "message": "Actually, ignore my earlier preference. What I need is: leather.",
                    },
                },
            }
            products = {product["parent_asin"]: product for product in PRODUCTS}
            session = ReplaySession(
                sample=sample,
                agent=agent,
                catalog_ids=set(products),
                categories={key: value["categories"] for key, value in products.items()},
                products=products,
            )
            first = session.step()
            self.assertTrue(first["contains_target"])
            self.assertFalse(first["scored_hit"])
            self.assertFalse(first["finished"])

            second = session.step()
            self.assertFalse(second["scored_hit"])
            self.assertIn("Actually, ignore", second["next_user_message"])

            third = session.step()
            self.assertTrue(third["scored_hit"])
            self.assertTrue(third["finished"])


if __name__ == "__main__":
    unittest.main()
