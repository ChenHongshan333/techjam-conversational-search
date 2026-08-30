from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from shopping_agent import ShoppingAgent
from shopping_agent.config import RetrievalSettings
from shopping_agent.conversation.answers import HybridAnswerInterpreter
from shopping_agent.conversation.intent import HybridIntentClassifier
from shopping_agent.conversation.parser import parse_message
from shopping_agent.conversation.profile import distill_user_profile
from shopping_agent.conversation.questions import ClarificationPolicy
from shopping_agent.conversation.state import ingest_message
from shopping_agent.models import Constraint, SessionState
from shopping_agent.providers.openrouter import OpenRouterError, OpenRouterResponse
from shopping_agent.retrieval.catalog import CatalogIndex
from shopping_agent.retrieval.exploration import select_diverse_candidates
from shopping_agent.retrieval.query import QueryBuilder
from shopping_agent.retrieval.semantic import (
    DenseSearchResult,
    DenseProductRetriever,
    embedding_query_document,
    weighted_rrf,
)


def write_catalog(directory: str) -> Path:
    path = Path(directory) / "catalog.jsonl"
    products = [
        {
            "parent_asin": "TARGET",
            "title": "Women's black winter boot",
            "features": ["Genuine leather", "Warm fleece lining"],
            "details": {"Color": "Black", "Department": "Womens"},
            "description": ["Comfortable outdoor boot"],
            "categories": ["Clothing", "Shoes", "Boots"],
            "store": "Example",
            "price": 89.0,
        },
        {
            "parent_asin": "OTHER",
            "title": "Women's blue running shoe",
            "features": ["Breathable fabric", "Rubber sole"],
            "details": {"Color": "Blue", "Department": "Womens"},
            "description": ["Lightweight gym shoe"],
            "categories": ["Clothing", "Shoes", "Athletic"],
            "store": "Example",
            "price": 49.0,
        },
    ]
    path.write_text("".join(json.dumps(product) + "\n" for product in products), encoding="utf-8")
    return path


class ParserTest(unittest.TestCase):
    def test_parses_buying_message(self) -> None:
        parsed = parse_message("I'm looking for Shoes Boots. A key requirement is: leather.")
        self.assertEqual(parsed.category, "Shoes Boots")
        self.assertEqual(parsed.constraints, ["leather"])

    def test_parses_multiple_revealed_constraints(self) -> None:
        parsed = parse_message("For that, what matters is: leather; color: black.")
        self.assertEqual(parsed.constraints, ["leather", "color: black"])

    def test_parses_boundary_without_exhausting_attribute(self) -> None:
        parsed = parse_message("I don't have a preference for other; please use your judgment.")
        self.assertTrue(parsed.boundary_response)
        self.assertEqual(parsed.rejected_attribute, "other")


class StateTest(unittest.TestCase):
    def test_override_deactivates_initial_preference_only(self) -> None:
        state = SessionState(user_profile={})
        ingest_message(state, "I'm looking for Shoes Boots. Warm fleece lining", turn=1)
        ingest_message(state, "For that, what matters is: leather; color: black.", turn=2)
        ingest_message(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
            turn=3,
        )
        active = {item.value for item in state.active_constraints}
        superseded = {item.value for item in state.superseded_constraints}
        self.assertEqual(active, {"leather", "color: black"})
        self.assertEqual(superseded, {"Warm fleece lining"})

    def test_marks_information_exhausted_after_other_has_no_answer(self) -> None:
        state = SessionState(user_profile={})
        ingest_message(
            state,
            "I don't have an additional preference for other.",
            turn=2,
        )
        self.assertTrue(state.information_exhausted)


class FakeIntentClient:
    def classify_shopping_intent(self, model: str, message: str) -> OpenRouterResponse:
        return OpenRouterResponse(
            payload={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "intent": "buying",
                            "confidence": 0.86,
                            "reason": "concrete upcoming need",
                        })
                    }
                }]
            },
            prompt_tokens=12,
            completion_tokens=8,
        )


class IntentTest(unittest.TestCase):
    def test_rules_handle_fixed_evaluator_language(self) -> None:
        classifier = HybridIntentClassifier()
        buying = classifier.classify(
            "I'm looking for Shoes. A key requirement is: leather.", 1
        )
        browsing = classifier.classify(
            "I'm looking for Shoes, but I'm still exploring.", 1
        )
        self.assertEqual((buying.mode, buying.source), ("buying", "rule"))
        self.assertEqual((browsing.mode, browsing.source), ("browsing", "rule"))

    def test_unclear_language_uses_optional_llm(self) -> None:
        classifier = HybridIntentClassifier(FakeIntentClient(), "fake-model")
        decision = classifier.classify("I have a trip coming up and need some ideas.", 1)
        self.assertEqual(decision.mode, "buying")
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.prompt_tokens, 12)


class FakeAnswerClient:
    def extract_shopping_answer(
        self, model: str, message: str, current_category: str, question_focus: str
    ) -> OpenRouterResponse:
        return OpenRouterResponse(
            payload={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "category": "",
                            "constraints": ["Grandma text", "long sleeve"],
                            "rejected_attribute": "",
                            "override": False,
                            "confidence": 0.91,
                        })
                    }
                }]
            },
            prompt_tokens=14,
            completion_tokens=9,
        )


class AnswerInterpreterTest(unittest.TestCase):
    def test_natural_first_turn_extracts_concrete_request(self) -> None:
        decision = HybridAnswerInterpreter().interpret(
            "I need black hiking shoes with a wide fit.",
            turn=1,
        )
        self.assertEqual(decision.source, "natural_rule")
        self.assertTrue(any("hiking shoes" in item for item in decision.parsed.constraints))

    def test_natural_follow_up_extracts_searchable_constraints(self) -> None:
        decision = HybridAnswerInterpreter().interpret(
            "I'd like Grandma text, long sleeves, and a Mother's Day gift.",
            turn=2,
            current_category="Women Novelty",
            question_focus="style",
        )
        self.assertEqual(decision.source, "natural_rule")
        self.assertIn("Grandma text", decision.parsed.constraints)
        self.assertTrue(any("long sleeve" in item for item in decision.parsed.constraints))

    def test_unclear_natural_answer_uses_optional_llm(self) -> None:
        interpreter = HybridAnswerInterpreter(FakeAnswerClient(), "fake-model")
        decision = interpreter.interpret(
            "It is for someone very important to me.",
            turn=1,
            question_focus="use_case",
        )
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.parsed.constraints, ["Grandma text", "long sleeve"])
        self.assertEqual(decision.prompt_tokens, 14)


class ProfileTest(unittest.TestCase):
    def test_dimensions_are_not_treated_as_concrete_preferences(self) -> None:
        signals = distill_user_profile({
            "preference_tags": ["material", "fit"],
            "explicit_preferences": ["cotton", "relaxed fit"],
        })
        self.assertEqual(signals.important_dimensions, ("material", "fit"))
        self.assertEqual(signals.positive_preferences, ("cotton", "relaxed fit"))


class ClarificationPolicyTest(unittest.TestCase):
    def test_profile_and_known_constraints_produce_focused_compatible_question(self) -> None:
        state = SessionState(
            user_profile={"preference_tags": ["material", "fit"]},
            intent_mode="buying",
        )
        state.constraints.append(Constraint("cotton", "material", 1, "user"))
        plan = ClarificationPolicy().plan(state, 1, [])
        self.assertEqual(plan.ask_attribute, "other")
        self.assertNotEqual(plan.focus_attribute, "material")
        self.assertTrue(plan.topics)

    def test_natural_conversation_can_use_specific_attribute(self) -> None:
        state = SessionState(
            user_profile={"preference_tags": ["fit"]},
            profile_dimensions=["fit"],
            intent_mode="buying",
            last_answer_source="natural_rule",
        )
        plan = ClarificationPolicy().plan(state, 2, [])
        self.assertEqual(plan.ask_attribute, "style")
        self.assertFalse(plan.protocol_fallback)

    def test_exhausted_state_stops_asking(self) -> None:
        state = SessionState(user_profile={}, information_exhausted=True)
        plan = ClarificationPolicy().plan(state, 3, [])
        self.assertIsNone(plan.ask_attribute)
        self.assertTrue(plan.information_exhausted)


class QueryTest(unittest.TestCase):
    def test_deterministic_query_keeps_fields_and_source_values(self) -> None:
        state = SessionState(user_profile={}, category="Women Hoodies")
        state.constraints.extend([
            Constraint("color: grey", "color", 1, "user"),
            Constraint("80% Cotton, 20% Polyester", "material", 2, "user"),
        ])
        query = QueryBuilder().build(state)
        self.assertIn("Product type: Women Hoodies", query.semantic_query)
        self.assertIn("80% Cotton, 20% Polyester", query.semantic_query)
        self.assertFalse(query.rewrite_used)

    def test_only_explicit_profile_values_enter_query(self) -> None:
        state = SessionState(
            user_profile={"preference_tags": ["material"]},
            category="Women Hoodies",
            profile_dimensions=["material"],
            profile_preferences=["organic cotton"],
        )
        query = QueryBuilder().build(state)
        self.assertIn("organic cotton", query.semantic_query)
        self.assertNotIn("Historical preferences (soft): material", query.semantic_query)

    def test_weighted_rrf_rewards_agreement(self) -> None:
        result = weighted_rrf([(1.0, ["A", "B"]), (1.0, ["B", "C"])])
        self.assertEqual(result[0], "B")

    def test_qwen_query_uses_retrieval_instruction(self) -> None:
        value = embedding_query_document("qwen/qwen3-embedding-8b", "red cotton hoodie")
        self.assertIn("Instruct:", value)
        self.assertTrue(value.endswith("Query: red cotton hoodie"))


class ExplorationTest(unittest.TestCase):
    def test_facet_exploration_promotes_a_deep_distinct_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            products = [
                {
                    "parent_asin": f"GENERIC{i:02d}",
                    "title": f"Generic cotton shirt {i}",
                    "features": ["cotton"],
                    "categories": ["Women", "Novelty"],
                }
                for i in range(20)
            ]
            products.append({
                "parent_asin": "TARGET",
                "title": "Grandma long sleeve gift shirt",
                "features": ["cotton"],
                "categories": ["Women", "Novelty"],
            })
            path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            catalog = CatalogIndex(path)
            ranked = [product["parent_asin"] for product in products]
            selection = select_diverse_candidates(
                ranked,
                catalog.products,
                SessionState(user_profile={}),
                limit=10,
            )
            self.assertIn("TARGET", selection.identifiers)
            self.assertIn("style:long_sleeve", selection.facets)


class FakeEmbeddingClient:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embeddings(self, model: str, texts: list[str], dimensions: int, input_type: str):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise OpenRouterError("simulated provider failure")
        data = []
        for index, text in enumerate(texts):
            vector = [0.0] * dimensions
            vector[(len(text) + index) % dimensions] = 1.0
            data.append({"index": index, "embedding": vector})
        return OpenRouterResponse(payload={"data": data}, prompt_tokens=len(texts))


class FakeDenseRetriever:
    def search(self, query: str, limit: int = 500) -> DenseSearchResult:
        return DenseSearchResult(
            identity_ranking=["TARGET", "OTHER"],
            attribute_ranking=["OTHER", "TARGET"],
        )


class SemanticIndexTest(unittest.TestCase):
    def settings(self, directory: str) -> RetrievalSettings:
        return replace(
            RetrievalSettings.from_environment(),
            api_key="fake",
            embedding_model="qwen/qwen3-embedding-8b",
            embedding_dimensions=32,
            embedding_batch_size=1,
            embedding_workers=1,
            embedding_job_retries=1,
            cache_directory=Path(directory) / "cache",
        )

    def test_missing_index_falls_back_without_starting_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = write_catalog(directory)
            client = FakeEmbeddingClient()
            retriever = DenseProductRetriever(
                CatalogIndex(catalog_path), catalog_path, client, self.settings(directory)
            )
            result = retriever.search("winter boot")
            self.assertIn("not built", result.error or "")
            self.assertEqual(client.calls, 0)

    def test_builder_resumes_completed_batches(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = write_catalog(directory)
            catalog = CatalogIndex(catalog_path)
            settings = self.settings(directory)
            first_client = FakeEmbeddingClient(fail_on_call=2)
            first = DenseProductRetriever(catalog, catalog_path, first_client, settings)
            with self.assertRaises(OpenRouterError):
                first.build_index()

            resumed_client = FakeEmbeddingClient()
            resumed = DenseProductRetriever(catalog, catalog_path, resumed_client, settings)
            cache_path, _ = resumed.build_index()
            self.assertTrue(cache_path.exists())
            self.assertLess(resumed_client.calls, 4)

    def test_portable_index_is_reused_when_catalog_path_changes(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional")
        with tempfile.TemporaryDirectory() as directory:
            first_directory = Path(directory) / "first"
            second_directory = Path(directory) / "second"
            first_directory.mkdir()
            second_directory.mkdir()
            first_catalog_path = write_catalog(str(first_directory))
            second_catalog_path = write_catalog(str(second_directory))
            settings = self.settings(directory)

            first_client = FakeEmbeddingClient()
            first = DenseProductRetriever(
                CatalogIndex(first_catalog_path), first_catalog_path, first_client, settings
            )
            cache_path, _ = first.build_index()
            self.assertEqual(
                cache_path.name,
                "catalog_qwen3_embedding_8b_32_v1.npz",
            )
            with np.load(cache_path, allow_pickle=False) as stored:
                self.assertEqual(int(stored["format_version"].item()), 1)
                self.assertEqual(
                    str(stored["embedding_model"].item()),
                    "qwen/qwen3-embedding-8b",
                )
                self.assertIn("catalog_fingerprint", stored.files)

            second_client = FakeEmbeddingClient()
            second = DenseProductRetriever(
                CatalogIndex(second_catalog_path), second_catalog_path, second_client, settings
            )
            reused_path, tokens = second.build_index()
            self.assertEqual(reused_path, cache_path)
            self.assertEqual(tokens, 0)
            self.assertEqual(second_client.calls, 0)

    def test_explicit_semantic_index_path_is_respected(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = write_catalog(directory)
            explicit_path = Path(directory) / "distributed" / "catalog.npz"
            settings = replace(
                self.settings(directory),
                semantic_index_path=explicit_path,
            )
            retriever = DenseProductRetriever(
                CatalogIndex(catalog_path), catalog_path, FakeEmbeddingClient(), settings
            )
            cache_path, _ = retriever.build_index()
            self.assertEqual(cache_path, explicit_path)
            self.assertTrue(explicit_path.exists())


class AgentTest(unittest.TestCase):
    def test_accumulates_constraints_and_retrieves_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            agent.reset("session", {"preference_tags": ["comfort"]})
            first = agent.respond(
                "session",
                "I'm looking for Shoes Boots, but I'm still exploring.",
                turn=1,
                top_k=10,
            )
            self.assertEqual(first["ask_attribute"], "other")

            second = agent.respond(
                "session",
                "For that, what matters is: leather; color: black.",
                turn=2,
                top_k=10,
            )
            self.assertEqual(second["recommendations"][0]["parent_asin"], "TARGET")

    def test_requires_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            with self.assertRaises(RuntimeError):
                agent.respond("missing", "hello", turn=1, top_k=10)

    def test_natural_follow_up_changes_state_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            agent.reset(
                "natural",
                {
                    "preference_tags": ["material", "fit"],
                    "explicit_preferences": ["warm fleece"],
                },
            )
            agent.respond(
                "natural",
                "I'm looking for Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )
            response = agent.respond(
                "natural",
                "I'd like a black leather winter boot.",
                turn=2,
                top_k=10,
            )
            state = agent.sessions["natural"]
            self.assertEqual(state.last_answer_source, "natural_rule")
            self.assertTrue(any("black leather" in item.value for item in state.active_constraints))
            self.assertEqual(response["recommendations"][0]["parent_asin"], "TARGET")
            self.assertIn(response["ask_attribute"], {"style", "use_case", "feature", "other"})

    def test_intent_selects_dense_fusion_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            agent.dense_retriever = FakeDenseRetriever()

            agent.reset("browse", {})
            agent.respond(
                "browse",
                "I'm looking for Shoes, but I'm still exploring.",
                turn=1,
                top_k=10,
            )
            browse_weights = agent.get_diagnostics("browse")["active_fusion_weights"]
            self.assertEqual(browse_weights, {
                "lexical": 30.0,
                "dense_identity": 2.0,
                "dense_attribute": 2.0,
            })

            agent.reset("buy", {})
            agent.respond(
                "buy",
                "I'm looking for Shoes. A key requirement is: leather.",
                turn=1,
                top_k=10,
            )
            buy_weights = agent.get_diagnostics("buy")["active_fusion_weights"]
            self.assertEqual(buy_weights, {
                "lexical": 50.0,
                "dense_identity": 1.0,
                "dense_attribute": 1.0,
            })

    def test_catalog_resolves_semicolons_inside_a_single_feature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            product = {
                "parent_asin": "TARGET",
                "title": "Grey Hoodie",
                "features": [
                    "Solid colors: 80% Cotton, 20% Polyester; Heather Grey: 78% Cotton, 22% Poly",
                    "Imported",
                ],
                "details": {},
                "categories": ["Women", "Hoodies"],
            }
            path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            agent = ShoppingAgent(path)
            payload = (
                "Solid colors: 80% Cotton, 20% Polyester; "
                "Heather Grey: 78% Cotton, 22% Poly; Imported"
            )
            self.assertEqual(
                agent.catalog.resolve_constraint_payload(payload),
                [product["features"][0], "Imported"],
            )

    def test_rotates_unseen_candidates_after_other_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            products = [
                {
                    "parent_asin": f"ITEM{i:02d}",
                    "title": f"Women Hoodie {i}",
                    "features": ["cotton"],
                    "details": {"Color": "Grey"},
                    "categories": ["Women", "Hoodies"],
                }
                for i in range(15)
            ]
            path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            agent = ShoppingAgent(path)
            agent.reset("rotation", {"preference_tags": []})
            first = agent.respond(
                "rotation",
                "I'm looking for Women Hoodies, but I'm still exploring.",
                1,
                10,
            )
            second = agent.respond(
                "rotation",
                "I don't have an additional preference for other.",
                2,
                10,
            )
            first_ids = {item["parent_asin"] for item in first["recommendations"]}
            second_ids = {item["parent_asin"] for item in second["recommendations"]}
            self.assertTrue(second_ids)
            self.assertTrue(first_ids.isdisjoint(second_ids))
            self.assertTrue(agent.get_diagnostics("rotation")["candidate_rotation_active"])
            self.assertTrue(agent.get_diagnostics("rotation")["information_exhausted"])


if __name__ == "__main__":
    unittest.main()
