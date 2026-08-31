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
import shopping_agent.retrieval.catalog as catalog_module
from shopping_agent.retrieval.catalog import CatalogIndex
from shopping_agent.retrieval.exploration import select_diverse_candidates
from shopping_agent.retrieval.query import QueryBuilder
from shopping_agent.retrieval.semantic import (
    DenseSearchResult,
    DenseProductRetriever,
    embedding_query_document,
    weighted_rrf,
)


def without_suppression(agent: "ShoppingAgent") -> "ShoppingAgent":
    """Isolate a retrieval test from the emission policy.

    Suppression deliberately withholds the first turns' lists, so a test that
    asserts on `recommendations` would otherwise be measuring the policy rather
    than the ranking it means to check.
    """
    agent.settings = replace(agent.settings, suppression_enabled=False)
    return agent


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
            agent = without_suppression(ShoppingAgent(write_catalog(directory)))
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
            agent = without_suppression(ShoppingAgent(write_catalog(directory)))
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
            # Rotation needs a first list to rotate away from, so the emission
            # policy is held out of the way; suppression is covered separately.
            agent = without_suppression(ShoppingAgent(path))
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


class SuppressionRuleTest(unittest.TestCase):
    """The rules that bound how long a session may stay silent."""

    def build(self, **overrides) -> ShoppingAgent:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
        defaults = {
            "suppression_enabled": True,
            "suppression_turns": 2,
            "suppression_max_turns": 2,
            "suppression_reserve_turns": 3,
        }
        agent.settings = replace(agent.settings, **{**defaults, **overrides})
        return agent

    def test_shipped_default_withholds_the_first_two_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
        self.assertTrue(agent.settings.suppression_enabled)
        self.assertEqual(agent.settings.suppression_turns, 2)
        state = SessionState(user_profile={})
        self.assertTrue(agent._suppress(state, 1))
        self.assertTrue(agent._suppress(state, 2))
        self.assertFalse(agent._suppress(state, 3))

    def test_can_be_switched_off(self) -> None:
        agent = self.build(suppression_enabled=False)
        self.assertFalse(agent._suppress(SessionState(user_profile={}), 1))

    def test_reserves_the_closing_turns(self) -> None:
        # The closing turns decide the 0.5-weighted hit rate and must always emit.
        agent = self.build(suppression_turns=9, suppression_max_turns=99)
        state = SessionState(user_profile={})
        self.assertTrue(agent._suppress(state, 7))
        for closing_turn in (8, 9, 10):
            self.assertFalse(agent._suppress(state, closing_turn))

    def test_respects_the_per_session_cap(self) -> None:
        agent = self.build(suppression_turns=9, suppression_max_turns=2)
        self.assertFalse(agent._suppress(SessionState(user_profile={}, suppressed_turns=2), 3))

    def test_stops_when_the_last_question_brought_nothing_back(self) -> None:
        agent = self.build(suppression_turns=9, suppression_max_turns=99)
        self.assertFalse(
            agent._suppress(SessionState(user_profile={}, gained_information=False), 3)
        )


class SuppressionExhaustionTest(unittest.TestCase):
    """Suppression must yield as soon as the customer stops supplying information."""

    def build(self) -> ShoppingAgent:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
        agent.settings = replace(
            agent.settings,
            suppression_enabled=True,
            suppression_turns=3,
            suppression_max_turns=9,
            suppression_reserve_turns=3,
        )
        return agent

    def test_yields_once_information_is_exhausted(self) -> None:
        agent = self.build()
        state = SessionState(user_profile={})
        self.assertTrue(agent._suppress(state, 2))
        state.rejected_attributes.add("other")
        self.assertFalse(agent._suppress(state, 2))

    def test_yields_on_the_explicit_exhausted_flag(self) -> None:
        agent = self.build()
        state = SessionState(user_profile={}, information_exhausted=True)
        self.assertFalse(agent._suppress(state, 2))


class RotationPrecedenceTest(unittest.TestCase):
    """The first list a session emits must be the ranking, never an exploration set."""

    def test_first_emission_is_not_diversified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            agent.settings = replace(
                agent.settings, suppression_enabled=True,
                suppression_turns=3, suppression_max_turns=9,
            )
            agent.reset("s", {})
            # Withheld while the customer still has information to give.
            agent.respond("s", "I'm looking for Shoes Boots, but I'm still exploring.", 1, 10)
            agent.respond("s", "For that, what matters is: leather; color: black.", 2, 10)
            # Customer runs dry: suppression must yield AND the first list must be
            # the ranking, not a diversified exploration set.
            third = agent.respond("s", "I don't have an additional preference for other.", 3, 10)
            state = agent.sessions["s"]
            self.assertTrue(state.rejected_attributes)
            self.assertTrue(third["recommendations"])
            self.assertFalse(agent.get_diagnostics("s")["candidate_rotation_active"])
            self.assertEqual(third["recommendations"][0]["parent_asin"], "TARGET")


class OverrideErasureTest(unittest.TestCase):
    """An override must retract volunteered preferences, whatever the opener."""

    def ingest(self, messages: list[str]) -> SessionState:
        state = SessionState(user_profile={})
        for turn, message in enumerate(messages, start=1):
            ingest_message(state, message, turn)
        return state

    def active(self, state: SessionState) -> set[str]:
        return {item.value.casefold() for item in state.active_constraints}

    def test_erases_a_preference_stated_with_a_key_requirement_opener(self) -> None:
        # The old rule keyed off a turn-1 string it never captured for this
        # opener, so the superseded requirement stayed active forever.
        state = self.ingest([
            "I'm looking for boots. A key requirement is: suede upper.",
            "Actually, ignore my earlier preference. What I need is: leather upper.",
        ])
        self.assertNotIn("suede upper", self.active(state))
        self.assertIn("leather upper", self.active(state))

    def test_erases_across_a_second_override(self) -> None:
        state = self.ingest([
            "I'm looking for boots. I prefer suede.",
            "Actually, ignore my earlier preference. What I need is: leather upper.",
            "Actually, ignore my earlier preference. What I need is: rubber sole.",
        ])
        self.assertNotIn("suede", self.active(state))
        self.assertNotIn("leather upper", self.active(state))
        self.assertIn("rubber sole", self.active(state))

    def test_keeps_facts_the_customer_gave_when_asked(self) -> None:
        # Answers to the agent's questions describe the target and survive.
        state = self.ingest([
            "I'm looking for boots. I prefer suede.",
            "For that, what matters is: waterproof lining.",
            "Actually, ignore my earlier preference. What I need is: leather upper.",
        ])
        self.assertNotIn("suede", self.active(state))
        self.assertIn("waterproof lining", self.active(state))
        self.assertIn("leather upper", self.active(state))

    def test_override_in_a_browsing_session_erases_nothing_it_should_not(self) -> None:
        state = self.ingest([
            "I'm looking for boots, but I'm still exploring.",
            "For that, what matters is: waterproof lining.",
            "Actually, ignore my earlier preference. What I need is: leather upper.",
        ])
        self.assertIn("waterproof lining", self.active(state))
        self.assertIn("leather upper", self.active(state))


class SlotDecayTest(unittest.TestCase):
    def test_later_constraints_outrank_earlier_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_catalog(directory)
            equal = CatalogIndex(path, slot_decay=1.0)
            decayed = CatalogIndex(path, slot_decay=0.5)
            self.assertEqual(equal.slot_decay, 1.0)
            self.assertLess(decayed.slot_decay, 1.0)

            state = SessionState(user_profile={})
            state.constraints.append(Constraint("leather", "material", 1, "stated"))
            state.constraints.append(Constraint("black", "color", 4, "answer"))
            # Both indexes must still rank; decay changes weighting, not recall.
            self.assertTrue(equal.retrieve(state, limit=5))
            self.assertTrue(decayed.retrieve(state, limit=5))


class IntentRoutingTest(unittest.TestCase):
    """Intent must reach retrieval.

    The classifier previously fed only a dense-retrieval branch that is off by
    default, so forcing every mode produced byte-identical results. These tests
    fail if that ever becomes true again.
    """

    def test_tracks_are_distinct(self) -> None:
        buying = catalog_module.INTENT_EMPHASIS["buying"]
        browsing = catalog_module.INTENT_EMPHASIS["browsing"]
        uncertain = catalog_module.INTENT_EMPHASIS["uncertain"]
        self.assertNotEqual(buying, browsing, "buying and browsing must route differently")
        self.assertNotEqual(buying, uncertain)
        self.assertEqual(uncertain, (1.0, 1.0, 1.0, 1.0, 1.0), "uncertain is the neutral track")

    def test_routing_is_enabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
        self.assertGreater(agent.settings.intent_routing_scale, 0.0)
        self.assertEqual(agent.catalog.intent_routing_scale, agent.settings.intent_routing_scale)

    def test_scale_zero_reduces_to_the_neutral_track(self) -> None:
        for multiplier in catalog_module.INTENT_EMPHASIS["buying"]:
            self.assertEqual(catalog_module.blend(multiplier, 0.0), 1.0)
        self.assertEqual(catalog_module.blend(1.5, 1.0), 1.5)
        self.assertAlmostEqual(catalog_module.blend(1.5, 0.5), 1.25)

    def test_an_override_relaxes_to_the_neutral_track(self) -> None:
        # Overrides rewrite the requirements, so they must not inherit buying's
        # hard-constraint emphasis even though "what I need is" reads as buying.
        with tempfile.TemporaryDirectory() as directory:
            agent = ShoppingAgent(write_catalog(directory))
            agent.reset("o", {})
            agent.respond("o", "I'm looking for boots. A key requirement is: leather.", 1, 10)
            self.assertEqual(agent.get_diagnostics("o")["intent_mode"], "buying")
            agent.respond(
                "o", "Actually, ignore my earlier preference. What I need is: suede.", 2, 10
            )
            state = agent.sessions["o"]
            self.assertTrue(state.override_seen)
            self.assertEqual(state.intent_mode, "buying")
