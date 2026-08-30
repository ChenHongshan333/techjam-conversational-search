from __future__ import annotations

from pathlib import Path

from .config import RetrievalSettings
from .conversation.answers import HybridAnswerInterpreter
from .conversation.intent import HybridIntentClassifier
from .conversation.profile import distill_user_profile
from .conversation.questions import ClarificationPolicy, render_question
from .conversation.state import ingest_message
from .models import SessionState
from .providers.openrouter import OpenRouterClient
from .retrieval.catalog import CatalogIndex
from .retrieval.exploration import select_diverse_candidates
from .retrieval.query import QueryBuilder
from .retrieval.semantic import DenseProductRetriever, ProductReranker, weighted_rrf


class ShoppingAgent:
    """Stateful, offline conversational catalog retrieval agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = CatalogIndex(catalog_path)
        self.settings = RetrievalSettings.from_environment()
        client = (
            OpenRouterClient(self.settings.api_key, self.settings.request_timeout_seconds)
            if self.settings.remote_enabled else None
        )
        self.query_builder = QueryBuilder(
            client if client and self.settings.rewrite_enabled else None,
            self.settings.rewrite_model,
        )
        self.intent_classifier = HybridIntentClassifier(
            client if client and self.settings.intent_enabled else None,
            self.settings.intent_model,
            self.settings.intent_confidence_threshold,
        )
        self.answer_interpreter = HybridAnswerInterpreter(
            client if client and self.settings.answer_enabled else None,
            self.settings.answer_model,
            self.settings.answer_confidence_threshold,
        )
        self.clarification_policy = ClarificationPolicy()
        self.dense_retriever = (
            DenseProductRetriever(self.catalog, Path(catalog_path), client, self.settings)
            if client and self.settings.dense_enabled else None
        )
        self.reranker = (
            ProductReranker(self.catalog, client, self.settings)
            if client and self.settings.rerank_enabled else None
        )
        self.sessions: dict[str, SessionState] = {}
        self.diagnostics: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile = distill_user_profile(user_profile)
        self.sessions[session_id] = SessionState(
            user_profile=dict(user_profile),
            profile_dimensions=list(profile.important_dimensions),
            profile_preferences=list(profile.positive_preferences),
            profile_avoidances=list(profile.negative_preferences),
        )
        self.diagnostics.pop(session_id, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        answer = self.answer_interpreter.interpret(
            user_message,
            turn,
            state.category,
            state.last_question_focus,
        )
        ingest_message(
            state,
            user_message,
            turn,
            constraint_resolver=self.catalog.resolve_constraint_payload,
            parsed=answer.parsed,
        )
        state.last_answer_source = answer.source
        state.last_answer_confidence = answer.confidence
        intent = self.intent_classifier.classify(
            user_message,
            turn,
            state.intent_mode,
            state.intent_confidence,
        )
        state.intent_mode = intent.mode
        state.intent_confidence = intent.confidence
        state.intent_source = intent.source
        if intent.mode in {"buying", "browsing"}:
            state.browsing = intent.mode == "browsing"
        search_query = self.query_builder.build(state)
        prompt_tokens = search_query.prompt_tokens + intent.prompt_tokens + answer.prompt_tokens
        completion_tokens = (
            search_query.completion_tokens
            + intent.completion_tokens
            + answer.completion_tokens
        )

        ranked_pool, retrieval_diagnostics = self.catalog.retrieve_with_diagnostics(
            state,
            limit=max(1000, top_k),
        )
        semantic_diagnostics = {
            "semantic_query": search_query.semantic_query,
            "llm_rewrite_used": search_query.rewrite_used,
            "llm_rewrite_error": search_query.error,
            "dense_retrieval_enabled": self.dense_retriever is not None,
            "dense_retrieval_error": None,
            "dense_identity_candidate_count": 0,
            "dense_attribute_candidate_count": 0,
            "rerank_enabled": self.reranker is not None,
            "rerank_error": None,
        }
        if self.dense_retriever is not None:
            dense = self.dense_retriever.search(search_query.semantic_query, limit=600)
            prompt_tokens += dense.prompt_tokens
            semantic_diagnostics.update({
                "dense_retrieval_error": dense.error,
                "dense_identity_candidate_count": len(dense.identity_ranking),
                "dense_attribute_candidate_count": len(dense.attribute_ranking),
            })
            if dense.identity_ranking or dense.attribute_ranking:
                if state.intent_mode == "browsing":
                    fusion_weights = (30.0, 2.0, 2.0)
                elif state.intent_mode == "buying":
                    fusion_weights = (
                        self.settings.lexical_fusion_weight,
                        self.settings.dense_identity_fusion_weight,
                        self.settings.dense_attribute_fusion_weight,
                    )
                else:
                    fusion_weights = (40.0, 1.0, 1.0)
                ranked_pool = weighted_rrf([
                    (fusion_weights[0], ranked_pool),
                    (fusion_weights[1], dense.identity_ranking),
                    (fusion_weights[2], dense.attribute_ranking),
                ])
                semantic_diagnostics["active_fusion_weights"] = {
                    "lexical": fusion_weights[0],
                    "dense_identity": fusion_weights[1],
                    "dense_attribute": fusion_weights[2],
                }

        if self.reranker is not None and ranked_pool:
            depth = min(self.settings.rerank_depth, len(ranked_pool))
            reranked = self.reranker.rerank(search_query.semantic_query, ranked_pool[:depth])
            prompt_tokens += reranked.prompt_tokens
            semantic_diagnostics["rerank_error"] = reranked.error
            reranked_head = weighted_rrf([
                (self.settings.rerank_base_fusion_weight, ranked_pool[:depth]),
                (self.settings.rerank_model_fusion_weight, reranked.ranking),
            ])
            ranked_pool = reranked_head + ranked_pool[depth:]
        query_signature = (
            state.category or "",
            *(f"{item.attribute}:{item.value.casefold()}" for item in state.active_constraints),
            *(f"profile:{item.casefold()}" for item in state.profile_preferences),
        )
        exhausted = state.information_exhausted or "other" in state.rejected_attributes
        rotating = exhausted and query_signature == state.last_query_signature
        exploration_facets: tuple[str, ...] = ()
        if rotating:
            selection = select_diverse_candidates(
                ranked_pool,
                self.catalog.products,
                state,
                top_k,
            )
            recommendations = selection.identifiers
            exploration_facets = selection.facets
            state.exploration_turns += 1
        else:
            recommendations = ranked_pool[:top_k]
        question_plan = self.clarification_policy.plan(
            state,
            turn,
            [self.catalog.products[item] for item in ranked_pool[:80]],
        )
        ask_attribute = question_plan.ask_attribute
        if ask_attribute:
            state.asked_attributes.append(ask_attribute)
        if question_plan.focus_attribute:
            state.last_question_focus = question_plan.focus_attribute
            state.asked_question_focuses.append(question_plan.focus_attribute)
            state.question_topics = list(question_plan.topics)
        state.previous_recommendations = recommendations
        state.seen_recommendations.update(recommendations)
        state.last_query_signature = query_signature
        self.diagnostics[session_id] = {
            "category": state.category,
            "browsing": state.browsing,
            "intent_mode": state.intent_mode,
            "intent_confidence": state.intent_confidence,
            "intent_source": state.intent_source,
            "intent_reason": intent.reason,
            "intent_error": intent.error,
            "intent_retrieval_mode": {
                "buying": "precision",
                "browsing": "semantic_diversity",
            }.get(state.intent_mode, "conservative"),
            "answer_source": answer.source,
            "answer_confidence": answer.confidence,
            "answer_error": answer.error,
            "override_seen": state.override_seen,
            "boundary_observed": state.boundary_observed,
            "active_constraints": [
                {
                    "value": constraint.value,
                    "attribute": constraint.attribute,
                    "turn": constraint.turn,
                    "source": constraint.source,
                }
                for constraint in state.active_constraints
            ],
            "superseded_constraints": [
                {
                    "value": constraint.value,
                    "attribute": constraint.attribute,
                    "turn": constraint.turn,
                }
                for constraint in state.superseded_constraints
            ],
            "asked_attributes": list(state.asked_attributes),
            "question_focus": question_plan.focus_attribute,
            "question_topics": list(question_plan.topics),
            "question_reason": question_plan.reason,
            "question_confidence": question_plan.confidence,
            "question_protocol_fallback": question_plan.protocol_fallback,
            "information_exhausted": question_plan.information_exhausted,
            "candidate_rotation_active": rotating,
            "exploration_facets": list(exploration_facets),
            "exploration_turns": state.exploration_turns,
            "profile_dimensions": list(state.profile_dimensions),
            "profile_preferences": list(state.profile_preferences),
            "profile_avoidances": list(state.profile_avoidances),
            "seen_recommendation_count": len(state.seen_recommendations),
            **semantic_diagnostics,
            **retrieval_diagnostics,
        }

        return {
            "message": render_question(question_plan, recommendations),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin}
                for parent_asin in recommendations
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def get_diagnostics(self, session_id: str) -> dict:
        return dict(self.diagnostics.get(session_id) or {})
