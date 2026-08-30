from __future__ import annotations

from pathlib import Path

from .catalog import CatalogIndex
from .config import RetrievalSettings
from .models import SessionState
from .openrouter import OpenRouterClient
from .policy import build_message, choose_question
from .query import QueryBuilder
from .semantic import DenseProductRetriever, ProductReranker, weighted_rrf
from .state import ingest_message


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
        self.sessions[session_id] = SessionState(user_profile=dict(user_profile))
        self.diagnostics.pop(session_id, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        ingest_message(
            state,
            user_message,
            turn,
            constraint_resolver=self.catalog.resolve_constraint_payload,
        )
        search_query = self.query_builder.build(state)
        prompt_tokens = search_query.prompt_tokens
        completion_tokens = search_query.completion_tokens

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
                ranked_pool = weighted_rrf([
                    (self.settings.lexical_fusion_weight, ranked_pool),
                    (self.settings.dense_identity_fusion_weight, dense.identity_ranking),
                    (self.settings.dense_attribute_fusion_weight, dense.attribute_ranking),
                ])

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
        )
        exhausted = "other" in state.rejected_attributes
        rotating = exhausted and query_signature == state.last_query_signature
        if rotating:
            unseen = [item for item in ranked_pool if item not in state.seen_recommendations]
            recommendations = (unseen or ranked_pool)[:top_k]
        else:
            recommendations = ranked_pool[:top_k]
        ask_attribute = choose_question(state, turn, len(recommendations))
        if ask_attribute:
            state.asked_attributes.append(ask_attribute)
        state.previous_recommendations = recommendations
        state.seen_recommendations.update(recommendations)
        state.last_query_signature = query_signature
        self.diagnostics[session_id] = {
            "category": state.category,
            "browsing": state.browsing,
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
            "candidate_rotation_active": rotating,
            "seen_recommendation_count": len(state.seen_recommendations),
            **semantic_diagnostics,
            **retrieval_diagnostics,
        }

        return {
            "message": build_message(ask_attribute, recommendations),
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
