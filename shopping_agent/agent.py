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
from .retrieval.override import rerank_override_candidates
from .retrieval.query import QueryBuilder
from .retrieval.semantic import DenseProductRetriever, ProductReranker, weighted_rrf


class ShoppingAgent:
    """Stateful, offline conversational catalog retrieval agent."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.settings = RetrievalSettings.from_environment()
        self.catalog = CatalogIndex(
            catalog_path,
            self.settings.slot_decay,
            self.settings.intent_routing_scale,
            self.settings.retracted_weight,
            self.settings.constraint_lock,
            self.settings.lock_tracks,
            self.settings.dynamic_truncation,
            self.settings.truncation_strong_evidence,
            self.settings.truncation_floor,
            self.settings.retracted_context_weight,
            self.settings.exact_category_suffix_bonus,
        )
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
        # Full fused pool from the most recent turn, kept so an offline harness
        # can locate the target beyond the ten emitted ids. Never used to rank.
        self.last_pool: dict[str, list[str]] = {}

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
        constraints_before = len(state.active_constraints)
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
        exposure_reset = False
        if answer.parsed.override and self.settings.exposure_demotion_enabled:
            # Products exposed under a superseded intent are not negative
            # evidence for the replacement intent. Reset only the exposure
            # state; the conversation-state layer separately erases the old
            # volunteered constraint.
            state.previous_recommendations.clear()
            state.seen_recommendations.clear()
            state.recommendation_exposures.clear()
            exposure_reset = True
        state.gained_information = len(state.active_constraints) > constraints_before
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
            "dense_retrieval_applied": False,
            "dense_filtered": self.settings.dense_filtered,
            "dense_min_turn": self.settings.dense_min_turn,
            "dense_tracks": list(self.settings.dense_tracks),
            "dense_candidate_pool_count": 0,
            "dense_retrieval_error": None,
            "dense_identity_candidate_count": 0,
            "dense_attribute_candidate_count": 0,
            "rerank_enabled": self.reranker is not None,
            "rerank_error": None,
        }
        dense_track = "override" if state.override_seen else state.intent_mode
        dense_track_allowed = (
            "all" in self.settings.dense_tracks
            or dense_track in self.settings.dense_tracks
        )
        semantic_diagnostics["dense_active_track"] = dense_track
        semantic_diagnostics["dense_track_allowed"] = dense_track_allowed
        if (
            self.dense_retriever is not None
            and turn >= self.settings.dense_min_turn
            and dense_track_allowed
        ):
            dense_candidates = (
                ranked_pool[:self.settings.dense_candidate_pool_size]
                if self.settings.dense_filtered
                else None
            )
            dense = self.dense_retriever.search(
                search_query.semantic_query,
                limit=min(600, len(dense_candidates)) if dense_candidates is not None else 600,
                candidate_ids=dense_candidates,
            )
            prompt_tokens += dense.prompt_tokens
            semantic_diagnostics.update({
                "dense_retrieval_applied": True,
                "dense_candidate_pool_count": dense.candidate_count,
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

        override_likelihood = rerank_override_candidates(
            ranked_pool,
            self.catalog.products,
            state,
            enabled=self.settings.override_likelihood_enabled,
        )
        ranked_pool = override_likelihood.ranking
        semantic_diagnostics.update({
            "override_likelihood_enabled": self.settings.override_likelihood_enabled,
            "override_likelihood_applied": override_likelihood.applied,
            "override_likelihood_exact_hard_candidates": (
                override_likelihood.exact_hard_candidates
            ),
            "override_likelihood_exact_context_candidates": (
                override_likelihood.exact_context_candidates
            ),
        })
        ranked_pool, exposure_demoted = self._demote_exposed(ranked_pool, state)
        query_signature = (
            state.category or "",
            *(f"{item.attribute}:{item.value.casefold()}" for item in state.active_constraints),
            *(f"profile:{item.casefold()}" for item in state.profile_preferences),
        )
        exhausted = state.information_exhausted or "other" in state.rejected_attributes
        # Rotation exists to move past products already shown. With nothing shown
        # yet there is nothing to rotate away from, and diversifying the very
        # first list trades the precision ranking for coverage the session has
        # not earned -- which costs the hit outright.
        rotating = (
            exhausted
            and query_signature == state.last_query_signature
            and bool(state.seen_recommendations)
        )
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
        self.last_pool[session_id] = list(ranked_pool)
        suppressed = self._suppress(
            state, turn, retrieval_diagnostics.get("fused_candidate_count", 0)
        )
        if suppressed and turn > self.settings.suppression_turns:
            state.overload_cutoffs += 1

        early_limit: int | None = None
        if (
            self.settings.suppression_enabled
            and turn <= self.settings.suppression_turns
            and (self.settings.early_recommendation_limit > 0 or suppressed)
        ):
            # Keep one best-effort result visible while the opening turns gather
            # evidence. A zero limit reproduces the previous fully silent policy.
            early_limit = min(top_k, self.settings.early_recommendation_limit)
            recommendations = ranked_pool[:early_limit]
        elif suppressed:
            recommendations = []

        if suppressed:
            state.suppressed_turns += 1

        policy_limit: int | None = None
        if not (suppressed and not recommendations):
            policy_limit = self._recommendation_limit(
                state,
                turn,
                top_k,
                question_plan.ask_attribute,
                retrieval_diagnostics,
            )
            if policy_limit is not None:
                # Preserve a facet-diverse ordering when rotation is active;
                # otherwise take the precision head of the exposure-adjusted
                # fused ranking.
                recommendations = (
                    recommendations[:policy_limit]
                    if rotating
                    else ranked_pool[:policy_limit]
                )

        recommendations_suppressed = suppressed and not recommendations
        active_limit = policy_limit if policy_limit is not None else early_limit
        recommendations_narrowed = (
            active_limit is not None and len(ranked_pool) > active_limit
        )
        state.previous_recommendations = recommendations
        state.seen_recommendations.update(recommendations)
        for parent_asin in recommendations:
            state.recommendation_exposures[parent_asin] = (
                state.recommendation_exposures.get(parent_asin, 0) + 1
            )
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
            "exposure_demotion_enabled": self.settings.exposure_demotion_enabled,
            "exposure_demoted_count": exposure_demoted,
            "exposure_reset_on_override": exposure_reset,
            "recommendation_policy": self.settings.recommendation_policy,
            "fused_pool_size": len(ranked_pool),
            "gathering_turn": suppressed,
            "recommendations_suppressed": recommendations_suppressed,
            "recommendations_narrowed": recommendations_narrowed,
            "recommendation_limit": active_limit if active_limit is not None else top_k,
            "overload_cutoffs": state.overload_cutoffs,
            "suppressed_turns": state.suppressed_turns,
            "gained_information": state.gained_information,
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

    def _demote_exposed(
        self,
        ranking: list[str],
        state: SessionState,
    ) -> tuple[list[str], int]:
        """Apply a recoverable rank-space penalty to previously exposed items.

        This is intentionally softer than deleting a product: newly accumulated
        evidence can still pull an exposed candidate back into the visible head.
        The default configuration leaves the ranking byte-for-byte unchanged.
        """
        settings = self.settings
        if (
            not settings.exposure_demotion_enabled
            or settings.exposure_rank_penalty <= 0.0
            or not state.recommendation_exposures
        ):
            return ranking, 0

        indexed = list(enumerate(ranking))
        demoted = sum(
            1 for _, parent_asin in indexed
            if state.recommendation_exposures.get(parent_asin, 0) > 0
        )
        indexed.sort(key=lambda item: (
            item[0]
            + settings.exposure_rank_penalty
            * state.recommendation_exposures.get(item[1], 0),
            item[0],
        ))
        return [parent_asin for _, parent_asin in indexed], demoted

    def _recommendation_limit(
        self,
        state: SessionState,
        turn: int,
        top_k: int,
        ask_attribute: str | None,
        diagnostics: dict,
    ) -> int | None:
        """Return an experimental disclosure width, or None for shipped policy.

        ``adaptive`` narrows only while another useful clarification remains,
        then widens for coverage. ``sequential`` is retained as an explicit
        benchmark ablation of L-GPT's one-at-a-time schedule; it is not the
        compliance-first policy.
        """
        policy = self.settings.recommendation_policy
        if policy == "current":
            return None
        if turn >= 10:
            return top_k
        if policy == "sequential":
            return min(1, top_k)
        if policy != "adaptive":
            return None

        exhausted = state.information_exhausted or "other" in state.rejected_attributes
        if exhausted or ask_attribute is None:
            return top_k

        relative_margin = float(diagnostics.get("fused_relative_margin") or 0.0)
        intersection = int(diagnostics.get("intersection_candidate_count") or 0)
        pool_size = int(diagnostics.get("fused_candidate_count") or 0)
        high_confidence = relative_margin >= 0.02 or 0 < intersection <= 3
        overloaded = pool_size >= 500
        if high_confidence or overloaded or state.gained_information:
            return min(1, top_k)
        return min(3, top_k)

    def _suppress(self, state: SessionState, turn: int, pool_size: int = 0) -> bool:
        """Decide whether to withhold this turn's recommendations.

        Withholding the opening turns lets a session convert on accumulated
        evidence instead of on turn 1, which is worth far more MRR than the
        extra turns cost in efficiency. Four rules bound it.
        """
        settings = self.settings
        if not settings.suppression_enabled:
            return False
        # Reserve the closing turns. The 0.5-weighted hit rate is decided there,
        # and a session that spends its last turns silent can lose a hit it had.
        if turn > 10 - settings.suppression_reserve_turns:
            return False
        if state.suppressed_turns >= settings.suppression_max_turns:
            return False
        if turn > 1 and not state.gained_information:
            return False
        # Stop the moment the customer has no more information to give. Past that
        # point the ranking cannot improve, and the first emitted list would be a
        # diversified exploration list rather than a precision one -- which costs
        # hit rate outright. This makes the rule self-limiting: it never depends
        # on how deep a particular session's intent card happens to be.
        if state.information_exhausted or "other" in state.rejected_attributes:
            return False
        if turn <= settings.suppression_turns:
            return True
        # Over-generality cutoff: the candidate pool is still overloaded, so the
        # ranking is not yet worth acting on. Cut retrieval short for one more
        # turn and spend it on a clarification instead. Gated by the same cap and
        # reserve as the opening turns, so an overloaded session can never talk
        # itself past the point where it must emit.
        return (
            settings.overload_threshold > 0
            and pool_size >= settings.overload_threshold
        )

    def get_diagnostics(self, session_id: str) -> dict:
        return dict(self.diagnostics.get(session_id) or {})
