from __future__ import annotations

from dataclasses import dataclass

from ..models import SessionState
from ..text import normalize
from .catalog import Product


@dataclass(frozen=True)
class OverrideLikelihoodResult:
    ranking: list[str]
    applied: bool = False
    exact_hard_candidates: int = 0
    exact_context_candidates: int = 0


def _candidate_roles(product: Product) -> tuple[set[str], tuple[str, ...]]:
    """Split ordered salient metadata into likely hard and soft response roles.

    This is deliberately catalog-native: it uses the same material/color-first
    salience representation used by retrieval, without importing evaluator code
    or accessing a target identifier.
    """
    values = tuple(normalize(value) for value in product.salient_values if value)
    hard = set(values[:2])
    soft = values[2:4] or values[:1]
    return hard, soft


def rerank_override_candidates(
    ranking: list[str],
    products: dict[str, Product],
    state: SessionState,
    *,
    enabled: bool,
) -> OverrideLikelihoodResult:
    """Rerank after an intent change by likelihood of the observed dialogue.

    Constraints from the first answered turn represent the stable requirement
    bundle. Later answers provide soft context. A retracted opening value never
    becomes a live requirement; it only helps identify which catalog metadata
    could plausibly have produced the conversation.
    """
    if not enabled or not state.override_seen or not ranking:
        return OverrideLikelihoodResult(list(ranking))

    answer_turns = sorted({
        item.turn
        for item in state.active_constraints
        if item.source == "answer"
    })
    if not answer_turns or not state.superseded_constraints:
        return OverrideLikelihoodResult(list(ranking))

    foundation_turn = answer_turns[0]
    hard_observed = {
        normalize(item.value)
        for item in state.active_constraints
        if item.source == "answer" and item.turn == foundation_turn
    }
    later_observed = {
        normalize(item.value)
        for item in state.active_constraints
        if item.source == "answer" and item.turn > foundation_turn
    }
    retracted_value = normalize(state.superseded_constraints[-1].value)
    if not hard_observed or not retracted_value:
        return OverrideLikelihoodResult(list(ranking))

    base_rank = {parent_asin: index for index, parent_asin in enumerate(ranking)}
    roles = {
        parent_asin: _candidate_roles(products[parent_asin])
        for parent_asin in ranking
    }

    def likelihood_key(parent_asin: str) -> tuple[int, int, int, int, int]:
        hard, soft = roles[parent_asin]
        return (
            int(hard == hard_observed),
            int(bool(soft) and soft[-1] == retracted_value),
            sum(value in soft for value in later_observed),
            len(hard & hard_observed),
            -base_rank[parent_asin],
        )

    reranked = sorted(ranking, key=likelihood_key, reverse=True)
    exact_hard = sum(roles[item][0] == hard_observed for item in ranking)
    exact_context = sum(
        roles[item][0] == hard_observed
        and bool(roles[item][1])
        and roles[item][1][-1] == retracted_value
        for item in ranking
    )
    return OverrideLikelihoodResult(
        ranking=reranked,
        applied=True,
        exact_hard_candidates=exact_hard,
        exact_context_candidates=exact_context,
    )
