from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..models import SessionState
from ..retrieval.catalog import Product
from ..text import COLORS, MATERIALS


STYLE_MARKERS = (
    "fit", "sleeve", "neck", "closure", "length", "waist", "heel", "hood", "collar",
)
USE_CASE_MARKERS = (
    "gift", "work", "running", "hiking", "outdoor", "winter", "wedding", "party",
    "travel", "gym", "school", "mother", "father", "birthday",
)
PROFILE_ATTRIBUTE_MAP = {
    "material": "material",
    "fit": "style",
    "style": "style",
    "comfort": "feature",
    "color": "color",
    "brand": "brand",
    "budget": "budget",
    "price": "budget",
    "size": "size",
    "use": "use_case",
    "occasion": "use_case",
}
TOPICS = {
    "category": ("more specific product type",),
    "material": ("material", "fabric"),
    "color": ("color",),
    "size": ("size", "width"),
    "style": ("fit", "sleeve length", "specific style"),
    "brand": ("brand",),
    "budget": ("budget", "price range"),
    "feature": ("design", "printed message", "must-have feature"),
    "use_case": ("occasion", "intended use", "gift recipient"),
}


@dataclass(frozen=True)
class QuestionPlan:
    ask_attribute: str | None
    focus_attribute: str | None
    topics: tuple[str, ...] = ()
    reason: str = ""
    confidence: float = 0.0
    information_exhausted: bool = False


class ClarificationPolicy:
    """Choose a question focus while keeping `other` as a protocol-safe fallback."""

    def plan(
        self,
        state: SessionState,
        turn: int,
        candidates: list[Product],
    ) -> QuestionPlan:
        if turn >= 10 or state.information_exhausted or "other" in state.rejected_attributes:
            return QuestionPlan(
                None,
                None,
                reason="no more answerable information",
                information_exhausted=True,
            )

        known = {item.attribute for item in state.active_constraints}
        rejected = set(state.rejected_attributes)
        scores = self._base_scores(state.intent_mode)
        reasons: dict[str, list[str]] = {name: [] for name in scores}

        for attribute in known | rejected:
            if attribute in scores:
                scores[attribute] -= 3.0

        for tag in state.user_profile.get("preference_tags") or []:
            mapped = self._profile_attribute(str(tag))
            if mapped in scores and mapped not in known:
                scores[mapped] += 0.8
                reasons[mapped].append("profile priority")

        variation = self._candidate_variation(candidates)
        for attribute, value in variation.items():
            if attribute in scores:
                scores[attribute] += value
                if value > 0.15:
                    reasons[attribute].append("separates current candidates")

        for attribute, count in Counter(state.asked_question_focuses).items():
            if attribute in scores:
                scores[attribute] -= 1.25 * count

        focus, score = max(scores.items(), key=lambda item: (item[1], item[0]))
        confidence = max(0.0, min(1.0, score / 3.0))
        reason = ", ".join(reasons[focus]) or "best unanswered clarification"
        return QuestionPlan(
            # `other` lets the deterministic evaluator reveal any remaining card
            # value, while the natural-language message still has a useful focus.
            ask_attribute="other",
            focus_attribute=focus,
            topics=TOPICS[focus],
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _base_scores(intent_mode: str) -> dict[str, float]:
        if intent_mode == "browsing":
            order = ("style", "use_case", "category", "feature", "color", "brand", "budget", "material", "size")
        elif intent_mode == "buying":
            order = ("feature", "style", "use_case", "size", "color", "brand", "budget", "material", "category")
        else:
            order = ("use_case", "category", "style", "feature", "budget", "color", "material", "size", "brand")
        return {attribute: 1.4 - index * 0.08 for index, attribute in enumerate(order)}

    @staticmethod
    def _profile_attribute(tag: str) -> str | None:
        lowered = tag.casefold()
        for marker, attribute in PROFILE_ATTRIBUTE_MAP.items():
            if marker in lowered:
                return attribute
        return None

    @staticmethod
    def _candidate_variation(candidates: list[Product]) -> dict[str, float]:
        sample = candidates[:80]
        if len(sample) < 2:
            return {}

        def signature(product: Product, markers: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(marker for marker in markers if re.search(rf"\b{re.escape(marker)}\b", product.search_text))

        signatures = {
            "material": [signature(product, MATERIALS) for product in sample],
            "color": [signature(product, COLORS) for product in sample],
            "style": [signature(product, STYLE_MARKERS) for product in sample],
            "use_case": [signature(product, USE_CASE_MARKERS) for product in sample],
            "category": [tuple(value.casefold() for value in product.category_values[-2:]) for product in sample],
        }
        result: dict[str, float] = {"feature": 0.35}
        for attribute, values in signatures.items():
            nonempty = [value for value in values if value]
            if not nonempty:
                continue
            distinct = len(set(nonempty))
            coverage = len(nonempty) / len(values)
            result[attribute] = min(0.9, coverage * distinct / max(2.0, len(values) ** 0.5))
        return result


def render_question(plan: QuestionPlan, recommendations: list[str]) -> str:
    if plan.ask_attribute is None:
        if recommendations:
            return "Here are the closest matches based on everything you have told me."
        return "I could not find a confident match yet."

    prefix = "I found some initial matches. " if recommendations else ""
    topics = list(plan.topics)
    if len(topics) == 1:
        detail = topics[0]
    else:
        detail = ", ".join(topics[:-1]) + f", or {topics[-1]}"
    if plan.focus_attribute == "category":
        return prefix + f"Could you narrow this down by {detail}?"
    return prefix + f"Do you have a preference for {detail}?"
