from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from ..models import ParsedMessage
from ..providers.openrouter import OpenRouterClient, OpenRouterError
from ..text import clean_constraint
from .parser import parse_message


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
EMPTY_OR_COMPLAINT = re.compile(
    r"\b(?:not quite right|ask me about)\b",
    re.IGNORECASE,
)
NATURAL_REJECTION = re.compile(
    r"\b(?:no preference|don'?t care|anything is fine|doesn'?t matter|use your judgment)\b",
    re.IGNORECASE,
)
PREFERENCE_PREFIX = re.compile(
    r"^(?:actually[, ]+)?(?:i(?:'d| would)? (?:like|prefer|want|need)|"
    r"it should (?:be|have)|please (?:prioriti[sz]e|look for)|"
    r"my preference is)\s+",
    re.IGNORECASE,
)


def has_information(parsed: ParsedMessage) -> bool:
    return bool(
        parsed.category
        or parsed.constraints
        or parsed.override
        or parsed.rejected_attribute
        or parsed.boundary_response
        or parsed.browsing
    )


@dataclass(frozen=True)
class AnswerDecision:
    parsed: ParsedMessage
    source: str
    confidence: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class HybridAnswerInterpreter:
    """Use the fixed protocol first, then conservative rules, then optional LLM."""

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        model: str = "",
        confidence_threshold: float = 0.65,
    ) -> None:
        self.client = client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.cache: dict[tuple[str, str, str], AnswerDecision] = {}

    def interpret(
        self,
        message: str,
        turn: int,
        current_category: str | None = None,
        question_focus: str | None = None,
    ) -> AnswerDecision:
        fixed = parse_message(message)
        if has_information(fixed):
            return AnswerDecision(fixed, "fixed", 1.0)

        ruled = self._natural_rule(message, turn, question_focus)
        if ruled is not None:
            return AnswerDecision(ruled, "natural_rule", 0.76)
        if self.client is None:
            return AnswerDecision(ParsedMessage(), "fallback", 0.0)

        key = (message, current_category or "", question_focus or "")
        cached = self.cache.get(key)
        if cached is not None:
            return replace(cached, prompt_tokens=0, completion_tokens=0)
        try:
            response = self.client.extract_shopping_answer(
                self.model,
                message,
                current_category or "",
                question_focus or "",
            )
            content = response.payload["choices"][0]["message"]["content"]
            payload = json.loads(content)
            confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
            if confidence < self.confidence_threshold:
                parsed = ParsedMessage()
            else:
                category = clean_constraint(str(payload.get("category") or ""), 120) or None
                constraints = [
                    clean_constraint(str(value))
                    for value in (payload.get("constraints") or [])[:4]
                    if clean_constraint(str(value))
                ]
                rejected = str(payload.get("rejected_attribute") or "").casefold()
                parsed = ParsedMessage(
                    category=category,
                    constraints=constraints,
                    override=bool(payload.get("override")),
                    rejected_attribute=rejected if rejected in ALLOWED_ATTRIBUTES else None,
                )
            result = AnswerDecision(
                parsed,
                "llm",
                confidence,
                response.prompt_tokens,
                response.completion_tokens,
            )
        except (OpenRouterError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result = AnswerDecision(
                ParsedMessage(), "fallback", 0.0, error=str(exc)[:300]
            )
        self.cache[key] = result
        return result

    @staticmethod
    def _natural_rule(
        message: str,
        turn: int,
        question_focus: str | None,
    ) -> ParsedMessage | None:
        text = clean_constraint(message)
        if not text or EMPTY_OR_COMPLAINT.search(text):
            return None
        if turn <= 1 and PREFERENCE_PREFIX.match(text) is None:
            return None
        if NATURAL_REJECTION.search(text) and question_focus in ALLOWED_ATTRIBUTES:
            return ParsedMessage(rejected_attribute=question_focus)
        lowered = text.casefold()
        override = any(marker in lowered for marker in ("actually", "instead", "ignore my earlier"))
        stripped = PREFERENCE_PREFIX.sub("", text)
        stripped = re.sub(r"^(?:something|one)\s+(?:that|with)\s+", "", stripped, flags=re.I)
        values = [
            clean_constraint(value)
            for value in re.split(r"\s*(?:;|,|\band\b)\s*", stripped, flags=re.I)
        ]
        values = [value for value in values if len(value) >= 2][:4]
        if not values:
            return None
        return ParsedMessage(constraints=values, override=override)
