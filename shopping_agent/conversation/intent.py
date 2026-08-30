from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from ..providers.openrouter import OpenRouterClient, OpenRouterError


BUYING_PATTERNS = (
    re.compile(r"\ba key requirement is\b", re.IGNORECASE),
    re.compile(r"\b(?:i need|i must have|has to be|must be|required)\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|purchase|order)\b", re.IGNORECASE),
)
BROWSING_PATTERNS = (
    re.compile(r"\bstill exploring\b", re.IGNORECASE),
    re.compile(r"\b(?:just browsing|looking around|open to ideas|not sure yet)\b", re.IGNORECASE),
    re.compile(r"\b(?:show me|inspire me|any suggestions)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class IntentDecision:
    mode: str
    confidence: float
    source: str
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class HybridIntentClassifier:
    """Prefer deterministic signals and use an LLM only for an unclear first turn."""

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        model: str = "",
        confidence_threshold: float = 0.65,
    ) -> None:
        self.client = client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.cache: dict[str, IntentDecision] = {}

    @staticmethod
    def _rule_decision(message: str) -> IntentDecision | None:
        if any(pattern.search(message) for pattern in BROWSING_PATTERNS):
            return IntentDecision("browsing", 1.0, "rule", "explicit exploration language")
        if any(pattern.search(message) for pattern in BUYING_PATTERNS):
            return IntentDecision("buying", 1.0, "rule", "explicit purchase requirement")
        return None

    def classify(
        self,
        message: str,
        turn: int,
        current_mode: str = "uncertain",
        current_confidence: float = 0.0,
    ) -> IntentDecision:
        rule = self._rule_decision(message)
        if rule is not None:
            return rule
        if turn > 1 and current_mode in {"buying", "browsing"}:
            return IntentDecision(current_mode, current_confidence, "state", "kept prior intent")
        if self.client is None:
            return IntentDecision("uncertain", 0.0, "fallback", "no decisive rule")

        cached = self.cache.get(message)
        if cached is not None:
            return replace(cached, prompt_tokens=0, completion_tokens=0)
        try:
            response = self.client.classify_shopping_intent(self.model, message)
            content = response.payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            mode = str(parsed.get("intent") or "uncertain").casefold()
            if mode not in {"buying", "browsing", "uncertain"}:
                raise ValueError(f"unsupported intent: {mode}")
            confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
            if confidence < self.confidence_threshold:
                mode = "uncertain"
            result = IntentDecision(
                mode=mode,
                confidence=confidence,
                source="llm",
                reason=str(parsed.get("reason") or "")[:160],
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        except (OpenRouterError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result = IntentDecision(
                "uncertain",
                0.0,
                "fallback",
                "intent model unavailable",
                error=str(exc)[:300],
            )
        self.cache[message] = result
        return result
