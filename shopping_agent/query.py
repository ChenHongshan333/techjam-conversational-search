from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from .models import SessionState
from .openrouter import OpenRouterClient, OpenRouterError


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class SearchQuery:
    category: str
    constraints: tuple[str, ...]
    lexical_query: str
    semantic_query: str
    rewrite_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class QueryBuilder:
    def __init__(self, client: OpenRouterClient | None = None, rewrite_model: str = "") -> None:
        self.client = client
        self.rewrite_model = rewrite_model
        self.cache: dict[tuple[str, tuple[str, ...]], SearchQuery] = {}

    def build(self, state: SessionState) -> SearchQuery:
        category = state.category or ""
        constraints = tuple(item.value for item in state.active_constraints)
        key = (category, constraints)
        cached = self.cache.get(key)
        if cached is not None:
            return replace(cached, prompt_tokens=0, completion_tokens=0)

        lexical_query = " ".join((category, *constraints)).strip()
        clauses = [f"Product type: {category}." if category else ""]
        if constraints:
            clauses.append("Requirements: " + "; ".join(constraints) + ".")
        deterministic = " ".join(part for part in clauses if part).strip()
        result = SearchQuery(
            category=category,
            constraints=constraints,
            lexical_query=lexical_query,
            semantic_query=deterministic or lexical_query,
        )
        if self.client is None or not constraints:
            self.cache[key] = result
            return result

        try:
            response = self.client.rewrite_query(self.rewrite_model, category, list(constraints))
            content = response.payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            rewrite = str(parsed.get("semantic_query") or "").strip()
            used = parsed.get("used_constraint_indexes")
            expected = set(range(len(constraints)))
            used_indexes = {int(index) for index in used} if isinstance(used, list) else set()
            input_numbers = set(NUMBER_RE.findall(lexical_query))
            output_numbers = set(NUMBER_RE.findall(rewrite))
            if not rewrite or len(rewrite) > 400:
                raise ValueError("empty or oversized semantic query")
            if used_indexes != expected:
                raise ValueError("rewrite omitted a disclosed constraint")
            if not output_numbers.issubset(input_numbers):
                raise ValueError("rewrite introduced a new number")
            result = SearchQuery(
                category=category,
                constraints=constraints,
                lexical_query=lexical_query,
                semantic_query=rewrite,
                rewrite_used=True,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        except (OpenRouterError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result = SearchQuery(
                category=category,
                constraints=constraints,
                lexical_query=lexical_query,
                semantic_query=deterministic or lexical_query,
                error=str(exc)[:300],
            )
        self.cache[key] = result
        return result
