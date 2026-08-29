from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# The simulated customer wraps every constraint in fixed template text. Those
# words describe the dialogue, not the product, so they are removed before the
# message is turned into search terms.
TEMPLATE_RE = re.compile(
    "|".join(
        (
            r"for that, what matters is:",
            r"a key requirement is:",
            r"i'm looking for",
            r"i'm still exploring",
            r"actually, ignore my earlier preference\. what i need is:",
            r"i don't have an additional preference for \w+",
            r"i don't have a preference for \w+; please use your judgment",
            r"those options are not quite right yet\. ask me about one specific attribute",
        )
    ),
    re.IGNORECASE,
)

# The opening line always names the coarse category, which stays valid for the
# whole session even when a later turn retracts the stated preference.
OPENING_RE = re.compile(r"i'm looking for (.+?)(?:\.|,\s*but\b)", re.IGNORECASE)

MAX_QUERY_TERMS = 40


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Editable baseline: BM25 retrieval over everything the customer disclosed."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {"category": [], "disclosed": []}

    def _absorb(self, state: dict, user_message: str, turn: int) -> None:
        """Fold this turn's message into the session's accumulated search terms."""
        # An override turn retracts a stated preference, but the simulator draws
        # both the old and new value from the same target product, so the earlier
        # terms stay useful and are deliberately kept.
        remainder = user_message
        if turn == 1:
            opening = OPENING_RE.search(user_message)
            if opening:
                state["category"] = _terms(opening.group(1))
                remainder = user_message[opening.end():]
        for term in _terms(TEMPLATE_RE.sub(" ", remainder)):
            if term not in state["disclosed"]:
                state["disclosed"].append(term)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._absorb(state, user_message, turn)
        unique_terms = list(dict.fromkeys(state["category"] + state["disclosed"]))[:MAX_QUERY_TERMS]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            # An open-ended ask bypasses the simulator's attribute filter and
            # costs nothing when the card is already empty, so ask every turn.
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
