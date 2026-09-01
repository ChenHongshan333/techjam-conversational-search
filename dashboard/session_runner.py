from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from shopping_agent import ShoppingAgent

from .explanations import (
    build_intent_summary,
    build_product_explanation,
    build_ranking_explanation,
)


@dataclass
class ReplaySession:
    sample: dict
    agent: ShoppingAgent
    catalog_ids: set[str]
    categories: dict[str, list[str]]
    products: dict[str, dict]
    session_id: str = field(default_factory=lambda: f"dashboard_{uuid.uuid4().hex}")

    def __post_init__(self) -> None:
        self.target = str(self.sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(self.sample, self.products)
        self.effective_sample = {
            **self.sample,
            "intent_card": intent_card,
            "behavior": behavior,
        }
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = self.sample["scenario_type"] != "intent_override"
        self.user_message = initial_message(
            self.effective_sample,
            coarse_category(self.categories.get(self.target, [])),
            self.disclosed,
        )
        self.turn = 1
        self.finished = False
        self.hit_turn: int | None = None
        self.best_rank: int | None = None
        self.events: list[dict] = []
        self.agent.reset(self.session_id, self.sample["user_profile"])

    def snapshot(self) -> dict:
        target_product = self.products[self.target]
        return {
            "session_id": self.session_id,
            "sample_id": self.sample["sample_id"],
            "scenario_type": self.sample["scenario_type"],
            "difficulty_bucket": self.sample.get("difficulty_bucket"),
            "user_profile": self.sample.get("user_profile") or {},
            "turn": self.turn,
            "max_turns": MAX_TURNS,
            "current_user_message": self.user_message,
            "finished": self.finished,
            "target": {
                "parent_asin": self.target,
                "title": target_product.get("title") or self.target,
            },
            "events": self.events,
            "summary": self._summary(),
        }

    def step(self) -> dict:
        if self.finished:
            raise RuntimeError("This replay has already finished")

        current_message = self.user_message
        response = self.agent.respond(self.session_id, current_message, self.turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
        contains_target = self.target in ranked
        scored_hit = self.override_applied and contains_target
        if scored_hit:
            self.best_rank = ranked.index(self.target) + 1
            self.hit_turn = self.turn
            self.finished = True

        diagnostics = self.agent.get_diagnostics(self.session_id)
        recommendation_rows = []
        for rank, parent_asin in enumerate(ranked, start=1):
            product = self.products.get(parent_asin) or {}
            category_match = self.agent.catalog.exact_category_suffix(
                parent_asin,
                str(diagnostics.get("category") or ""),
            )
            recommendation_rows.append({
                "rank": rank,
                "parent_asin": parent_asin,
                "title": product.get("title") or parent_asin,
                "categories": product.get("categories") or [],
                "average_rating": product.get("average_rating") or 0.0,
                "rating_number": product.get("rating_number") or 0,
                "explanation": build_product_explanation(product, diagnostics, category_match),
                "is_target": parent_asin == self.target,
                "is_scored_hit": scored_hit and parent_asin == self.target,
            })

        next_user_message: str | None = None
        if not self.finished and self.turn == MAX_TURNS:
            self.finished = True
        elif not self.finished:
            override = self.effective_sample.get("behavior", {}).get("override") or {}
            if not self.override_applied and self.turn + 1 == int(override.get("turn", 3)):
                self.override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    self.disclosed.add(new_value)
                next_user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                next_user_message, self.boundary_used = customer_reply(
                    self.effective_sample,
                    response.get("ask_attribute"),
                    self.disclosed,
                    self.boundary_used,
                )

        event = {
            "turn": self.turn,
            "user_message": current_message,
            "agent_message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": recommendation_rows,
            "diagnostics": diagnostics,
            "intent": build_intent_summary(self.sample["scenario_type"], diagnostics),
            "ranking_explanation": build_ranking_explanation(diagnostics, recommendation_rows),
            "contains_target": contains_target,
            "scored_hit": scored_hit,
            "override_applied": self.override_applied,
            "next_user_message": next_user_message,
            "finished": self.finished,
        }
        self.events.append(event)

        if next_user_message is not None:
            self.user_message = next_user_message
            self.turn += 1
        event["summary"] = self._summary()
        return event

    def run_all(self) -> list[dict]:
        events: list[dict] = []
        while not self.finished:
            events.append(self.step())
        return events

    def _summary(self) -> dict:
        if not self.finished:
            return {
                "finished": False,
                "hit": False,
                "first_hit_turn": None,
                "best_rank": None,
                "reciprocal_rank": None,
                "efficiency": None,
                "technical_score": None,
            }
        hit = self.hit_turn is not None
        reciprocal_rank = 0.0 if self.best_rank is None else 1.0 / self.best_rank
        mttc = self.hit_turn if self.hit_turn is not None else MAX_TURNS + 1
        efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        technical_score = 0.50 * int(hit) + 0.30 * reciprocal_rank + 0.20 * efficiency
        return {
            "finished": True,
            "hit": hit,
            "first_hit_turn": self.hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": round(reciprocal_rank, 6),
            "efficiency": round(efficiency, 6),
            "technical_score": round(technical_score, 6),
        }
