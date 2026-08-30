from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Constraint:
    value: str
    attribute: str
    turn: int
    source: str
    active: bool = True


@dataclass
class ParsedMessage:
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    constraint_payload: str | None = None
    override: bool = False
    rejected_attribute: str | None = None
    boundary_response: bool = False
    browsing: bool = False


@dataclass
class SessionState:
    user_profile: dict
    messages: list[str] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    category: str | None = None
    browsing: bool = False
    boundary_observed: bool = False
    rejected_attributes: set[str] = field(default_factory=set)
    asked_attributes: list[str] = field(default_factory=list)
    previous_recommendations: list[str] = field(default_factory=list)
    seen_recommendations: set[str] = field(default_factory=set)
    last_query_signature: tuple[str, ...] | None = None
    initial_preference: str | None = None
    override_seen: bool = False
    intent_mode: str = "uncertain"
    intent_confidence: float = 0.0
    intent_source: str = "default"
    information_exhausted: bool = False
    last_question_focus: str | None = None
    asked_question_focuses: list[str] = field(default_factory=list)
    question_topics: list[str] = field(default_factory=list)
    profile_dimensions: list[str] = field(default_factory=list)
    profile_preferences: list[str] = field(default_factory=list)
    profile_avoidances: list[str] = field(default_factory=list)
    last_answer_source: str = "fixed"
    last_answer_confidence: float = 1.0
    exploration_turns: int = 0

    @property
    def active_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if constraint.active]

    @property
    def superseded_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if not constraint.active]
