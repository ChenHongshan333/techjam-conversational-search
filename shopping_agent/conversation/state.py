from __future__ import annotations

from collections.abc import Callable

from ..models import Constraint, ParsedMessage, SessionState
from ..text import classify_constraint, normalize
from .parser import parse_message


# A constraint the customer offered unprompted, which a later override retracts.
VOLUNTEERED = frozenset({"stated", "override"})


def constraint_source(override: bool, turn: int) -> str:
    """Where a constraint came from, which decides whether an override drops it.

    Turn 1 is unprompted; every later turn answers a question the agent asked,
    so those facts survive an override of the opening preference.
    """
    if override:
        return "override"
    return "stated" if turn == 1 else "answer"


def ingest_message(
    state: SessionState,
    message: str,
    turn: int,
    constraint_resolver: Callable[[str], list[str]] | None = None,
    parsed: ParsedMessage | None = None,
) -> None:
    state.messages.append(message)
    parsed = parsed or parse_message(message)

    if parsed.category:
        state.category = parsed.category
    if parsed.browsing:
        state.browsing = True
    if parsed.boundary_response:
        state.boundary_observed = True
    elif parsed.rejected_attribute:
        state.rejected_attributes.add(parsed.rejected_attribute)
        if parsed.rejected_attribute == "other":
            state.information_exhausted = True

    if parsed.override:
        state.override_seen = True
        # "Ignore my earlier preference" retracts what the customer volunteered,
        # not the facts they supplied when asked. Anchoring on the volunteered
        # constraints rather than on a remembered turn-1 string means a second
        # override, and an override in a session that opened without a stated
        # preference, both erase the right thing.
        for constraint in state.constraints:
            if constraint.turn < turn and constraint.source in VOLUNTEERED:
                constraint.active = False

    resolved_constraints = parsed.constraints
    if parsed.constraint_payload and constraint_resolver is not None:
        resolved = constraint_resolver(parsed.constraint_payload)
        if resolved:
            resolved_constraints = resolved

    known_values = {normalize(item.value) for item in state.active_constraints}
    for value in resolved_constraints:
        key = normalize(value)
        if not key or key in known_values:
            continue
        source = constraint_source(parsed.override, turn)
        state.constraints.append(Constraint(
            value=value,
            attribute=classify_constraint(value),
            turn=turn,
            source=source,
        ))
        known_values.add(key)
