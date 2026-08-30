from __future__ import annotations

from collections.abc import Callable

from .models import Constraint, SessionState
from .parser import parse_message
from .text import classify_constraint, normalize


def ingest_message(
    state: SessionState,
    message: str,
    turn: int,
    constraint_resolver: Callable[[str], list[str]] | None = None,
) -> None:
    state.messages.append(message)
    parsed = parse_message(message)

    if parsed.category:
        state.category = parsed.category
    if parsed.browsing:
        state.browsing = True
    if parsed.boundary_response:
        state.boundary_observed = True
    elif parsed.rejected_attribute:
        state.rejected_attributes.add(parsed.rejected_attribute)

    if parsed.override:
        state.override_seen = True
        if state.initial_preference:
            old_value = normalize(state.initial_preference)
            for constraint in state.constraints:
                if normalize(constraint.value) == old_value:
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
        source = "override" if parsed.override else "user"
        state.constraints.append(Constraint(
            value=value,
            attribute=classify_constraint(value),
            turn=turn,
            source=source,
        ))
        known_values.add(key)

    if turn == 1 and parsed.constraints and not parsed.browsing:
        if "A key requirement is:" not in message:
            state.initial_preference = parsed.constraints[0]
