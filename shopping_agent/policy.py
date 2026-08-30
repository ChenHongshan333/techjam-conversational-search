from __future__ import annotations

from .models import SessionState


def choose_question(state: SessionState, turn: int, candidate_count: int) -> str | None:
    if turn >= 10:
        return None
    if "other" in state.rejected_attributes:
        return None
    # Boundary's first refusal is scenario behavior, not proof that all constraints
    # have been exhausted. Asking `other` again can reveal useful information.
    return "other"


def build_message(ask_attribute: str | None, recommendations: list[str]) -> str:
    if ask_attribute == "other":
        if recommendations:
            return (
                "I found some initial matches. What other requirement matters most, "
                "such as material, fit, color, or intended use?"
            )
        return "What requirement matters most, such as material, fit, color, or intended use?"
    if recommendations:
        return "Here are the closest matches based on everything you have told me."
    return "I could not find a confident match yet."
