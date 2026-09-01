from __future__ import annotations

from shopping_agent.text import normalize


INTENT_LABELS = {
    "buying": "Ready to buy",
    "browsing": "Exploring options",
    "uncertain": "Intent forming",
}


def build_intent_summary(scenario_type: str, diagnostics: dict) -> dict:
    """Translate classifier state into concise, customer-safe language."""

    mode = str(diagnostics.get("intent_mode") or "uncertain")
    confidence = max(0.0, min(1.0, float(diagnostics.get("intent_confidence") or 0.0)))
    override_seen = bool(diagnostics.get("override_seen"))
    boundary_observed = bool(diagnostics.get("boundary_observed"))

    if override_seen:
        label = "Intent updated"
        detail = "A changed preference was detected, so the earlier signal was retired before ranking again."
    elif boundary_observed:
        label = "Preference left open"
        detail = "One preference was deliberately left undecided and is not being used as a hard filter."
    elif mode == "buying":
        label = INTENT_LABELS[mode]
        detail = "The customer is expressing concrete requirements, so precision is prioritised."
    elif mode == "browsing":
        label = INTENT_LABELS[mode]
        detail = "The customer is still exploring, so the result set stays broader and more diverse."
    else:
        label = INTENT_LABELS["uncertain"]
        detail = "There is not enough evidence yet to choose a strong buying or browsing strategy."

    return {
        "mode": mode,
        "label": label,
        "confidence": round(confidence, 3),
        "source": str(diagnostics.get("intent_source") or "state"),
        "reason": str(diagnostics.get("intent_reason") or "")[:180],
        "detail": detail,
        "scenario": scenario_type,
        "strategy": str(diagnostics.get("intent_retrieval_mode") or "conservative"),
        "override_seen": override_seen,
        "boundary_observed": boundary_observed,
    }


def build_product_explanation(product: dict, diagnostics: dict, category_match: bool) -> dict:
    active_constraints = diagnostics.get("active_constraints") or []
    atomic_values = {normalize(str(value)) for value in product.get("atomic_values") or []}
    matched = [
        str(constraint.get("value") or "")
        for constraint in active_constraints
        if normalize(str(constraint.get("value") or "")) in atomic_values
    ]
    matched = [value for value in matched if value]

    reasons: list[str] = []
    if matched:
        reasons.append(
            f"Matches {len(matched)} of {len(active_constraints)} active requirement"
            f"{'s' if len(active_constraints) != 1 else ''}: {', '.join(matched[:3])}."
        )
    elif active_constraints:
        reasons.append("Provides a close lexical/category match while stronger attribute evidence is gathered.")
    else:
        reasons.append("Matches the current product category while the customer is still narrowing the search.")

    category_values = [str(value) for value in product.get("categories") or [] if value]
    if category_match and category_values:
        reasons.append(f"Exact category path: {' › '.join(category_values[-2:])}.")

    rating = float(product.get("average_rating") or 0.0)
    rating_count = int(product.get("rating_number") or 0)
    if rating and rating_count:
        reasons.append(f"Rated {rating:.1f} from {rating_count:,} customer ratings.")

    return {
        "matched_constraints": matched,
        "matched_count": len(matched),
        "constraint_count": len(active_constraints),
        "category_match": category_match,
        "reasons": reasons,
    }


def build_ranking_explanation(diagnostics: dict, recommendations: list[dict]) -> dict:
    active_constraints = diagnostics.get("active_constraints") or []
    superseded = diagnostics.get("superseded_constraints") or []
    profile_dimensions = [str(value) for value in diagnostics.get("profile_dimensions") or []]
    profile_preferences = [str(value) for value in diagnostics.get("profile_preferences") or []]
    fused_count = int(
        diagnostics.get("fused_candidate_count")
        or diagnostics.get("fused_pool_size")
        or 0
    )
    exact_count = int(diagnostics.get("intersection_candidate_count") or 0)

    if diagnostics.get("override_seen"):
        summary = "I removed the superseded preference and rebuilt this ranking around the customer's latest request."
    elif active_constraints and exact_count:
        summary = (
            f"I compared {fused_count:,} candidates and found {exact_count:,} that match every active "
            "requirement. Exact matches are prioritised before broader alternatives."
        )
    elif active_constraints:
        summary = (
            f"I compared {fused_count:,} candidates using the category and disclosed requirements. "
            "The list includes the strongest close matches while clarification continues."
        )
    else:
        summary = (
            f"I started with {fused_count:,} category candidates and kept the ranking broad because "
            "the customer is still exploring."
        )

    signals: list[str] = []
    if active_constraints:
        values = ", ".join(str(item.get("value") or "") for item in active_constraints if item.get("value"))
        if values:
            signals.append(f"Current requirements: {values}")
    if superseded:
        values = ", ".join(str(item.get("value") or "") for item in superseded if item.get("value"))
        if values:
            signals.append(f"No longer used: {values}")
    if diagnostics.get("category"):
        signals.append(f"Requested category: {diagnostics['category']}")
    if diagnostics.get("dense_retrieval_applied"):
        signals.append("Semantic similarity contributed to candidate retrieval")
    else:
        signals.append("Structured matching and lexical retrieval produced this ranking")

    if profile_preferences:
        profile_note = f"Explicit profile preferences used in retrieval: {', '.join(profile_preferences)}."
    elif profile_dimensions:
        profile_note = (
            f"Profile priorities ({', '.join(profile_dimensions)}) guide which question to ask next; "
            "they are not assumed to be product requirements."
        )
    else:
        profile_note = "The profile contains no concrete preference value to use as a product filter."

    return {
        "summary": summary,
        "signals": signals,
        "profile_note": profile_note,
        "candidate_count": fused_count,
        "exact_match_count": exact_count,
        "returned_count": len(recommendations),
    }
