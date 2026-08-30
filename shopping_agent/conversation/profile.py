from __future__ import annotations

from dataclasses import dataclass

from ..text import clean_constraint, normalize


DIMENSION_ALIASES = {
    "material": "material",
    "fabric": "material",
    "fit": "fit",
    "style": "style",
    "comfort": "comfort",
    "color": "color",
    "colour": "color",
    "brand": "brand",
    "budget": "budget",
    "price": "budget",
    "size": "size",
    "occasion": "use_case",
    "use": "use_case",
}


@dataclass(frozen=True)
class ProfileSignals:
    important_dimensions: tuple[str, ...] = ()
    positive_preferences: tuple[str, ...] = ()
    negative_preferences: tuple[str, ...] = ()


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    return []


def _unique(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_constraint(value)
        key = normalize(cleaned)
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return tuple(result)


def distill_user_profile(profile: dict) -> ProfileSignals:
    """Separate decision dimensions from evidence-backed preference values.

    Public profiles usually contain abstract tags such as ``material`` and ``fit``.
    Those guide clarification only. Concrete values enter retrieval only when the
    profile exposes them in an explicit preference field.
    """

    dimensions: list[str] = []
    for tag in _strings(profile.get("preference_tags")):
        lowered = tag.casefold()
        mapped = next((value for marker, value in DIMENSION_ALIASES.items() if marker in lowered), None)
        if mapped:
            dimensions.append(mapped)

    positives: list[str] = []
    for key in ("explicit_preferences", "positive_preferences", "preferred_values"):
        positives.extend(_strings(profile.get(key)))

    negatives: list[str] = []
    for key in ("negative_preferences", "avoid", "avoidances"):
        negatives.extend(_strings(profile.get(key)))

    return ProfileSignals(
        important_dimensions=_unique(dimensions),
        positive_preferences=_unique(positives),
        negative_preferences=_unique(negatives),
    )
