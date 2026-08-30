from __future__ import annotations

from dataclasses import dataclass

from ..models import SessionState
from .catalog import Product


# Broad, reusable shopping facets. A product may belong to several queues; selecting
# one unseen item from each queue prevents a single popularity-sorted identity from
# consuming every recommendation slot.
FACETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("style:long_sleeve", ("long sleeve",)),
    ("recipient:grandma", ("grandma",)),
    ("occasion:gift", ("gift",)),
    ("style:short_sleeve", ("short sleeve",)),
    ("style:sleeveless", ("sleeveless", "tank top")),
    ("style:hooded", ("hood", "hoodie")),
    ("neck:v_neck", ("v neck", "v-neck")),
    ("recipient:mother", ("mother", "mom", "mama")),
    ("recipient:father", ("father", "dad", "papa")),
    ("occasion:birthday", ("birthday",)),
    ("occasion:wedding", ("wedding",)),
    ("occasion:holiday", ("christmas", "holiday")),
    ("use:work", ("work", "office")),
    ("use:outdoor", ("outdoor", "hiking", "running")),
    ("fit:classic", ("classic fit",)),
    ("fit:relaxed", ("relaxed fit", "loose fit")),
)


@dataclass(frozen=True)
class DiverseSelection:
    identifiers: list[str]
    facets: tuple[str, ...] = ()


def select_diverse_candidates(
    ranked: list[str],
    products: dict[str, Product],
    state: SessionState,
    limit: int,
    preserve_top: int = 0,
) -> DiverseSelection:
    unseen = [item for item in ranked if item not in state.seen_recommendations]
    pool = unseen or list(ranked)
    selected = pool[:max(0, min(preserve_top, limit))]
    selected_set = set(selected)
    used_facets: list[str] = []

    for name, markers in FACETS:
        if len(selected) >= limit:
            break
        candidate = next(
            (
                item for item in pool
                if item not in selected_set
                and any(marker in products[item].search_text for marker in markers)
            ),
            None,
        )
        if candidate is None:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
        used_facets.append(name)

    # Fill any unused slots from different depth bands before returning to the
    # ranking head. This explores beyond the first page without discarding rank.
    if len(selected) < limit and pool:
        stride = max(1, len(pool) // max(1, (limit - len(selected)) * 3))
        for index in range(stride - 1, len(pool), stride):
            candidate = pool[index]
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        selected.extend(item for item in pool if item not in selected_set)
    return DiverseSelection(selected[:limit], tuple(used_facets))
