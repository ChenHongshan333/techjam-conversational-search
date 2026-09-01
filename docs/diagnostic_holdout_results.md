# Catalog-derived 100-case diagnostic holdout

This suite tests whether public-set gains generalize to unseen catalog targets.
It is diagnostic only and is not an estimate of the private leaderboard.

## Construction

`tools/generate_diagnostic_set.py` deterministically selects 100 products that
do not occur as targets in `data/public_set.jsonl`. It uses only the visible
catalog and the local simulator contract.

The set preserves the official scenario proportions:

| Scenario | Cases |
|---|---:|
| Buying | 40 |
| Browsing | 40 |
| Intent Override | 15 |
| Boundary | 5 |

It also balances 40 clothing, 35 shoes and 25 jewelry products across 67 leaf
categories. After all four simulator-visible constraints are intersected, 53
targets are unique, 30 have 2–10 matching products, and 17 deliberately
ambiguous stress cases have 11–100 matching products.

## Current performance

| Configuration | HR@10 | MRR | MTTC | Score | Query tokens |
|---|---:|---:|---:|---:|---:|
| Offline + provenance likelihood | 0.990 | 0.902595 | 2.570 | 0.934378 | 0 |
| Filtered, override-gated Qwen | 0.990 | 0.902595 | 2.570 | 0.934378 | 947 |

Qwen changed zero of the 100 outcomes. Its small public-set improvement did not
generalize to this holdout, so dense retrieval should remain optional.

Breakdown by evidence ambiguity:

| Evidence pool | Cases | HR@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|---:|
| Identifiable (1–10 products) | 83 | 1.000 | 0.947504 | 2.469880 | 0.954854 |
| Ambiguous (11–100 products) | 17 | 0.941176 | 0.683333 | 3.058824 | 0.834412 |

The agent generalizes well when the disclosed evidence identifies a small set.
The remaining weakness is tie-breaking and exploration inside metadata-identical
candidate groups, especially for clothing and shoes. Jewelry scored `0.971600`
because its disclosed attributes were usually more distinctive.

## Discovered miss

`diagnostic_0087` targets `B078BB4ZJZ`, a low-review-count leather belt. After
all disclosed constraints, 18 products remain indistinguishable by exact
metadata. The target stabilizes at fused-pool rank 13. On exhausted turns the
current explorer fills all ten positions with broad facet candidates such as
gift, birthday and outdoor products, skipping the nearest unseen ranks 11–20.

This is not a questioning failure: the simulator has no additional target
attribute to reveal. It is an exploitation-versus-diversity allocation issue.

## Unshipped exploration ablations

| Exploration policy | Public score | Diagnostic HR@10 | Diagnostic score |
|---|---:|---:|---:|
| Current facet/depth exploration | 0.959793 | 0.990 | 0.934378 |
| Preserve three nearest first | 0.958668 | 1.000 | 0.941779 |
| Alternate facet and nearest turns | 0.958668 | 1.000 | 0.942079 |
| **Maximum five facets, then nearest unseen** | **0.959793** | **1.000** | **0.941154** |

The balanced five-facet quota preserves the public result and public case 20's
facet-based recovery while exposing the diagnostic belt on turn 4 at rank 8.
It was tested by runtime monkeypatch only and has not been added to production
code. It is the strongest next implementation candidate from this holdout.

Raw evaluator outputs remain ignored under `artifacts/`:

- `diagnostic_set_100_offline_evaluation.json`
- `diagnostic_set_100_dense_evaluation.json`
