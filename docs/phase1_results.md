# Phase 1: does semantic reranking earn its place?

Measured 2026-08-31. Baseline: Hit Rate@10 1.000, MRR 0.739940, MTTC 2.035,
score 0.901282 (`artifacts/baseline.json`).

## Question

The competition brief names "Multi-Route Retrieval → LLM Semantic Ranking" as a
core pillar, and `ProductReranker` had shipped disabled and unmeasured. With
hit rate already at 1.000, MRR holds 79% of the remaining score headroom, so
reranking was the obvious lever to test.

## What could actually be measured

| Component | Status | Why |
|---|---|---|
| Qwen dense retrieval | **not measured** | OpenRouter account returns HTTP 402, insufficient credits |
| Qwen reranker | **not measured** | same |
| Claude semantic reranking | **measured** | runs on the Anthropic key |

Anthropic serves no embeddings and no rerank endpoint, so the Qwen two-stage
design cannot be run on an Anthropic key at all. A Claude reranker was written
against the Messages API instead -- same `.rerank(query, identifiers)` contract,
structured outputs returning a permutation of the candidate window -- measured,
and then **removed from the tree** once the result came back negative. The
numbers below are what it produced; the implementation is not carried as dead
code.

## Result: reranking makes ranking worse

Measured in isolation — one API call per session
against the lexical top-50, no dialogue, so nothing is confounded by conversion
timing or rotation.

| | sessions | reciprocal rank | rank 1 | inside top 10 | up / down / same | cost |
|---|---|---|---|---|---|---|
| `effort=low`, 600-char docs | 59 | 0.8271 → **0.7604** (−0.0667) | 44 → **40** | 57 → **52** | 5 / 12 / 42 | $3.55 |
| `effort=high`, 1200-char docs | 29 | 0.7977 → **0.7430** (−0.0547) | 21 → **20** | 27 → **25** | 3 / 5 / 21 | $3.70 |

The model is roughly 2× more likely to demote the target than promote it, and
it pushes targets **out of the scored top 10** — which would cost hit rate, the
one metric currently at a perfect 1.000. Raising effort and doubling document
length did not change the sign, so this is not an artifact of a deliberately
cheap configuration; it cost 2.5× more per session and 3× the wall clock.

End-to-end confirmation on whole sessions agrees:

| fusion (base:model) | MRR | rank 1 |
|---|---|---|
| baseline, no rerank | 0.83333 | 7 |
| 25:1 (shipped default) | 0.83333 (+0.00000) | 7 |
| 5:1 | 0.76111 (−0.07222) | 6 |
| 0:1 (model only) | 0.69762 (−0.13571) | 5 |

The shipped 25:1 fusion is a no-op — the base ranking outvotes the model 25 to 1,
which is why the component looked harmless while contributing nothing. The more
authority the model is given, the worse the result gets, monotonically.

## Why this is the expected result

`tools/oracle_bound.py` explains it. Given every constraint the simulator will
ever disclose, the existing lexical retriever already reaches:

```
state given to retriever   hit@10     MRR   rank1
category_only               0.215   0.133   0.095
turn1_buying                0.595   0.385   0.290
hard_only                   0.915   0.770   0.700
full_oracle                 0.990   0.943   0.915
```

At full disclosure the target is already at rank 1 in 91.5% of sessions. There
is almost no ranking error left for a reranker to correct, so a reranker can
mostly only add noise. The live gap (MRR 0.740 vs 0.943) is **not** a ranking
defect — it is that the agent converts at turn 1–2, before enough constraints
have been disclosed.

## Decision

Keep all reranking disabled, and do not carry a Claude reranker in the tree. The
measurement stands as the record; the code was deleted rather than left as an
off-by-default path nobody runs. The Qwen `ProductReranker` remains only because
it predates this work and is still referenced as an optional ablation -- it too
is disabled and, on this evidence, unlikely to be worth enabling.

Total measurement spend: ~$8.62 of Anthropic credit.

## What this redirects to

Ranking quality is not the bottleneck; **information timing** is. The scoring
formula makes one extra turn cost `0.2 × 0.1/200 = 0.0001` while moving one
session from rank 3 to rank 1 gains `0.3 × 0.667/200 = 0.001` — roughly 10× in
favour of gathering more information before converting. Quantifying that
tradeoff needs no API credit and is the recommended next measurement.
