# Category hierarchy and post-override provenance reranking

Validated on all 200 public sessions with remote components disabled.

| Configuration | HR@10 | MRR | MTTC | Score | Override MRR |
|---|---:|---:|---:|---:|---:|
| Progressive-opening baseline | 1.000 | 0.949056 | 2.345 | 0.957817 | 0.898611 |
| Exact category suffix (`0.5`) | 1.000 | 0.949464 | 2.325 | 0.958339 | 0.898611 |
| Retracted-context RRF route (`1.0`) | 1.000 | 0.951556 | 2.345 | 0.958567 | 0.915278 |
| **Both, conservative default** | **1.000** | **0.951964** | **2.325** | **0.959089** | **0.915278** |
| Both + provenance likelihood | 1.000 | 0.953978 | 2.320 | 0.959793 | 0.928704 |

## Conservative default

The category bonus requires an exact normalized match between the requested
category and the product's final two taxonomy nodes. The retracted-context route
searches attribute fields independently at RRF weight `1.0`; retracted values do
not enter live constraint intersection or coverage.

## Optional provenance likelihood

`TECHJAM_OVERRIDE_LIKELIHOOD=1` enables a deterministic post-override reranker.
It models whether a candidate's ordered salient catalog attributes could
plausibly have produced the observed conversation:

- the first answered constraint bundle is compared with likely hard attributes;
- later answered values are compared with likely soft attributes;
- the retracted opening value is context, never a live requirement;
- the original fused rank breaks every remaining tie.

The implementation uses catalog metadata and the agent's own constraint
provenance only. It does not import evaluator code, inspect ground truth, or use
a target identifier. It remains opt-in because ordered metadata salience is more
simulator-sensitive than the conservative category and RRF changes.

## Rejected ablations

- Removing the product quality prior: score `0.925848`, HR@10 `0.990`.
- Exact-constraint IDF weighting: no measurable change.
- Extending the hard constraint lock to the override track: no measurable change.
- Naive override-time diversity: severe hit-rate regression because products
  shown before the override were incorrectly excluded as already seen.
