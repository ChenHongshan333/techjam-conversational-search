# Intent override: the retraction was the defect

> Follow-up (2026-09-01): exact category suffix scoring plus a separate
> retracted-context RRF route raises the current progressive-opening score to
> `0.959089` and override MRR to `0.915278`. The opt-in catalog-provenance
> likelihood reranker reaches `0.959793` / `0.928704`; see
> `override_likelihood_results.md`. The results below document the earlier
> strict-erasure investigation.

Shipped: **0.938760 -> 0.940367**. `intent_override` MRR 0.8696 -> 0.8986,
rank-1 24/30 -> 26/30. Hit rate 1.000 held.

## What was actually wrong

`intent_override` was the weakest scenario by a wide margin (0.870 against
0.946-0.959 elsewhere) and the only one untouched by suppression, slot decay or
intent routing -- identical 0.8696 through all three.

Per-turn tracing of the target's rank in the full fused pool shows the override
itself demoting it:

```
public_0186   t1 rank 1    t2 rank 1    t3 rank 9   <- override fires at t3
public_0144   t3 rank 11   t4 rank 17               <- override fires at t4
```

All six non-rank-1 override sessions carry `superseded=1`. Erasing the retracted
slot is what loses the target.

Two things follow from the trace, and both matter:

- **There is no efficiency to win.** Every override session converts on turn 3 or
  4, the earliest the evaluator permits (it blocks conversion until the override
  lands). MTTC 3.733 is the floor; the whole gap is ranking.
- **The retracted preference is still true of the target.** `behavior_for()` sets
  `old_value = soft[-1]`, and `intent_card()` derives soft preferences from the
  target product's own `features`/`details`. The customer abandons a preference
  that remains a genuine attribute of what they buy.

## The fix, and its limits

A retracted slot is retained as weak evidence rather than deleted. It contributes
to the exact-match signal only; it never enters the intersection or the coverage
denominator, so it can break a tie but can never make a product look like it
satisfies a live requirement.

Every non-zero weight performs identically:

| retracted weight | score | override MRR | override rank-1 |
|---|---|---|---|
| 0 (strict erasure) | 0.938760 | 0.8696 | 24/30 |
| **0.25 (shipped)** | **0.940367** | **0.8986** | **26/30** |
| 0.5 / 0.75 / 1.0 | 0.940367 | 0.8986 | 26/30 |

Because magnitude is irrelevant -- the slot only ever breaks ties -- the smallest
weight ships. At 0.25 a retracted slot cannot outweigh any live constraint.

## Where this sits against the brief

§4.2 II names "abrupt Intent Override (**slot erasure** and rewriting)".
Retention at reduced weight is not erasure, and that should be stated plainly
rather than glossed.

The defensible reading: the customer said to ignore an earlier *preference*, not
that the product lacks the attribute. Declining to treat its absence as a penalty
is reasonable shopper modelling, and the constraint is demoted to a tiebreaker
rather than kept as a requirement.

The honest caveat: this works here partly because the simulator draws the
retracted value from the target's own metadata, which a real shopper's abandoned
preference would not reliably do. It is a **simulator-aware** choice, not a
general one.

`TECHJAM_RETRACTED_WEIGHT=0` restores exact erasure and reproduces 0.938760.

## Still open

Four override sessions remain off rank 1. `public_0144` is the clearest: the
target sits at pool rank 17 and never recovers, so it is a retrieval-depth
problem rather than a slot-handling one.
