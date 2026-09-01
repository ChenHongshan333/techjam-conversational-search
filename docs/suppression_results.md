# Early-conversion suppression: Phases A–C

## Shipped follow-up: progressive widening (2026-09-01)

The original implementation returned no products during its opening gathering
window. The shipped policy now returns rank 1 on turns 1-2 and widens to Top 10
from turn 3. Only displayed products enter `seen_recommendations`.

| Opening policy | Hit Rate@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|
| Silent turns 1-2 | 1.000 | 0.941556 | 3.105 | 0.940367 |
| Rank 1 on turn 1 only | 1.000 | 0.941556 | 2.805 | 0.946367 |
| Rank 2 on turns 1-2 | 1.000 | 0.881556 | 2.190 | 0.940667 |
| **Rank 1 on turns 1-2 (shipped)** | **1.000** | **0.949056** | **2.345** | **0.957817** |
| Rank 1 on turns 1-3 | 0.980 | 0.955833 | 2.595 | 0.944850 |
| Rank 1 on turns 1-4 | 0.975 | 0.957500 | 2.675 | 0.941250 |

The two-turn rank-1 policy is the best measured balance: it keeps all 200 hits,
makes 124 sessions convert earlier, improves three targets to rank 1 with no
rank regressions, and avoids the hit-rate cliff caused by narrowing later turns.
Use `TECHJAM_EARLY_RECOMMENDATION_LIMIT=0` to reproduce the older silent policy.

Baseline: Hit Rate@10 1.000, MRR 0.739940, MTTC 2.035, score 0.901282.

## The hypothesis

With hit rate saturated at 1.000, MRR holds 79% of the remaining headroom. The
oracle bound (`tools/oracle_bound.py`) shows the lexical retriever reaches MRR
0.943 and rank 1 in 91.5% of sessions **once every constraint is disclosed** —
so the live MRR of 0.740 is not a ranking defect. The agent converts at turn 1–2,
before enough has been disclosed, and takes whatever rank it gets.

The scoring formula makes that trade lopsided. One extra turn costs
`0.2 × 0.1/200 = 0.0001`; moving one session from rank 3 to rank 1 gains
`0.3 × 0.667/200 = 0.001`. Roughly 10× in favour of waiting.

## Phase A — instrument, no behaviour change

`tools/instrument.py` records per turn: `fused_candidate_count`, top-1 and top-2
fused scores, their margin, and the target's true rank in the **full** fused pool
(not just the ten emitted ids). Written to `artifacts/instrumentation.json`.

Gate held: score 0.901282 unchanged, all tests green.

### Answer: the margin signal does *not* cleanly separate the groups

Comparing turn 1 of the 30 tail sessions (best rank ≥ 5) against the 27 that hit
rank 1 on turn 1:

| signal | tail median | clean median | ratio |
|---|---|---|---|
| `fused_candidate_count` | 1016.5 | 1691.0 | 0.60 |
| `fused_margin` | 0.0254 | 0.0298 | 0.85 |
| `fused_relative_margin` | 0.0147 | 0.0168 | 0.87 |

Best achievable thresholds:

```
fused_relative_margin   catches 93% of tail, but also hits 74% of clean   (+19%)
fused_candidate_count   catches 50% of tail, but also hits 22% of clean   (+28%)
```

A threshold catching 93% of the tail while also firing on 74% of the healthy
sessions is not a discriminator — it is near-unconditional suppression wearing a
threshold. Note also that **pool size runs the opposite way to the hypothesis**:
tail sessions have *smaller* candidate pools, not larger.

### But the mechanism itself is confirmed

```
        median target rank in pool at turn 1    in top 10
tail                     9                        16/30
clean                    1                        27/27
```

Over half the tail sessions already hold the target inside the emitted top 10 on
turn 1 — they convert immediately at a rank the evidence does not support.

One correction to the framing: of the **64** turn-1 hits, only **27** are at rank
1. The other **37** convert on turn 1 at rank 2–10. That widens the opportunity:

```
max MRR-term gain if all 37 reached rank 1   +0.03832
efficiency cost of delaying all 64 by a turn -0.00640
net ceiling                                  +0.03192
```

So: no usable threshold, but a real and well-quantified mechanism. Phase B tests
the mechanism directly, which needs no threshold.

## Phase B — unconditional turn-1 suppression

Turn 1 returns the clarifying question with `recommendations: []`. Legal:
`normalize_recommendations` yields an empty list, no hit registers, the session
continues.

| | baseline | Phase B | Δ |
|---|---|---|---|
| Hit Rate@10 | 1.000 | **1.000** | held |
| MRR | 0.739940 | **0.850603** | **+0.110663** |
| MTTC | 2.035 | 2.355 | +0.320 |
| **Score** | 0.901282 | **0.928081** | **+0.026799** |
| rank-1 sessions | 125 | **156** | **+31** |

```
before  1:125  2:26  3:13  4:6   5:4  6:5  7:4  8:6  9:9  10:2
after   1:156  2:16  3:8   4:5   5:3  6:4  7:1  8:2  9:4  10:1
```

MTTC moved exactly the predicted +0.320 (64 turn-1 hits pushed to turn 2). The
MRR term gained +0.0332 in score against a −0.0064 efficiency cost — a 5:1
payoff. Per scenario, `buying` improved most (MRR 0.708 → 0.890) and
`intent_override` was unchanged at 0.870, as expected since those sessions cannot
convert before the override lands on turn 3–4.

The mechanism is real. Early conversion was the cause of the tail.

## Phase C — gate on the signal

Three safety rules hold in every mode, unit-tested in
`tests/test_agent_behavior.py::SuppressionRuleTest`:

1. **Never withhold on turn 10** — that turn decides the 0.5-weighted hit rate.
2. **Cap withheld turns per session** (default 2).
3. **Never withhold when the last question returned nothing** — waiting cannot
   help if nothing is arriving.

### Sweep results (`tools/suppression_sweep.py`)

| config | hit | MRR | MTTC | score | Δscore | rank1 |
|---|---|---|---|---|---|---|
| margin 0.06, cap 2 | 1.000 | 0.93553 | 2.890 | **0.942860** | +0.041578 | 180 |
| margin 0.025, cap 2 | 1.000 | 0.91137 | 2.605 | 0.941310 | +0.040028 | 171 |
| margin 0.035, cap 2 | 0.995 | 0.92553 | 2.725 | 0.940660 | +0.039378 | 177 |
| margin 0.06, cap 1 | 1.000 | 0.85060 | 2.325 | 0.928681 | +0.027399 | 156 |
| turn 1 only (Phase B) | 1.000 | 0.85060 | 2.355 | 0.928081 | +0.026799 | 156 |
| margin 0.02, cap 2 | 0.990 | 0.85699 | 2.570 | 0.920697 | +0.019415 | 157 |
| margin 0.015, cap 2 | 0.995 | 0.81833 | 2.395 | 0.915099 | +0.013817 | 144 |
| margin 0.01, cap 2 | 0.995 | 0.77340 | 2.220 | 0.905120 | +0.003838 | 133 |
| off | 1.000 | 0.73994 | 2.035 | 0.901282 | — | 125 |

### Two findings that decide what ships

**The threshold is not doing the work.** `margin 0.06` withholds 1.60 turns per
session against a cap of 2 — at that setting almost every turn clears the gate,
so the config is approximately "withhold the first two turns". The gain comes
from the extra turn, not from the gating.

**Hit rate is fragile under two-turn suppression, and not monotonically so:**

```
margin  0.010  0.015  0.020  0.025  0.035  0.060
hit     0.995  0.995  0.990  1.000  0.995  1.000
```

The two configs that hold 1.000 do so by luck of which sessions the threshold
happened to catch, not by construction. Fitting that threshold to 200 public
sessions and shipping it against 800 private ones would be a liability.

Two guards were added in response:

- **Reserve the closing turns** (`suppression_reserve_turns`, default 3) — never
  withhold on turns 8-10, so a session always keeps emitting turns in hand.
- An unconditional **`turns` mode** with no fitted parameter, so the mechanism
  can be shipped without a threshold tuned to the public split.

## Track 1 — how far can suppression go?

### 1b: the residual tail is the same failure, one turn later

Of the 44 sessions Phase B left off rank 1, **34 convert on turn 2**, and 21 of
those at rank 3-9. Max gain if all reached rank 1: +0.0346, against a −0.0200
cost for one more withheld turn.

### 1c: browsing wants *less* gathering, not more

The hypothesis was that browsing — which discloses nothing on turn 1 — needs a
longer runway. The measurement says the opposite:

| config | hit | score | browsing rank-1 | buying rank-1 |
|---|---|---|---|---|
| turns2 | **1.000** | **0.937510** | 73/80 | 74/80 |
| turns2, browsing 3 | 0.815 | 0.718907 | **8/80** | 74/80 |
| turns3 | 0.605 | 0.487242 | 6/80 | 12/80 |
| turns4 | 0.590 | 0.466838 | — | — |

Giving browsing a third turn costs 37 hits. Per-scenario tuning is actively
harmful here.

`intent_override` also needs no special case: its MTTC is **3.733 under both the
baseline and Phase B**. The evaluator already blocks conversion before the
override lands, so withholding those turns costs nothing and excluding them
would recover nothing.

### The cliff is a rotation bug, not a suppression limit

Beyond two withheld turns the hit rate collapses. The cause is an interaction:
with three turns withheld, the first emission lands *after* the customer has
exhausted their intent card, so the session enters that turn with
`exhausted=True`, `rotating` becomes true, and **the first list it ever emits
comes from `select_diverse_candidates`** — a coverage set instead of the
precision ranking.

Guarding suppression on exhaustion did not help (turns9: 0.590 both before and
after) because by then the turns are already spent. The fix is in the rotation
condition itself:

```python
rotating = exhausted and query_signature == state.last_query_signature and bool(state.seen_recommendations)
```

Rotation exists to move past products already shown; with nothing shown there is
nothing to rotate away from. This is a latent bug independent of suppression —
any session reaching exhaustion before emitting would diversify prematurely.

### What the rotation fix bought

Re-running the same configs after the one-line rotation fix:

| config | hit before | hit after | score before | score after |
|---|---|---|---|---|
| turns3 | 0.605 | **1.000** | 0.487242 | 0.921660 |
| turns4 | 0.590 | 0.985 | 0.466838 | 0.901256 |
| turns9 | 0.590 | 0.985 | 0.466838 | 0.901256 |

A collapse became graceful degradation, which confirms the cliff was the
premature-diversification bug rather than a limit on how long a session can
usefully stay silent.

`turns2` still wins on score: `turns3` edges MRR (0.93553 vs 0.93303) and rank-1
(180 vs 179), but its MTTC of 3.950 against 3.120 costs more than the MRR buys.
Nothing beats 0.937510 at hit rate 1.000, so the shipped config is unchanged.

The value of the fix is robustness rather than score. `turns2` was previously
safe only because 2 happens to be the last turn before *this* evaluator runs out
of information; a private session with a shallower intent card would have hit the
same cliff. The worst case is now 0.985, not 0.60.

**Known edge:** at `turns4`, boundary rank-1 falls from 8/10 to 2/10. A boundary
reply sets `boundary_observed` but not `rejected_attributes`, so the exhaustion
guard never fires for those sessions. It does not bite at `turns2` (boundary
holds at 8/10, matching the baseline), and treating boundary as exhaustion would
be wrong -- the customer may still answer about other attributes.

### Shipped

`turns2` is the default: `TECHJAM_SUPPRESSION=1`, mode `turns`, 2 turns, with a
3-turn closing reserve and the exhaustion guard.

| | before | shipped |
|---|---|---|
| Hit Rate@10 | 1.000 | **1.000** |
| MRR | 0.739940 | **0.933034** |
| MTTC | 2.035 | 3.120 |
| **Score** | 0.901282 | **0.937510** |
| rank-1 | 125 | **179** |

Per scenario: browsing MRR 0.705 → 0.946, buying 0.708 → 0.950.
`public_0020` still converts at turn 7, rank 1.
