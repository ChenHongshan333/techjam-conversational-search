# Dual-track intent routing

Shipped: **0.937510 → 0.938760**, hit rate 1.000 held, rank-1 179 → 180.

## The defect this fixes

The classifier was well built and completely inert. Forcing every mode produced
byte-identical results:

```
forced=buying     0.937510
forced=browsing   0.937510
forced=uncertain  0.937510
```

It fed exactly two consumers. One was the dense-retrieval fusion weights, inside
`if self.dead_retriever is not None` -- dense is off by default, so that branch
never ran. The other was clarification-question ordering, which cannot change
retrieval because `ask_attribute` is pinned to `other` on the fixed protocol.

Intent now feeds `legacy_score`, the route carrying weight 100.0 against <=2.5
for every other route, which alone determines the final order.

## What the measurements actually said

Routing buying and browsing in *opposite* directions -- the intuitive reading of
the brief -- loses on every setting tried:

| variant | score | vs baseline |
|---|---|---|
| scale 0.25 | 0.937287 | −0.000223 |
| scale 0.50 | 0.936656 | −0.000854 |
| scale 1.00 | 0.933443 | −0.004067 |
| buying strong / browsing inverted | 0.935872 | −0.001638 |

The reason is structural: suppression withholds turns 1-2, and by the turn the
agent first emits, a browsing session has disclosed the same kind of catalog
constraints a buying session has. The label describes the opening message, not
the evidence. The two tracks therefore differ in **degree**, not direction.

## The override track

Intent modes across the public set are `buying: 112, browsing: 88` -- **no
`uncertain` at all**, because an override message ("what I *need* is:") trips the
buying pattern. Those sessions were inheriting buying's hard-constraint emphasis
for requirements that had just been rewritten, costing `intent_override` 0.024
MRR. They now relax to the neutral track, which recovers it in full.

## Shipped configuration

| track | exact | coverage | intersection | route bonus | category |
|---|---|---|---|---|---|
| buying | 1.50 | 1.50 | 1.50 | 0.70 | 0.60 |
| browsing | 1.30 | 1.30 | 1.30 | 0.80 | 0.75 |
| uncertain / post-override | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Per scenario against the pre-routing baseline:

| scenario | before | after |
|---|---|---|
| buying | 0.9503 | **0.9589** |
| boundary | 0.8833 | **0.9000** |
| browsing | 0.9457 | 0.9456 |
| intent_override | 0.8696 | 0.8696 |

## A deliberate trade worth recording

Giving browsing the *same* emphasis as buying scores higher -- **0.939510** --
because browsing MRR rises to 0.9518. That configuration was rejected: with both
live tracks sharing weights the classifier stops being consequential, and the
pillar reduces to a single global weighting. The routed configuration gives up
**0.00075** to keep buying and browsing genuinely distinct.

`TECHJAM_INTENT_ROUTING_SCALE=0` reduces every track to the neutral weighting and
reproduces the pre-routing 0.937510 exactly, so the routing is auditable.

Four tests in `IntentRoutingTest` fail if the tracks ever collapse back together
or the scale defaults to zero.
