# TechJam Conversational Product Search

A multi-turn shopping agent for retrieving a hidden product from a frozen
50,000-item Amazon catalog. The solution combines structured conversation state,
exact attribute matching, BM25, and Qwen3 dense retrieval while preserving a
fully offline lexical fallback.

## Results

Validated on all 200 public sessions:

| Configuration | Hit Rate@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|
| Starter BM25 | 0.125 | 0.068034 | 9.810 | — |
| Structured lexical agent | 0.995 | 0.741480 | 2.055 | 0.898844 |
| Clarification + facet exploration (offline) | 1.000 | 0.739940 | 2.035 | 0.901282 |
| Qwen8B dense hybrid | 0.995 | 0.744899 | 2.055 | 0.899870 |
| + early-conversion suppression | 1.000 | 0.933034 | 3.120 | 0.937510 |
| + dual-track intent routing | 1.000 | 0.937200 | 3.120 | 0.938760 |
| + weak retracted-preference evidence | 1.000 | 0.941556 | 3.105 | 0.940367 |
| + Qwen8B dense retrieval on prior silent policy (ablation) | 1.000 | 0.937083 | 3.105 | 0.939025 |
| + rank-1 turns 1-2, Top 10 from turn 3 | 1.000 | 0.949056 | 2.345 | 0.957817 |
| + category suffix + retracted-context route | 1.000 | 0.951964 | 2.325 | 0.959089 |
| + post-override provenance likelihood (opt-in) | 1.000 | 0.953978 | 2.320 | 0.959793 |
| + filtered, override-gated Qwen dense fusion (opt-in) | 1.000 | 0.954395 | 2.320 | 0.959918 |
| **+ exposure demotion + override provenance (shipped)** | **1.000** | **0.965853** | **2.305** | **0.963656** |

The shipped deterministic path finds 200/200 targets. Previously exposed
products receive a recoverable rank penalty on later turns, so fresh candidates
can surface without permanently deleting any product; exposure state resets on
an intent override.
Facet-diverse exploration recovers the formerly missed ambiguous item without
changing the evaluator. Turns 1-2 expose only the best candidate while the agent
gathers evidence; turn 3 onward exposes the full Top 10. This improves both MRR
and time to conversion over the previous fully silent opening policy -- see
`docs/suppression_results.md`.

An additional deterministic 100-case catalog-derived holdout uses unseen target
products and the same 40/40/15/5 scenario proportions. The current
offline/provenance configuration scores `0.940612` with 99/100 hits; the 83
cases whose disclosed evidence narrows to at most ten products score `0.954854`.
Filtered Qwen retrieval changed none of the 100 outcomes. See
`docs/diagnostic_holdout_results.md` for construction, failure analysis, and an
unshipped exploration-quota ablation that recovers the miss without changing
the public score.

Unrestricted semantic retrieval lowers ranking quality, so it stays disabled by
default. On the current progressive-opening agent, running filtered dense fusion
for every track scored `0.957513`; restricting it to post-override turns scored
`0.959918`, a small gain over the `0.959793` lexical/provenance configuration.
See `docs/filtered_dense_results.md`. Buying and browsing route through different
retrieval weightings; see `docs/intent_routing_results.md`.

## Architecture

```text
Customer message → fixed/rule/optional-LLM answer interpreter
      ↓
Conversation state: category, constraints, overrides, intent, distilled profile
      ↓
┌──────────────────────────┬─────────────────────────────┐
│ Exact + fielded BM25     │ Qwen3-Embedding-8B         │
│ lexical retrieval        │ identity + attribute index │
└──────────────────────────┴─────────────────────────────┘
      ↓ weighted reciprocal-rank fusion (50:1:1)
Ranked candidates → exact taxonomy suffix + weak historical-context route
                  → optional post-override provenance likelihood
                  → facet-diverse exhausted-state exploration
                  → rank 1 on turns 1-2; Top 10 from turn 3
```

The clarification policy separates natural-language focus from the structured
protocol. Fixed evaluator conversations keep the backwards-compatible `other`
fallback; natural conversations can ask a high-confidence specific attribute.
Fixed answers use Regex, natural follow-ups use conservative rules, and unclear
answers can optionally use an LLM extractor. Fixed intent phrases use rules;
unclear first-turn messages can optionally use an LLM intent classifier.
The question focus combines buying/browsing mode, profile priorities, known and
rejected constraints, and differences among the current candidates. The LLM is
not allowed to freely choose questions, and every remote failure falls back to
the deterministic policy.

Dense retrieval supports ranking but does not override strong exact metadata
matches. GPT-5.6 Luna rewriting and Qwen3-Reranker-8B are implemented as optional
ablations; neither improved the validated final score, so both remain disabled.
When enabled, Qwen dense retrieval waits until turn 3, scores only the top 1,000
structured candidates, and by default runs only after a preference override.

## Setup

Python 3.10+ is recommended. Place the supplied catalog at
`data/catalog.jsonl`, then install the semantic dependency:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-semantic.txt
```

Create `.env`:

```bash
OPENROUTER_API_KEY=your_key
TECHJAM_LLM_ANSWER=0
TECHJAM_LLM_INTENT=0
TECHJAM_DENSE_RETRIEVAL=0
TECHJAM_DENSE_FILTERED=1
TECHJAM_DENSE_MIN_TURN=3
TECHJAM_DENSE_CANDIDATE_POOL_SIZE=1000
TECHJAM_DENSE_TRACKS=override
TECHJAM_LLM_REWRITE=0
TECHJAM_RERANK=0
TECHJAM_RETRACTED_WEIGHT=0.25
TECHJAM_EXACT_CATEGORY_SUFFIX_BONUS=0.5
TECHJAM_RETRACTED_CONTEXT_WEIGHT=1.0
TECHJAM_OVERRIDE_LIKELIHOOD=1
TECHJAM_EARLY_RECOMMENDATION_LIMIT=1
TECHJAM_RECOMMENDATION_POLICY=current
TECHJAM_EXPOSURE_DEMOTION=1
TECHJAM_EXPOSURE_RANK_PENALTY=10
```

Never commit `.env`.

## Build the Semantic Index

```bash
.venv/bin/python -m shopping_agent.build_semantic_index
```

The one-time build embeds two views of every product using
`qwen/qwen3-embedding-8b` at 512 dimensions. It costs approximately $0.13–$0.20
at current OpenRouter pricing and produces a roughly 95 MB cache under
`artifacts/semantic_cache/` named
`catalog_qwen3_embedding_8b_512_v1.npz`.

Every successful batch is checkpointed. Re-running the command resumes after
rate limits, disconnections, or interruption. Product embeddings and repeated
query vectors are reused locally.

### Use the Prebuilt Submission Asset

The product index is packaged in the repository-root `submission-assets.zip`;
the extracted generated cache remains ignored by Git. From the repository root,
unpack the archive to restore the index at the correct relative path:

```bash
unzip submission-assets.zip
```

Then configure:

```bash
export OPENROUTER_API_KEY=your_key
export TECHJAM_SEMANTIC_INDEX_PATH=artifacts/semantic_cache/catalog_qwen3_embedding_8b_512_v1.npz
export TECHJAM_LLM_ANSWER=0
export TECHJAM_LLM_INTENT=0
export TECHJAM_DENSE_RETRIEVAL=1
export TECHJAM_DENSE_FILTERED=1
export TECHJAM_DENSE_MIN_TURN=3
export TECHJAM_DENSE_CANDIDATE_POOL_SIZE=1000
export TECHJAM_DENSE_TRACKS=override
export TECHJAM_LLM_REWRITE=0
export TECHJAM_RERANK=0
export TECHJAM_OVERRIDE_LIKELIHOOD=1
```

The index contains all 50,000 product vectors and does not need rebuilding.
OpenRouter access is still required online to embed each new conversation query
with the same Qwen model. The loader verifies the catalog content, model,
dimensions, document schema, and index format before using the asset.

## Evaluate

```bash
.venv/bin/python -m evaluator.local_evaluator \
  --output artifacts/evaluation.json

# Windows PowerShell
.venv\Scripts\python.exe -m evaluator.local_evaluator --output artifacts\latest_evaluation.json
```

The validated default is deterministic and offline (`0.963656`). It combines
recoverable exposure demotion with the post-override provenance route. Enabling
filtered, override-gated Qwen retrieval is optional and is not required for the
best measured configuration.
Do not set `TECHJAM_DENSE_TRACKS=all`: applying semantic fusion to every track
scored `0.957513`, below the offline path.

## Dashboard

```bash
# macOS / Linux
.venv/bin/python -m dashboard.app --port 8000

# Windows PowerShell
.venv\Scripts\python.exe -m dashboard.app --port 8000
```

Open `http://127.0.0.1:8000`. The minimalist dashboard can replay any public
case turn by turn and display aggregate scoring and retrieval diagnostics. Use a
different port if 8000 is already occupied.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Main Files

```text
starter/agent.py                         competition entry point
shopping_agent/agent.py                  conversation and retrieval pipeline
shopping_agent/conversation/              intent, parsing, state, and questions
shopping_agent/retrieval/                 lexical, dense, fusion, and reranking
shopping_agent/providers/                 external model-provider clients
shopping_agent/models.py                  shared conversation data structures
shopping_agent/config.py                  environment-backed settings
shopping_agent/build_semantic_index.py   resumable index builder
evaluator/local_evaluator.py             public simulator and scorer
dashboard/                               local replay and metrics UI
```

## Data and Compliance

The catalog is derived from Amazon Reviews 2023, `Clothing_Shoes_and_Jewelry`,
using `parent_asin` as the product key. Offline embeddings are generated only
from participant-visible catalog text; no private labels or holdout sessions are
used. See `DATA_ATTRIBUTION.md`, `docs/competition_specification.md`, and
`docs/agent_api_contract.json`.
