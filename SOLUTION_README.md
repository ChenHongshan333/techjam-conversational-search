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
| **Qwen8B dense hybrid** | **0.995** | **0.744899** | **2.055** | **0.899870** |

The final system finds 199/200 targets. The remaining miss is an ambiguous item
whose distinguishing title text is never disclosed by the simulator.

## Architecture

```text
Customer message
      ↓
Conversation state: category, constraints, overrides, profile
      ↓
┌──────────────────────────┬─────────────────────────────┐
│ Exact + fielded BM25     │ Qwen3-Embedding-8B         │
│ lexical retrieval        │ identity + attribute index │
└──────────────────────────┴─────────────────────────────┘
      ↓ weighted reciprocal-rank fusion (50:1:1)
Ranked candidates → unseen-result rotation → Top 10
```

The agent asks `other` until the evaluator's finite intent card is exhausted.
An adaptive clarification policy now separates the natural-language question
focus from that protocol fallback. Fixed evaluator phrases are classified by
rules; unclear first-turn messages can optionally use an LLM intent classifier.
The question focus combines buying/browsing mode, profile priorities, known and
rejected constraints, and differences among the current candidates. The LLM is
not allowed to freely choose questions, and every remote failure falls back to
the deterministic policy.

Dense retrieval supports ranking but does not override strong exact metadata
matches. GPT-5.6 Luna rewriting and Qwen3-Reranker-8B are implemented as optional
ablations; neither improved the validated final score, so both remain disabled.

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
TECHJAM_LLM_INTENT=0
TECHJAM_DENSE_RETRIEVAL=1
TECHJAM_LLM_REWRITE=0
TECHJAM_RERANK=0
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

The product index is distributed separately because generated artifacts are not
tracked by Git. From the repository root, unpack `submission-assets.zip`; it
restores the index at the correct relative path:

```bash
unzip submission-assets.zip
```

Then configure:

```bash
export OPENROUTER_API_KEY=your_key
export TECHJAM_SEMANTIC_INDEX_PATH=artifacts/semantic_cache/catalog_qwen3_embedding_8b_512_v1.npz
export TECHJAM_LLM_INTENT=0
export TECHJAM_DENSE_RETRIEVAL=1
export TECHJAM_LLM_REWRITE=0
export TECHJAM_RERANK=0
```

The index contains all 50,000 product vectors and does not need rebuilding.
OpenRouter access is still required online to embed each new conversation query
with the same Qwen model. The loader verifies the catalog content, model,
dimensions, document schema, and index format before using the asset.

## Evaluate

```bash
TECHJAM_DENSE_RETRIEVAL=1 \
  .venv/bin/python -m evaluator.local_evaluator \
  --output artifacts/evaluation.json

# Windows PowerShell
$env:TECHJAM_DENSE_RETRIEVAL = "1"
.venv\Scripts\python.exe -m evaluator.local_evaluator --output artifacts\latest_evaluation.json
```

Run the deterministic offline fallback by setting
`TECHJAM_DENSE_RETRIEVAL=0`.

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
