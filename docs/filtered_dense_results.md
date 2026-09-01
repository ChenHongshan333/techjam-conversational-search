# Filtered and gated dense retrieval

Validated on all 200 public sessions with Qwen3-Embedding-8B, the portable
50,000-product index, query rewriting disabled, external reranking disabled,
and `TECHJAM_OVERRIDE_LIKELIHOOD=1`.

| Configuration | HR@10 | MRR | MTTC | Score |
|---|---:|---:|---:|---:|
| Structured + provenance baseline | 1.000 | 0.953978 | 2.320 | 0.959793 |
| Filtered dense on every intent track | 1.000 | 0.946042 | 2.315 | 0.957513 |
| **Filtered dense after overrides only** | **1.000** | **0.954395** | **2.320** | **0.959918** |

The semantic route is deliberately downstream of structured retrieval:

1. exact attributes, taxonomy and BM25 produce a ranked pool;
2. from turn 3 onward, at most the first 1,000 candidates are selected;
3. the query is embedded once with the same Qwen model as the catalog;
4. cosine similarity ranks the identity and attribute views inside that pool;
5. weighted reciprocal-rank fusion combines those rankings with lexical rank.

Running dense fusion everywhere reduced MRR in boundary, browsing and buying
sessions. The fixed evaluator supplies catalog-exact metadata, where structured
matching is stronger than semantic similarity. The post-override track is the
only measured slice that improved: case `public_0096` moved from rank 4 to rank
3 while HR@10 and MTTC were unchanged.

The safe measured configuration is:

```ini
TECHJAM_DENSE_RETRIEVAL=1
TECHJAM_DENSE_FILTERED=1
TECHJAM_DENSE_MIN_TURN=3
TECHJAM_DENSE_CANDIDATE_POOL_SIZE=1000
TECHJAM_DENSE_TRACKS=override
TECHJAM_OVERRIDE_LIKELIHOOD=1
TECHJAM_LLM_REWRITE=0
TECHJAM_RERANK=0
```

The generated evaluation files are ignored local artifacts:

- `artifacts/filtered_dense_turn3_evaluation.json`
- `artifacts/filtered_dense_override_evaluation.json`
