# Shopping Agent Package

```text
shopping_agent/
├── agent.py                 public orchestration entry point
├── config.py                environment-backed runtime settings
├── models.py                shared conversation data structures
├── text.py                  shared text normalization helpers
├── build_semantic_index.py  semantic-index CLI
├── conversation/
│   ├── answers.py           fixed/rule/LLM answer interpretation
│   ├── intent.py            buying/browsing intent classification
│   ├── parser.py            deterministic message parsing
│   ├── profile.py           profile dimension/value distillation
│   ├── state.py             conversation-state updates
│   └── questions.py         clarification planning and rendering
├── retrieval/
│   ├── catalog.py           catalog loading, exact match, and BM25
│   ├── exploration.py       facet-diverse exhausted-state exploration
│   ├── query.py             lexical and semantic query construction
│   └── semantic.py          dense retrieval, fusion, and reranking
└── providers/
    └── openrouter.py        OpenRouter API client
```

External callers should import `ShoppingAgent` from `shopping_agent`. Internal
modules import concrete components from the relevant subpackage.
