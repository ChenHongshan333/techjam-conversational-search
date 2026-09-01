# Seekly Vercel add-on

This directory contains the optional Vercel presentation layer. It does not
replace or modify the competition entry point, evaluator, retrieval code, or
local dashboard.

The add-on differs from the local dashboard in two deployment-specific ways:

- a complete public-case replay is calculated in one serverless request and
  then revealed turn by turn in the browser, so no server session is required;
- the published metrics are read from `validated_results.json`; live evaluator
  runs remain local-only.

Vercel uses the repository-root `vercel.json`, `api/index.py`, and
`.vercelignore`. During the build, `prepare.py` extracts the frozen catalog from
`catalog.jsonl.gz` into the generated, ignored `data/catalog.jsonl` path.

The winning deterministic configuration requires no environment variables or
API key. The Vercel deployment intentionally excludes the optional Qwen index
archive and all ignored evaluation artifacts.
