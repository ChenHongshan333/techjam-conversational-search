from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .catalog import CatalogIndex
from .config import RetrievalSettings
from .openrouter import OpenRouterClient
from .semantic import DenseProductRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cached two-view product embedding index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retries", type=int)
    args = parser.parse_args()

    settings = RetrievalSettings.from_environment()
    overrides = {
        "embedding_model": args.model,
        "embedding_dimensions": args.dimensions,
        "embedding_batch_size": args.batch_size,
        "embedding_workers": args.workers,
        "embedding_job_retries": args.retries,
    }
    settings = replace(settings, **{key: value for key, value in overrides.items() if value is not None})
    if not settings.api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    if settings.embedding_dimensions < 32 or settings.embedding_dimensions > 4096:
        raise SystemExit("Embedding dimensions must be between 32 and 4096")
    if settings.embedding_batch_size < 1 or settings.embedding_workers < 1:
        raise SystemExit("Batch size and worker count must be positive")
    catalog_path = Path(args.catalog)
    catalog = CatalogIndex(catalog_path)
    client = OpenRouterClient(settings.api_key, settings.request_timeout_seconds)
    retriever = DenseProductRetriever(catalog, catalog_path, client, settings)
    print(
        f"Building {settings.embedding_model} index at {settings.embedding_dimensions} dimensions "
        f"with batch size {settings.embedding_batch_size} and {settings.embedding_workers} workers",
        flush=True,
    )
    cache_path, prompt_tokens = retriever.build_index()
    print(f"Semantic index ready: {cache_path}")
    print(f"Embedding input tokens used during this run: {prompt_tokens}")


if __name__ == "__main__":
    main()
