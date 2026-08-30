from __future__ import annotations

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from ..config import RetrievalSettings
from ..providers.openrouter import OpenRouterClient, OpenRouterError
from .catalog import CatalogIndex, Product


INDEX_FORMAT_VERSION = 1
DOCUMENT_SCHEMA = "two-view-identity-attributes-v2"


def product_identity_document(product: Product) -> str:
    return f"Title and identity: {product.identity_text}"


def product_attribute_document(product: Product) -> str:
    return f"Title: {product.title}\nAttributes: {product.attribute_text}"


def product_rerank_document(product: Product) -> str:
    return (
        f"Title: {product.title}\n"
        f"Category: {product.categories}\n"
        f"Attributes: {product.attribute_text}"
    )[:6000]


def embedding_query_document(model: str, query: str) -> str:
    """Apply Qwen's retrieval instruction on queries, never on catalog documents."""
    if "qwen3-embedding" not in model.casefold():
        return query
    return (
        "Instruct: Given a conversational shopping request, retrieve the most relevant "
        "products from an e-commerce catalog. Preserve all disclosed product constraints.\n"
        f"Query: {query}"
    )


def rerank_query_document(model: str, query: str) -> str:
    if "qwen3-reranker" not in model.casefold():
        return query
    return (
        "Instruct: Rank e-commerce products by compatibility with the disclosed shopping "
        "request. Treat explicit category and product attributes as hard relevance evidence. "
        "Do not invent undisclosed preferences or reward popularity by itself.\n"
        f"Query: {query}"
    )


@dataclass
class DenseSearchResult:
    identity_ranking: list[str]
    attribute_ranking: list[str]
    prompt_tokens: int = 0
    error: str | None = None


@dataclass
class RerankResult:
    ranking: list[str]
    prompt_tokens: int = 0
    error: str | None = None


class DenseProductRetriever:
    """Checkpointed two-view dense index backed by OpenRouter embeddings."""

    def __init__(
        self,
        catalog: CatalogIndex,
        catalog_path: Path,
        client: OpenRouterClient,
        settings: RetrievalSettings,
    ) -> None:
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.client = client
        self.settings = settings
        self.identifiers = list(catalog.products)
        self.identity_matrix = None
        self.attribute_matrix = None
        self.search_cache: dict[tuple[str, int], DenseSearchResult] = {}

        self.catalog_fingerprint = self._catalog_fingerprint()
        model_name = settings.embedding_model.rsplit("/", 1)[-1].casefold()
        model_slug = re.sub(r"[^a-z0-9]+", "_", model_name).strip("_")
        stable_name = (
            f"catalog_{model_slug}_{settings.embedding_dimensions}_"
            f"v{INDEX_FORMAT_VERSION}.npz"
        )
        self.cache_path = settings.semantic_index_path or settings.cache_directory / stable_name
        self.checkpoint_directory = (
            settings.cache_directory / f"{Path(stable_name).stem}.checkpoint"
        )

    @staticmethod
    def _update_fingerprint(digest, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    def _catalog_fingerprint(self) -> str:
        """Hash the embedded documents, independent of their local file location."""
        digest = hashlib.sha256()
        self._update_fingerprint(digest, DOCUMENT_SCHEMA)
        for parent_asin in self.identifiers:
            product = self.catalog.products[parent_asin]
            self._update_fingerprint(digest, parent_asin)
            self._update_fingerprint(digest, product_identity_document(product))
            self._update_fingerprint(digest, product_attribute_document(product))
        return digest.hexdigest()

    @staticmethod
    def _numpy():
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires NumPy; install requirements-semantic.txt"
            ) from exc
        return np

    @staticmethod
    def _atomic_save_array(np, path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)

    @staticmethod
    def _atomic_write_json(path: Path, value: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _legacy_manifest_matches(self, path: Path) -> bool:
        manifest_path = path.with_suffix(".checkpoint") / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        identifiers_hash = hashlib.sha256("\n".join(self.identifiers).encode()).hexdigest()
        return (
            manifest.get("model") == self.settings.embedding_model
            and manifest.get("dimensions") == self.settings.embedding_dimensions
            and manifest.get("product_count") == len(self.identifiers)
            and manifest.get("identifiers_sha256") == identifiers_hash
        )

    def _load_cache_file(self, path: Path, allow_legacy: bool = False) -> str | None:
        np = self._numpy()
        try:
            with np.load(path, allow_pickle=False) as stored:
                cached_ids = stored["identifiers"].tolist()
                identity = stored["identity"]
                attributes = stored["attributes"]
                expected_shape = (len(self.identifiers), self.settings.embedding_dimensions)
                if cached_ids != self.identifiers:
                    return None
                if identity.shape != expected_shape or attributes.shape != expected_shape:
                    return None

                metadata_fields = {
                    "format_version",
                    "catalog_fingerprint",
                    "embedding_model",
                    "embedding_dimensions",
                    "document_schema",
                }
                if metadata_fields.issubset(stored.files):
                    if (
                        int(stored["format_version"].item()) != INDEX_FORMAT_VERSION
                        or str(stored["catalog_fingerprint"].item())
                        != self.catalog_fingerprint
                        or str(stored["embedding_model"].item())
                        != self.settings.embedding_model
                        or int(stored["embedding_dimensions"].item())
                        != self.settings.embedding_dimensions
                        or str(stored["document_schema"].item()) != DOCUMENT_SCHEMA
                    ):
                        return None
                    cache_kind = "portable"
                elif allow_legacy and self._legacy_manifest_matches(path):
                    cache_kind = "legacy"
                else:
                    return None

                self.identity_matrix = identity
                self.attribute_matrix = attributes
                return cache_kind
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_final_cache(self) -> None:
        np = self._numpy()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.asarray(INDEX_FORMAT_VERSION, dtype=np.int32),
                catalog_fingerprint=np.asarray(self.catalog_fingerprint),
                embedding_model=np.asarray(self.settings.embedding_model),
                embedding_dimensions=np.asarray(
                    self.settings.embedding_dimensions, dtype=np.int32
                ),
                document_schema=np.asarray(DOCUMENT_SCHEMA),
                identifiers=np.asarray(self.identifiers),
                identity=self.identity_matrix,
                attributes=self.attribute_matrix,
            )
        temporary.replace(self.cache_path)

    def _load_cache(self) -> bool:
        if self.identity_matrix is not None and self.attribute_matrix is not None:
            return True
        if self.cache_path.exists() and self._load_cache_file(self.cache_path) == "portable":
            return True

        # Migrate a locally built, path-dependent v0 cache without any API calls.
        for candidate in sorted(
            self.settings.cache_directory.glob("catalog_*.npz"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        ):
            if candidate == self.cache_path:
                continue
            if self._load_cache_file(candidate, allow_legacy=True) == "legacy":
                self._write_final_cache()
                print(
                    f"Migrated legacy semantic index {candidate} to {self.cache_path}",
                    flush=True,
                )
                return True
        return False

    def _checkpoint(self):
        np = self._numpy()
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        count = len(self.identifiers)
        dimensions = self.settings.embedding_dimensions
        identifiers_hash = hashlib.sha256("\n".join(self.identifiers).encode()).hexdigest()
        manifest = {
            "version": 2,
            "model": self.settings.embedding_model,
            "dimensions": dimensions,
            "product_count": count,
            "identifiers_sha256": identifiers_hash,
            "catalog_fingerprint": self.catalog_fingerprint,
            "document_schema": DOCUMENT_SCHEMA,
        }
        manifest_path = self.checkpoint_directory / "manifest.json"
        if manifest_path.exists():
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if stored_manifest != manifest:
                raise RuntimeError(
                    f"Embedding checkpoint does not match this build: {self.checkpoint_directory}"
                )
        else:
            self._atomic_write_json(manifest_path, manifest)

        shape = (count, dimensions)
        matrices = {}
        masks = {}
        for kind in ("identity", "attributes"):
            matrix_path = self.checkpoint_directory / f"{kind}.float32"
            mask_path = self.checkpoint_directory / f"{kind}.done.npy"
            mode = "r+" if matrix_path.exists() else "w+"
            matrices[kind] = np.memmap(matrix_path, dtype=np.float32, mode=mode, shape=shape)
            if mask_path.exists():
                mask = np.load(mask_path, allow_pickle=False)
                if mask.shape != (count,) or mask.dtype != np.bool_:
                    raise RuntimeError(f"Invalid checkpoint mask: {mask_path}")
                masks[kind] = mask
            else:
                masks[kind] = np.zeros(count, dtype=np.bool_)
                self._atomic_save_array(np, mask_path, masks[kind])
        return matrices, masks

    def _build(self) -> int:
        if self._load_cache():
            return 0

        self.settings.cache_directory.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np = self._numpy()
        matrices, masks = self._checkpoint()
        prompt_tokens = 0
        products = [self.catalog.products[parent_asin] for parent_asin in self.identifiers]
        batch_size = max(1, self.settings.embedding_batch_size)
        jobs: list[tuple[str, int, list[Product]]] = []
        for start in range(0, len(products), batch_size):
            batch = products[start:start + batch_size]
            stop = start + len(batch)
            if not bool(masks["identity"][start:stop].all()):
                jobs.append(("identity", start, batch))
            if not bool(masks["attributes"][start:stop].all()):
                jobs.append(("attributes", start, batch))

        def embed_job(kind: str, start: int, batch: list[Product]):
            formatter = product_identity_document if kind == "identity" else product_attribute_document
            attempts = max(1, self.settings.embedding_job_retries)
            for attempt in range(attempts):
                try:
                    response = self.client.embeddings(
                        self.settings.embedding_model,
                        [formatter(product) for product in batch],
                        self.settings.embedding_dimensions,
                        "search_document",
                    )
                    rows = sorted(response.payload.get("data") or [], key=lambda item: item["index"])
                    if len(rows) != len(batch):
                        raise OpenRouterError("Embedding API returned an incomplete batch")
                    vectors = np.asarray([item["embedding"] for item in rows], dtype=np.float32)
                    expected_shape = (len(batch), self.settings.embedding_dimensions)
                    if vectors.shape != expected_shape:
                        raise OpenRouterError(
                            f"Embedding batch has shape {vectors.shape}; expected {expected_shape}"
                        )
                    if not bool(np.isfinite(vectors).all()):
                        raise OpenRouterError("Embedding API returned non-finite values")
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    vectors /= np.maximum(norms, 1e-12)
                    return kind, start, vectors, response.prompt_tokens
                except (OpenRouterError, KeyError, TypeError, ValueError) as exc:
                    if attempt + 1 >= attempts:
                        raise
                    delay = min(
                        30.0,
                        max(0.1, self.settings.embedding_retry_base_seconds) * (2 ** attempt),
                    ) + random.uniform(0.0, 0.5)
                    print(
                        f"Retrying {kind} rows {start}-{start + len(batch) - 1} "
                        f"in {delay:.1f}s after: {str(exc)[:120]}",
                        flush=True,
                    )
                    time.sleep(delay)
            raise OpenRouterError("Embedding batch exhausted retries")

        already_completed = sum(int(mask.sum()) for mask in masks.values())
        total_views = len(self.identifiers) * 2
        print(
            f"Embedding checkpoint: {already_completed}/{total_views} product views complete; "
            f"{len(jobs)} batches remaining",
            flush=True,
        )
        if not jobs:
            print("All checkpoint batches are present; consolidating final index", flush=True)
        completed = 0
        workers = max(1, self.settings.embedding_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(embed_job, *job) for job in jobs]
            for future in as_completed(futures):
                kind, start, vectors, tokens = future.result()
                stop = start + len(vectors)
                matrices[kind][start:stop] = vectors
                matrices[kind].flush()
                masks[kind][start:stop] = True
                self._atomic_save_array(
                    np,
                    self.checkpoint_directory / f"{kind}.done.npy",
                    masks[kind],
                )
                prompt_tokens += tokens
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                    completed_views = sum(int(mask.sum()) for mask in masks.values())
                    print(
                        f"Embedded {completed_views}/{total_views} product views "
                        f"({completed_views / total_views:.1%}); "
                        f"{completed}/{len(jobs)} batches this run",
                        flush=True,
                    )

        if not all(bool(mask.all()) for mask in masks.values()):
            raise OpenRouterError("Embedding build finished with missing vectors")

        self.identity_matrix = np.asarray(matrices["identity"], dtype=np.float16)
        self.attribute_matrix = np.asarray(matrices["attributes"], dtype=np.float16)

        self._write_final_cache()
        return prompt_tokens

    def build_index(self) -> tuple[Path, int]:
        prompt_tokens = self._build()
        return self.cache_path, prompt_tokens

    def _top(self, matrix, query_vector, limit: int) -> list[str]:
        np = self._numpy()
        scores = matrix @ query_vector
        count = min(max(1, limit), len(scores))
        if count == len(scores):
            indexes = np.argsort(scores)[::-1]
        else:
            indexes = np.argpartition(scores, -count)[-count:]
            indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [self.identifiers[int(index)] for index in indexes]

    def _query_vector(self, query: str):
        np = self._numpy()
        query_document = embedding_query_document(self.settings.embedding_model, query)
        cache_namespace = hashlib.sha256(
            f"{self.settings.embedding_model}:{self.settings.embedding_dimensions}:v1".encode()
        ).hexdigest()[:16]
        query_directory = self.settings.cache_directory / f"queries_{cache_namespace}"
        query_path = query_directory / f"{hashlib.sha256(query_document.encode()).hexdigest()}.npy"
        if query_path.exists():
            try:
                vector = np.load(query_path, allow_pickle=False).astype(np.float32)
                if vector.shape == (self.settings.embedding_dimensions,) and bool(np.isfinite(vector).all()):
                    vector /= max(float(np.linalg.norm(vector)), 1e-12)
                    return vector, 0
            except (OSError, ValueError):
                pass

        response = self.client.embeddings(
            self.settings.embedding_model,
            [query_document],
            self.settings.embedding_dimensions,
            "search_query",
        )
        rows = response.payload.get("data") or []
        if not rows:
            raise OpenRouterError("Embedding API returned no query vector")
        vector = np.asarray(rows[0]["embedding"], dtype=np.float32)
        if vector.shape != (self.settings.embedding_dimensions,):
            raise OpenRouterError(
                f"Query embedding has shape {vector.shape}; expected "
                f"({self.settings.embedding_dimensions},)"
            )
        if not bool(np.isfinite(vector).all()):
            raise OpenRouterError("Embedding API returned a non-finite query vector")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        query_directory.mkdir(parents=True, exist_ok=True)
        self._atomic_save_array(np, query_path, vector)
        return vector, response.prompt_tokens

    def search(self, query: str, limit: int = 500) -> DenseSearchResult:
        cache_key = (query, limit)
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            return replace(cached, prompt_tokens=0)
        try:
            if not self._load_cache():
                return DenseSearchResult(
                    [],
                    [],
                    error=(
                        "Dense catalog index is not built. Run "
                        "python -m shopping_agent.build_semantic_index first."
                    ),
                )
            query_vector, prompt_tokens = self._query_vector(query)
            result = DenseSearchResult(
                identity_ranking=self._top(self.identity_matrix, query_vector, limit),
                attribute_ranking=self._top(self.attribute_matrix, query_vector, limit),
                prompt_tokens=prompt_tokens,
            )
            self.search_cache[cache_key] = result
            return result
        except (OpenRouterError, RuntimeError, KeyError, IndexError, TypeError, ValueError) as exc:
            return DenseSearchResult([], [], error=str(exc)[:300])


class ProductReranker:
    def __init__(
        self,
        catalog: CatalogIndex,
        client: OpenRouterClient,
        settings: RetrievalSettings,
    ) -> None:
        self.catalog = catalog
        self.client = client
        self.settings = settings
        self.cache: dict[tuple[str, tuple[str, ...]], RerankResult] = {}

    def rerank(self, query: str, identifiers: list[str]) -> RerankResult:
        if not identifiers:
            return RerankResult([])
        cache_key = (query, tuple(identifiers))
        cached = self.cache.get(cache_key)
        if cached is not None:
            return replace(cached, prompt_tokens=0)
        try:
            documents = [product_rerank_document(self.catalog.products[item]) for item in identifiers]
            effective_query = rerank_query_document(self.settings.rerank_model, query)
            namespace = hashlib.sha256(
                f"{self.settings.rerank_model}:documents-v1".encode()
            ).hexdigest()[:16]
            persistent_directory = self.settings.cache_directory / f"rerank_{namespace}"
            persistent_key = hashlib.sha256(json.dumps(
                [effective_query, list(zip(identifiers, documents))],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()).hexdigest()
            persistent_path = persistent_directory / f"{persistent_key}.json"
            if persistent_path.exists():
                try:
                    stored = json.loads(persistent_path.read_text(encoding="utf-8"))
                    ranking = stored.get("ranking") if isinstance(stored, dict) else None
                    if (
                        isinstance(ranking, list)
                        and len(ranking) == len(identifiers)
                        and set(ranking) == set(identifiers)
                    ):
                        result = RerankResult(ranking=[str(item) for item in ranking])
                        self.cache[cache_key] = result
                        return result
                except (OSError, ValueError, TypeError):
                    pass
            response = self.client.rerank(
                self.settings.rerank_model,
                effective_query,
                documents,
                len(documents),
            )
            results = response.payload.get("results") or []
            ranking = [identifiers[int(item["index"])] for item in results]
            if len(ranking) != len(identifiers):
                ranking.extend(item for item in identifiers if item not in set(ranking))
            result = RerankResult(ranking=ranking, prompt_tokens=response.prompt_tokens)
            persistent_directory.mkdir(parents=True, exist_ok=True)
            temporary = persistent_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"ranking": ranking}) + "\n", encoding="utf-8")
            temporary.replace(persistent_path)
            self.cache[cache_key] = result
            return result
        except (OpenRouterError, KeyError, IndexError, TypeError, ValueError) as exc:
            return RerankResult(ranking=identifiers, error=str(exc)[:300])


def weighted_rrf(rankings: list[tuple[float, list[str]]]) -> list[str]:
    scores: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, parent_asin in enumerate(ranking, start=1):
            scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (60.0 + rank)
    return sorted(scores, key=lambda item: (scores[item], item), reverse=True)
