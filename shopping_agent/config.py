from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without replacing exported variables."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.casefold().strip() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetrievalSettings:
    api_key: str
    answer_enabled: bool
    intent_enabled: bool
    rewrite_enabled: bool
    dense_enabled: bool
    rerank_enabled: bool
    slot_decay: float
    intent_routing_scale: float
    suppression_enabled: bool
    suppression_max_turns: int
    suppression_turns: int
    suppression_reserve_turns: int
    rewrite_model: str
    embedding_model: str
    rerank_model: str
    answer_model: str
    intent_model: str
    answer_confidence_threshold: float
    intent_confidence_threshold: float
    embedding_dimensions: int
    embedding_batch_size: int
    embedding_workers: int
    embedding_job_retries: int
    embedding_retry_base_seconds: float
    lexical_fusion_weight: float
    dense_identity_fusion_weight: float
    dense_attribute_fusion_weight: float
    rerank_depth: int
    rerank_base_fusion_weight: float
    rerank_model_fusion_weight: float
    request_timeout_seconds: float
    cache_directory: Path
    semantic_index_path: Path | None

    @classmethod
    def from_environment(cls, env_path: str | Path = ".env") -> "RetrievalSettings":
        load_env_file(env_path)
        semantic_index_value = os.environ.get("TECHJAM_SEMANTIC_INDEX_PATH", "").strip()
        return cls(
            api_key=os.environ.get("CLAUDE_API_KEY", "").strip(),
            answer_enabled=_enabled("TECHJAM_LLM_ANSWER"),
            intent_enabled=_enabled("TECHJAM_LLM_INTENT"),
            rewrite_enabled=_enabled("TECHJAM_LLM_REWRITE"),
            dense_enabled=_enabled("TECHJAM_DENSE_RETRIEVAL"),
            rerank_enabled=_enabled("TECHJAM_RERANK"),
            slot_decay=float(os.environ.get("TECHJAM_SLOT_DECAY", "0.9")),
            intent_routing_scale=float(
                os.environ.get("TECHJAM_INTENT_ROUTING_SCALE", "1.0")
            ),
            suppression_enabled=_enabled("TECHJAM_SUPPRESSION", True),
            suppression_max_turns=int(os.environ.get("TECHJAM_SUPPRESSION_MAX_TURNS", "2")),
            suppression_turns=int(os.environ.get("TECHJAM_SUPPRESSION_TURNS", "2")),
            suppression_reserve_turns=int(
                os.environ.get("TECHJAM_SUPPRESSION_RESERVE_TURNS", "3")
            ),
            rewrite_model=os.environ.get("TECHJAM_REWRITE_MODEL", "openai/gpt-5.6-luna"),
            embedding_model=os.environ.get("TECHJAM_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
            rerank_model=os.environ.get("TECHJAM_RERANK_MODEL", "qwen/qwen3-reranker-8b"),
            answer_model=os.environ.get("TECHJAM_ANSWER_MODEL", "openai/gpt-5.6-luna"),
            intent_model=os.environ.get("TECHJAM_INTENT_MODEL", "openai/gpt-5.6-luna"),
            answer_confidence_threshold=float(
                os.environ.get("TECHJAM_ANSWER_CONFIDENCE_THRESHOLD", "0.65")
            ),
            intent_confidence_threshold=float(
                os.environ.get("TECHJAM_INTENT_CONFIDENCE_THRESHOLD", "0.65")
            ),
            embedding_dimensions=int(os.environ.get("TECHJAM_EMBEDDING_DIMENSIONS", "512")),
            embedding_batch_size=int(os.environ.get("TECHJAM_EMBEDDING_BATCH_SIZE", "64")),
            embedding_workers=int(os.environ.get("TECHJAM_EMBEDDING_WORKERS", "2")),
            embedding_job_retries=int(os.environ.get("TECHJAM_EMBEDDING_JOB_RETRIES", "8")),
            embedding_retry_base_seconds=float(
                os.environ.get("TECHJAM_EMBEDDING_RETRY_BASE_SECONDS", "2")
            ),
            lexical_fusion_weight=float(os.environ.get("TECHJAM_LEXICAL_FUSION_WEIGHT", "50")),
            dense_identity_fusion_weight=float(
                os.environ.get("TECHJAM_DENSE_IDENTITY_FUSION_WEIGHT", "1")
            ),
            dense_attribute_fusion_weight=float(
                os.environ.get("TECHJAM_DENSE_ATTRIBUTE_FUSION_WEIGHT", "1")
            ),
            rerank_depth=int(os.environ.get("TECHJAM_RERANK_DEPTH", "50")),
            rerank_base_fusion_weight=float(
                os.environ.get("TECHJAM_RERANK_BASE_FUSION_WEIGHT", "25")
            ),
            rerank_model_fusion_weight=float(
                os.environ.get("TECHJAM_RERANK_MODEL_FUSION_WEIGHT", "1")
            ),
            request_timeout_seconds=float(os.environ.get("TECHJAM_API_TIMEOUT_SECONDS", "45")),
            cache_directory=Path(os.environ.get("TECHJAM_CACHE_DIR", "artifacts/semantic_cache")),
            semantic_index_path=(
                Path(semantic_index_value) if semantic_index_value else None
            ),
        )

    @property
    def remote_enabled(self) -> bool:
        return bool(self.api_key) and (
            self.answer_enabled
            or self.intent_enabled
            or self.rewrite_enabled
            or self.dense_enabled
            or self.rerank_enabled
        )
