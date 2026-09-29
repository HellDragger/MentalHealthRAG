"""Configuration: configs/default.yaml overridden by MHRAG_* environment variables.

Priority (highest first): explicit kwargs > environment (MHRAG_<SECTION>__<KEY>) > YAML file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class PathsCfg(BaseModel):
    raw_data: str = "data/raw_data.zip"
    artifacts: str = "artifacts"
    results: str = "results"


class IngestCfg(BaseModel):
    include_pdfs: bool = True
    include_web_articles: bool = True
    include_faq: bool = True
    include_kb_facts: bool = True
    include_counselling: bool = False
    exclude_faq_from_index: bool = False
    near_dup_threshold: float = 0.85
    boilerplate_min_docs: int = 3


class ChunkingCfg(BaseModel):
    chunk_tokens: int = 256
    overlap_ratio: float = 0.15
    min_chunk_tokens: int = 24


class IndexCfg(BaseModel):
    embedder: str = "bge-small"
    embed_batch_size: int = 64
    use_faiss: bool = False


class EmbedderCfg(BaseModel):
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_tokens: int = 512


class RetrievalCfg(BaseModel):
    mode: str = "hybrid_rerank"
    candidates: int = 30
    rrf_k: int = 60
    top_k: int = 5
    reranker: str = "minilm-ce"
    context_token_budget: int = 1800
    query_rewrite: str = "heuristic"


class RerankerCfg(BaseModel):
    model: str
    fastembed_model: str | None = None


class LLMCfg(BaseModel):
    model: str = "qwen2.5-1.5b-gguf"  # or "auto": first available model in auto_preference
    auto_preference: list[str] = Field(default_factory=lambda: ["gpt-oss-120b-groq", "llama-3.3-70b-groq", "qwen2.5-1.5b-gguf", "mock"])
    served_models: list[str] = Field(default_factory=lambda: ["*"])
    # Models that need an API key (Groq, OpenRouter, HF router) are hidden from the app and refused by the server
    # unless this is true. Research scripts load the catalogue directly and are not affected.
    allow_api_models: bool = False
    max_new_tokens: int = 400
    temperature: float = 0.2
    top_p: float = 0.9
    history_turns: int = 6
    timeout_s: float = 120


class SafetyCfg(BaseModel):
    region: str = "IN"
    classifier: str = "tfidf_lr"
    crisis_threshold: float | None = None
    elevated_threshold: float | None = None
    output_checks: bool = True


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit: str = "20/minute"
    max_message_chars: int = 2000
    max_concurrent_generations: int = 2
    log_content: bool = False
    warmup: bool = True


class _YamlSource(PydanticBaseSettingsSource):
    """Reads MHRAG_CONFIG (or configs/default.yaml)."""

    def get_field_value(self, field, field_name):  # pragma: no cover - required by ABC
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path = Path(os.environ.get("MHRAG_CONFIG", CONFIG_DIR / "default.yaml"))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MHRAG_", env_nested_delimiter="__", extra="ignore", env_file=None
    )

    paths: PathsCfg = PathsCfg()
    ingest: IngestCfg = IngestCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    index: IndexCfg = IndexCfg()
    embedders: dict[str, EmbedderCfg] = Field(default_factory=dict)
    embedder_backend: str = "auto"
    retrieval: RetrievalCfg = RetrievalCfg()
    rerankers: dict[str, RerankerCfg] = Field(default_factory=dict)
    llm: LLMCfg = LLMCfg()
    safety: SafetyCfg = SafetyCfg()
    server: ServerCfg = ServerCfg()

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return (init_settings, env_settings, _YamlSource(settings_cls))

    # ---- helpers -------------------------------------------------------
    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.paths.artifacts)

    @property
    def results_dir(self) -> Path:
        return self.resolve(self.paths.results)

    def embedder_cfg(self, key: str | None = None) -> EmbedderCfg:
        key = key or self.index.embedder
        if key not in self.embedders:
            raise KeyError(f"Unknown embedder '{key}'. Known: {sorted(self.embedders)}")
        return self.embedders[key]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_yaml(name: str) -> Any:
    """Load a YAML file from configs/ (e.g. 'models.yaml')."""
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
