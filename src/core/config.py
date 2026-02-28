"""Application configuration loaded from settings.yaml and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_yaml_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


class ServerSettings(BaseSettings):
    host: str = _yaml.get("server", {}).get("host", "0.0.0.0")
    port: int = _yaml.get("server", {}).get("port", 8000)
    workers: int = _yaml.get("server", {}).get("workers", 1)
    reload: bool = _yaml.get("server", {}).get("reload", True)


class LLMSettings(BaseSettings):
    provider: str = _yaml.get("llm", {}).get("provider", "anthropic")
    model: str = _yaml.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens: int = _yaml.get("llm", {}).get("max_tokens", 4096)
    temperature: float = _yaml.get("llm", {}).get("temperature", 0.1)
    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")


class EmbeddingSettings(BaseSettings):
    model: str = _yaml.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
    dimension: int = _yaml.get("embeddings", {}).get("dimension", 384)


class VectorStoreSettings(BaseSettings):
    provider: str = _yaml.get("vector_store", {}).get("provider", "chromadb")
    persist_directory: str = _yaml.get("vector_store", {}).get(
        "persist_directory", "./data/chroma_db"
    )
    collection_name: str = _yaml.get("vector_store", {}).get(
        "collection_name", "enterprise_docs"
    )


class ChunkingSettings(BaseSettings):
    strategy: str = _yaml.get("chunking", {}).get("strategy", "recursive")
    chunk_size: int = _yaml.get("chunking", {}).get("chunk_size", 512)
    chunk_overlap: int = _yaml.get("chunking", {}).get("chunk_overlap", 50)
    separators: list[str] = _yaml.get("chunking", {}).get(
        "separators", ["\n\n", "\n", ". ", " "]
    )


class RetrievalSettings(BaseSettings):
    top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)
    score_threshold: float = _yaml.get("retrieval", {}).get("score_threshold", 0.3)
    rerank: bool = _yaml.get("retrieval", {}).get("rerank", False)


class AuthSettings(BaseSettings):
    enabled: bool = _yaml.get("auth", {}).get("enabled", False)
    secret_key: str = _yaml.get("auth", {}).get("secret_key", "change-me-in-production")
    algorithm: str = _yaml.get("auth", {}).get("algorithm", "HS256")
    access_token_expire_minutes: int = _yaml.get("auth", {}).get(
        "access_token_expire_minutes", 60
    )


class LoggingSettings(BaseSettings):
    level: str = _yaml.get("logging", {}).get("level", "INFO")
    format: str = _yaml.get("logging", {}).get("format", "json")


class MonitoringSettings(BaseSettings):
    prometheus_enabled: bool = _yaml.get("monitoring", {}).get("prometheus_enabled", True)
    metrics_path: str = _yaml.get("monitoring", {}).get("metrics_path", "/metrics")


class Settings(BaseSettings):
    server: ServerSettings = ServerSettings()
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    auth: AuthSettings = AuthSettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
