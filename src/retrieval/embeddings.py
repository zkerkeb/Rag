"""Embedding model wrapper for generating vector representations of text."""

from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings().embeddings
    logger.info("loading_embedding_model", model=settings.model)
    model = SentenceTransformer(settings.model)
    logger.info("embedding_model_loaded", model=settings.model)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query string."""
    model = _load_model()
    embedding = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
    return embedding.tolist()
