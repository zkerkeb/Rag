"""ChromaDB-backed vector store for chunk storage and similarity search."""

from __future__ import annotations

from functools import lru_cache

import chromadb
import structlog

from src.core.config import get_settings
from src.models.schemas import Chunk, SourceDocument
from src.retrieval.embeddings import embed_texts, embed_query

logger = structlog.get_logger(__name__)


class VectorStore:
    """Wraps a ChromaDB collection with typed helpers."""

    def __init__(self, collection_name: str | None = None) -> None:
        settings = get_settings().vector_store
        self._client = chromadb.Client()  # in-memory for demo; use PersistentClient for prod
        name = collection_name or settings.collection_name
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_ready", collection=name)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [{**c.metadata, "document_id": c.document_id} for c in chunks]

        embeddings = embed_texts(texts)

        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("chunks_stored", count=len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SourceDocument]:
        query_embedding = embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        sources: list[SourceDocument] = []
        if not results["ids"] or not results["ids"][0]:
            return sources

        for idx, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity score
            distance = results["distances"][0][idx]
            score = 1.0 - distance

            if score < score_threshold:
                continue

            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            document_id = meta.pop("document_id", "unknown")

            sources.append(
                SourceDocument(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=results["documents"][0][idx],
                    score=round(score, 4),
                    metadata=meta,
                )
            )

        sources.sort(key=lambda s: s.score, reverse=True)
        return sources

    def delete_by_document_id(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        logger.info("chunks_deleted", document_id=document_id)

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def name(self) -> str:
        return self._collection.name


# ── Singleton per collection ─────────────────────────────────────────────────

_stores: dict[str, VectorStore] = {}


def get_vector_store(collection: str | None = None) -> VectorStore:
    key = collection or get_settings().vector_store.collection_name
    if key not in _stores:
        _stores[key] = VectorStore(collection_name=key)
    return _stores[key]


def reset_stores() -> None:
    """Clear all cached stores (useful for testing)."""
    _stores.clear()
