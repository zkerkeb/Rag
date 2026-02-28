"""Text chunking strategies for splitting documents into indexable pieces."""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
import structlog

from src.core.config import get_settings
from src.models.schemas import Chunk

logger = structlog.get_logger(__name__)


def chunk_text(
    text: str,
    document_id: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into chunks using the configured strategy."""
    settings = get_settings().chunking
    extra_meta = metadata or {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.separators,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    logger.info("text_chunked", document_id=document_id, chunk_count=len(raw_chunks))

    chunks: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        chunk_id = f"{document_id}::{uuid.uuid4().hex[:12]}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=raw,
                metadata={
                    **extra_meta,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                },
            )
        )

    return chunks
