"""End-to-end document ingestion pipeline: parse → chunk → embed → store."""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.models.schemas import DocumentMetadata, DocumentResponse, DocumentStatus
from src.retrieval.vector_store import get_vector_store

logger = structlog.get_logger(__name__)

# In-memory document registry (swap for a database in production)
_document_registry: dict[str, DocumentResponse] = {}


def ingest_document(
    content: bytes,
    filename: str,
    content_type: str,
    collection: str | None = None,
) -> DocumentResponse:
    """Ingest a single document through the full pipeline."""
    document_id = uuid.uuid4().hex

    metadata = DocumentMetadata(
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        status=DocumentStatus.PROCESSING,
        created_at=datetime.utcnow(),
    )

    doc_response = DocumentResponse(document_id=document_id, metadata=metadata)
    _document_registry[document_id] = doc_response

    try:
        # 1. Parse
        logger.info("ingestion_parse", document_id=document_id, filename=filename)
        text = parse_document(content, content_type, filename)

        if not text.strip():
            raise ValueError("Document produced no extractable text")

        # 2. Chunk
        logger.info("ingestion_chunk", document_id=document_id)
        chunks = chunk_text(
            text,
            document_id,
            metadata={"filename": filename, "content_type": content_type},
        )

        # 3. Embed & store
        logger.info("ingestion_store", document_id=document_id, chunk_count=len(chunks))
        store = get_vector_store(collection=collection)
        store.add_chunks(chunks)

        # Update metadata
        metadata.chunk_count = len(chunks)
        metadata.status = DocumentStatus.INDEXED
        doc_response.metadata = metadata
        _document_registry[document_id] = doc_response

        logger.info("ingestion_complete", document_id=document_id)

    except Exception as exc:
        logger.error("ingestion_failed", document_id=document_id, error=str(exc))
        metadata.status = DocumentStatus.FAILED
        metadata.error = str(exc)
        doc_response.metadata = metadata
        _document_registry[document_id] = doc_response

    return doc_response


def get_document(document_id: str) -> DocumentResponse | None:
    return _document_registry.get(document_id)


def list_documents() -> list[DocumentResponse]:
    return list(_document_registry.values())


def delete_document(document_id: str) -> bool:
    if document_id in _document_registry:
        store = get_vector_store()
        store.delete_by_document_id(document_id)
        del _document_registry[document_id]
        return True
    return False
