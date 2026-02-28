"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Document models ──────────────────────────────────────────────────────────


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None


class DocumentResponse(BaseModel):
    document_id: str
    metadata: DocumentMetadata


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ── Chunk models ─────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


# ── Query models ─────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    collection: str | None = None


class SourceDocument(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    model: str
    latency_ms: float


# ── Auth models ──────────────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Health models ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    components: dict[str, str] = Field(default_factory=dict)


# ── Collection models ────────────────────────────────────────────────────────


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    chunk_count: int


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo]
