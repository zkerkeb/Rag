"""REST API routes for the Enterprise RAG system."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
import structlog

from src.enterprise.auth import get_current_user
from src.enterprise.monitoring import (
    DOCUMENTS_INGESTED,
    QUERY_COUNT,
    QUERY_LATENCY,
    metrics_response,
)
from src.ingestion.parser import SUPPORTED_TYPES
from src.ingestion.pipeline import (
    delete_document,
    get_document,
    ingest_document,
    list_documents,
)
from src.models.schemas import (
    CollectionInfo,
    CollectionListResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TokenRequest,
    TokenResponse,
)
from src.retrieval.rag_engine import query as rag_query
from src.retrieval.vector_store import get_vector_store, _stores

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    store = get_vector_store()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "vector_store": "ok",
            "chunks_indexed": str(store.count),
        },
    )


# ── Metrics ──────────────────────────────────────────────────────────────────


@router.get("/metrics", tags=["System"], include_in_schema=False)
async def prometheus_metrics():
    return metrics_response()


# ── Auth ─────────────────────────────────────────────────────────────────────


@router.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(request: TokenRequest):
    from src.enterprise.auth import authenticate_user, create_access_token

    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token)


# ── Documents ────────────────────────────────────────────────────────────────


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    user: dict | None = Depends(get_current_user),
):
    content_type = file.content_type or "application/octet-stream"
    content = await file.read()
    filename = file.filename or "untitled"

    logger.info("upload_request", filename=filename, content_type=content_type, size=len(content))

    result = ingest_document(content, filename, content_type)

    DOCUMENTS_INGESTED.labels(status=result.metadata.status.value).inc()

    if result.metadata.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {result.metadata.error}",
        )

    return result


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_documents(user: dict | None = Depends(get_current_user)):
    docs = list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document_by_id(
    document_id: str,
    user: dict | None = Depends(get_current_user),
):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def remove_document(
    document_id: str,
    user: dict | None = Depends(get_current_user),
):
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


# ── Query ────────────────────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    user: dict | None = Depends(get_current_user),
):
    import time

    QUERY_COUNT.inc()
    start = time.perf_counter()

    result = rag_query(request)

    elapsed = time.perf_counter() - start
    QUERY_LATENCY.observe(elapsed)

    return result


# ── Collections ──────────────────────────────────────────────────────────────


@router.get("/collections", response_model=CollectionListResponse, tags=["Collections"])
async def list_collections(user: dict | None = Depends(get_current_user)):
    infos = []
    for name, store in _stores.items():
        infos.append(
            CollectionInfo(
                name=name,
                document_count=0,
                chunk_count=store.count,
            )
        )
    return CollectionListResponse(collections=infos)
