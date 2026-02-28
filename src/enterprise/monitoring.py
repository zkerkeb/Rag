"""Prometheus metrics and monitoring middleware."""

from __future__ import annotations

import time

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

# ── Metrics ──────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
)

QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query end-to-end latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
    ["status"],
)

CHUNKS_STORED = Counter(
    "rag_chunks_stored_total",
    "Total chunks stored in vector store",
)


# ── Middleware ────────────────────────────────────────────────────────────────


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)

        return response


def metrics_response() -> Response:
    """Generate Prometheus-compatible metrics response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
