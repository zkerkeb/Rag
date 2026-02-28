"""RAG query engine: retrieve context, build prompt, and generate an answer."""

from __future__ import annotations

import time

import anthropic
import structlog

from src.core.config import get_settings
from src.models.schemas import QueryRequest, QueryResponse, SourceDocument
from src.retrieval.vector_store import get_vector_store

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are an enterprise AI assistant. Answer questions using ONLY the provided context documents.

Rules:
1. Base your answer strictly on the context provided below.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite which source documents support your answer when possible.
4. Be concise, accurate, and professional.
"""

CONTEXT_TEMPLATE = """\
## Retrieved Context

{context}

## User Question

{question}
"""


def _build_context_block(sources: list[SourceDocument]) -> str:
    parts: list[str] = []
    for i, src in enumerate(sources, 1):
        header = f"[Source {i} | score={src.score}]"
        meta_str = ", ".join(f"{k}={v}" for k, v in src.metadata.items() if k != "chunk_index")
        if meta_str:
            header += f" ({meta_str})"
        parts.append(f"{header}\n{src.text}")
    return "\n\n---\n\n".join(parts)


def query(request: QueryRequest) -> QueryResponse:
    """Execute a full RAG query: retrieve → augment → generate."""
    settings = get_settings()
    start = time.perf_counter()

    # 1. Retrieve relevant chunks
    store = get_vector_store(collection=request.collection)
    sources = store.search(
        query=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    logger.info(
        "retrieval_complete",
        question=request.question[:80],
        sources_found=len(sources),
    )

    # 2. Build augmented prompt
    if sources:
        context_block = _build_context_block(sources)
        user_message = CONTEXT_TEMPLATE.format(
            context=context_block,
            question=request.question,
        )
    else:
        user_message = (
            f"No relevant documents were found for this question.\n\n"
            f"Question: {request.question}\n\n"
            f"Please let the user know that no matching documents were found "
            f"and suggest they upload relevant documents first."
        )

    # 3. Generate answer via Claude
    api_key = settings.llm.api_key
    if not api_key:
        # Fallback: return sources without LLM generation
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResponse(
            answer=(
                "LLM API key not configured. Here are the most relevant document "
                "excerpts found:\n\n"
                + "\n\n---\n\n".join(s.text for s in sources[:3])
                if sources
                else "No API key configured and no relevant documents found."
            ),
            sources=sources,
            query=request.question,
            model="none (no API key)",
            latency_ms=round(elapsed, 1),
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    elapsed = (time.perf_counter() - start) * 1000

    logger.info("generation_complete", latency_ms=round(elapsed, 1))

    return QueryResponse(
        answer=answer,
        sources=sources,
        query=request.question,
        model=settings.llm.model,
        latency_ms=round(elapsed, 1),
    )
