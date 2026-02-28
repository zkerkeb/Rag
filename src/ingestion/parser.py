"""Document parsers for extracting text from various file formats."""

from __future__ import annotations

import io
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Supported MIME types → parser mapping
SUPPORTED_TYPES: dict[str, str] = {
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "text",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def parse_document(content: bytes, content_type: str, filename: str) -> str:
    """Extract plain text from a document based on its content type."""
    parser_key = SUPPORTED_TYPES.get(content_type)

    if parser_key is None:
        # Fallback: try to decode as text
        logger.warning("unsupported_content_type", content_type=content_type, filename=filename)
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported content type: {content_type}")

    if parser_key == "text":
        return _parse_text(content)
    elif parser_key == "pdf":
        return _parse_pdf(content)
    elif parser_key == "docx":
        return _parse_docx(content)
    else:
        raise ValueError(f"No parser for: {parser_key}")


def _parse_text(content: bytes) -> str:
    return content.decode("utf-8")


def _parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
