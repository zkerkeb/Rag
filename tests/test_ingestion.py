"""Tests for the document ingestion pipeline."""

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text


class TestParser:
    def test_parse_plain_text(self):
        content = b"Hello, this is a test document."
        result = parse_document(content, "text/plain", "test.txt")
        assert result == "Hello, this is a test document."

    def test_parse_markdown(self):
        content = b"# Title\n\nSome content here."
        result = parse_document(content, "text/markdown", "test.md")
        assert "# Title" in result

    def test_parse_unsupported_type_fallback(self):
        content = b"Fallback text content"
        result = parse_document(content, "application/unknown", "test.bin")
        assert result == "Fallback text content"

    def test_parse_unsupported_binary_raises(self):
        content = bytes(range(256))
        try:
            parse_document(content, "application/unknown", "test.bin")
            assert False, "Should have raised ValueError"
        except (ValueError, UnicodeDecodeError):
            pass


class TestChunker:
    def test_chunk_short_text(self):
        text = "This is a short text."
        chunks = chunk_text(text, "doc-001")
        assert len(chunks) == 1
        assert chunks[0].document_id == "doc-001"
        assert chunks[0].text == text

    def test_chunk_long_text(self):
        # Create text longer than default chunk_size (512)
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, "doc-002")
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == "doc-002"
            assert chunk.chunk_id.startswith("doc-002::")

    def test_chunk_preserves_metadata(self):
        text = "Some text content for testing."
        chunks = chunk_text(text, "doc-003", metadata={"source": "test"})
        assert chunks[0].metadata["source"] == "test"
        assert "chunk_index" in chunks[0].metadata
