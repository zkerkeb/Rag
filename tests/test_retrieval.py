"""Tests for the vector store and retrieval system."""

from src.models.schemas import Chunk
from src.retrieval.vector_store import VectorStore, reset_stores


class TestVectorStore:
    def setup_method(self):
        reset_stores()
        self.store = VectorStore(collection_name="test_collection")

    def test_add_and_count(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="Python is a programming language used for web development and AI.",
                metadata={"source": "test"},
            ),
            Chunk(
                chunk_id="c2",
                document_id="d1",
                text="FastAPI is a modern Python web framework for building APIs.",
                metadata={"source": "test"},
            ),
        ]
        self.store.add_chunks(chunks)
        assert self.store.count == 2

    def test_search_returns_results(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="Machine learning models require training data to learn patterns.",
                metadata={},
            ),
            Chunk(
                chunk_id="c2",
                document_id="d1",
                text="The weather forecast predicts rain tomorrow afternoon.",
                metadata={},
            ),
        ]
        self.store.add_chunks(chunks)

        results = self.store.search("What is machine learning?", top_k=2, score_threshold=0.0)
        assert len(results) > 0
        # The ML-related chunk should score higher
        assert "machine learning" in results[0].text.lower() or "training data" in results[0].text.lower()

    def test_search_empty_store(self):
        results = self.store.search("test query")
        assert results == []

    def test_delete_by_document_id(self):
        chunks = [
            Chunk(chunk_id="c1", document_id="d1", text="Document one content.", metadata={}),
            Chunk(chunk_id="c2", document_id="d2", text="Document two content.", metadata={}),
        ]
        self.store.add_chunks(chunks)
        assert self.store.count == 2

        self.store.delete_by_document_id("d1")
        assert self.store.count == 1
