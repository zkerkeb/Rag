"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.vector_store import reset_stores


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestDocumentEndpoints:
    def setup_method(self):
        reset_stores()

    def test_upload_text_file(self):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"This is test content for indexing.", "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["metadata"]["status"] == "indexed"
        assert data["metadata"]["chunk_count"] >= 1

    def test_list_documents_empty(self):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0

    def test_get_nonexistent_document(self):
        resp = client.get("/api/v1/documents/nonexistent")
        assert resp.status_code == 404


class TestQueryEndpoint:
    def setup_method(self):
        reset_stores()

    def test_query_empty_store(self):
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the meaning of life?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_after_upload(self):
        # Upload a document first
        client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "info.txt",
                    b"The capital of France is Paris. It is known for the Eiffel Tower.",
                    "text/plain",
                )
            },
        )

        # Query about the content
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the capital of France?", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) > 0


class TestCollectionEndpoints:
    def test_list_collections(self):
        resp = client.get("/api/v1/collections")
        assert resp.status_code == 200


class TestAuthEndpoint:
    def test_login_invalid_credentials(self):
        resp = client.post(
            "/api/v1/auth/token",
            json={"username": "wrong", "password": "wrong"},
        )
        assert resp.status_code == 401
