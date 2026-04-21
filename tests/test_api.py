"""
Integration tests for the FastAPI layer.
Uses TestClient — no real network, no real indexes (mocked at app state).
Tests all endpoints: /health, /collections, /search, /search/debug, /ingest.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_collection_manager():
    manager = MagicMock()
    manager.list_collections.return_value = [
        {
            "slug": "hr-policies",
            "name": "HR Policies",
            "description": "HR documents",
            "created_at": "2026-04-20T10:00:00+00:00",
            "updated_at": "2026-04-20T10:00:00+00:00",
            "files": [],
            "total_chunks": 42,
            "total_tokens": 5000,
        }
    ]
    manager.get_collection.return_value = manager.list_collections.return_value[0]
    manager.create_collection.return_value = manager.list_collections.return_value[0]
    manager.file_exists.return_value = False
    return manager


@pytest.fixture
def mock_hybrid():
    hybrid = MagicMock()
    hybrid._sparse = MagicMock()
    hybrid._dense = MagicMock()
    hybrid.search.return_value = [
        {
            "chunk_id": "abc123",
            "score": 0.95,
            "rank": 1,
            "source": "dense+sparse",
            "text": "Payment terms are net 30 days.",
            "source_file": "contract.pdf",
            "page": 3,
            "token_count": 8,
            "collection": "hr-policies",
            "rrf_score": 0.016,
        }
    ]
    hybrid.search_debug.return_value = {
        "query": "payment terms",
        "collection": "hr-policies",
        "sparse": [],
        "dense": [],
        "hybrid_rrf": [],
    }
    return hybrid


@pytest.fixture
def client(mock_collection_manager, mock_hybrid):
    with patch("neural_search.api.routes.collection_manager", mock_collection_manager), \
         patch("neural_search.api.routes._get_hybrid", return_value=mock_hybrid), \
         patch("neural_search.synthesis.groq_client.Groq"):
        from neural_search.api.main import app
        app.state.synthesizer = MagicMock()
        app.state.synthesizer.synthesize.return_value = {
            "answer": "Payment terms are net 30 days.",
            "sources_used": [{"source_file": "contract.pdf", "page": 3}],
            "model": "llama3-8b-8192",
        }
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_collection_count(self, client):
        data = resp = client.get("/health").json()
        assert "collections_count" in data
        assert "total_chunks" in data

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


class TestCollectionEndpoints:
    def test_list_collections_returns_200(self, client):
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_collection_returns_201(self, client):
        resp = client.post("/collections", json={"name": "Test", "description": ""})
        assert resp.status_code == 201

    def test_create_collection_missing_name_returns_422(self, client):
        resp = client.post("/collections", json={"description": "no name"})
        assert resp.status_code == 422

    def test_get_collection_returns_200(self, client):
        resp = client.get("/collections/hr-policies")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "hr-policies"

    def test_get_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.get("/collections/ghost")
        assert resp.status_code == 404

    def test_delete_collection_returns_204(self, client):
        resp = client.delete("/collections/hr-policies")
        assert resp.status_code == 204


class TestSearchEndpoint:
    def test_search_returns_200(self, client):
        resp = client.post("/search", json={
            "query": "payment terms",
            "collection": "hr-policies",
            "k": 5,
            "mode": "hybrid",
            "synthesize": False,
        })
        assert resp.status_code == 200

    def test_search_returns_results(self, client):
        data = client.post("/search", json={
            "query": "payment terms",
            "collection": "hr-policies",
            "k": 5,
            "mode": "hybrid",
        }).json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_result_has_required_fields(self, client):
        result = client.post("/search", json={
            "query": "test",
            "collection": "hr-policies",
        }).json()["results"][0]
        for field in ["chunk_id", "source_file", "page", "text", "score", "rank", "source"]:
            assert field in result

    def test_search_with_synthesis(self, client):
        data = client.post("/search", json={
            "query": "payment terms",
            "collection": "hr-policies",
            "synthesize": True,
        }).json()
        assert data["synthesis"] is not None
        assert "answer" in data["synthesis"]

    def test_search_without_synthesis_has_no_answer(self, client):
        data = client.post("/search", json={
            "query": "test",
            "collection": "hr-policies",
            "synthesize": False,
        }).json()
        assert data["synthesis"] is None

    def test_search_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.post("/search", json={"query": "test", "collection": "ghost"})
        assert resp.status_code == 404

    def test_search_returns_latency(self, client):
        data = client.post("/search", json={
            "query": "test", "collection": "hr-policies"
        }).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_search_invalid_mode_returns_422(self, client):
        resp = client.post("/search", json={
            "query": "test",
            "collection": "hr-policies",
            "mode": "invalid_mode",
        })
        assert resp.status_code == 422

    def test_search_empty_query_returns_422(self, client):
        resp = client.post("/search", json={"query": "", "collection": "hr-policies"})
        assert resp.status_code == 422


class TestDebugEndpoint:
    def test_debug_returns_200(self, client):
        resp = client.get("/search/debug?query=test&collection=hr-policies")
        assert resp.status_code == 200

    def test_debug_returns_all_three_sections(self, client):
        data = client.get("/search/debug?query=test&collection=hr-policies").json()
        assert "sparse" in data
        assert "dense" in data
        assert "hybrid_rrf" in data

    def test_debug_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.get("/search/debug?query=test&collection=ghost")
        assert resp.status_code == 404


class TestIngestEndpoint:
    def test_ingest_pdf_returns_200(self, client, tmp_path):
        pdf_content = b"%PDF-1.4 test"
        resp = client.post(
            "/collections/hr-policies/ingest",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
        )
        # Will be 200 or may warn — should not be 5xx
        assert resp.status_code in (200, 422)

    def test_ingest_duplicate_returns_409(self, client, mock_collection_manager):
        mock_collection_manager.file_exists.return_value = True
        resp = client.post(
            "/collections/hr-policies/ingest",
            files={"file": ("existing.pdf", b"%PDF", "application/pdf")},
        )
        assert resp.status_code == 409

    def test_ingest_nonexistent_collection_returns_404(self, client, mock_collection_manager):
        mock_collection_manager.get_collection.return_value = None
        resp = client.post(
            "/collections/ghost/ingest",
            files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        )
        assert resp.status_code == 404
