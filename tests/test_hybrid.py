"""
Unit tests for retrieval/hybrid.py
Tests: RRF score calculation, source attribution, ranking order, edge cases.
"""
import pytest
from unittest.mock import MagicMock
from neural_search.retrieval.hybrid import HybridRetriever, _rrf


def make_result(chunk_id, rank, source, score=0.9):
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "score": score,
        "source": source,
        "text": f"text for {chunk_id}",
        "source_file": "test.pdf",
        "page": 1,
        "token_count": 10,
        "collection": "test",
    }


class TestRRF:
    def test_chunk_in_both_gets_higher_score(self):
        sparse = [make_result("a", 1, "sparse"), make_result("b", 2, "sparse")]
        dense  = [make_result("a", 1, "dense"),  make_result("c", 2, "dense")]
        fused = _rrf(sparse, dense, k=60)
        chunk_ids = [r["chunk_id"] for r in fused]
        # "a" appears in both — must rank first
        assert chunk_ids[0] == "a"

    def test_source_attribution_both(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("a", 1, "dense")]
        fused = _rrf(sparse, dense, k=60)
        assert fused[0]["source"] in ("dense+sparse", "sparse+dense")

    def test_source_attribution_sparse_only(self):
        sparse = [make_result("x", 1, "sparse")]
        dense  = [make_result("y", 1, "dense")]
        fused = _rrf(sparse, dense, k=60)
        sources = {r["chunk_id"]: r["source"] for r in fused}
        assert sources["x"] == "sparse"
        assert sources["y"] == "dense"

    def test_rrf_scores_descending(self):
        sparse = [make_result(f"c{i}", i + 1, "sparse") for i in range(5)]
        dense  = [make_result(f"d{i}", i + 1, "dense")  for i in range(5)]
        fused = _rrf(sparse, dense, k=60)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_is_sequential(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("b", 1, "dense")]
        fused = _rrf(sparse, dense, k=60)
        ranks = [r["rank"] for r in fused]
        assert ranks == list(range(1, len(fused) + 1))

    def test_empty_inputs_returns_empty(self):
        assert _rrf([], [], k=60) == []

    def test_rrf_k_affects_scores(self):
        sparse = [make_result("a", 1, "sparse")]
        dense  = [make_result("a", 1, "dense")]
        score_low_k  = _rrf(sparse, dense, k=1)[0]["rrf_score"]
        score_high_k = _rrf(sparse, dense, k=100)[0]["rrf_score"]
        assert score_low_k > score_high_k


class TestHybridRetriever:
    def _make_retriever(self, sparse_results, dense_results):
        sparse = MagicMock()
        sparse.search.return_value = sparse_results
        dense = MagicMock()
        dense.search.return_value = dense_results
        return HybridRetriever(sparse=sparse, dense=dense)

    def test_search_calls_both_retrievers(self):
        sparse_r = [make_result("a", 1, "sparse")]
        dense_r  = [make_result("b", 1, "dense")]
        hybrid = self._make_retriever(sparse_r, dense_r)
        results = hybrid.search("test query", k=5)
        hybrid._sparse.search.assert_called_once_with("test query", k=5)
        hybrid._dense.search.assert_called_once_with("test query", k=5)

    def test_search_returns_fused_results(self):
        sparse_r = [make_result("a", 1, "sparse")]
        dense_r  = [make_result("b", 1, "dense")]
        hybrid = self._make_retriever(sparse_r, dense_r)
        results = hybrid.search("test query", k=5)
        chunk_ids = {r["chunk_id"] for r in results}
        assert "a" in chunk_ids
        assert "b" in chunk_ids

    def test_search_debug_returns_all_three(self):
        sparse_r = [make_result("a", 1, "sparse")]
        dense_r  = [make_result("b", 1, "dense")]
        hybrid = self._make_retriever(sparse_r, dense_r)
        debug = hybrid.search_debug("test query", k=5)
        assert "sparse" in debug
        assert "dense" in debug
        assert "hybrid_rrf" in debug
        assert debug["query"] == "test query"

    def test_search_respects_k_limit(self):
        sparse_r = [make_result(f"s{i}", i + 1, "sparse") for i in range(10)]
        dense_r  = [make_result(f"d{i}", i + 1, "dense")  for i in range(10)]
        hybrid = self._make_retriever(sparse_r, dense_r)
        results = hybrid.search("query", k=3)
        assert len(results) <= 3
