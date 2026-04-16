import asyncio
from loguru import logger
from neural_search.config import settings
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever


def _rrf(sparse_results: list[dict], dense_results: list[dict], k: int = None) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    score(d) = Σ 1 / (rrf_k + rank(d))
    """
    rrf_k = k or settings.rrf_k
    scores: dict[str, float] = {}
    sources: dict[str, set] = {}
    meta: dict[str, dict] = {}

    for result in sparse_results:
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1 / (rrf_k + result["rank"])
        sources.setdefault(cid, set()).add("sparse")
        meta[cid] = result

    for result in dense_results:
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1 / (rrf_k + result["rank"])
        sources.setdefault(cid, set()).add("dense")
        meta[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        entry = {**meta[cid]}
        entry["rrf_score"] = round(score, 6)
        entry["rank"] = rank
        entry["source"] = "+".join(sorted(sources[cid]))   # sparse | dense | dense+sparse
        fused.append(entry)
    return fused


class HybridRetriever:
    def __init__(self, sparse: BM25sRetriever, dense: QdrantRetriever):
        self._sparse = sparse
        self._dense = dense

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or settings.top_k
        sparse_results = self._sparse.search(query, k=k)
        dense_results = self._dense.search(query, k=k)
        fused = _rrf(sparse_results, dense_results)
        return fused[:k]

    def search_debug(self, query: str, k: int = None) -> dict:
        """Returns raw results from each retriever + fused output for debugging."""
        k = k or settings.top_k
        sparse_results = self._sparse.search(query, k=k)
        dense_results = self._dense.search(query, k=k)
        fused = _rrf(sparse_results, dense_results)[:k]
        return {
            "query": query,
            "sparse": sparse_results,
            "dense": dense_results,
            "hybrid_rrf": fused,
        }
