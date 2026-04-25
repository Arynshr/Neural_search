"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Pipeline: retrieve top-N candidates → rerank → return top-k.

Design decisions:
- Model loaded once at module level via thread-safe singleton (mirrors dense.py pattern)
- Accepts the same list[dict] contract that sparse/dense/hybrid return
- Adds `rerank_score` and `rerank_rank` fields; preserves all original fields
- Raises on empty candidate list — caller must guard
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from loguru import logger
from sentence_transformers import CrossEncoder

from neural_search.config import get_settings

settings = get_settings()

# ── Singleton ─────────────────────────────────────────────────────────────────

_MODEL: Optional[CrossEncoder] = None
_MODEL_LOCK = threading.Lock()

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                model_name = getattr(settings, "reranker_model", _DEFAULT_MODEL)
                logger.info(f"Loading cross-encoder: {model_name}")
                t0 = time.perf_counter()
                _MODEL = CrossEncoder(model_name)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                logger.info(f"Cross-encoder loaded in {elapsed}ms")
    return _MODEL


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks a list of retrieved chunks using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self) -> None:
        self._model = _get_model()

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank candidates and return top_k results.

        Args:
            query:      The search query string.
            candidates: List of chunk dicts from any retriever. Must contain 'text' and 'chunk_id'.
            top_k:      Number of results to return. Defaults to len(candidates).

        Returns:
            List of chunk dicts sorted by rerank_score descending, each enriched with:
              - rerank_score (float): cross-encoder relevance score
              - rerank_rank (int):    1-based rank after reranking
        """
        if not candidates:
            logger.warning("rerank() called with empty candidate list — returning []")
            return []

        top_k = top_k or len(candidates)
        top_k = min(top_k, len(candidates))

        pairs = [(query, c["text"]) for c in candidates]

        t0 = time.perf_counter()
        scores: list[float] = self._model.predict(pairs).tolist()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            f"Reranked {len(candidates)} candidates in {elapsed_ms}ms "
            f"| query='{query[:60]}'"
        )

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append({
                **chunk,
                "rerank_score": round(score, 6),
                "rerank_rank": rank,
                "rerank_latency_ms": elapsed_ms,
            })

        return results
