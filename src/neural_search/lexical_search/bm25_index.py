"""
BM25Index — Lexical search pipeline for MS MARCO passage retrieval.

Design principles:
- Identical preprocessing at index time and query time (no train/test skew)
- Lazy NLTK resource loading with graceful fallback
- Atomic pickle writes (write-then-rename) to prevent corrupt index files
- Full type annotations for IDE support and runtime clarity
- Structured logging via loguru; degrades to stdlib logging if unavailable
"""

from __future__ import annotations

import os
import pickle
import re
import string
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependency guards — fail loudly at import, not at runtime
# ---------------------------------------------------------------------------
try:
    import bm25s
except ImportError as e:
    raise ImportError(
        "bm25s is required. Install with: uv add bm25s"
    ) from e

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BM25_K1: float = 1.5   # Term frequency saturation
_BM25_B: float  = 0.75  # Length normalisation

_MIN_TOKEN_LEN: int = 2

# Pickle protocol 5 for Python ≥3.8 (supports large buffers)
_PICKLE_PROTOCOL: int = 5


# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if missing. Silent if already present."""
    if not _NLTK_AVAILABLE:
        return
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt",     "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


def _load_stopwords() -> frozenset:
    """
    Load English stopwords from NLTK, with a hardcoded fallback for
    environments where NLTK data cannot be downloaded.
    """
    if _NLTK_AVAILABLE:
        try:
            _ensure_nltk_resources()
            return frozenset(nltk_stopwords.words("english"))
        except Exception as exc:
            logger.warning(f"NLTK stopwords unavailable ({exc}); using fallback set.")

    # Minimal fallback — top-50 English stopwords by corpus frequency
    return frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "it", "its", "this", "that", "these", "those", "i", "we", "you",
        "he", "she", "they", "what", "which", "who", "whom", "as", "if",
    })


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Pre-compiled pattern — compiled once at module load, reused across all calls
_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _preprocess(
    text: str,
    stopwords: frozenset,
    min_token_len: int = _MIN_TOKEN_LEN,
) -> List[str]:
    """
    Normalise and tokenise a single text string.

    Pipeline:
        1. Null / type guard
        2. Lowercase
        3. Punctuation removal (replace with space to avoid token merging)
        4. Whitespace tokenisation
        5. Stopword removal
        6. Minimum token length filter

    Args:
        text:          Raw passage or query string.
        stopwords:     Frozenset of lowercase stopword strings.
        min_token_len: Tokens shorter than this are discarded (default 2).

    Returns:
        List of normalised tokens. Returns [] for empty/non-string input.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return []

    text = text.strip()
    if not text:
        return []

    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    tokens = [
        t for t in tokens
        if t not in stopwords and len(t) >= min_token_len
    ]
    return tokens


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25-based passage retrieval index.

    Wraps bm25s.BM25 with:
    - Deterministic preprocessing (same at index and query time)
    - Persistent storage with atomic writes
    - Structured search results with scores and passage IDs
    - Latency instrumentation on every search call

    Typical usage::

        # Build
        index = BM25Index()
        index.build(passages=[(pid, text), ...])
        index.save("data/indices/bm25.pkl")

        # Query
        index = BM25Index.load("data/indices/bm25.pkl")
        results = index.search("cardiac arrest", top_k=10)
    """

    def __init__(
        self,
        k1: float = _BM25_K1,
        b: float = _BM25_B,
        min_token_len: int = _MIN_TOKEN_LEN,
    ) -> None:
        """
        Args:
            k1:            BM25 term-frequency saturation parameter.
            b:             BM25 document-length normalisation parameter.
            min_token_len: Discard tokens shorter than this after preprocessing.
        """
        self.k1 = k1
        self.b = b
        self.min_token_len = min_token_len

        self._stopwords: frozenset = _load_stopwords()
        self._bm25: Optional[bm25s.BM25] = None
        self._passage_ids: List[str] = []      # parallel to BM25 corpus rows
        self._corpus_size: int = 0
        self._build_time_s: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Public preprocessing entry point.
        Identical behaviour at index time and query time — no skew possible.
        """
        return _preprocess(text, self._stopwords, self.min_token_len)

    def build(
        self,
        passages: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> "BM25Index":
        """
        Build the BM25 index from a list of (passage_id, text) tuples.

        Args:
            passages:      Iterable of (pid, text) pairs. PIDs must be unique.
            show_progress: Show tqdm progress bar if tqdm is available.

        Returns:
            self — allows chaining: index.build(...).save(...)

        Raises:
            ValueError: If passages is empty.
        """
        if not passages:
            raise ValueError("Cannot build index from empty passage list.")

        logger.info(f"Building BM25 index over {len(passages):,} passages …")
        t0 = time.perf_counter()

        iterator = self._progress(passages, desc="Tokenising", unit="passage") \
            if show_progress else passages

        tokenised: List[List[str]] = []
        ids: List[str] = []
        empty_count = 0

        for pid, text in iterator:
            tokens = self.preprocess(text)
            if not tokens:
                # Keep empty token lists — bm25s handles them gracefully,
                # and dropping rows would break the pid↔row index alignment.
                empty_count += 1
            tokenised.append(tokens)
            ids.append(str(pid))

        if empty_count:
            logger.warning(
                f"{empty_count:,} passages produced zero tokens after preprocessing. "
                "They will score 0 for all queries (expected for very short/noisy passages)."
            )

        logger.info("Fitting bm25s.BM25 …")
        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(tokenised)

        self._bm25 = retriever
        self._passage_ids = ids
        self._corpus_size = len(ids)
        self._build_time_s = time.perf_counter() - t0

        logger.info(
            f"Index built in {self._build_time_s:.1f}s — "
            f"{self._corpus_size:,} passages"
        )
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[str], List[float], float]:
        """
        Retrieve the top-k passages for a query.

        Args:
            query: Raw query string (preprocessing applied internally).
            top_k: Number of results to return.

        Returns:
            Tuple of:
                - passage_ids: List[str] ranked by descending score
                - scores:      List[float] corresponding BM25 scores
                - latency_ms:  float query latency in milliseconds

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
            ValueError:   If top_k < 1.
        """
        self._require_built()

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        t0 = time.perf_counter()

        query_tokens = self.preprocess(query)

        # Edge case: query reduces to zero tokens after preprocessing
        # (e.g. query is entirely stopwords like "is the a").
        if not query_tokens:
            logger.warning(
                f"Query produced zero tokens after preprocessing: {query!r}. "
                "Returning empty results."
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return [], [], latency_ms

        effective_k = min(top_k, self._corpus_size)

        # bm25s.retrieve expects List[List[str]] — wrap single query in a batch
        top_indices, result_scores_arr = self._bm25.retrieve(  # type: ignore[union-attr]
            [query_tokens], k=effective_k
        )
        # retrieve returns shape (n_queries, k) — take first query row
        top_indices   = top_indices[0].tolist()
        result_ids    = [self._passage_ids[i] for i in top_indices]
        result_scores = result_scores_arr[0].tolist()

        latency_ms = (time.perf_counter() - t0) * 1000
        return result_ids, result_scores, latency_ms

    def search_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run multiple queries and return results in ranx-compatible format.

        Args:
            queries:       List of (query_id, query_text) tuples.
            top_k:         Results per query.
            show_progress: Show tqdm progress bar if tqdm is available.

        Returns:
            Dict[query_id, Dict[passage_id, score]] — ready for ranx evaluation.
        """
        self._require_built()

        results: Dict[str, Dict[str, float]] = {}

        iterator = self._progress(queries, desc="Searching", unit="query") \
            if show_progress else queries

        for qid, query_text in iterator:
            pids, scores, _ = self.search(query_text, top_k=top_k)
            results[str(qid)] = {pid: score for pid, score in zip(pids, scores)}

        return results

    def save(self, path: str | Path) -> None:
        """
        Persist the index to disk using an atomic write (temp file → rename).

        Atomic write prevents a corrupt/partial file if the process is killed
        mid-write. The rename is atomic on POSIX systems.

        Args:
            path: Target path for the pickle file.
        """
        self._require_built()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(self, f, protocol=_PICKLE_PROTOCOL)
            os.replace(tmp_path, path)  # atomic on POSIX
            logger.info(f"Index saved → {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """
        Load a persisted index from disk.

        Args:
            path: Path to the pickle file produced by save().

        Returns:
            A fully initialised BM25Index instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            pickle.UnpicklingError: If the file is corrupt or incompatible.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        logger.info(f"Loading index from {path} …")
        t0 = time.perf_counter()

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, expected {cls.__name__}."
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Index loaded in {elapsed:.0f}ms — "
            f"{obj._corpus_size:,} passages"
        )
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    @property
    def build_time_seconds(self) -> Optional[float]:
        return self._build_time_s

    def __repr__(self) -> str:
        state = (
            f"corpus_size={self._corpus_size:,}, "
            f"k1={self.k1}, b={self.b}"
        ) if self.is_built else "not built"
        return f"BM25Index({state})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_built(self) -> None:
        if not self.is_built:
            raise RuntimeError(
                "Index has not been built. Call build() or load() first."
            )

    @staticmethod
    def _progress(iterable, **kwargs):
        """Wrap iterable with tqdm if available; else return as-is."""
        try:
            from tqdm import tqdm
            return tqdm(iterable, **kwargs)
        except ImportError:
            return iterable
