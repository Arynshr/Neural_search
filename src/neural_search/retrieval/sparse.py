import pickle
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.chunker import Chunk

try:
    import bm25s
except ImportError as e:
    raise ImportError(
        "bm25s is required for sparse retrieval. Run: pip install bm25s"
    ) from e

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        _STOPWORDS = list(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = list(stopwords.words("english"))
except Exception:
    logger.warning("NLTK stopwords unavailable — proceeding without stopword filtering")
    _STOPWORDS = []


class BM25sRetriever:
    def __init__(self, collection_slug: str):
        self._slug = collection_slug
        self._index = None
        self._chunks: list[Chunk] = []
        self._dir = settings.bm25_path_for(collection_slug)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._dir / "bm25.pkl"
        self._chunks_file = self._dir / "chunks.pkl"

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return [
            [t for t in text.lower().split() if t.isalpha() and t not in _STOPWORDS]
            for text in texts
        ]

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus_tokens = self._tokenize([c.text for c in chunks])
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens)

        with open(self._index_file, "wb") as f:
            pickle.dump(self._index, f)
        with open(self._chunks_file, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info(f"[{self._slug}] BM25 index built — {len(chunks)} chunks")

    def load(self) -> bool:
        try:
            if self._index_file.exists() and self._chunks_file.exists():
                with open(self._index_file, "rb") as f:
                    self._index = pickle.load(f)
                with open(self._chunks_file, "rb") as f:
                    self._chunks = pickle.load(f)
                logger.info(f"[{self._slug}] BM25 index loaded — {len(self._chunks)} chunks")
                return True
        except Exception as e:
            logger.error(f"[{self._slug}] Failed to load BM25 index: {e}")
            self.reset()
        return False

    def search(self, query: str, k: int | None = None) -> list[dict]:
        if not query.strip():
            return []

        if self._index is None and not self.load():
            raise RuntimeError(f"[{self._slug}] BM25 index not found. Call index() first.")

        if not self._chunks:
            return []

        k = min(k or settings.top_k, len(self._chunks))

        query_tokens = self._tokenize([query])[0]
        results, scores = self._index.retrieve(query_tokens, corpus=None, k=k)

        raw_scores = scores[0]
        max_score = float(max(raw_scores)) if raw_scores and max(raw_scores) > 0 else 1.0

        output = []
        for rank, (idx, score) in enumerate(zip(results[0], raw_scores), start=1):
            if idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]
            output.append({
                "chunk_id": chunk.chunk_id,
                "score": float(score) / max_score,
                "rank": rank,
                "source": "sparse",
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "token_count": chunk.token_count,
                "collection": self._slug,
            })

        return output

    def count(self) -> int:
        if not self._chunks:
            self.load()
        return len(self._chunks)

    def reset(self) -> None:
        self._index = None
        self._chunks = []
        for f in [self._index_file, self._chunks_file]:
            if f.exists():
                f.unlink()
        logger.info(f"[{self._slug}] BM25 index wiped")
