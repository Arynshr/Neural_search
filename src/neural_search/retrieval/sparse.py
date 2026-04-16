import pickle
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.chunker import Chunk

try:
    import bm25s
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = list(stopwords.words("english"))
except ImportError:
    _STOPWORDS = []


class BM25sRetriever:
    def __init__(self):
        self._index = None
        self._chunks: list[Chunk] = []
        self._index_file = settings.bm25_index_path / "bm25.pkl"
        self._chunks_file = settings.bm25_index_path / "chunks.pkl"
        settings.bm25_index_path.mkdir(parents=True, exist_ok=True)

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        tokenized = []
        for text in texts:
            tokens = text.lower().split()
            tokens = [t for t in tokens if t.isalpha() and t not in _STOPWORDS]
            tokenized.append(tokens)
        return tokenized

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus_tokens = self._tokenize([c.text for c in chunks])
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens)
        with open(self._index_file, "wb") as f:
            pickle.dump(self._index, f)
        with open(self._chunks_file, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"BM25 index built and saved — {len(chunks)} chunks")

    def load(self) -> bool:
        if self._index_file.exists() and self._chunks_file.exists():
            with open(self._index_file, "rb") as f:
                self._index = pickle.load(f)
            with open(self._chunks_file, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info(f"BM25 index loaded — {len(self._chunks)} chunks")
            return True
        logger.warning("BM25 index not found — run ingestion first")
        return False

    def search(self, query: str, k: int = None) -> list[dict]:
        if self._index is None:
            self.load()
        k = k or settings.top_k
        query_tokens = self._tokenize([query])[0]
        results, scores = self._index.retrieve(
            bm25s.tokenize(query, stopwords=_STOPWORDS), corpus=None, k=min(k, len(self._chunks))
        )
        max_score = max(scores[0]) if max(scores[0]) > 0 else 1.0
        output = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0]), start=1):
            chunk = self._chunks[idx]
            output.append({
                "chunk_id": chunk.chunk_id,
                "score": float(score) / max_score,   # normalized to [0,1]
                "rank": rank,
                "source": "sparse",
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "token_count": chunk.token_count,
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
        logger.info("BM25 index wiped")
