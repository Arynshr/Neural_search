from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neural_search.config import settings
from neural_search.ingestion.chunker import Chunk

# Single shared model instance loaded once at import time
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _MODEL = SentenceTransformer(settings.embedding_model)
    return _MODEL


class QdrantRetriever:
    def __init__(self, collection_slug: str):
        self._slug = collection_slug
        self._model = _get_model()
        self._dim = self._model.get_sentence_embedding_dimension()
        self._client = QdrantClient(path=str(settings.qdrant_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._slug not in existing:
            self._client.create_collection(
                collection_name=self._slug,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(f"Qdrant collection '{self._slug}' created (dim={self._dim})")

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        logger.info(f"[{self._slug}] Embedding {len(chunks)} chunks...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            vectors = self._model.encode(
                [c.text for c in batch],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self._client.upsert(
                collection_name=self._slug,
                points=[
                    PointStruct(
                        id=abs(hash(c.chunk_id)) % (2 ** 63),
                        vector=vectors[j].tolist(),
                        payload={
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "source_file": c.source_file,
                            "page": c.page,
                            "chunk_index": c.chunk_index,
                            "token_count": c.token_count,
                            "text": c.text,
                            "collection": self._slug,
                            "metadata": c.metadata,
                        },
                    )
                    for j, c in enumerate(batch)
                ],
            )
            logger.debug(f"[{self._slug}] Upserted batch {i // batch_size + 1}")
        logger.info(f"[{self._slug}] Qdrant upsert complete — {len(chunks)} chunks")

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or settings.top_k
        query_vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.search(
            collection_name=self._slug,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )
        return [
            {
                "chunk_id": hit.payload["chunk_id"],
                "score": hit.score,
                "rank": rank,
                "source": "dense",
                "text": hit.payload["text"],
                "source_file": hit.payload["source_file"],
                "page": hit.payload["page"],
                "token_count": hit.payload["token_count"],
                "collection": self._slug,
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    def count(self) -> int:
        return self._client.get_collection(self._slug).points_count

    def reset(self) -> None:
        self._client.delete_collection(self._slug)
        self._ensure_collection()
        logger.info(f"[{self._slug}] Qdrant collection wiped and recreated")
