from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter
)
from neural_search.config import settings
from neural_search.ingestion.chunker import Chunk


class QdrantRetriever:
    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._client = QdrantClient(path=str(settings.qdrant_path))
        self._collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(f"Qdrant collection '{self._collection}' created (dim={self._dim})")

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            points = [
                PointStruct(
                    id=abs(hash(c.chunk_id)) % (2**63),
                    vector=vectors[j].tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source_file": c.source_file,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                        "text": c.text,
                        "metadata": c.metadata,
                    },
                )
                for j, c in enumerate(batch)
            ]
            self._client.upsert(collection_name=self._collection, points=points)
            logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")
        logger.info(f"Qdrant upsert complete — {len(chunks)} chunks")

    def search(self, query: str, k: int = None) -> list[dict]:
        k = k or settings.top_k
        query_vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.search(
            collection_name=self._collection,
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
            }
            for rank, hit in enumerate(hits, start=1)
        ]

    def count(self) -> int:
        return self._client.get_collection(self._collection).points_count

    def reset(self) -> None:
        self._client.delete_collection(self._collection)
        self._ensure_collection()
        logger.info(f"Qdrant collection '{self._collection}' wiped and recreated")
