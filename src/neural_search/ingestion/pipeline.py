import json
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.parser import parse_document, parse_directory
from neural_search.ingestion.chunker import chunk_pages, Chunk


def _export_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Exported {len(chunks)} chunks to {path}")


def run_ingestion(
    source: Path,
    sparse_retriever=None,
    dense_retriever=None,
    reset: bool = False,
    export_snapshot: bool = True,
) -> list[Chunk]:
    """
    Full ingestion pipeline: parse → chunk → index → (optional) JSONL snapshot.

    Args:
        source: Path to a single file or directory.
        sparse_retriever: BM25sRetriever instance (optional).
        dense_retriever: QdrantRetriever instance (optional).
        reset: Wipe existing indexes before ingestion.
        export_snapshot: Write chunks to JSONL for offline eval and debugging.

    Returns:
        List of all Chunk objects produced.
    """
    if reset:
        logger.warning("Reset flag set — wiping existing indexes")
        if sparse_retriever:
            sparse_retriever.reset()
        if dense_retriever:
            dense_retriever.reset()

    # Parse
    if source.is_dir():
        pages = parse_directory(source)
    elif source.is_file():
        pages = parse_document(source)
    else:
        logger.error(f"Source path does not exist: {source}")
        return []

    if not pages:
        logger.warning("No pages parsed — check document content and format")
        return []

    # Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        logger.warning("No chunks produced — check chunking config")
        return []

    # Export JSONL snapshot — always under data_dir/snapshots/
    if export_snapshot:
        snapshot_path = settings.data_dir / "snapshots" / "chunks.jsonl"
        _export_jsonl(chunks, snapshot_path)

    # Index sparse
    if sparse_retriever:
        logger.info("Indexing into BM25 sparse retriever...")
        sparse_retriever.index(chunks)

    # Index dense
    if dense_retriever:
        logger.info("Upserting into Qdrant dense retriever...")
        dense_retriever.upsert(chunks)

    logger.info(f"Ingestion complete — {len(chunks)} chunks from {source}")
    return chunks
