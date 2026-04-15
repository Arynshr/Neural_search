from pathlib import Path
from loguru import logger
from neural_search.ingestion.parser import parse_document, parse_directory
from neural_search.ingestion.chunker import chunk_pages, Chunk


def run_ingestion(
    source: Path,
    sparse_retriever=None,
    dense_retriever=None,
    reset: bool = False,
) -> list[Chunk]:
    """
    Full ingestion pipeline: parse → chunk → index.

    Args:
        source: Path to a single file or directory.
        sparse_retriever: BM25sRetriever instance (optional, indexed if provided).
        dense_retriever: QdrantRetriever instance (optional, upserted if provided).
        reset: If True, wipe existing indexes before ingestion.

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
