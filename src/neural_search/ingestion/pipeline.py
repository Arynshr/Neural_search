import json
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.parser import parse_document, parse_directory
from neural_search.ingestion.chunker import chunk_pages, Chunk


def _export_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp.replace(path)
    logger.info(f"Exported {len(chunks)} chunks to {path}")


def run_ingestion(
    source: Path,
    sparse_retriever=None,
    dense_retriever=None,
    reset: bool = False,
    export_snapshot: bool = True,
    collection_slug: str = "default",
) -> list[Chunk]:
    if not source.exists():
        logger.error(f"Source path does not exist: {source}")
        return []

    if reset:
        logger.warning("Reset flag set — wiping existing indexes")
        if sparse_retriever:
            sparse_retriever.reset()
        if dense_retriever:
            dense_retriever.reset()

    try:
        if source.is_dir():
            pages = parse_directory(source)
        else:
            pages = parse_document(source)
    except Exception as e:
        logger.error(f"Parsing failed for {source}: {e}")
        return []

    if not pages:
        logger.warning("No pages parsed — check document content and format")
        return []

    try:
        chunks = chunk_pages(pages)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return []

    if not chunks:
        logger.warning("No chunks produced — check chunking config")
        return []

    if export_snapshot:
        try:
            snapshot_path = settings.snapshot_path_for(collection_slug)
            _export_jsonl(chunks, snapshot_path)
        except Exception as e:
            logger.error(f"Snapshot export failed: {e}")

    if sparse_retriever:
        try:
            logger.info("Indexing into BM25 sparse retriever...")
            sparse_retriever.index(chunks)
        except Exception as e:
            logger.error(f"Sparse indexing failed: {e}")
            raise

    if dense_retriever:
        try:
            logger.info("Upserting into Qdrant dense retriever...")
            dense_retriever.upsert(chunks)
        except Exception as e:
            logger.error(f"Dense indexing failed: {e}")
            raise

    logger.info(f"Ingestion complete — {len(chunks)} chunks from {source}")
    return chunks
