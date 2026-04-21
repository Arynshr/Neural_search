import json
from pathlib import Path
from loguru import logger
from neural_search.config import settings as global_settings
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
    collection_slug: str = "default",
    settings_obj=None,   # ✅ injection for testability
) -> list[Chunk]:
    """
    Full ingestion pipeline: parse → chunk → index → (optional) JSONL snapshot.
    """
    settings = settings_obj or global_settings

    if reset:
        logger.warning("Reset flag set — wiping existing indexes")
        if sparse_retriever:
            sparse_retriever.reset()
        if dense_retriever:
            dense_retriever.reset()

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

    chunks = chunk_pages(pages)
    if not chunks:
        logger.warning("No chunks produced — check chunking config")
        return []

    if export_snapshot:
        if collection_slug == "default":
            snapshot_path = settings.data_dir / "snapshots" / "chunks.jsonl"
        else:
            snapshot_path = settings.snapshot_path_for(collection_slug)

        _export_jsonl(chunks, snapshot_path)

    if sparse_retriever:
        logger.info("Indexing into BM25 sparse retriever...")
        sparse_retriever.index(chunks)

    if dense_retriever:
        logger.info("Upserting into Qdrant dense retriever...")
        dense_retriever.upsert(chunks)

    logger.info(f"Ingestion complete — {len(chunks)} chunks from {source}")
    return chunks
