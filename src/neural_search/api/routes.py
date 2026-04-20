import time
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from loguru import logger

from neural_search.api.schemas import (
    CreateCollectionRequest, CollectionMeta,
    SearchRequest, SearchResponse, DebugResponse, ChunkResult,
    IngestResponse, HealthResponse,
)
from neural_search.collections.manager import CollectionManager
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.synthesis.groq_client import GroqSynthesizer
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.config import settings

router = APIRouter()
collection_manager = CollectionManager()


def _get_hybrid(slug: str) -> HybridRetriever:
    sparse = BM25sRetriever(collection_slug=slug)
    sparse.load()
    dense = QdrantRetriever(collection_slug=slug)
    return HybridRetriever(sparse=sparse, dense=dense)


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health():
    cols = collection_manager.list_collections()
    return HealthResponse(
        status="ok",
        collections_count=len(cols),
        total_chunks=sum(c["total_chunks"] for c in cols),
    )


# ── Collections ───────────────────────────────────────────────────────────────
@router.get("/collections", response_model=list[CollectionMeta])
def list_collections():
    return collection_manager.list_collections()


@router.post("/collections", response_model=CollectionMeta, status_code=201)
def create_collection(body: CreateCollectionRequest):
    try:
        return collection_manager.create_collection(body.name, body.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/collections/{slug}", response_model=CollectionMeta)
def get_collection(slug: str):
    col = collection_manager.get_collection(slug)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


@router.delete("/collections/{slug}", status_code=204)
def delete_collection(slug: str):
    try:
        # Also wipe Qdrant collection
        dense = QdrantRetriever(collection_slug=slug)
        dense.reset()
        collection_manager.delete_collection(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Ingest ────────────────────────────────────────────────────────────────────
@router.post("/collections/{slug}/ingest", response_model=IngestResponse)
async def ingest(slug: str, file: UploadFile = File(...), force: bool = False):
    col = collection_manager.get_collection(slug)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")

    filename = file.filename
    warnings = []

    # Duplicate detection
    if collection_manager.file_exists(slug, filename) and not force:
        raise HTTPException(
            status_code=409,
            detail=f"'{filename}' already exists in this collection. Use force=true to re-ingest.",
        )

    # Save file
    dest_dir = settings.documents_path_for(slug)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    dest.write_bytes(await file.read())

    # Ingest
    sparse = BM25sRetriever(collection_slug=slug)
    sparse.load()
    dense = QdrantRetriever(collection_slug=slug)
    chunks = run_ingestion(source=dest, sparse_retriever=sparse, dense_retriever=dense)

    if not chunks:
        warnings.append("No chunks extracted — check document content")

    total_tokens = sum(c.token_count for c in chunks)
    pages = max((c.page for c in chunks), default=0)

    # Update metadata
    collection_manager.add_file_record(slug, {
        "filename": filename,
        "pages": pages,
        "chunks": len(chunks),
        "tokens": total_tokens,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if chunks else "empty",
    })

    logger.info(f"[{slug}] Ingested '{filename}' — {len(chunks)} chunks, {total_tokens} tokens")
    return IngestResponse(
        status="ok",
        collection=slug,
        filename=filename,
        chunks_indexed=len(chunks),
        tokens=total_tokens,
        pages=pages,
        warnings=warnings,
    )


# ── Search ────────────────────────────────────────────────────────────────────
@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    col = collection_manager.get_collection(body.collection)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")

    t0 = time.perf_counter()
    hybrid = _get_hybrid(body.collection)
    sparse = hybrid._sparse
    dense = hybrid._dense

    if body.mode == "sparse":
        results = sparse.search(body.query, k=body.k)
    elif body.mode == "dense":
        results = dense.search(body.query, k=body.k)
    else:
        results = hybrid.search(body.query, k=body.k)

    synthesis = None
    if body.synthesize:
        synthesizer: GroqSynthesizer = request.app.state.synthesizer
        synthesis = synthesizer.synthesize(body.query, results)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Search | collection={body.collection} | mode={body.mode} | latency={latency_ms}ms")

    return SearchResponse(
        query=body.query,
        collection=body.collection,
        mode=body.mode,
        results=[ChunkResult(**r) for r in results],
        synthesis=synthesis,
        latency_ms=latency_ms,
    )


@router.get("/search/debug", response_model=DebugResponse)
def search_debug(query: str, collection: str, k: int = 10):
    col = collection_manager.get_collection(collection)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    hybrid = _get_hybrid(collection)
    debug = hybrid.search_debug(query, k=k)
    debug["collection"] = collection
    return DebugResponse(**debug)
