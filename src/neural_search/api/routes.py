import time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from loguru import logger
from neural_search.api.schemas import (
    SearchRequest, SearchResponse, DebugResponse,
    HealthResponse, IngestResponse, ChunkResult,
)
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.synthesis.groq_client import GroqSynthesizer
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.config import settings

router = APIRouter()


# ── Dependency: shared retriever instances (injected from app state) ──────────
def get_sparse(request) -> BM25sRetriever:
    return request.app.state.sparse


def get_dense(request) -> QdrantRetriever:
    return request.app.state.dense


def get_hybrid(request) -> HybridRetriever:
    return request.app.state.hybrid


def get_synthesizer(request) -> GroqSynthesizer:
    return request.app.state.synthesizer


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health(request: SearchRequest = None, sparse: BM25sRetriever = Depends(get_sparse),
           dense: QdrantRetriever = Depends(get_dense)):
    from fastapi import Request
    pass  # implemented inline in main.py for simplicity


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request=None):
    sparse: BM25sRetriever = request.app.state.sparse
    dense: QdrantRetriever = request.app.state.dense
    hybrid: HybridRetriever = request.app.state.hybrid
    synthesizer: GroqSynthesizer = request.app.state.synthesizer

    t0 = time.perf_counter()

    if body.mode == "sparse":
        results = sparse.search(body.query, k=body.k)
    elif body.mode == "dense":
        results = dense.search(body.query, k=body.k)
    else:
        results = hybrid.search(body.query, k=body.k)

    synthesis = None
    if body.synthesize:
        synthesis = synthesizer.synthesize(body.query, results)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"Search | mode={body.mode} | query='{body.query}' | latency={latency_ms}ms")

    return SearchResponse(
        query=body.query,
        mode=body.mode,
        results=[ChunkResult(**r) for r in results],
        synthesis=synthesis,
        latency_ms=latency_ms,
    )


@router.get("/search/debug", response_model=DebugResponse)
def search_debug(query: str, k: int = 10, request=None):
    hybrid: HybridRetriever = request.app.state.hybrid
    debug = hybrid.search_debug(query, k=k)
    logger.info(f"Debug search | query='{query}'")
    return DebugResponse(**debug)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...), request=None):
    sparse: BM25sRetriever = request.app.state.sparse
    dense: QdrantRetriever = request.app.state.dense
    warnings = []
    total_chunks = 0

    upload_dir = settings.documents_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        dest = upload_dir / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        chunks = run_ingestion(source=dest, sparse_retriever=sparse, dense_retriever=dense)
        if not chunks:
            warnings.append(f"No chunks extracted from {upload.filename}")
        total_chunks += len(chunks)

    return IngestResponse(
        status="ok",
        chunks_indexed=total_chunks,
        files_processed=len(files),
        warnings=warnings,
    )


@router.delete("/index")
def reset_index(request=None):
    sparse: BM25sRetriever = request.app.state.sparse
    dense: QdrantRetriever = request.app.state.dense
    sparse.reset()
    dense.reset()
    logger.warning("Both indexes wiped via API")
    return {"status": "indexes reset"}
