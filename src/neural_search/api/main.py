from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from loguru import logger
from neural_search.config import settings
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.synthesis.groq_client import GroqSynthesizer
from neural_search.api.routes import router
from neural_search.api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — initialise all shared state
    settings.ensure_dirs()
    logger.info("Initialising Neural Search API...")

    sparse = BM25sRetriever()
    sparse.load()

    dense = QdrantRetriever()
    hybrid = HybridRetriever(sparse=sparse, dense=dense)
    synthesizer = GroqSynthesizer()

    app.state.sparse = sparse
    app.state.dense = dense
    app.state.hybrid = hybrid
    app.state.synthesizer = synthesizer

    logger.info("Neural Search API ready")
    yield
    # Shutdown
    logger.info("Shutting down Neural Search API")


app = FastAPI(
    title="Neural Search API",
    description="Hybrid sparse + dense semantic search over PDF/DOCX documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    sparse: BM25sRetriever = request.app.state.sparse
    dense: QdrantRetriever = request.app.state.dense
    bm25_count = sparse.count()
    qdrant_count = dense.count()
    return HealthResponse(
        status="ok",
        bm25_chunks=bm25_count,
        qdrant_points=qdrant_count,
        index_in_sync=(bm25_count == qdrant_count),
    )
