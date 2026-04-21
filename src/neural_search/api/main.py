from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from neural_search.config import settings
from neural_search.synthesis.groq_client import GroqSynthesizer
from neural_search.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("Starting Neural Search API...")
    # Groq synthesizer is the only shared singleton — retrievers are per-collection
    app.state.synthesizer = GroqSynthesizer()
    logger.info("Neural Search API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Neural Search API",
    description="Hybrid semantic search over named document collections",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")
