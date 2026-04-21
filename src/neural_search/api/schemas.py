from pydantic import BaseModel, Field
from typing import Literal, Optional


# ── Collections ───────────────────────────────────────────────────────────────
class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=256)


class FileRecord(BaseModel):
    filename: str
    pages: int
    chunks: int
    tokens: int
    ingested_at: str
    status: str = "ok"


class CollectionMeta(BaseModel):
    slug: str
    name: str
    description: str
    created_at: str
    updated_at: str
    files: list[FileRecord]
    total_chunks: int
    total_tokens: int


# ── Search ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str
    k: int = Field(default=10, ge=1, le=50)
    synthesize: bool = False
    mode: Literal["hybrid", "sparse", "dense"] = "hybrid"


class ChunkResult(BaseModel):
    chunk_id: str
    source_file: str
    page: int
    token_count: int
    text: str
    score: float
    rank: int
    source: str
    collection: str
    # #3: rrf_score is optional — only present for hybrid results
    rrf_score: Optional[float] = None


class SynthesisResult(BaseModel):
    answer: str
    sources_used: list[dict]
    model: str


class SearchResponse(BaseModel):
    query: str
    collection: str
    mode: str
    results: list[ChunkResult]
    synthesis: Optional[SynthesisResult] = None
    latency_ms: float


class DebugResponse(BaseModel):
    query: str
    collection: str
    sparse: list[dict]
    dense: list[dict]
    hybrid_rrf: list[dict]


# ── Ingest ────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    status: str
    collection: str
    filename: str
    chunks_indexed: int
    tokens: int
    pages: int
    warnings: list[str] = []


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    collections_count: int
    total_chunks: int
