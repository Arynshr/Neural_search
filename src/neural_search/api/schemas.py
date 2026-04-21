from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


# ── Collections ───────────────────────────────────────────────────────────────
class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=256)


class FileRecord(BaseModel):
    filename: str
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    tokens: int = Field(ge=0)
    ingested_at: datetime
    status: str = "ok"


class CollectionMeta(BaseModel):
    slug: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    files: list[FileRecord]
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


# ── Search ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field(..., min_length=1, max_length=64)
    k: int = Field(default=10, ge=1, le=50)
    synthesize: bool = False
    mode: Literal["hybrid", "sparse", "dense"] = "hybrid"


class ChunkResult(BaseModel):
    chunk_id: str
    source_file: str
    page: int = Field(ge=0)
    token_count: int = Field(ge=0)
    text: str
    score: float = Field(ge=0)
    rank: int = Field(ge=1)
    source: str
    collection: str
    rrf_score: Optional[float] = Field(default=None, ge=0)


class SynthesisResult(BaseModel):
    answer: str
    sources_used: list[ChunkResult]
    model: str


class SearchResponse(BaseModel):
    query: str
    collection: str = Field(..., min_length=1, max_length=64)
    mode: Literal["hybrid", "sparse", "dense"]
    results: list[ChunkResult]
    synthesis: Optional[SynthesisResult] = None
    latency_ms: float = Field(ge=0)


class DebugResponse(BaseModel):
    query: str
    collection: str = Field(..., min_length=1, max_length=64)
    sparse: list[ChunkResult]
    dense: list[ChunkResult]
    hybrid_rrf: list[ChunkResult]


# ── Ingest ────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    status: str
    collection: str = Field(..., min_length=1, max_length=64)
    filename: str
    chunks_indexed: int = Field(ge=0)
    tokens: int = Field(ge=0)
    pages: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    collections_count: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
