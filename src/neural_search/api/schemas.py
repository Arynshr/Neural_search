from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    # Phase 4: "learned" added
    mode: Literal["hybrid", "sparse", "dense", "learned"] = "hybrid"
    # Phase 4: optional reranking
    rerank: bool = False
    rerank_top_k: int = Field(default=5, ge=1, le=50)


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
    rrf_score: Optional[float] = None
    # Phase 4: only populated when rerank=True
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None


class SynthesisResult(BaseModel):
    answer: str
    sources_used: list[dict]
    model: str


# Phase 4: split latency breakdown — kept backwards-compat via latency_ms
class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    rerank_ms: Optional[float] = None
    synthesis_ms: Optional[float] = None
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    mode: str
    reranked: bool = False              # Phase 4
    results: list[ChunkResult]
    synthesis: Optional[SynthesisResult] = None
    latency_ms: float                   # kept for backwards compat
    latency: Optional[LatencyBreakdown] = None  # Phase 4


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    collection: str
    filename: str
    chunks_indexed: int
    tokens: int
    pages: int
    warnings: list[str] = []


# ── Debug ─────────────────────────────────────────────────────────────────────

class DebugResponse(BaseModel):
    query: str
    collection: str
    sparse: list[dict]
    dense: list[dict]
    hybrid_rrf: list[dict]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    collections_count: int
    total_chunks: int
