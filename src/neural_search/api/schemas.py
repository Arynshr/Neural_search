from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=50)
    synthesize: bool = Field(default=False)
    mode: Literal["hybrid", "sparse", "dense"] = Field(default="hybrid")


class ChunkResult(BaseModel):
    chunk_id: str
    source_file: str
    page: int
    token_count: int
    text: str
    score: float
    rank: int
    source: str       # sparse | dense | dense+sparse


class SynthesisResult(BaseModel):
    answer: str
    sources_used: list[dict]
    model: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[ChunkResult]
    synthesis: SynthesisResult | None = None
    latency_ms: float


class DebugResponse(BaseModel):
    query: str
    sparse: list[dict]
    dense: list[dict]
    hybrid_rrf: list[dict]


class HealthResponse(BaseModel):
    status: str
    bm25_chunks: int
    qdrant_points: int
    index_in_sync: bool


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    files_processed: int
    warnings: list[str] = []
