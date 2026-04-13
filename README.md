# IMPLEMENTATION.md — Neural Search

**Objective:** Replace keyword search with a hybrid sparse + dense retrieval system (BM25 + embeddings + RRF) over client-ingested PDF/DOCX documents, with Groq-powered answer synthesis and a Streamlit interface.

---

## Final Stack

| Layer | Tool |
|-------|------|
| Document parsing | `pymupdf`, `python-docx` |
| Chunking | `langchain-text-splitters` |
| Sparse retrieval | `BM25s` |
| Embedding model | `all-MiniLM-L6-v2` |
| Dense retrieval | `Qdrant` (local mode) |
| Fusion | RRF (custom) |
| Inference / synthesis | `Groq API` (`llama3-8b-8192`) |
| API | `FastAPI` |
| UI | `Streamlit` |
| Config | `pydantic-settings` + `.env` |
| Logging | `loguru` |

---

## Project Structure

```
neural_search/
  src/
    neural_search/
      __init__.py
      config.py
      ingestion/
        parser.py          # PDF + DOCX → raw text with metadata
        chunker.py         # Recursive chunking with overlap
        pipeline.py        # Orchestrates parse → chunk → index
      retrieval/
        sparse.py          # BM25sRetriever
        dense.py           # QdrantRetriever
        hybrid.py          # RRF fusion
      synthesis/
        groq_client.py     # Groq API wrapper
        prompt.py          # Prompt templates
      api/
        main.py            # FastAPI app
        schemas.py         # Pydantic models
        routes.py          # /ingest, /search, /health
  ui/
    app.py                 # Streamlit app
    components/
      upload.py            # Document upload widget
      results.py           # Search results renderer
      sidebar.py           # Config + index stats panel
  scripts/
    ingest_documents.py    # CLI ingestion
    verify_index.py        # Index health check
  .env.example
  pyproject.toml
  README.md
```

---

## Phase 1 — Data Foundation
**Covers:** Project setup, document ingestion pipeline, both indexes built and verified.

### 1.1 Project Bootstrap
- Rewrite `pyproject.toml` with final stack deps
- `config.py` — typed pydantic-settings for all tunables (chunk size, overlap, top-k, RRF k, model names, paths)
- `.env.example` with `GROQ_API_KEY`, `QDRANT_PATH`, `BM25_INDEX_PATH`

### 1.2 Document Ingestion
- `parser.py` — PDF (pymupdf, page by page) + DOCX (python-docx, paragraph level); output `{doc_id, source_file, page, text}`
- `chunker.py` — `RecursiveCharacterTextSplitter` (512 tokens, 64 overlap); `chunk_id = sha256(source_file + chunk_index)`
- `pipeline.py` — parse → chunk → yield; per-document error handling (log and continue)
- `scripts/ingest_documents.py` — CLI with `--input-dir` and `--reset` flags

### 1.3 Index Build
- `sparse.py` — BM25s index: tokenize (lowercase, stopwords, punctuation strip), serialize to disk, load on startup
- `dense.py` — Qdrant local: embed chunks in batches via sentence-transformers, upsert with full metadata payload, idempotent by `chunk_id`
- `scripts/verify_index.py` — assert BM25 chunk count == Qdrant point count

### Exit Criteria
- Mixed PDF/DOCX folder ingested cleanly with no silent data loss
- Both indexes populated, verified, and survive a restart
- Embedding throughput benchmarked on target CPU machine

---

## Phase 2 — Search & Synthesis
**Covers:** Hybrid retrieval, Groq synthesis, FastAPI layer.

### 2.1 Sparse Retrieval
- `BM25sRetriever.search(query, k)` → ranked `[{chunk_id, score, rank}]`
- Scores normalized to [0,1] for debuggability

### 2.2 Dense Retrieval
- `QdrantRetriever.search(query, k)` → ranked `[{chunk_id, score, rank, metadata}]`
- Chunk text stored in Qdrant payload — no secondary lookup at query time

### 2.3 Hybrid RRF Fusion
- `HybridRetriever` — runs sparse + dense in parallel (asyncio)
- RRF formula: `score(d) = Σ 1 / (60 + rank(d))`
- Result includes source attribution per chunk: `sparse`, `dense`, or `both`
- `alpha` parameter exposed for future weighting tuning (default: equal)

### 2.4 Groq Answer Synthesis
- `groq_client.py` — async client with exponential backoff
- `prompt.py` — system prompt enforces answer-from-context-only; explicitly instructs "say I don't know" if answer absent
- Top-5 chunks passed as context (~2500 tokens); synthesis optional per request (`synthesize=true/false`)

### 2.5 FastAPI
- `POST /ingest` — file upload(s), runs ingestion pipeline
- `POST /search` — hybrid retrieval + optional synthesis; returns chunks with source file, page, retrieval source, and answer
- `DELETE /index` — wipe and rebuild both indexes
- `GET /health` — liveness + chunk count + collection stats
- Lifespan event: load BM25 index + Qdrant client on startup

### Exit Criteria
- `/search` returns relevant results with full provenance (file, page, retrieval source)
- Hybrid consistently outperforms either system alone on 20 manual test queries
- End-to-end latency (retrieval + synthesis) <5s on CPU
- Groq handles "not in context" without hallucinating

---

## Phase 3 — Streamlit UI & Hardening
**Covers:** User-facing interface, observability, validation, handoff-ready.

### 3.1 Streamlit UI
- `ui/app.py` — single-page app, talks to FastAPI backend
- `components/upload.py` — drag-and-drop PDF/DOCX upload, ingestion status indicator, file list
- `components/results.py`
  - Query input box
  - Synthesized answer card (if enabled) with source citations
  - Expandable ranked result cards: chunk text, source file, page, retrieval source badge (`BM25` / `Neural` / `Both`), RRF score
- `components/sidebar.py` — index stats (total chunks, collection size), `synthesize` toggle, top-k slider, reset index button

### 3.2 Observability
- Every ingestion logged: doc count, chunk count, duration (loguru)
- Every query logged: query text, retrieval latency, synthesis latency, top result source file
- Streamlit sidebar shows last 5 queries with latency

### 3.3 Validation & Hardening
- Manual relevance eval: 30–50 real client queries scored for top-3 relevance (target: >80% hit rate)
- Document and test failure modes: very short queries, cross-document queries, out-of-corpus queries
- Re-ingestion of same file must not duplicate chunks (idempotency test)
- API handles concurrent requests without index corruption

### Exit Criteria
- Client can upload documents and query entirely through the UI
- >80% of real client queries return at least one relevant result in top-3
- Zero index drift between BM25 and Qdrant stores
- System runs stably on CPU machine with no intervention

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| CPU embedding slow on large batches | Batch encode with progress bar; upsert is resumable |
| Qdrant data loss on crash | WAL enabled by default; document backup path |
| BM25 full rebuild on new docs | Acceptable at this scale; incremental indexing is v2 |
| Groq rate limits | Exponential backoff; synthesis is optional |
| Chunk boundary splits context | 64-token overlap default; tune on real queries |
| DOCX tables/images | Skip non-text elements, flag in metadata |

---
