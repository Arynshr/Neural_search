# Neural Search

A hybrid sparse + dense retrieval system for PDF and DOCX documents, with multi-collection support, a FastAPI backend, Streamlit UI, and LLM-based answer synthesis via Groq.

---

## Stack

| Layer | Technology |
|---|---|
| Document parsing | `pymupdf`, `python-docx` |
| Chunking | `langchain-text-splitters` (512 tokens, 64 overlap) |
| Sparse retrieval | `BM25s` + NLTK |
| Dense retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) + `Qdrant` (local) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Answer synthesis | `Groq` (`llama-3.1-8b-instant`) |
| API | `FastAPI` + `Uvicorn` |
| UI | `Streamlit` |
| Config | `pydantic-settings` + `.env` |
| Logging | `loguru` |

---

## Project Structure

```
Neural_search/
├── src/neural_search/
│   ├── api/            # FastAPI routes, schemas, app entrypoint
│   ├── ingestion/      # Document parsing and chunking pipeline
│   ├── retrieval/
│   │   ├── sparse.py   # BM25s retriever
│   │   ├── dense.py    # Qdrant dense retriever
│   │   └── hybrid.py   # RRF fusion + debug output
│   ├── collections/    # Multi-collection manager (CRUD + metadata)
│   ├── synthesis/      # Groq LLM answer synthesis
│   ├── evaluation/     # Metrics (P@K, Recall@K, MRR, nDCG) + dataset loader
│   └── config.py       # Pydantic settings (all paths & model params)
├── scripts/
│   ├── ingest_documents.py   # CLI: bulk ingest documents
│   ├── run_eval.py           # Evaluation runner (BM25 vs dense vs hybrid)
│   ├── build_lexical_index.py
│   └── BM25_benchmark.py
├── evaluation/
│   ├── queries.json          # Evaluation query set
│   └── relevance.json        # Ground-truth relevance labels
├── tests/                    # Unit + integration tests (pytest)
└── data/                     # Runtime data (qdrant, bm25_index, documents, snapshots)
```

---

## Quickstart

**Requirements:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env   # set GROQ_API_KEY

# Start the API
uvicorn neural_search.api.main:app --reload

# Start the UI (separate terminal)
streamlit run src/ui/app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status + collection count |
| `GET` | `/collections` | List all collections |
| `POST` | `/collections` | Create a named collection |
| `DELETE` | `/collections/{slug}` | Delete collection + its indexes |
| `POST` | `/collections/{slug}/ingest` | Upload and index a PDF/DOCX file |
| `POST` | `/search` | Search with mode: `sparse`, `dense`, or `hybrid` |
| `GET` | `/search/debug` | Side-by-side BM25 / dense / RRF rank breakdown |

**Search request example:**
```json
{
  "query": "what is attention mechanism",
  "collection": "ml-papers",
  "mode": "hybrid",
  "k": 5,
  "synthesize": true
}
```

---

## Configuration (`.env`)

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=10
RRF_K=60
DATA_DIR=./data
QDRANT_PATH=./data/qdrant
BM25_INDEX_PATH=./data/bm25_index
```

---

## Evaluation

Run quantitative retrieval comparison across BM25, dense, and hybrid modes:

```bash
python scripts/run_eval.py
```

Outputs **Precision@K, Recall@K, MRR, and nDCG** for each retrieval mode against labeled ground-truth in `evaluation/relevance.json`.

**Targets:**

| Metric | Target |
|---|---|
| Precision@3 | > 80% |
| Hybrid vs BM25 | measurable improvement |
| Search latency | < 5s |

---

## Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/ --ignore=tests/integration

# Integration tests
pytest tests/integration/
```
