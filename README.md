# Neural-IR

**Objective:** Build a hybrid sparse + dense retrieval system with measurable improvements over baseline retrieval, incorporating evaluation, reranking, and adaptive retrieval strategies.

---

# Core Philosophy Update

This project now follows two parallel tracks:

- **Track A — System Engineering** (ingestion, indexing, API, UI)
- **Track B — Retrieval Intelligence** (evaluation, optimization, learning)

The goal is not just to build a working system, but to **prove improvements quantitatively**.

---

# Final Stack (Updated)

| Layer | Tool |
|------|------|
| Document parsing | `pymupdf`, `python-docx` |
| Chunking | `langchain-text-splitters` |
| Sparse retrieval | `BM25s` |
| Embedding model | `all-MiniLM-L6-v2` |
| Dense retrieval | `Qdrant` (local mode) |
| Fusion | RRF + Learned Hybrid (LogReg) |
| Reranking | Cross-Encoder (`ms-marco-MiniLM`) |
| Inference / synthesis | `Groq API` (`llama3-8b-8192`) |
| API | `FastAPI` |
| UI | `Streamlit` |
| Config | `pydantic-settings` + `.env` |
| Logging | `loguru` |

---

# Project Structure (Additions Only)

```
neural_search/
  evaluation/
    queries.json          # query set
    relevance.json        # labeled relevance
  scripts/
    run_eval.py           # evaluation runner
  src/
    neural_search/
      retrieval/
        reranker.py       # cross-encoder reranker
        learned.py        # learned hybrid model
      evaluation/
        metrics.py        # P@K, MRR, nDCG
        dataset.py        # load eval data
```

---

# Phase 1 — Data Foundation (Week 1–2)

**Unchanged core, with additions for evaluation readiness**

### Additions
- Store both **raw text + cleaned text**
- Store **token count per chunk**
- Export dataset snapshot (JSONL)

### Exit Criteria
- Same as before + dataset ready for evaluation

---

# Phase 2 — Baseline Retrieval (Week 3)

**Focus: Build clean, debuggable baselines before optimization**

### Additions
- `/search/debug` endpoint returning:
  - BM25 ranks
  - Dense ranks
  - RRF output

### Exit Criteria
- Clear visibility into retrieval behavior

---

# Phase 3 — Evaluation Framework (Week 4) ⚠️

**New critical phase**

### 3.1 Labeled Dataset
- 30–50 queries
- Each query mapped to relevant chunk_ids

### 3.2 Metrics
Implement:
- Precision@K
- Recall@K
- MRR
- nDCG

### 3.3 Benchmark Script

`scripts/run_eval.py`

Outputs comparative metrics:
- BM25
- Dense
- Hybrid (RRF)

### Exit Criteria
- Quantitative comparison of retrieval methods

---

# Phase 4 — Retrieval Optimization (Week 5–6)

## 4.1 Learned Hybrid Fusion

Replace static RRF with model-based scoring.

### Features
- BM25 score
- Dense score
- Rank positions
- Query length
- Chunk length

### Model
- Logistic Regression

### Goal
- Predict probability of relevance

---

## 4.2 Reranking Layer

Pipeline:

retrieve top-20 → rerank → top-5

### Implementation
- Cross-encoder model
- Compare performance vs baseline

---

## 4.3 Context Optimization

- Remove redundant chunks (MMR)
- Merge adjacent chunks

---

# Phase 5 — Adaptive Retrieval (Week 7)

### Query Classification

Classify queries into:
- keyword-heavy
- semantic
- vague

### Dynamic Strategy

Adjust hybrid weights dynamically.

---

# Phase 6 — API & Synthesis (Integrated)

### Enhancements
- Support retrieval mode selection:
  - BM25
  - Dense
  - Hybrid
  - Learned
- Include reranked outputs

---

# Phase 7 — Streamlit UI & Observability (Week 8)

### UI Additions
- Toggle retrieval modes
- Show score breakdown
- Show reranking impact

### Observability
- Log retrieval + reranking latency separately
- Display last queries with metrics

---

# Evaluation Goals (Updated)

| Metric | Target |
|------|--------|
| Precision@3 | >80% |
| Latency | <5s |
| Hybrid vs BM25 improvement | measurable |

---

# Updated Timeline

| Phase | Deliverable | Duration |
|------|------------|----------|
| 1 | Data foundation | 2 weeks |
| 2 | Baseline retrieval | 1 week |
| 3 | Evaluation framework | 1 week |
| 4 | Retrieval optimization | 2 weeks |
| 5 | Adaptive retrieval | 1 week |
| 6 | API + synthesis | parallel |
| 7 | UI + hardening | 1 week |

**Total: 8 weeks**

---

# Final Outcome

System evolves from:

- Basic RAG pipeline

To:

- Evaluated retrieval system
- Learned ranking model
- Adaptive hybrid search
- Reranked and optimized context

---
