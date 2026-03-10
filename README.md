# Project Context: Traditional vs Neural Search — Comparison Prototype

## 1. Project Overview

This project is a working demo prototype that directly compares two fundamentally different information retrieval paradigms on the same dataset:

- **Traditional Search** — lexical, string-based retrieval using BM25 and TF-IDF
- **Neural Search** — semantic, embedding-based retrieval using dense vector representations

The goal is not to build a production search engine, but to construct a rigorous, demonstrable side-by-side comparison that surfaces where each approach excels, where it fails, and what a hybrid of both looks like. The output is an interactive demo with a live metrics dashboard.

---

## 2. Motivation & Research Question

Traditional search systems have powered the web for decades. They are fast, interpretable, and highly effective for keyword-dense queries. Neural search, powered by transformer-based sentence encoders, understands *meaning* rather than surface form — enabling it to match semantically related content even when no keywords overlap.

**Core question:** Under what conditions does semantic understanding outperform exact string matching, and vice versa?

**Secondary question:** Does a hybrid combination of both reliably outperform either system alone?

This prototype answers these questions empirically, using a public benchmark dataset with ground truth relevance labels.

---

## 3. Scope

### In Scope
- BM25-based lexical search pipeline (traditional)
- Dense retrieval pipeline using sentence transformers (neural)
- Shared preprocessing and evaluation on a public dataset (MS MARCO or SQuAD)
- Head-to-head evaluation: MRR@10, NDCG@10, Precision@k, Recall@k, latency
- Hybrid retrieval via Reciprocal Rank Fusion (RRF)
- Interactive side-by-side UI and live metrics dashboard
- Query-level failure analysis and edge case documentation

## 4. Dataset

**Primary candidate:** MS MARCO Passage Ranking  
**Alternative:** SQuAD v1.1 / v2.0

| Property | MS MARCO | SQuAD |
|---|---|---|
| Task type | Passage retrieval | Reading comprehension |
| Corpus size | ~8.8M passages | ~500 articles |
| Queries | ~1M (use subset) | ~100K |
| Ground truth | Sparse relevance judgments | Answer spans |
| Best for | Retrieval benchmarking | QA-style comparison |

A fixed evaluation set of 200–500 queries will be defined at project start and held constant across all systems to ensure fair comparison.

---

## 5. System Architecture

### 5.1 Shared Layer
Both systems operate on the same preprocessed corpus and receive identical queries.

```
Raw Dataset → Preprocessing → Indexed Corpus
                                   ↓
              Fixed Query Set → [BM25 Pipeline] → Results A
                             → [Neural Pipeline] → Results B
                                                       ↓
                                           Evaluation & Dashboard
```

### 5.2 Lexical Search Pipeline (Traditional)
1. **Tokenization** — lowercase, punctuation removal, stopword filtering
2. **Stemming** (optional) — Porter or Snowball stemmer
3. **Inverted index construction** — BM25 via `rank_bm25` or Elasticsearch
4. **BM25 scoring** — tuned k1 and b parameters against dev set
5. **TF-IDF baseline** — secondary baseline using `sklearn`
6. **Result serialization** — top-k results + scores stored as JSON per query

### 5.3 Neural Search Pipeline (Semantic)
1. **Sentence encoding** — encode all corpus passages offline using a sentence transformer
2. **Vector index construction** — FAISS flat or HNSW index
3. **Query encoding** — encode query at runtime using the same model
4. **ANN search** — cosine similarity top-k retrieval
5. **Optional reranker** — cross-encoder for precision boost on top-50 candidates
6. **Result serialization** — top-k results + similarity scores stored as JSON per query

### 5.4 Hybrid Layer
Reciprocal Rank Fusion (RRF) combines ranked lists from both systems:

```
RRF_score(d) = Σ 1 / (k + rank(d))    where k = 60 (standard)
```

### 5.5 Evaluation Layer
All metrics computed against ground truth relevance labels:

| Metric | Description |
|---|---|
| MRR@10 | Mean Reciprocal Rank — how high is the first relevant result? |
| NDCG@10 | Graded ranking quality |
| Precision@k | Fraction of top-k results that are relevant |
| Recall@k | Fraction of relevant docs found in top-k |
| Latency p50/p95 | Query time percentiles for both systems |

### 5.6 Demo Layer
- **Side-by-side UI** — single search input, dual result columns
- **Highlighting** — matched terms highlighted in lexical results; similarity score shown in neural results
- **Live metrics cards** — per-query MRR, P@k, latency update on each search
- **Query history** — saved comparisons for walkthrough and presentation

---

## 6. Project Phases

| Phase | Title | Duration |
|---|---|---|
| 1 | Dataset & Baseline Setup | Week 1–2 |
| 2 | Traditional Search (Lexical) | Week 3–4 |
| 3 | Neural Search (Semantic) | Week 5–7 |
| 4 | Evaluation & Benchmarking | Week 8–9 |
| 5 | Demo Interface | Week 10–11 |
| 6 | Analysis & Presentation | Week 12–13 |

**Total estimated timeline:** ~13 weeks

---

## 7. Tools & Stack

### 7.1 Data & Preprocessing

| Tool | Purpose | Notes |
|---|---|---|
| `datasets` (HuggingFace) | Load MS MARCO / SQuAD | `pip install datasets` |
| `nltk` | Tokenization, stopwords, stemming | Download punkt & stopwords corpus |
| `pandas` | Data manipulation, query/corpus management | — |
| `jsonlines` | Serializing results per query | Lightweight, fast |

### 7.2 Traditional Search (Lexical)

| Tool | Purpose | Notes |
|---|---|---|
| `rank_bm25` | BM25 implementation in Python | Lightweight, no server required |
| `Elasticsearch` | Production-grade inverted index with BM25 | Optional — use if scale demands it |
| `sklearn` (TfidfVectorizer) | TF-IDF baseline | Part of `scikit-learn` |

### 7.3 Neural Search (Semantic)

| Tool | Purpose | Notes |
|---|---|---|
| `sentence-transformers` | Encode queries and passages into dense vectors | Use `all-MiniLM-L6-v2` or `msmarco-distilbert-base-v4` |
| `faiss-cpu` | Fast ANN vector search (flat or HNSW index) | Use `faiss-gpu` if CUDA available |
| `torch` | Backend for transformer inference | Required by sentence-transformers |
| `transformers` (HuggingFace) | Cross-encoder reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### 7.4 Evaluation

| Tool | Purpose | Notes |
|---|---|---|
| `pytrec_eval` | TREC-standard IR metrics (MRR, NDCG, MAP) | Industry standard for retrieval eval |
| `ranx` | Alternative: clean Python ranking metrics | Easier API than pytrec_eval |
| `scipy` | Wilcoxon / paired t-test for significance | — |
| `time` / `perf_counter` | Latency measurement | Built-in Python |

### 7.5 Backend & API

| Tool | Purpose | Notes |
|---|---|---|
| `FastAPI` | REST API serving both search pipelines | Async, fast, auto-docs via Swagger |
| `uvicorn` | ASGI server for FastAPI | — |
| `pydantic` | Request/response schema validation | Bundled with FastAPI |

### 7.6 Frontend & Demo UI

| Tool | Purpose | Notes |
|---|---|---|
| `React` | Side-by-side search UI + metrics dashboard | Vite or CRA scaffolding |
| `Tailwind CSS` | Styling | Utility-first, fast iteration |
| `Recharts` | Metric visualizations (bar, scatter, line) | React-native charting |
| **Alternative:** `Streamlit` | All-in-one Python demo UI | Faster to prototype; less polished |

### 7.7 Experiment Tracking & Observability

| Tool | Purpose | Notes |
|---|---|---|
| `mlflow` | Log eval runs, parameters, metrics | Lightweight local tracking |
| `loguru` | Structured logging | Drop-in replacement for `logging` |

### 7.8 Environment & Infra

| Tool | Purpose | Notes |
|---|---|---|
| `Python 3.10+` | Primary language | — |
| `conda` / `venv` | Environment isolation | Recommended: conda for FAISS GPU compatibility |
| `Docker` | Containerize backend + frontend | For eventual portability when deployment target is decided |
| `git` + `git-lfs` | Version control; LFS for index artifacts | — |
| `Jupyter` | Exploration, eval analysis, visualizations | — |

---

## 8. Key Design Decisions

**Why MS MARCO over SQuAD?**  
MS MARCO is a passage retrieval benchmark — structurally identical to what this project compares. SQuAD is a reading comprehension dataset; retrieval is a secondary concern. MS MARCO gives more realistic and interpretable retrieval metrics.

**Why `rank_bm25` over Elasticsearch?**  
For a prototype, `rank_bm25` is zero-infrastructure — no server to run or configure. Elasticsearch becomes relevant if the corpus is large enough that in-memory indexing is impractical (>1M passages).

**Why FAISS over Pinecone/Weaviate?**  
Deployment target is undecided. FAISS is local, free, and portable. A cloud vector DB can be swapped in later without changing the retrieval logic.

**Why FastAPI over Flask?**  
Async support is important when both pipelines run concurrently per query. FastAPI's Pydantic integration also enforces clean request/response contracts from day one.

**Streamlit vs React UI?**  
Use Streamlit for rapid early demos and stakeholder walkthroughs. Migrate to React + FastAPI if the demo needs to be embedded, shared publicly, or extended with richer interactivity.

---

## 9. Open Decisions

| Decision | Status | Impact |
|---|---|---|
| MS MARCO vs SQuAD | **Pending** — confirm based on compute available | Affects index build time and eval complexity |
| Deployment target | **Pending** — Cloud / On-prem / Hybrid | Affects FAISS vs cloud vector DB, Docker strategy |
| Cross-encoder reranker | **Optional** — adds quality, adds latency | Decide after baseline neural results are evaluated |
| Streamlit vs React UI | **Pending** — depends on timeline and polish required | Can start with Streamlit, migrate later |

---

## 10. Success Criteria

The prototype is considered complete when:

1. Both BM25 and neural pipelines return ranked results on the same query set
2. MRR@10, NDCG@10, P@k, and latency are computed and logged for both systems
3. A side-by-side UI allows live querying and comparison
4. At least one class of queries is identified where each system clearly outperforms the other
5. Hybrid RRF results are computed and compared against both individual systems
6. A curated set of demonstration queries is prepared for walkthrough

---

*This document should be updated at the end of each phase to reflect decisions made, tools confirmed, and scope changes.*
