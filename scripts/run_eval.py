"""
Evaluation runner — compares BM25, Dense, and Hybrid RRF.

Requires:
  evaluation/queries.json   — [{"id": "q1", "text": "..."}]
  evaluation/relevance.json — {"q1": ["chunk_id_1", "chunk_id_2"], ...}

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --k 5
"""
import json
import argparse
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.evaluation.metrics import evaluate_run

EVAL_DIR = Path("evaluation")


def load_eval_data():
    queries_path = EVAL_DIR / "queries.json"
    relevance_path = EVAL_DIR / "relevance.json"
    if not queries_path.exists() or not relevance_path.exists():
        logger.error("Eval data not found — create evaluation/queries.json and evaluation/relevance.json")
        raise FileNotFoundError
    with open(queries_path) as f:
        queries = json.load(f)
    with open(relevance_path) as f:
        relevance = json.load(f)
    return queries, relevance


def main():
    parser = argparse.ArgumentParser(description="Neural Search — Evaluation Runner")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    settings.ensure_dirs()
    queries, relevance = load_eval_data()

    sparse = BM25sRetriever()
    sparse.load()
    dense = QdrantRetriever()
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    retrievers = {
        "BM25 (sparse)": lambda q, k: [r["chunk_id"] for r in sparse.search(q, k=k)],
        "Dense (Qdrant)": lambda q, k: [r["chunk_id"] for r in dense.search(q, k=k)],
        "Hybrid (RRF)":   lambda q, k: [r["chunk_id"] for r in hybrid.search(q, k=k)],
    }

    results_agg: dict[str, list[dict]] = {name: [] for name in retrievers}

    for query in queries:
        qid = query["id"]
        qtext = query["text"]
        relevant = set(relevance.get(qid, []))
        if not relevant:
            logger.warning(f"No relevance labels for query '{qid}' — skipping")
            continue
        for name, fn in retrievers.items():
            retrieved = fn(qtext, args.k)
            metrics = evaluate_run(retrieved, relevant, k=args.k)
            results_agg[name].append(metrics)

    # Aggregate and print
    print(f"\n{'='*60}")
    print(f"Evaluation Results  |  k={args.k}  |  queries={len(queries)}")
    print(f"{'='*60}")
    for name, runs in results_agg.items():
        if not runs:
            continue
        avg = {
            metric: round(sum(r[metric] for r in runs) / len(runs), 4)
            for metric in runs[0]
        }
        print(f"\n{name}")
        for metric, val in avg.items():
            print(f"  {metric:<20} {val}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
