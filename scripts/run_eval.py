"""
Evaluation runner — compares BM25, Dense, and Hybrid RRF.

Requires:
  evaluation/queries.json   — [{"id": "q1", "text": "..."}]
  evaluation/relevance.json — {"q1": ["chunk_id_1", ...], ...}

Usage:
    python scripts/run_eval.py --collection <slug>
    python scripts/run_eval.py --collection <slug> --k 5
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
    # #21: support both spellings — relevance.json (correct) and relevence.json (legacy)
    queries_path = EVAL_DIR / "queries.json"
    relevance_path = EVAL_DIR / "relevance.json"
    if not relevance_path.exists():
        relevance_path = EVAL_DIR / "relevence.json"   # legacy typo fallback
    if not queries_path.exists() or not relevance_path.exists():
        logger.error(
            "Eval data not found. Expected:\n"
            "  evaluation/queries.json\n"
            "  evaluation/relevance.json"
        )
        raise FileNotFoundError("Missing eval data files")
    with open(queries_path) as f:
        queries = json.load(f)
    with open(relevance_path) as f:
        relevance = json.load(f)

    # #10: warn clearly if placeholder chunk IDs are still present
    placeholder = "<chunk_id_from_verify_index>"
    for qid, ids in relevance.items():
        if any(i == placeholder for i in ids):
            logger.warning(
                f"Query '{qid}' has placeholder relevance IDs — "
                "metrics will be 0.0. Run verify_index.py and update relevance.json."
            )
    return queries, relevance


def main():
    parser = argparse.ArgumentParser(description="Neural Search — Evaluation Runner")
    parser.add_argument("--k", type=int, default=10)
    # #1: collection_slug is now required — retrievers need it
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection slug to evaluate against (e.g. hr-policies)",
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    queries, relevance = load_eval_data()

    # #1: pass collection_slug to all retrievers
    sparse = BM25sRetriever(collection_slug=args.collection)
    sparse.load()
    dense = QdrantRetriever(collection_slug=args.collection)
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

    print(f"\n{'='*60}")
    print(f"Evaluation Results  |  k={args.k}  |  collection={args.collection}")
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
