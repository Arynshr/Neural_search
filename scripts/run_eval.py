"""
scripts/run_eval.py

Runs retrieval evaluation across BM25, Dense, and Hybrid modes.
Outputs a comparison table and saves results to logs/eval_results.json.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --k 5
    python scripts/run_eval.py --collection base --k 3 --output logs/my_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_search.evaluation.dataset import load_dataset
from neural_search.evaluation.metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever

MODES = ["sparse", "dense", "hybrid", "learned"]

def _build_retrievers(collection: str):
    sparse = BM25sRetriever(collection_slug=collection)
    dense = QdrantRetriever(collection_slug=collection)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    if not sparse.load():
        print(f"ERROR: BM25 index not found for collection '{collection}'. Run ingestion first.")
        sys.exit(1)

    return sparse, dense, hybrid


def _retrieve(mode: str, query: str, k: int, sparse, dense, hybrid) -> list[str]:
    """Returns ordered list of chunk_ids."""
    if mode == "sparse":
        results = sparse.search(query, k=k)
    elif mode == "dense":
        results = dense.search(query, k=k)
    else:
        results = hybrid.search(query, k=k)
    return [r["chunk_id"] for r in results]


def _run_mode(mode: str, dataset, k: int, sparse, dense, hybrid) -> dict:
    queries = dataset.labeled_queries()
    if not queries:
        print("WARNING: No labeled queries found. Populate evaluation/relevance.json first.")
        return {}

    p_scores, r_scores, mrr_scores, ndcg_scores, latencies = [], [], [], [], []

    for q in queries:
        relevant = dataset.get_relevant(q.id)
        t0 = time.perf_counter()
        retrieved = _retrieve(mode, q.text, k, sparse, dense, hybrid)
        latency = (time.perf_counter() - t0) * 1000

        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        mrr_scores.append(mrr(retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))
        latencies.append(latency)

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        f"precision@{k}": avg(p_scores),
        f"recall@{k}": avg(r_scores),
        "mrr": avg(mrr_scores),
        f"ndcg@{k}": avg(ndcg_scores),
        "avg_latency_ms": round(avg(latencies), 2),
        "queries_evaluated": len(queries),
    }


def _print_table(results: dict[str, dict], k: int):
    col_w = 14
    metrics = [f"precision@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "avg_latency_ms"]
    header = f"{'Metric':<20}" + "".join(f"{m:>{col_w}}" for m in MODES)
    print("\n" + "=" * (20 + col_w * 3))
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * (20 + col_w * 3))
    print(header)
    print("-" * (20 + col_w * 3))

    for metric in metrics:
        row = f"{metric:<20}"
        for mode in MODES:
            val = results.get(mode, {}).get(metric, "—")
            row += f"{str(val):>{col_w}}"
        print(row)

    print("-" * (20 + col_w * 3))
    for mode in MODES:
        n = results.get(mode, {}).get("queries_evaluated", 0)
        print(f"  {mode}: {n} queries evaluated")
    print()


def _improvement(base: dict, compare: dict, metric: str) -> str:
    b = base.get(metric, 0)
    c = compare.get(metric, 0)
    if b == 0:
        return "N/A"
    delta = ((c - b) / b) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _print_improvements(results: dict[str, dict], k: int):
    metric = f"precision@{k}"
    sparse = results.get("sparse", {})
    dense = results.get("dense", {})
    hybrid = results.get("hybrid", {})

    print("  Improvements vs BM25 (sparse baseline):")
    print(f"    Dense  {metric}: {_improvement(sparse, dense, metric)}")
    print(f"    Hybrid {metric}: {_improvement(sparse, hybrid, metric)}")
    print(f"    Dense  MRR:          {_improvement(sparse, dense, 'mrr')}")
    print(f"    Hybrid MRR:          {_improvement(sparse, hybrid, 'mrr')}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="base", help="Collection slug")
    parser.add_argument("--k", type=int, default=5, help="Evaluation depth (default: 5)")
    parser.add_argument(
        "--queries", default="evaluation/queries.json", help="Path to queries file"
    )
    parser.add_argument(
        "--relevance", default="evaluation/relevance.json", help="Path to relevance file"
    )
    parser.add_argument(
        "--output", default="logs/eval_results.json", help="Path to save results JSON"
    )
    args = parser.parse_args()

    print(f"\nLoading eval dataset...")
    dataset = load_dataset(queries_path=args.queries, relevance_path=args.relevance)
    print(f"Coverage: {dataset.coverage}")

    if not dataset.labeled_queries():
        print(
            "\nERROR: No labeled queries. Run the following to label:\n"
            "  python scripts/label_relevance.py\n"
        )
        sys.exit(1)

    print(f"\nBuilding retrievers for collection: '{args.collection}'...")
    sparse, dense, hybrid = _build_retrievers(args.collection)

    print(f"Running evaluation at k={args.k}...\n")
    mode_results = {}
    for mode in MODES:
        print(f"  Evaluating {mode}...")
        mode_results[mode] = _run_mode(mode, dataset, args.k, sparse, dense, hybrid)

    _print_table(mode_results, args.k)
    _print_improvements(mode_results, args.k)

    # Breakdown by query type
    for qtype in ("keyword", "semantic", "vague"):
        type_queries = dataset.by_type(qtype)
        if not type_queries:
            continue
        print(f"  [{qtype.upper()}] {len(type_queries)} queries:")
        for mode in MODES:
            p_scores = []
            for q in type_queries:
                relevant = dataset.get_relevant(q.id)
                retrieved = _retrieve(mode, q.text, args.k, sparse, dense, hybrid)
                p_scores.append(precision_at_k(retrieved, relevant, args.k))
            avg_p = round(sum(p_scores) / len(p_scores), 4) if p_scores else 0.0
            print(f"    {mode:<10} precision@{args.k}: {avg_p}")
        print()

    # Save to logs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection,
        "k": args.k,
        "queries_total": len(dataset.queries),
        "queries_labeled": len(dataset.labeled_queries()),
        "results": mode_results,
    }

    # Append to history if file exists
    history = []
    if output_path.exists():
        with output_path.open() as f:
            existing = json.load(f)
            history = existing if isinstance(existing, list) else [existing]

    history.append(record)
    with output_path.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Results saved → {output_path}")
    dense._client.close()


if __name__ == "__main__":
    main()
