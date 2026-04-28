#!/usr/bin/env python3
"""
run_eval.py

Evaluation runner. Compares BM25, Dense, and Hybrid retrieval modes
against the labeled relevance dataset.

Usage:
    python scripts/run_eval.py --collection base --k 5
    python scripts/run_eval.py --collection base --k 5 --output evaluation/results/phase3.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever


def load_eval_data(eval_dir: Path) -> tuple[list[dict], dict]:
    queries_path = eval_dir / "queries.json"
    relevance_path = eval_dir / "relevance.json"

    if not queries_path.exists():
        print(f"ERROR: queries.json not found at {queries_path}")
        sys.exit(1)
    if not relevance_path.exists():
        print(f"ERROR: relevance.json not found at {relevance_path}")
        sys.exit(1)

    queries = json.loads(queries_path.read_text())
    relevance = json.loads(relevance_path.read_text())

    # Validate no placeholder chunk IDs
    for qid, chunk_ids in relevance.items():
        for cid in chunk_ids:
            if "<chunk_id" in cid:
                print(f"ERROR: Placeholder chunk_id found in relevance.json for {qid}")
                print("Run build_eval_dataset.py to label real chunk IDs first.")
                sys.exit(1)

    labeled = {qid: set(ids) for qid, ids in relevance.items() if ids}
    print(f"Loaded {len(queries)} queries, {len(labeled)} with relevance labels")
    return queries, labeled


def evaluate_mode(
    queries: list[dict],
    relevance: dict[str, set],
    retrieve_fn,
    k: int,
) -> dict:
    p_at_k_scores, recall_scores, mrr_scores, ndcg_scores = [], [], [], []

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue

        relevant = relevance[qid]
        results = retrieve_fn(query["text"], k)
        result_ids = [r["chunk_id"] for r in results]

        p_at_k_scores.append(precision_at_k(result_ids, relevant, k))
        recall_scores.append(recall_at_k(result_ids, relevant, k))
        mrr_scores.append(mrr(result_ids, relevant))
        ndcg_scores.append(ndcg_at_k(result_ids, relevant, k))

    n = len(p_at_k_scores)
    return {
        f"P@{k}": round(sum(p_at_k_scores) / n, 4) if n else 0.0,
        f"Recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        "MRR": round(sum(mrr_scores) / n, 4) if n else 0.0,
        f"nDCG@{k}": round(sum(ndcg_scores) / n, 4) if n else 0.0,
        "queries_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-dir", default="evaluation")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    queries, relevance = load_eval_data(eval_dir)

    print(f"\nLoading index for collection: {args.collection}")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print(f"ERROR: BM25 index not found for '{args.collection}'")
        sys.exit(1)
    dense = QdrantRetriever(collection_slug=args.collection)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    def sparse_fn(q, k): return sparse.search(q, k=k)
    def dense_fn(q, k): return dense.search(q, k=k)
    def hybrid_fn(q, k): return hybrid.search_full(q, k=k, expand=False, web_search=False, rerank=False)["results"]
    def hybrid_rerank_fn(q, k): return hybrid.search_full(q, k=k, expand=False, web_search=False, rerank=True, rerank_top_k=k)["results"]

    modes = {
        "BM25 (sparse)": sparse_fn,
        "Dense (Qdrant)": dense_fn,
        "Hybrid RRF": hybrid_fn,
        "Hybrid RRF + Reranker": hybrid_rerank_fn,
    }

    results = {}
    print(f"\nEvaluating at k={args.k} across {len(relevance)} labeled queries...\n")

    for mode_name, fn in modes.items():
        t0 = time.perf_counter()
        metrics = evaluate_mode(queries, relevance, fn, args.k)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        results[mode_name] = {**metrics, "eval_latency_ms": elapsed}

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 26
    metric_keys = [f"P@{args.k}", f"Recall@{args.k}", "MRR", f"nDCG@{args.k}"]

    header = f"{'Mode':<{col_w}}" + "".join(f"{m:>12}" for m in metric_keys)
    print(header)
    print("-" * len(header))

    for mode_name, metrics in results.items():
        row = f"{mode_name:<{col_w}}" + "".join(
            f"{metrics.get(m, 0.0):>12.4f}" for m in metric_keys
        )
        print(row)

    print()

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection": args.collection,
            "k": args.k,
            "queries_labeled": len(relevance),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
