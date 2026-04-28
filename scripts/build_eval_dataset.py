#!/usr/bin/env python3
"""
build_eval_dataset.py

Semi-automated evaluation dataset builder.

Usage:
    python scripts/build_eval_dataset.py \
        --collection base \
        --output evaluation/ \
        [--queries evaluation/queries.json]   # optional: resume from existing queries

Workflow:
    1. Loads (or generates) queries from queries.json
    2. For each query, retrieves top-10 candidates from the index
    3. Displays each candidate and prompts for manual 0/1 relevance judgment
    4. Writes labeled results to evaluation/relevance.json

Target: 50+ queries across keyword-heavy, semantic, and vague types.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever


def load_queries(path: Path) -> list[dict]:
    if path.exists():
        queries = json.loads(path.read_text())
        print(f"Loaded {len(queries)} queries from {path}")
        return queries
    return []


def load_relevance(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved → {path}")


def label_interactively(
    query: dict,
    results: list[dict],
    existing_labels: dict,
) -> list[str]:
    """
    Show retrieval results for a query and collect relevance judgments.
    Returns list of relevant chunk_ids.
    """
    qid = query["id"]
    qtext = query["text"]

    if qid in existing_labels:
        print(f"\n[SKIP] {qid} already labeled ({len(existing_labels[qid])} relevant chunks)")
        return existing_labels[qid]

    print(f"\n{'='*70}")
    print(f"Query [{qid}]: {qtext}")
    print(f"{'='*70}")

    relevant_ids = []
    for i, result in enumerate(results, start=1):
        print(f"\n  [{i}] chunk_id: {result['chunk_id']}")
        print(f"       source:   {result['source_file']} (page {result['page']})")
        print(f"       score:    {result.get('rrf_score', result.get('score', 0)):.4f}")
        print(f"       text:     {result['text'][:200]}...")

        while True:
            answer = input("  Relevant? [1=yes / 0=no / s=skip query / q=quit]: ").strip().lower()
            if answer in {"0", "1", "s", "q"}:
                break
            print("  Please enter 1, 0, s, or q")

        if answer == "q":
            print("\nSaving progress and exiting...")
            return relevant_ids
        if answer == "s":
            print(f"  Skipping {qid}")
            return relevant_ids
        if answer == "1":
            relevant_ids.append(result["chunk_id"])

    print(f"\n  → {len(relevant_ids)} relevant chunks for [{qid}]")
    return relevant_ids


def main():
    parser = argparse.ArgumentParser(description="Build evaluation dataset interactively")
    parser.add_argument("--collection", required=True, help="Collection slug")
    parser.add_argument("--output", default="evaluation", help="Output directory")
    parser.add_argument("--queries", help="Path to existing queries.json (optional)")
    parser.add_argument("--k", type=int, default=10, help="Candidates per query")
    args = parser.parse_args()

    output_dir = Path(args.output)
    queries_path = Path(args.queries) if args.queries else output_dir / "queries.json"
    relevance_path = output_dir / "relevance.json"

    queries = load_queries(queries_path)
    relevance = load_relevance(relevance_path)

    if not queries:
        print("No queries found. Add queries to evaluation/queries.json first.")
        print('Example format: [{"id": "q1", "text": "What is agent memory?"}]')
        sys.exit(1)

    # ── Load retriever ────────────────────────────────────────────────────────
    print(f"\nLoading index for collection: {args.collection}")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print(f"ERROR: BM25 index not found for collection '{args.collection}'")
        sys.exit(1)
    dense = QdrantRetriever(collection_slug=args.collection)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    # ── Label loop ────────────────────────────────────────────────────────────
    labeled_count = len(relevance)
    print(f"\nStarting labeling. {labeled_count}/{len(queries)} already done.")
    print("Target: 50+ queries. Type 'q' at any time to save and exit.\n")

    try:
        for query in queries:
            outcome = hybrid.search(query["text"], k=args.k, expand=False, web_search=False, rerank_results=False)
            results = outcome["results"]
            relevant_ids = label_interactively(query, results, relevance)
            relevance[query["id"]] = relevant_ids
            save_json(relevance, relevance_path)
    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved.")

    total_labeled = sum(1 for v in relevance.values() if v is not None)
    print(f"\nDone. {total_labeled}/{len(queries)} queries labeled.")
    print(f"Relevance file: {relevance_path}")

    if total_labeled < 50:
        print(f"WARNING: Only {total_labeled} queries labeled. Target is 50+.")
        print("Add more queries to evaluation/queries.json and re-run.")


if __name__ == "__main__":
    main()
