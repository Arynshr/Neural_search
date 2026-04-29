#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever


class _Quit(Exception):
    pass


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_retriever(collection: str, mode: str):
    sparse = BM25sRetriever(collection_slug=collection)
    sparse.load()
    dense = QdrantRetriever(collection_slug=collection)

    if mode == "sparse":
        return sparse
    if mode == "dense":
        return dense
    return HybridRetriever(sparse=sparse, dense=dense)


def print_result(i, r):
    print(f"\n[{i}] chunk_id: {r.get('chunk_id')}")

    print(
        f"  scores → "
        f"bm25: {r.get('bm25_score', 0):.3f} | "
        f"dense: {r.get('dense_score', 0):.3f} | "
        f"rrf: {r.get('rrf_score', r.get('score', 0)):.3f}"
    )

    src = r.get("source_file", "unknown")
    page = r.get("page", "-")
    print(f"  source: {src} (page {page})")

    text = r.get("text", "")
    print(f"  text: {text[:200]}...")


def label_query(query, results, relevance, allow_override=False):
    qid = query["id"]

    if qid in relevance and not allow_override:
        print(f"[SKIP] {qid} already labeled")
        return relevance[qid]

    print("\n" + "=" * 80)
    print(f"{qid} [{query.get('type','?')}] → {query['text']}")
    print("=" * 80)

    relevant = []

    i = 0
    while i < len(results):
        r = results[i]
        print_result(i, r)

        cmd = input("1=yes 0=no s=skip r=restart q=quit >> ").strip().lower()

        if cmd == "q":
            raise _Quit
        elif cmd == "s":
            return relevant
        elif cmd == "r":
            print("Restarting query...\n")
            return label_query(query, results, relevance, allow_override=True)
        elif cmd == "1":
            relevant.append(r["chunk_id"])
            i += 1
        elif cmd == "0":
            i += 1
        else:
            print("Invalid input")

    print(f"\n→ {len(relevant)} relevant chunks")
    return relevant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    queries_path = Path(args.queries) if args.queries else out / "queries.json"
    relevance_path = out / "relevance.json"

    queries = load_json(queries_path, [])
    relevance = load_json(relevance_path, {})

    if not queries:
        print("No queries found.")
        sys.exit(1)

    retriever = get_retriever(args.collection, args.mode)

    print(f"\nMode: {args.mode}")
    print(f"Labeled: {len(relevance)}/{len(queries)}\n")

    try:
        for q in queries:
            if q["id"] in relevance and not args.override:
                continue

            results = retriever.search(q["text"], k=args.k)

            labels = label_query(q, results, relevance, args.override)
            relevance[q["id"]] = labels

            save_json(relevance, relevance_path)
            print(f"Saved → {relevance_path}")

    except _Quit:
        print("\nExiting early... progress saved.")
    except KeyboardInterrupt:
        print("\nInterrupted... progress saved.")

    print(f"\nDone: {len(relevance)}/{len(queries)} labeled")


if __name__ == "__main__":
    main()
