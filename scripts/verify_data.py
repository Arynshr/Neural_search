import jsonlines
import json
from pathlib import Path

BASE = Path(__file__).parent.parent

# Check corpus
with jsonlines.open(BASE / "data" / "raw" / "corpus.jsonl") as f:
    corpus = list(f)
print(f"Corpus size:    {len(corpus):,} passages")
print(f"Sample passage: {corpus[0]}")

# Check queries
with jsonlines.open(BASE / "data" / "processed" / "eval_queries.jsonl") as f:
    queries = list(f)
print(f"\nQuery count:  {len(queries):,}")
print(f"Sample query: {queries[0]}")

# Check qrels
with open(BASE / "data" / "processed" / "qrels.json") as f:
    qrels = json.load(f)
total_relevant = sum(len(v) for v in qrels.values())
print(f"\nQrels coverage: {len(qrels):,} queries with judgments")
print(f"Total relevant: {total_relevant:,} passage-query pairs")

# Sanity: every qrels query_id should exist in queries
query_ids = {q["id"] for q in queries}
orphan_qrels = [qid for qid in qrels if qid not in query_ids]
print(f"\nOrphan qrels: {len(orphan_qrels)} (should be 0)")

# Summary
print("\n--- Phase 1 Verification ---")
print(f"{'PASS' if len(corpus) > 0 else 'FAIL'} Corpus non-empty")
print(f"{'PASS' if len(queries) == 500 else 'FAIL'} Query count = {len(queries)} (target 500)")
print(f"{'PASS' if len(orphan_qrels) == 0 else 'FAIL'} No orphan qrels")
print(f"{'PASS' if len(qrels) == len(queries) else 'FAIL'} Qrels covers all queries")
