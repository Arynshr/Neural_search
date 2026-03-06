from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import jsonlines
import json
import os
import random

BASE = Path(__file__).parent.parent
os.makedirs(BASE / "data" / "processed", exist_ok=True)
random.seed(42)

ds = load_dataset("ms_marco", "v1.1", split="validation", streaming=True)
queries = []
qrels = {}
MAX_QUERIES = 500

for item in tqdm(ds, desc="Processing queries", total=MAX_QUERIES):
    qid = str(item["query_id"])
    query_text = item["query"].strip()
    passages = item["passages"]["passage_text"]
    is_selected = item["passages"]["is_selected"]

    if 1 not in is_selected:          
        continue

    queries.append({"id": qid, "text": query_text})

    qrels[qid] = {}
    for j, selected in enumerate(is_selected):
        if selected == 1:
            qrels[qid][f"{qid}_{j}"] = 1

    if len(queries) >= MAX_QUERIES:
        break

with jsonlines.open(BASE / "data" / "processed" / "eval_queries.jsonl", "w") as f:
    for q in queries:
        f.write(q)

with open(BASE / "data" / "processed" / "qrels.json", "w") as f:   
    json.dump(qrels, f, indent=2)

total_rels = sum(len(v) for v in qrels.values())
print(f"Saved {len(queries)} queries and {total_rels} relevance judgments")
print(f"Avg relevant passages per query: {total_rels / len(queries):.2f}" if queries else "No queries saved")
