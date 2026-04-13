from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import jsonlines
import os

BASE = Path(__file__).parent.parent
os.makedirs(BASE / "data" / "raw", exist_ok=True)

ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

total_passages = 0
with jsonlines.open(BASE / "data" / "raw" / "corpus.jsonl", "w") as f:
    for i, item in enumerate(tqdm(ds, desc="Downloading corpus", total=50_000)):
        for j, text in enumerate(item["passages"]["passage_text"]):
            f.write({"id": f"{item['query_id']}_{j}", "text": text})
            total_passages += 1
        if i >= 50_000:
            break

print(f"Saved {total_passages:,} passages from {i+1:,} queries → data/raw/corpus.jsonl")
