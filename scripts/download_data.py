from datasets import load_dataset
import jsonlines
import os

os.makedirs("data/raw", exist_ok= True)

ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
with jsonlines.open("data/raw/corpus.jsonl", "w") as f:
    for i, item in enumerate(ds):
        for j,text in enumerate(item["passages"]["passage_text"]):
            f.write({"id": f"{i}_{j}", "text": text})
        if i >= 50_000:
            break
