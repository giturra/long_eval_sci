import json
from pathlib import Path
INPUT_FOLDER = Path("data/longeval_sci_testing_2025_fulltext/documents/")
docs = []
with open("data/longeval_sci_testing_2025_fulltext/documents/documents_000001.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

print(len(docs))   # first JSON object