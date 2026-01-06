import os
import json
import glob
from datetime import datetime
from collections import Counter
from tqdm import tqdm



# --------------------------------------------------
# Paths (ADJUST IF NEEDED)
# --------------------------------------------------

DOCUMENTS_DIR = "../../data/longeval_sci_testing_2025_abstract/documents"
IDS_FILE = "cluster_38_ids.txt"     # one ID per line
OUTPUT_JSONL = "cluster_38_documents_sorted.jsonl"
STATS_JSON = "cluster_38_year_stats.json"


# --------------------------------------------------
# Load target IDs
# --------------------------------------------------

with open(IDS_FILE, "r", encoding="utf-8") as f:
    TARGET_IDS = set(line.strip() for line in f)

print(f"Loaded {len(TARGET_IDS)} target IDs")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_year(published_date: str):
    """
    Extract year from ISO datetime string.
    Example: 2012-01-01T00:00:00 -> 2012
    """
    if not published_date:
        return None
    try:
        return datetime.fromisoformat(published_date[:19]).year
    except Exception:
        return None


def sortable_date(published_date: str):
    """
    Convert date to sortable datetime.
    Missing dates go to the end.
    """
    try:
        return datetime.fromisoformat(published_date[:19])
    except Exception:
        return datetime.max


# --------------------------------------------------
# Main extraction
# --------------------------------------------------

documents = []
year_counter = Counter()

jsonl_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.jsonl")))

for path in tqdm(jsonl_files, desc="Scanning document shards"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)

            doc_id = str(doc.get("id"))
            if doc_id not in TARGET_IDS:
                continue

            published_date = doc.get("publishedDate")
            year = extract_year(published_date)

            if year:
                year_counter[year] += 1

            documents.append({
                "id": doc_id,
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "publishedDate": published_date,
                "year": year
            })

print(f"Matched documents: {len(documents)}")


# --------------------------------------------------
# Sort by publication date
# --------------------------------------------------

documents.sort(key=lambda d: sortable_date(d["publishedDate"]))


# --------------------------------------------------
# Write filtered JSONL
# --------------------------------------------------

with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
    for doc in documents:
        out.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"Saved sorted documents to: {OUTPUT_JSONL}")


# --------------------------------------------------
# Save yearly statistics
# --------------------------------------------------

with open(STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(dict(sorted(year_counter.items())), f, indent=2)

print(f"Saved year statistics to: {STATS_JSON}")


# --------------------------------------------------
# Pretty console summary
# --------------------------------------------------

print("\nPapers per year:")
for year, count in sorted(year_counter.items()):
    print(f"{year}: {count}")
