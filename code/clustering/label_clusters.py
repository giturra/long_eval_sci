import os
import json
import glob
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Aggressive cleaning for scientific corpora.
    Removes URLs, digits, symbols, and normalizes text.
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove digits
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove non-letter characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# Stopwords (NLTK multilingual + academic)
# --------------------------------------------------

EXTRA_STOPWORDS = {
    # Generic academic
    "study", "studies", "research", "paper", "results", "analysis",
    "based", "using", "used", "method", "methods", "data",
    "new", "different", "effect", "effects", "model", "models",

    # Publishing boilerplate
    "abstract", "introduction", "review", "editorial",
    "volume", "issue", "table", "contents", "page", "pages",
    "conference", "proceedings", "journal",

    # Repositories / URLs
    "https", "http", "www", "pdf", "html",
    "digitalcommons", "scholarworks", "doi",

    # English glue
    "also", "may", "can", "one", "two", "three",

    # Extremely common corpus junk
    "available", "copyright", "author", "authors"
}


def build_stopwords() -> List[str]:
    """
    Build a multilingual stopword list using NLTK,
    merged with academic / corpus-specific stopwords.
    """
    languages = [
        "english",
        "spanish",
        "french",
        "german",
        "portuguese",
        "italian",
        "norwegian",
        "swedish",
        "danish",
    ]

    sw = set()
    for lang in languages:
        try:
            sw.update(stopwords.words(lang))
        except OSError:
            pass

    sw.update(EXTRA_STOPWORDS)

    return sorted(sw)


STOPWORDS = build_stopwords()


# --------------------------------------------------
# Load cluster assignments
# --------------------------------------------------

def load_clusters(cluster_path: str) -> Dict[int, List[str]]:
    """
    Returns:
        cluster_id -> list of doc_ids
    """
    clusters = defaultdict(list)

    with open(cluster_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            clusters[obj["cluster"]].append(obj["id"])

    return clusters


# --------------------------------------------------
# Load documents (title + abstract)
# --------------------------------------------------

def load_documents(documents_dir: str) -> Dict[str, str]:
    """
    Returns:
        doc_id -> cleaned combined text (title + abstract)
    """
    docs = {}

    jsonl_files = sorted(glob.glob(os.path.join(documents_dir, "*.jsonl")))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {documents_dir}")

    for path in tqdm(jsonl_files, desc="Loading documents"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                doc = json.loads(line)
                doc_id = doc.get("id")
                if not doc_id:
                    continue

                title = (doc.get("title") or "").strip()
                abstract = (doc.get("abstract") or "").strip()

                if not title and not abstract:
                    continue

                text = title if not abstract else f"{title}. {abstract}"
                text = clean_text(text)

                if text:
                    docs[doc_id] = text

    return docs


# --------------------------------------------------
# Label clusters with TF-IDF
# --------------------------------------------------

def label_clusters(
    clusters: Dict[int, List[str]],
    documents: Dict[str, str],
    top_k: int = 10,
    min_docs: int = 5
) -> Dict[int, Dict]:
    """
    Returns:
        cluster_id -> {
            "keywords": [...],
            "num_docs": int
        }
    """
    cluster_labels = {}

    for cluster_id, doc_ids in tqdm(clusters.items(), desc="Labeling clusters"):
        texts = [
            documents[doc_id]
            for doc_id in doc_ids
            if doc_id in documents
        ]

        if len(texts) < min_docs:
            continue

        vectorizer = TfidfVectorizer(
            stop_words=STOPWORDS,
            max_df=0.7,
            min_df=2,                 # IMPORTANT: cluster-local
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b[a-z][a-z]+\b"
        )

        X = vectorizer.fit_transform(texts)

        if X.shape[1] == 0:
            continue

        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        top_idx = scores.argsort()[::-1][:top_k]
        keywords = [terms[i] for i in top_idx]

        cluster_labels[cluster_id] = {
            "num_docs": len(texts),
            "keywords": keywords
        }

    return cluster_labels


# --------------------------------------------------
# Save labels
# --------------------------------------------------

def save_cluster_labels(labels: Dict[int, Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for cluster_id, info in sorted(labels.items()):
            f.write(
                json.dumps(
                    {
                        "cluster": cluster_id,
                        **info
                    },
                    ensure_ascii=False
                ) + "\n"
            )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    CLUSTERS_PATH = "clusters_faiss_gpu_12_clusters.jsonl"
    DOCUMENTS_DIR = "../../data/longeval_sci_testing_2025_abstract/documents"
    OUTPUT_PATH = "cluster_labels_cleaned_12_clusters.jsonl"

    TOP_K_KEYWORDS = 10
    MIN_DOCS_PER_CLUSTER = 5

    print("🚀 Loading clusters")
    clusters = load_clusters(CLUSTERS_PATH)

    print("🚀 Loading documents")
    documents = load_documents(DOCUMENTS_DIR)

    print("🚀 Labeling clusters")
    labels = label_clusters(
        clusters,
        documents,
        top_k=TOP_K_KEYWORDS,
        min_docs=MIN_DOCS_PER_CLUSTER
    )

    save_cluster_labels(labels, OUTPUT_PATH)

    print(f"✅ Saved cleaned cluster labels → {OUTPUT_PATH}")
