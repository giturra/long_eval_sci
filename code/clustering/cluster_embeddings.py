import os
import glob
import json
import numpy as np
from typing import List, Tuple

from tqdm import tqdm
from sklearn.preprocessing import normalize
import faiss


# --------------------------------------------------
# Load embeddings from multiple .npz shards
# --------------------------------------------------

def load_embedding_shards(
    embeddings_dir: str,
    max_docs: int = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings + ids from .npz files.

    Returns:
        embeddings: (N, D)
        ids: list of document ids
    """
    embeddings_list = []
    ids_list = []

    shard_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npz")))
    if not shard_files:
        raise RuntimeError(f"No .npz files found in {embeddings_dir}")

    for shard in tqdm(shard_files, desc="Loading shards"):
        data = np.load(shard)
        embeddings = data["embeddings"]
        ids = data["ids"]

        embeddings_list.append(embeddings)
        ids_list.extend(ids.tolist())

        if max_docs and len(ids_list) >= max_docs:
            break

    embeddings = np.vstack(embeddings_list)
    ids_list = ids_list[: embeddings.shape[0]]

    if max_docs:
        embeddings = embeddings[:max_docs]
        ids_list = ids_list[:max_docs]

    return embeddings, ids_list


# --------------------------------------------------
# FAISS GPU clustering
# --------------------------------------------------

def cluster_embeddings_faiss_gpu(
    embeddings: np.ndarray,
    n_clusters: int = 50,
    n_iter: int = 30,
    pca_dim: int = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Ultra-fast KMeans clustering using FAISS on GPU.
    """

    print("🔹 Normalizing embeddings")
    embeddings = normalize(embeddings).astype("float32")
    n, d = embeddings.shape

    # Optional PCA (usually not needed)
    if pca_dim and pca_dim < d:
        print(f"🔹 Applying FAISS PCA → {pca_dim} dims")
        pca = faiss.PCAMatrix(d, pca_dim)
        pca.train(embeddings)
        embeddings = pca.apply_py(embeddings)
        d = pca_dim

    print(f"🚀 FAISS KMeans on GPU (k={n_clusters}, d={d})")

    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        seed=seed,
        verbose=True,
        gpu=True
    )

    kmeans.train(embeddings)

    print("🔹 Assigning clusters")
    _, labels = kmeans.index.search(embeddings, 1)

    return labels.squeeze()


# --------------------------------------------------
# Save cluster assignments
# --------------------------------------------------

def save_clusters(
    ids: List[str],
    labels: np.ndarray,
    output_path: str
):
    """
    Save (doc_id, cluster_id) pairs as JSONL.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, cluster_id in zip(ids, labels):
            f.write(
                json.dumps(
                    {"id": doc_id, "cluster": int(cluster_id)},
                    ensure_ascii=False
                ) + "\n"
            )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    EMBEDDINGS_DIR = "embeddings"
    OUTPUT_PATH = "clusters_faiss_gpu_12_clusters.jsonl"

    N_CLUSTERS = 12        # 🔧 tune this (50–200 typical)
    PCA_DIM = 128         # 🔧 usually None; try 128 if memory is tight
    MAX_DOCS = 1_524_045        # e.g. 100_000 for quick tests

    print("🚀 Loading embeddings")
    embeddings, ids = load_embedding_shards(
        EMBEDDINGS_DIR,
        max_docs=MAX_DOCS
    )

    print(f"Loaded {len(ids)} documents")

    labels = cluster_embeddings_faiss_gpu(
        embeddings,
        n_clusters=N_CLUSTERS,
        n_iter=30,
        pca_dim=PCA_DIM
    )

    save_clusters(ids, labels, OUTPUT_PATH)

    print(f"✅ Saved clusters → {OUTPUT_PATH}")
