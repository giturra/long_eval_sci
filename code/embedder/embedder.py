import os
import glob
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np

from tqdm import tqdm

class SciBERTEmbedder:
    """
    SciBERT embedder for paper title + abstract.
    Uses mean pooling over token embeddings.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: str = "cuda:0",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_text(self, title: str, abstract: str) -> str:
        title = title.strip() if title else ""
        abstract = abstract.strip() if abstract else ""

        if abstract:
            return f"{title}. {abstract}"
        return title

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode(
        self,
        documents: List[Dict],
        title_key: str = "title",
        abstract_key: str = "abstract",
    ) -> np.ndarray:
        texts = [
            self._prepare_text(doc.get(title_key, ""), doc.get(abstract_key, ""))
            for doc in documents
        ]

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                model_output = self.model(**encoded)
                batch_embeddings = self._mean_pooling(
                    model_output, encoded["attention_mask"]
                )

                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, p=2, dim=1
                    )

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
        ids: List[str] = None,
    ):
        """
        Save embeddings to disk.

        Args:
            embeddings (np.ndarray): Embedding matrix (N, D)
            output_path (str): Path to save (.npy or .npz)
            ids (List[str], optional): Document IDs aligned with embeddings
        """

        if output_path.endswith(".npy"):
            np.save(output_path, embeddings)

        elif output_path.endswith(".npz"):
            if ids is None:
                np.savez_compressed(output_path, embeddings=embeddings)
            else:
                np.savez_compressed(
                    output_path,
                    embeddings=embeddings,
                    ids=np.array(ids)
                )

        else:
            raise ValueError(
                "Unsupported format. Use .npy or .npz"
            )


def load_jsonl(path):
    """
    Generator that yields one JSON object per line from a .jsonl file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                yield json.loads(line)


if __name__ == "__main__":
    embedder = SciBERTEmbedder(batch_size=32)

    input_dir = "../../data/longeval_sci_testing_2025_abstract/documents"
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)

    jsonl_files = sorted(
        glob.glob(os.path.join(input_dir, "documents_*.jsonl"))
    )

    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {input_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")

    total_docs = 0

    for jsonl_path in tqdm(jsonl_files, desc="Embedding shards"):
        shard_name = os.path.basename(jsonl_path).replace(".jsonl", ".npz")
        output_path = os.path.join(output_dir, shard_name)

        print(f"\nProcessing {jsonl_path}")

        documents = []
        ids = []

        for doc in load_jsonl(jsonl_path):
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")

            # Skip truly empty docs
            if not title and not abstract:
                continue

            documents.append({
                "title": title,
                "abstract": abstract
            })
            ids.append(doc["id"])

        if not documents:
            print("  ⚠️ No valid documents found, skipping shard")
            continue

        embeddings = embedder.encode(documents)

        embedder.save_embeddings(
            embeddings=embeddings,
            output_path=output_path,
            ids=ids
        )

        total_docs += embeddings.shape[0]

        print(f"  ✅ Saved {embeddings.shape[0]} embeddings → {output_path}")

    print("\nDONE")
    print("Total embedded documents:", total_docs)
