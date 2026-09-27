"""ContractIQ - FAISS Vector Store & Semantic Search Engine.

Builds and queries high-dimensional dense vector representations of legal contracts
using FAISS (IndexFlatIP with normalized vectors for exact cosine similarity)
and SentenceTransformer (all-MiniLM-L6-v2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class VectorStore:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.dimension = EMBEDDING_DIM
        # IndexFlatIP with normalized vectors computes cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def count(self) -> int:
        return self.index.ntotal

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def add_documents(
        self,
        docs: List[Dict[str, Any]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> int:
        """Embed and add documents to the FAISS index."""
        if not docs:
            return 0

        texts = [d["text"] for d in docs]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        norm_embeddings = self._normalize(embeddings)
        self.index.add(norm_embeddings)
        self.documents.extend(docs)
        return len(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search the FAISS vector database for contracts matching a semantic query."""
        if self.count() == 0:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_norm = self._normalize(query_vec)

        k = min(top_k, self.count())
        scores, indices = self.index.search(query_norm, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            similarity = float(score)
            if similarity >= min_score:
                doc = dict(self.documents[idx])
                doc["similarity_score"] = round(similarity, 4)
                results.append(doc)

        return results

    def save(self, storage_dir: Path) -> None:
        """Persist FAISS index and document metadata to disk."""
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        index_file = storage_dir / "contracts.index"
        meta_file = storage_dir / "metadata.json"

        faiss.write_index(self.index, str(index_file))

        metadata_payload = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "documents": self.documents,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    def load(self, storage_dir: Path) -> bool:
        """Load FAISS index and document metadata from disk."""
        storage_dir = Path(storage_dir)
        index_file = storage_dir / "contracts.index"
        meta_file = storage_dir / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            return False

        self.index = faiss.read_index(str(index_file))

        with open(meta_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.model_name = payload.get("model_name", DEFAULT_MODEL_NAME)
        self.dimension = payload.get("dimension", EMBEDDING_DIM)
        self.documents = payload.get("documents", [])
        return True
