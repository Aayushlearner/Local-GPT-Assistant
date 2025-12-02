# rag_engine/vector_store.py

from typing import List, Dict, Any, Tuple

import numpy as np
import faiss


class VectorStore:
    """
    Simple in-memory FAISS-based vector store.

    Stores:
    - embeddings in a FAISS index
    - texts and metadata in Python lists
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # L2 distance index
        self.index = faiss.IndexFlatL2(embedding_dim)

        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    @property
    def is_empty(self) -> bool:
        return len(self.texts) == 0

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add embeddings + corresponding texts and metadatas to the store.
        """
        if embeddings.shape[0] != len(texts) or len(texts) != len(metadatas):
            raise ValueError("Embeddings, texts, and metadatas must have same length.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for the nearest neighbors of the query embedding.

        Returns:
            List of dicts: {"text": str, "metadata": dict, "distance": float}
        """
        if self.is_empty:
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(dist),
                }
            )
        return results
