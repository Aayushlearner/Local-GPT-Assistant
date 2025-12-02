# rag_engine/embedder.py

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer to embed texts and queries.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts into a NumPy array of shape (n_texts, dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype="float32")
        embeddings = self.model.encode(texts, batch_size=16, show_progress_bar=False)
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string, returning shape (1, dim).
        """
        embedding = self.model.encode([query], show_progress_bar=False)
        return np.asarray(embedding, dtype="float32")
