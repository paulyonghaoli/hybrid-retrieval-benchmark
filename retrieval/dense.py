from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Iterable, List, Tuple


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.doc_ids: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, documents: Iterable[dict]) -> None:
        self.doc_ids = [doc["doc_id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        self.embeddings = np.asarray(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False))

    def _score(self, query_text: str) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("Index has not been built")
        query_embedding = self.model.encode(query_text, convert_to_numpy=True, show_progress_bar=False)
        similarity = np.dot(self.embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        doc_norms = np.where(doc_norms == 0, 1e-9, doc_norms)
        if query_norm == 0:
            return np.zeros(len(self.doc_ids), dtype=float)
        return similarity / (doc_norms * query_norm)

    def retrieve(self, query_text: str, k: int) -> List[str]:
        scores = self._score(query_text)
        indices = np.argsort(-scores)
        return [self.doc_ids[idx] for idx in indices[:k]]

    def retrieve_all(self, query_text: str) -> List[str]:
        scores = self._score(query_text)
        indices = np.argsort(-scores)
        return [self.doc_ids[idx] for idx in indices]
