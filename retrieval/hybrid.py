from __future__ import annotations

from typing import List
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def retrieve(self, query_text: str, k: int) -> List[str]:
        dense_rank = self.dense.retrieve_all(query_text)
        sparse_rank = self.sparse.retrieve_all(query_text)
        scores: dict[str, float] = {}

        for rank_list in (dense_rank, sparse_rank):
            for rank, doc_id in enumerate(rank_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.keys(), key=lambda doc_id: -scores[doc_id])
        return ranked[:k]
