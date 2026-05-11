from __future__ import annotations

import re
from rank_bm25 import BM25Okapi
from typing import Iterable, List


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class SparseRetriever:
    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.bm25: BM25Okapi | None = None
        self.corpus_tokens: list[list[str]] = []

    def build_index(self, documents: Iterable[dict]) -> None:
        docs = list(documents)
        self.doc_ids = [doc["doc_id"] for doc in docs]
        self.corpus_tokens = [simple_tokenize(doc["text"]) for doc in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query_text: str, k: int) -> List[str]:
        if self.bm25 is None:
            raise ValueError("Index has not been built")
        query_tokens = simple_tokenize(query_text)
        if not query_tokens:
            return self.doc_ids[:k]
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda idx: -scores[idx])
        return [self.doc_ids[idx] for idx in ranked[:k]]

    def retrieve_all(self, query_text: str) -> List[str]:
        if self.bm25 is None:
            raise ValueError("Index has not been built")
        query_tokens = simple_tokenize(query_text)
        if not query_tokens:
            return self.doc_ids.copy()
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda idx: -scores[idx])
        return [self.doc_ids[idx] for idx in ranked]
