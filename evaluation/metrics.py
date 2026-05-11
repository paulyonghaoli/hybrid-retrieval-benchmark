from __future__ import annotations

import math
from typing import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for doc_id in retrieved_k if doc_id in relevant_set) / len(retrieved_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return sum(1 for doc_id in relevant_set if doc_id in retrieved_k) / len(relevant_set)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    score = 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            score += 1.0 / math.log2(index + 1)
    return score


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_list = list(relevant)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_list), k) + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg
