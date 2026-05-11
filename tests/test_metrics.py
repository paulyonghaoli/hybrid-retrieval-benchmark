from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, mrr_at_k


def test_precision_recall_mrr_ndcg():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert precision_at_k(retrieved, relevant, 1) == 0.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert mrr_at_k(retrieved, relevant, 4) == 1.0 / 2
    assert ndcg_at_k(retrieved, relevant, 4) > 0.0


def test_metrics_with_no_relevant_docs():
    retrieved = ["a", "b"]
    relevant: list[str] = []

    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert mrr_at_k(retrieved, relevant, 2) == 0.0
    assert ndcg_at_k(retrieved, relevant, 2) == 0.0
