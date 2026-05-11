from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever


def test_retrievers_return_exactly_k_results():
    documents = [
        {"doc_id": "doc_001", "text": "machine learning supervised algorithms"},
        {"doc_id": "doc_002", "text": "best practices for neural networks"},
        {"doc_id": "doc_003", "text": "bm25 and sparse retrieval"},
    ]
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense.build_index(documents)
    sparse.build_index(documents)

    assert len(dense.retrieve("machine learning", 2)) == 2
    assert len(sparse.retrieve("machine learning", 2)) == 2


def test_retrievers_handle_empty_query():
    documents = [{"doc_id": "doc_001", "text": "cluster analysis"}]
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense.build_index(documents)
    sparse.build_index(documents)

    empty_dense = dense.retrieve("", 1)
    empty_sparse = sparse.retrieve("", 1)

    assert len(empty_dense) == 1
    assert isinstance(empty_dense[0], str)
    assert empty_sparse == ["doc_001"]
