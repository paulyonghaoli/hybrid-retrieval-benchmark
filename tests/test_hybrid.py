from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseRetriever


def test_hybrid_ranks_documents_from_both_strategies():
    documents = [
        {"doc_id": "doc_001", "text": "machine learning algorithms semantic search"},
        {"doc_id": "doc_002", "text": "bm25 sparse exact term retrieval"},
        {"doc_id": "doc_003", "text": "hybrid ranking fusion"},
    ]
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense.build_index(documents)
    sparse.build_index(documents)
    hybrid = HybridRetriever(dense, sparse, rrf_k=60)

    result = hybrid.retrieve("semantic search", 3)
    assert len(result) == 3
    assert result[0] in {"doc_001", "doc_002", "doc_003"}


def test_hybrid_prefers_documents_ranked_high_by_both():
    documents = [
        {"doc_id": "doc_001", "text": "semantic matching and dense retrieval"},
        {"doc_id": "doc_002", "text": "simple keyword search bm25"},
        {"doc_id": "doc_003", "text": "extra unrelated content"},
    ]
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense.build_index(documents)
    sparse.build_index(documents)
    hybrid = HybridRetriever(dense, sparse, rrf_k=1)

    merged = hybrid.retrieve("semantic search", 2)
    assert merged[0] == "doc_001"
