# Hybrid Retrieval Benchmark

`hybrid-retrieval-benchmark` is a reproducible Python benchmark comparing three retrieval strategies on a shared corpus and query set.

## What it does
- Builds a synthetic corpus of ~200 machine learning documents
- Evaluates three retrieval strategies:
  - dense retrieval using sentence-transformers
  - sparse BM25 retrieval with rank_bm25
  - hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Computes Precision@k, Recall@k, MRR, and NDCG@k from scratch
- Measures query latency and produces a tradeoff analysis
- Generates results in `results/metrics.json`, `results/report.md`, and plots in `results/plots/`

## Why it matters
This benchmark isolates retrieval strategy behavior on the same corpus and queries. It helps compare exact-match vs semantic retrieval and identify the best strategy for different query types.

## Quick start
1. Create a Python environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Generate the corpus and queries:
   ```bash
   python data/prepare_corpus.py
   ```
3. Run the benchmark:
   ```bash
   python main.py
   ```

## RRF formula
The hybrid strategy uses Reciprocal Rank Fusion (RRF):

`score(d) = \sum_i 1 / (k + rank_i(d))`

With `k=60`, this formula rewards documents that are ranked highly by either dense or sparse retrieval.

## Swap in your own corpus and queries
- Place a JSONL corpus at `data/corpus.jsonl` with `doc_id`, `title`, and `text`
- Place queries at `data/queries.jsonl` with `query_id`, `query`, `relevant_doc_ids`, and `query_type`
- Update `config.yaml` if file paths or parameters change

## Tradeoff summary
- Dense retrieval is strongest on semantic queries that require conceptual similarity
- Sparse BM25 is strongest on exact term-match queries and short factual search
- Hybrid retrieval balances both, often improving NDCG and MRR across mixed query workloads
