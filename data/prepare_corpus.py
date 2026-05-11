from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

random.seed(42)

TOPICS = [
    "supervised learning",
    "unsupervised learning",
    "reinforcement learning",
    "neural networks",
    "decision trees",
    "support vector machines",
    "clustering",
    "feature engineering",
    "model evaluation",
    "natural language processing",
]

DOCUMENT_TEMPLATES = [
    "{topic} is a foundational machine learning approach that {detail}. It is often used with datasets that {example}.",
    "A key aspect of {topic} is the way it {detail}. Practitioners often combine it with {example}.",
    "The study of {topic} includes methods like {example} and {detail}. This helps models generalize when {scenario}.",
]

QUERY_TEMPLATES = [
    "Which algorithm is best for {topic}?",
    "How does {topic} differ from {related}?",
    "Explain {topic} in simple terms.",
    "What are practical examples of {topic}?",
    "When should you choose {topic} over {related}?",
]

QUERY_TYPES = ["term_match", "semantic", "balanced"]

CORPUS_SIZE = 200
QUERY_COUNT = 30


def make_document(doc_id: str, topic: str) -> dict[str, str]:
    detail = random.choice([
        "learns from labelled examples",
        "discovers clusters in data",
        "optimizes a reward objective",
        "uses backpropagation to adjust weights",
        "reduces overfitting with regularization",
        "measures distance between points",
        "identifies high-variance features",
        "estimates probability distributions",
        "builds hierarchical decision rules",
    ])
    example = random.choice([
        "large-scale training sets",
        "complex feature representations",
        "text embeddings for semantic search",
        "cross-validation for robust evaluation",
        "graph data or sequential signals",
        "dimensionality reduction first",
        "ensemble methods for prediction",
        "online learning from streaming inputs",
    ])
    scenario = random.choice([
        "labels are sparse",
        "the dataset contains noise",
        "the model must generalize to new data",
        "the feature space is high dimensional",
        "the task requires ranking documents",
    ])
    template = random.choice(DOCUMENT_TEMPLATES)
    text = template.format(topic=topic, detail=detail, example=example, scenario=scenario)
    text += " The document also discusses " + ", ".join(random.sample(TOPICS, 2)) + "."
    return {
        "doc_id": doc_id,
        "title": f"Document on {topic}",
        "text": text,
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"

    docs: list[dict[str, str]] = []
    for index in range(1, CORPUS_SIZE + 1):
        topic = TOPICS[(index - 1) % len(TOPICS)]
        docs.append(make_document(f"doc_{index:03d}", topic))

    with corpus_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    queries: list[dict[str, Any]] = []
    for index in range(QUERY_COUNT):
        query_type = QUERY_TYPES[index % len(QUERY_TYPES)]
        topic = random.choice(TOPICS)
        related = random.choice([t for t in TOPICS if t != topic])
        template = random.choice(QUERY_TEMPLATES)
        query_text = template.format(topic=topic, related=related)
        relevant_ids = [doc["doc_id"] for doc in docs if topic in doc["text"] or topic in doc["title"]]

        if query_type == "term_match":
            relevant_ids = [doc["doc_id"] for doc in docs if topic in doc["title"]]
        elif query_type == "semantic":
            relevant_ids = [doc["doc_id"] for doc in docs if any(term in doc["text"] for term in ["similarity", "concept", "semantic", topic.split()[0]])][:8]
        elif query_type == "balanced":
            relevant_ids = [doc["doc_id"] for doc in docs if topic in doc["text"] or related in doc["text"]][:8]

        if not relevant_ids:
            relevant_ids = [docs[0]["doc_id"]]

        queries.append({
            "query_id": f"q_{index+1:02d}",
            "query": query_text,
            "relevant_doc_ids": relevant_ids[:6],
            "query_type": query_type,
        })

    with queries_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} documents to {corpus_path}")
    print(f"Wrote {len(queries)} queries to {queries_path}")


if __name__ == "__main__":
    main()
