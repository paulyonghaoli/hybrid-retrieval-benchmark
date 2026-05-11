from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseRetriever
from evaluation.metrics import precision_at_k, recall_at_k, mrr_at_k, ndcg_at_k


class Benchmark:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.corpus = self._load_jsonl(config["corpus_file"])
        self.queries = self._load_jsonl(config["queries_file"])
        self.k_values = config["k_values"]
        self.warmup_runs = config.get("warmup_runs", 3)
        self.rrf_k = config.get("rrf_k", 60)
        self.results_folder = Path(config["results_folder"])
        self.plots_folder = Path(config["plots_folder"])
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        self.dense = DenseRetriever(config["model_name"])
        self.sparse = SparseRetriever()
        self.dense.build_index(self.corpus)
        self.sparse.build_index(self.corpus)
        self.hybrid = HybridRetriever(self.dense, self.sparse, self.rrf_k)

    def _load_jsonl(self, path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _evaluate(self, retrieved: list[str], relevant: list[str]) -> dict[str, float]:
        return {
            "precision": {k: precision_at_k(retrieved, relevant, k) for k in self.k_values},
            "recall": {k: recall_at_k(retrieved, relevant, k) for k in self.k_values},
            "mrr": {k: mrr_at_k(retrieved, relevant, k) for k in self.k_values},
            "ndcg": {k: ndcg_at_k(retrieved, relevant, k) for k in self.k_values},
        }

    def _run_strategy(self, name: str, retriever: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {"queries": []}
        latencies: list[float] = []

        # Warmup phase
        for _ in range(self.warmup_runs):
            for query in self.queries:
                retriever.retrieve(query["query"], max(self.k_values))

        for query in self.queries:
            start = time.perf_counter()
            retrieved = retriever.retrieve(query["query"], max(self.k_values))
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed)
            query_metrics = self._evaluate(retrieved, query["relevant_doc_ids"])
            metrics["queries"].append({
                "query_id": query["query_id"],
                "query_type": query["query_type"],
                "metrics": query_metrics,
            })

        aggregated = {
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "metrics": self._aggregate_strategy(metrics["queries"]),
            "breakdown": self._aggregate_by_type(metrics["queries"]),
        }
        return aggregated

    def _aggregate_strategy(self, query_entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        count = len(query_entries)
        for entry in query_entries:
            for metric_name, metric_values in entry["metrics"].items():
                for k, value in metric_values.items():
                    summary[metric_name][str(k)] += value
        return {metric: {k: value / count for k, value in values.items()} for metric, values in summary.items()}

    def _aggregate_by_type(self, query_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in query_entries:
            grouped[entry["query_type"]].append(entry["metrics"])

        breakdown: dict[str, dict[str, dict[str, float]]] = {}
        for query_type, entries in grouped.items():
            summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for metrics in entries:
                for metric_name, metric_values in metrics.items():
                    for k, value in metric_values.items():
                        summary[metric_name][str(k)] += value
            breakdown[query_type] = {
                metric_name: {k: value / len(entries) for k, value in values.items()}
                for metric_name, values in summary.items()
            }
        return breakdown

    def _best_strategy_by_type(self, results: dict[str, Any], k: int) -> dict[str, str]:
        best: dict[str, str] = {}
        strategy_names = list(results.keys())
        query_types = set()
        for strategy in strategy_names:
            for query_type in results[strategy]["breakdown"]:
                query_types.add(query_type)

        for query_type in sorted(query_types):
            best_strategy = None
            best_score = -1.0
            for strategy in strategy_names:
                breakdown = results[strategy]["breakdown"].get(query_type, {})
                score = breakdown.get("ndcg", {}).get(str(k), 0.0)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            best[query_type] = best_strategy or "unknown"
        return best

    def run_all(self) -> dict[str, Any]:
        results = {
            "dense": self._run_strategy("dense", self.dense),
            "sparse": self._run_strategy("sparse", self.sparse),
            "hybrid": self._run_strategy("hybrid", self.hybrid),
        }
        results["best_by_type_at_10"] = self._best_strategy_by_type(results, 10)
        return results

    def save_metrics(self, results: dict[str, Any]) -> None:
        target = self.results_folder / "metrics.json"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def write_report(self, results: dict[str, Any]) -> None:
        lines: list[str] = ["# Retrieval Benchmark Report", ""]
        lines.append("## Average metrics")
        lines.append("")
        lines.append("Strategy | Precision@1 | Precision@3 | Precision@5 | Precision@10 | NDCG@10 | Latency (ms)")
        lines.append("--- | --- | --- | --- | --- | --- | ---")
        for name, summary in results.items():
            if name.startswith("best_by_type"):
                continue
            avg = summary["metrics"]
            lines.append(
                f"{name} | {avg['precision']['1']:.3f} | {avg['precision']['3']:.3f} | {avg['precision']['5']:.3f} | {avg['precision']['10']:.3f} | {avg['ndcg']['10']:.3f} | {summary['average_latency_ms']:.2f}"
            )
        lines.append("")
        lines.append("## Best strategy by query type (NDCG@10)")
        lines.append("")
        for query_type, winner in results["best_by_type_at_10"].items():
            lines.append(f"- **{query_type}**: {winner}")
        lines.append("")
        lines.append("## Tradeoff recommendations")
        lines.append("")
        lines.append("- Use **sparse retrieval** when the query contains exact terminology or named algorithm names.")
        lines.append("- Use **dense retrieval** when the query requires conceptual or semantic reasoning beyond exact tokens.")
        lines.append("- Use **hybrid retrieval** when the workload mixes both query types and you need robust NDCG/MRR.")

        target = self.results_folder / "report.md"
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def plot_results(self, results: dict[str, Any]) -> None:
        import matplotlib.pyplot as plt

        k_values = self.k_values
        plt.figure(figsize=(8, 5))
        for name in ("dense", "sparse", "hybrid"):
            values = [results[name]["metrics"]["precision"][str(k)] for k in k_values]
            plt.plot(k_values, values, marker="o", label=name)
        plt.title("Average Precision@k")
        plt.xlabel("k")
        plt.ylabel("Precision")
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plots_folder / "precision_at_k.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        for name in ("dense", "sparse", "hybrid"):
            values = [results[name]["metrics"]["ndcg"][str(k)] for k in k_values]
            plt.plot(k_values, values, marker="o", label=name)
        plt.title("Average NDCG@k")
        plt.xlabel("k")
        plt.ylabel("NDCG")
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plots_folder / "ndcg_at_k.png")
        plt.close()
