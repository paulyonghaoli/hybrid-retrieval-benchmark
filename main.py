from pathlib import Path
import yaml
from evaluation.benchmark import Benchmark


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    config = load_config("config.yaml")
    benchmark = Benchmark(config)
    results = benchmark.run_all()
    benchmark.save_metrics(results)
    benchmark.write_report(results)
    benchmark.plot_results(results)
    target = Path(config["results_folder"]).resolve()
    print(f"Benchmark complete. Results written to {target}")


if __name__ == "__main__":
    main()
