"""Compare graph filter construction runtimes across implementations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gcn.utils import (  # type: ignore
    load_data,
    preprocess_graph_diff,
    process_closed_form_graph_diff,
    process_sumpow_graph_diff,
)


CLOSED_FORM_NODE_LIMIT = 100000

FILTERS = {
    "preprocess_graph_diff": preprocess_graph_diff,
    "process_closed_form_graph_diff": process_closed_form_graph_diff,
    "process_sumpow_graph_diff": process_sumpow_graph_diff,
}


def measure_runtime(filter_fn, adj, diff_n: int, diff_alpha: float) -> float:
    start = time.perf_counter()
    _ = filter_fn(adj, diff_n, diff_alpha)
    end = time.perf_counter()
    return end - start


def run_experiment(
    datasets: List[str],
    hops: List[int],
    diff_alphas: List[float],
    data_dir: Path,
    output_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    for dataset in datasets:
        adj, *_ = load_data(dataset, task="classification", feat_norm=False)
        num_nodes = adj.shape[0]
        dataset_results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

        for filter_name, filter_fn in FILTERS.items():
            filter_results: Dict[str, Dict[str, Optional[float]]] = {}
            skip_filter = (
                filter_name == "process_closed_form_graph_diff" and num_nodes > CLOSED_FORM_NODE_LIMIT
            )
            for alpha in diff_alphas:
                hop_times: Dict[str, Optional[float]] = {}
                if skip_filter:
                    print(
                        f"Dataset={dataset:10s} Filter={filter_name:30s} alpha={alpha:.2f} skipped (>{CLOSED_FORM_NODE_LIMIT} nodes)"
                    )
                    hop_times = {str(hop): None for hop in hops}
                    filter_results[f"{alpha:.4f}"] = hop_times
                    continue
                for hop in hops:
                    runtime = measure_runtime(filter_fn, adj, hop, alpha)
                    hop_times[str(hop)] = runtime
                    print(
                        f"Dataset={dataset:10s} Filter={filter_name:30s} alpha={alpha:.2f} hop={hop} time={runtime:.4f}s"
                    )
                filter_results[f"{alpha:.4f}"] = hop_times
            dataset_results[filter_name] = filter_results

        results[dataset] = dataset_results

        # Visualization per dataset & alpha for clarity
        for alpha in diff_alphas:
            fig, ax = plt.subplots(figsize=(8, 4))
            alpha_key = f"{alpha:.4f}"
            for filter_name, alpha_results in dataset_results.items():
                hop_times = alpha_results[alpha_key]
                hop_vals = [
                    hop_times[str(hop)] if hop_times[str(hop)] is not None else float("nan")
                    for hop in hops
                ]
                ax.plot(hops, hop_vals, marker="o", label=filter_name)
            ax.set_title(f"Filter Construction Time ({dataset}, alpha={alpha:.2f})")
            ax.set_xlabel("n-hop")
            ax.set_ylabel("Seconds")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig_path = output_dir / f"{dataset}_alpha{alpha:.2f}_graph_filter_timing.png"
            fig.savefig(fig_path)
            plt.close(fig)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"graph_filter_comparison_{timestamp}.json"
    payload = {
        "datasets": results,
        "metadata": {
            "hops": hops,
            "diff_alphas": diff_alphas,
            "created_utc": timestamp,
            "data_dir": str(data_dir),
            "datasets": datasets,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON results to {json_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare runtimes of different graph diffusion filters"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names (space or comma separated). Use 'all' or omit to run every *.graph file.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of diffusion alpha values to evaluate",
    )
    parser.add_argument(
        "--hops",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of hop counts (n) to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to directory containing *.graph files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "graph_filter_comparison",
        help="Directory to save JSON and figures",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args: Optional[List[str]], data_dir: Path) -> List[str]:
    available = sorted({p.stem for p in Path(data_dir).glob("*.graph")})
    if not available:
        raise ValueError(f"No *.graph files found under {data_dir}")

    if not dataset_args:
        return available

    requested: List[str] = []
    for entry in dataset_args:
        parts = [token.strip() for token in entry.split(",") if token.strip()]
        requested.extend(parts)

    if any(name.lower() == "all" for name in requested):
        return available

    unique: List[str] = []
    for name in requested:
        if name not in available:
            raise ValueError(
                f"Dataset '{name}' not found in {data_dir}. Available: {', '.join(available)}"
            )
        if name not in unique:
            unique.append(name)

    if not unique:
        raise ValueError("No valid dataset names provided.")
    return unique


def main() -> None:
    args = parse_args()
    hops = sorted(set(args.hops))
    diff_alphas = args.alphas

    # Ensure relative paths inside gcn.utils (e.g., data/...) resolve correctly
    os.chdir(PROJECT_ROOT)

    datasets = resolve_datasets(args.datasets, args.data_dir)

    run_experiment(
        datasets=datasets,
        hops=hops,
        diff_alphas=diff_alphas,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
