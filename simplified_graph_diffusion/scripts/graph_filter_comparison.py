"""Compare graph filter construction runtimes across implementations."""

from __future__ import annotations

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
from matplotlib.ticker import MultipleLocator
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gcn.utils import (  # type: ignore
    load_data,
    preprocess_graph_diff,
    process_closed_form_graph_diff,
    # process_sumpow_graph_diff,
)


CLOSED_FORM_NODE_LIMIT = 100000

FILTERS = {
    "preprocess_graph_diff": preprocess_graph_diff,
    "process_closed_form_graph_diff": process_closed_form_graph_diff,
    # "process_sumpow_graph_diff": process_sumpow_graph_diff,
}


# -------------------------------
# Experiment configuration
# -------------------------------

DATASETS: List[str] = [
    "cora",
    "citeseer",
    "pubmed",
    "arx"
]
HOPS: List[int] = [1, 2, 3, 4, 5]
DIFF_ALPHAS: List[float] = [0.5]
RUNS_PER_SETTING: int = 10
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "graph_filter_comparison"
Y_AXIS_TICKS: Dict[str, float] = {
    "pubmed": 50.0,
    "arx": 1.0,
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
            filter_results: Dict[str, Dict[str, Optional[List[float]]]] = {}
            skip_filter = (
                filter_name == "process_closed_form_graph_diff" and num_nodes > CLOSED_FORM_NODE_LIMIT
            )
            for alpha in diff_alphas:
                hop_times: Dict[str, Optional[List[float]]] = {}
                if skip_filter:
                    print(
                        f"Dataset={dataset:10s} Filter={filter_name:30s} alpha={alpha:.2f} skipped (>{CLOSED_FORM_NODE_LIMIT} nodes)"
                    )
                    hop_times = {str(hop): None for hop in hops}
                    filter_results[f"{alpha:.4f}"] = hop_times
                    continue
                for hop in hops:
                    runtimes: List[float] = []
                    for run_idx in range(RUNS_PER_SETTING):
                        runtime = measure_runtime(filter_fn, adj, hop, alpha)
                        runtimes.append(runtime)
                        print(
                            f"Dataset={dataset:10s} Filter={filter_name:30s} alpha={alpha:.2f} "
                            f"hop={hop} run={run_idx + 1}/{RUNS_PER_SETTING} time={runtime:.4f}s"
                        )
                    hop_times[str(hop)] = runtimes
                filter_results[f"{alpha:.4f}"] = hop_times
            dataset_results[filter_name] = filter_results

        results[dataset] = dataset_results

        # Visualization per dataset & alpha for clarity
        for alpha in diff_alphas:
            fig, ax = plt.subplots(figsize=(8, 4))
            alpha_key = f"{alpha:.4f}"
            for filter_name, alpha_results in dataset_results.items():
                hop_times = alpha_results[alpha_key]
                hop_vals = []
                for hop in hops:
                    samples = hop_times[str(hop)]
                    if not samples:
                        hop_vals.append(float("nan"))
                    else:
                        hop_vals.append(float(np.mean(samples)))
                style_kwargs = {"marker": "o"}
                axis_label = filter_name
                if filter_name == "process_closed_form_graph_diff":
                    axis_label = "Closed-form Polynomial"
                    style_kwargs["linestyle"] = (0, (6, 3))
                elif filter_name == "preprocess_graph_diff":
                    axis_label = "SimDiff graph filter"
                ax.plot(hops, hop_vals, label=axis_label, **style_kwargs)
            ax.set_title(f"Graph Filter Construction Time ({dataset}, alpha={alpha:.2f})")
            ax.set_xlabel("n-hop")
            ax.set_ylabel("Elapsed time (sec)")
            ax.set_xticks(hops)
            tick_interval = Y_AXIS_TICKS.get(dataset, 0.1)
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig_path = output_dir / f"{dataset}_alpha{alpha:.2f}_graph_filter_comparison_runs.png"
            fig.savefig(fig_path)
            plt.close(fig)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"graph_filter_comparison_runs_{timestamp}.json"
    payload = {
        "datasets": results,
        "metadata": {
            "hops": hops,
            "diff_alphas": diff_alphas,
            "runs_per_setting": RUNS_PER_SETTING,
            "created_utc": timestamp,
            "data_dir": str(data_dir),
            "datasets": datasets,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved JSON results to {json_path}")
    return payload


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
    os.chdir(PROJECT_ROOT)

    hops = sorted(set(HOPS))
    diff_alphas = DIFF_ALPHAS
    datasets = resolve_datasets(DATASETS, DATA_DIR)

    run_experiment(
        datasets=datasets,
        hops=hops,
        diff_alphas=diff_alphas,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
