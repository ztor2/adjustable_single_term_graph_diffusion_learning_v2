"""Regenerate graph filter comparison plots from an existing JSON result."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_JSON = PROJECT_ROOT / "results" / "graph_filter_comparison" / "graph_filter_comparison_20251110_025140.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "graph_filter_comparison"
Y_AXIS_TICKS = {
    "pubmed": 50.0,
    "arx": 1.0,
}


def plot_from_payload(payload: dict) -> None:
    metadata = payload["metadata"]
    hops = metadata["hops"]
    diff_alphas = metadata["diff_alphas"]
    datasets = metadata["datasets"]
    results = payload["datasets"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dataset_results = results[dataset]
        for alpha in diff_alphas:
            alpha_key = f"{alpha:.4f}"
            fig, ax = plt.subplots(figsize=(8, 4))
            for filter_name, alpha_results in dataset_results.items():
                hop_times = alpha_results.get(alpha_key)
                if hop_times is None:
                    continue
                hop_vals = [
                    hop_times.get(str(hop)) if hop_times.get(str(hop)) is not None else float("nan")
                    for hop in hops
                ]
                style_kwargs = {"marker": "o"}
                label = filter_name
                if filter_name == "process_closed_form_graph_diff":
                    label = "Closed-form Polynomial"
                    style_kwargs["linestyle"] = (0, (6, 3))
                elif filter_name == "preprocess_graph_diff":
                    label = "Single-term Diffusion"
                ax.plot(hops, hop_vals, label=label, **style_kwargs)

            ax.set_title(f"Graph Filter Construction Time ({dataset}, alpha={alpha:.2f})")
            ax.set_xlabel("n-hop")
            ax.set_ylabel("Elapsed time (sec)")
            ax.set_xticks(hops)
            tick_interval = Y_AXIS_TICKS.get(dataset, 0.1)
            ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig_path = OUTPUT_DIR / f"{dataset}_alpha{alpha:.2f}_graph_filter_comparison.png"
            fig.savefig(fig_path)
            plt.close(fig)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"JSON file not found: {RESULTS_JSON}")
    with RESULTS_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    plot_from_payload(payload)
    print(f"Plots regenerated from {RESULTS_JSON}")


if __name__ == "__main__":
    main()
