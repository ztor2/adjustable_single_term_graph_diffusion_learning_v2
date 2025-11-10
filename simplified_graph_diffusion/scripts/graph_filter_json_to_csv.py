"""Convert graph_filter_comparison JSON results into CSV tables."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "graph_filter_comparison"
RESULTS_JSON_NAME: str | None = None  # set to filename to override


def main() -> None:
    os.chdir(PROJECT_ROOT)
    results_path = resolve_results_path()
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    hops = payload["metadata"]["hops"]
    diff_alphas = payload["metadata"]["diff_alphas"]
    datasets = payload["metadata"]["datasets"]
    results = payload["datasets"]

    for dataset in datasets:
        dataset_results = results[dataset]
        for alpha in diff_alphas:
            alpha_key = f"{alpha:.4f}"
            rows = []
            for filter_name, alpha_results in dataset_results.items():
                hop_times = alpha_results.get(alpha_key)
                row = {"filter": filter_name}
                for hop in hops:
                    value = None
                    if hop_times is not None:
                        value = hop_times.get(str(hop))
                    row[str(hop)] = format_cell(value)
                rows.append(row)

            fieldnames = ["filter"] + [str(h) for h in hops]
            filename = OUTPUT_DIR / f"{dataset}_alpha{alpha:.2f}_summary_runs.csv"
            with filename.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"Saved {filename}")


def resolve_results_path() -> Path:
    if RESULTS_JSON_NAME:
        path = OUTPUT_DIR / RESULTS_JSON_NAME
        if not path.exists():
            raise FileNotFoundError(f"Specified JSON not found: {path}")
        return path

    candidates = sorted(OUTPUT_DIR.glob("graph_filter_comparison_runs_*.json"))
    if not candidates:
        candidates = sorted(OUTPUT_DIR.glob("graph_filter_comparison_*.json"))
    if not candidates:
        raise FileNotFoundError("No graph_filter_comparison JSON files found.")
    return candidates[-1]


def format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        arr = np.array(value, dtype=float)
        mean = float(arr.mean())
        var = float(arr.var())
        return f"{mean:.2f}±{var:.2f}"
    try:
        num = float(value)
        return f"{num:.2f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
