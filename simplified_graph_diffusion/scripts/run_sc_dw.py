#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sc_dw.model import deepwalk_scores, spectral_clustering_scores
from sc_dw.utils import edge_split, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run spectral clustering or DeepWalk baselines for link prediction."
    )
    parser.add_argument("--dataset", default="cora", help="Dataset name stored under data/.")
    parser.add_argument(
        "--model",
        choices=["spectral_clustering", "deepwalk"],
        default="spectral_clustering",
        help="Embedding method to evaluate.",
    )
    parser.add_argument(
        "--feat-norm",
        action="store_true",
        help="Apply row-normalisation to features when loading (unused, kept for parity).",
    )
    parser.add_argument("--n-iter", type=int, default=1, help="Number of Monte Carlo runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--walk-len", type=int, default=80, help="Random walk length for DeepWalk.")
    parser.add_argument("--num-walk", type=int, default=10, help="Number of walks per node for DeepWalk.")
    parser.add_argument("--window", type=int, default=10, help="Context window size for DeepWalk.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of edges for testing.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction of edges for validation.")
    parser.add_argument(
        "--prevent-disconnect",
        action="store_true",
        help="Prevent the train graph from becoming disconnected during edge split.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSON file to append experiment summaries.",
    )
    return parser.parse_args()


def run_baseline(
    adj_matrix,
    *,
    model_name: str,
    n_iter: int,
    seed: int,
    dim: int,
    walk_len: int,
    num_walk: int,
    window: int,
    test_ratio: float,
    val_ratio: float,
    prevent_disconnect: bool,
) -> Dict[str, List[float]]:
    roc_scores: List[float] = []
    ap_scores: List[float] = []
    runtimes: List[float] = []

    for run_idx in range(n_iter):
        iter_seed = seed + run_idx
        np.random.seed(iter_seed)
        split = edge_split(
            adj_matrix,
            test_frac=test_ratio,
            val_frac=val_ratio,
            prevent_disconnect=prevent_disconnect,
        )
        if model_name == "spectral_clustering":
            scores = spectral_clustering_scores(split, random_state=iter_seed, dim=dim)
        else:
            scores = deepwalk_scores(split, dim=dim, walk_len=walk_len, num_walk=num_walk, window=window)

        roc_scores.append(float(scores["test_roc"]))
        ap_scores.append(float(scores["test_ap"]))
        runtimes.append(float(scores["runtime"]))

        print(
            f"[{model_name}] iter {run_idx + 1}/{n_iter} | "
            f"test ROC {scores['test_roc']:.5f} | test AP {scores['test_ap']:.5f} "
            f"| runtime {scores['runtime']:.2f}s"
        )

    return {
        "roc": roc_scores,
        "ap": ap_scores,
        "runtime": runtimes,
        "roc_mean": float(np.mean(roc_scores)),
        "roc_std": float(np.std(roc_scores)),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores)),
        "runtime_mean": float(np.mean(runtimes)),
        "runtime_std": float(np.std(runtimes)),
    }


def append_logs(log_entry: Dict[str, float], log_file: Path) -> None:
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        try:
            with log_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(log_entry)
    with log_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    adj_matrix, _ = load_data(args.dataset, task="link_prediction", feat_norm=args.feat_norm)

    stats = run_baseline(
        adj_matrix,
        model_name=args.model,
        n_iter=args.n_iter,
        seed=args.seed,
        dim=args.dim,
        walk_len=args.walk_len,
        num_walk=args.num_walk,
        window=args.window,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        prevent_disconnect=args.prevent_disconnect,
    )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "timestamp": datetime.utcnow().isoformat(),
        "n_iter": args.n_iter,
        "dim": args.dim,
        "walk_len": args.walk_len if args.model == "deepwalk" else None,
        "num_walk": args.num_walk if args.model == "deepwalk" else None,
        "window": args.window if args.model == "deepwalk" else None,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "prevent_disconnect": args.prevent_disconnect,
        **stats,
    }

    print(
        f"[{args.model}] ROC {summary['roc_mean']:.4f}±{summary['roc_std']:.4f} "
        f"| AP {summary['ap_mean']:.4f}±{summary['ap_std']:.4f} "
        f"| runtime {summary['runtime_mean']:.2f}±{summary['runtime_std']:.2f}s"
    )

    if args.log_file:
        append_logs(summary, args.log_file)
        print(f"Appended baseline summary to {args.log_file}")


if __name__ == "__main__":
    main()
