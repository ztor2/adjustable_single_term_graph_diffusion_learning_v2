#!/usr/bin/env python
import argparse
import json
import pickle as pkl
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run logistic regression or MLP baselines for node classification."
    )
    parser.add_argument("--dataset", default="cora", help="Dataset name stored under data/.")
    parser.add_argument(
        "--model",
        choices=["logreg", "mlp"],
        default="mlp",
        help="Baseline model to train.",
    )
    parser.add_argument("--n-iter", type=int, default=1, help="Number of random restarts.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--test-size", type=float, default=0.45, help="Test split fraction.")
    parser.add_argument("--val-size", type=float, default=0.05, help="Validation fraction of test set.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training iterations.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="MLP learning rate (ignored for logistic regression).",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Hidden units per layer for MLP.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers for MLP.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSON file to append experiment summaries.",
    )
    return parser.parse_args()


def load_dataset(dataset: str):
    data_dir = PROJECT_ROOT / "data"
    with open(data_dir / f"{dataset}.feature", "rb") as fh:
        features = pkl.load(fh).todense()
    with open(data_dir / f"{dataset}.labels", "rb") as fh:
        labels = pkl.load(fh)
    return np.asarray(features), np.asarray(labels)


def run_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model_name: str,
    n_iter: int,
    seed: int,
    test_size: float,
    val_size: float,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    hidden_layers: int,
) -> Dict[str, float]:
    scores: List[float] = []
    val_scores: List[float] = []

    for run_idx in range(n_iter):
        iteration_seed = seed + run_idx
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            shuffle=True,
            random_state=iteration_seed,
            stratify=labels,
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=val_size,
            shuffle=True,
            random_state=iteration_seed,
            stratify=y_test,
        )

        if model_name == "logreg":
            clf = LogisticRegression(
                multi_class="ovr",
                max_iter=epochs,
                solver="lbfgs",
                random_state=iteration_seed,
            )
        else:
            hidden_shape = tuple([hidden_units] * hidden_layers)
            clf = MLPClassifier(
                max_iter=epochs,
                learning_rate_init=learning_rate,
                hidden_layer_sizes=hidden_shape,
                random_state=iteration_seed,
            )

        clf.fit(X_train, y_train)
        val_score = float(clf.score(X_val, y_val))
        test_score = float(clf.score(X_test, y_test))
        val_scores.append(val_score)
        scores.append(test_score)
        print(
            f"[{model_name}] iter {run_idx + 1}/{n_iter} | "
            f"val acc {val_score * 100:.2f}% | test acc {test_score * 100:.2f}%"
        )

    return {
        "val_acc": val_scores,
        "test_acc": scores,
        "val_acc_mean": float(np.mean(val_scores)),
        "val_acc_std": float(np.std(val_scores)),
        "test_acc_mean": float(np.mean(scores)),
        "test_acc_std": float(np.std(scores)),
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
    features, labels = load_dataset(args.dataset)

    stats = run_baseline(
        features,
        labels,
        model_name=args.model,
        n_iter=args.n_iter,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
    )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "timestamp": datetime.utcnow().isoformat(),
        "n_iter": args.n_iter,
        "epochs": args.epochs,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "learning_rate": args.learning_rate if args.model == "mlp" else None,
        "hidden_units": args.hidden_units if args.model == "mlp" else None,
        "hidden_layers": args.hidden_layers if args.model == "mlp" else None,
        **stats,
    }

    print(
        f"[{args.model}] test acc "
        f"{summary['test_acc_mean'] * 100:.2f}% ± {summary['test_acc_std'] * 100:.2f}% "
        f"| val acc {summary['val_acc_mean'] * 100:.2f}% ± {summary['val_acc_std'] * 100:.2f}%"
    )

    if args.log_file:
        append_logs(summary, args.log_file)
        print(f"Appended baseline summary to {args.log_file}")


if __name__ == "__main__":
    main()
