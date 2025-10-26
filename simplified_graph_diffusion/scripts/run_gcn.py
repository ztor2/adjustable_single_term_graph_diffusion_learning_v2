#!/usr/bin/env python
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gcn.model import GCN
from gcn.utils import accuracy, load_data, preprocess_graph_diff, split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GCN classification experiments with adjustable diffusion."
    )
    parser.add_argument("--dataset", default="cora", help="Dataset name stored under data/.")
    parser.add_argument(
        "--feat-norm",
        action="store_true",
        help="Apply row-normalisation to features when loading.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per run.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay factor.")
    parser.add_argument("--num-hidden", type=int, default=32, help="Hidden layer size.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Fraction of nodes used for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation fraction relative to the training set size.",
    )
    parser.add_argument(
        "--n-diff-range",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of diffusion powers to sweep.",
    )
    parser.add_argument(
        "--alpha-range",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of diffusion alpha values to sweep.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1,
        help="Number of Monte Carlo runs per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each iteration adds its index to this base.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Epoch interval for validation logging.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSON file to append experiment summaries.",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Suppress per-epoch validation logs.",
    )
    return parser.parse_args()


def resolve_device(option: str) -> torch.device:
    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_gcn_iteration(
    adj_matrix,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    dataset: str,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_hidden: int,
    dropout: float,
    train_ratio: float,
    val_ratio: float,
    n_diff: int,
    alpha: float,
    n_iter: int,
    seed: int,
    log_interval: int,
    quiet: bool,
) -> Dict[str, float]:
    acc_scores: List[float] = []
    loss_scores: List[float] = []
    num_nodes = features.shape[0]
    len_train = int(num_nodes * train_ratio)
    len_val = int(len_train * val_ratio)
    len_test = num_nodes - (len_train + len_val)
    if len_train <= 0 or len_val <= 0 or len_test <= 0:
        raise ValueError("Dataset split produced non-positive subset sizes.")

    start = time.time()
    adj_processed = preprocess_graph_diff(adj_matrix, n_diff, alpha).to(device)

    base_features = features.clone().detach()
    base_labels = labels.clone().detach()

    for run_idx in range(n_iter):
        iter_seed = seed + run_idx
        set_seed(iter_seed)

        idx_train, idx_val, idx_test = split(num_nodes, len_train, len_val, len_test)

        feat_tensor = base_features.to(device)
        label_tensor = base_labels.to(device)

        model = GCN(
            nfeat=feat_tensor.shape[1],
            nhid=num_hidden,
            nclass=int(label_tensor.max().item()) + 1,
            dropout=dropout,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(feat_tensor, adj_processed)
            train_loss = F.nll_loss(output[idx_train], label_tensor[idx_train])
            train_loss.backward()
            optimizer.step()

            if not quiet and (epoch + 1) % max(log_interval, 1) == 0:
                model.eval()
                with torch.no_grad():
                    val_output = model(feat_tensor, adj_processed)
                    val_loss = F.nll_loss(val_output[idx_val], label_tensor[idx_val])
                    val_acc = accuracy(val_output[idx_val], label_tensor[idx_val])
                print(
                    f"[{dataset}] iter {run_idx + 1}/{n_iter} | epoch {epoch + 1:04d}/{epochs} "
                    f"| val acc {val_acc:.4f} | val loss {val_loss:.4f}"
                )

        model.eval()
        with torch.no_grad():
            test_output = model(feat_tensor, adj_processed)
            test_loss = F.nll_loss(test_output[idx_test], label_tensor[idx_test])
            test_acc = accuracy(test_output[idx_test], label_tensor[idx_test])
        acc_scores.append(float(test_acc))
        loss_scores.append(float(test_loss))
        print(
            f"[{dataset}] iter {run_idx + 1}/{n_iter} | "
            f"test acc {test_acc:.5f} | test loss {test_loss:.5f} | n_diff={n_diff} | alpha={alpha}"
        )

    elapsed = time.time() - start
    return {
        "acc": acc_scores,
        "loss": loss_scores,
        "acc_mean": float(np.mean(acc_scores)),
        "acc_std": float(np.std(acc_scores)),
        "loss_mean": float(np.mean(loss_scores)),
        "loss_std": float(np.std(loss_scores)),
        "elapsed_time": elapsed,
    }


def append_logs(logs: List[Dict[str, float]], log_file: Path) -> None:
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
    data.extend(logs)
    with log_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    os.chdir(PROJECT_ROOT)

    data = load_data(args.dataset, task="classification", feat_norm=args.feat_norm)
    adj_matrix, features, labels = data

    all_logs: List[Dict[str, float]] = []
    for n_diff in args.n_diff_range:
        for alpha in args.alpha_range:
            stats = run_gcn_iteration(
                adj_matrix,
                features,
                labels,
                dataset=args.dataset,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_hidden=args.num_hidden,
                dropout=args.dropout,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                n_diff=n_diff,
                alpha=alpha,
                n_iter=args.n_iter,
                seed=args.seed,
                log_interval=args.log_interval,
                quiet=args.results_only,
            )

            summary = {
                "model": "gcn",
                "dataset": args.dataset,
                "timestamp": datetime.utcnow().isoformat(),
                "feat_norm": args.feat_norm,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_hidden": args.num_hidden,
                "dropout": args.dropout,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "n_iter": args.n_iter,
                "n_diff": n_diff,
                "alpha": alpha,
                **stats,
            }
            all_logs.append(summary)

            print(
                f"[{args.dataset}] n_diff={n_diff} alpha={alpha} "
                f"| Acc {stats['acc_mean']:.4f}±{stats['acc_std']:.4f} "
                f"| Loss {stats['loss_mean']:.4f}±{stats['loss_std']:.4f} "
                f"| elapsed {stats['elapsed_time']:.2f}s"
            )

    if args.log_file:
        append_logs(all_logs, args.log_file)
        print(f"Appended {len(all_logs)} entries to {args.log_file}")


if __name__ == "__main__":
    main()
