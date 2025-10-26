#!/usr/bin/env python
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gae.model import GCN_AE
from gae.optimizer import loss_function
from gae.utils import (
    get_roc_score_gae,
    load_data,
    mask_test_edges,
    preprocess_graph_diff,
    remove_diag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graph Auto-Encoder (GAE) experiments with adjustable diffusion."
    )
    parser.add_argument("--dataset", default="cora", help="Dataset name stored under data/.")
    parser.add_argument(
        "--feature-use",
        action="store_true",
        help="Use node features during training (otherwise identity features are used).",
    )
    parser.add_argument(
        "--feat-norm",
        action="store_true",
        help="Apply row-normalisation to features when loading.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per run.")
    parser.add_argument("--hidden1", type=int, default=32, help="Hidden dimension of first layer.")
    parser.add_argument("--hidden2", type=int, default=16, help="Embedding dimension.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of observed edges reserved for test evaluation.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of observed edges reserved for validation evaluation.",
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
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_ratios(test_ratio: float, val_ratio: float) -> Dict[str, float]:
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be in (0, 1).")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in (0, 1).")
    return {
        "test_factor": 1.0 / test_ratio,
        "val_factor": 1.0 / val_ratio,
    }


def run_gae_iteration(
    adj_matrix: sp.csr_matrix,
    features: torch.Tensor,
    *,
    dataset: str,
    device: torch.device,
    epochs: int,
    hidden1: int,
    hidden2: int,
    lr: float,
    dropout: float,
    test_factor: float,
    val_factor: float,
    n_diff: int,
    alpha: float,
    feature_use: bool,
    n_iter: int,
    seed: int,
    log_interval: int,
    quiet: bool,
) -> Dict[str, float]:
    roc_scores: List[float] = []
    ap_scores: List[float] = []
    start = time.time()

    for run_idx in range(n_iter):
        iter_seed = seed + run_idx
        set_seed(iter_seed)

        adj_orig = remove_diag(adj_matrix)
        while True:
            try:
                split = mask_test_edges(adj_matrix, test_factor, val_factor)
            except AssertionError:
                continue
            break

        (
            adj_train,
            train_edges,
            val_edges,
            val_edges_false,
            test_edges,
            test_edges_false,
        ) = split

        adj_norm = preprocess_graph_diff(adj_train, n_diff, alpha).to(device)
        if feature_use:
            feat_tensor = features.clone().detach().to(device)
        else:
            feat_tensor = torch.eye(adj_matrix.shape[0], device=device)

        n_nodes, feat_dim = feat_tensor.shape
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.as_tensor(adj_label.toarray(), dtype=torch.float32, device=device)

        pos_weight_value = float(adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) / float(
            adj_matrix.sum()
        )
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        norm = adj_matrix.shape[0] * adj_matrix.shape[0] / float(
            (adj_matrix.shape[0] * adj_matrix.shape[0] - adj_matrix.sum()) * 2
        )

        model = GCN_AE(feat_dim, hidden1, hidden2, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        recon_adj = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            recon_adj = model(feat_tensor, adj_norm)
            loss = loss_function(preds=recon_adj, labels=adj_label, norm=norm, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            if not quiet and (epoch + 1) % max(log_interval, 1) == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = recon_adj.detach().cpu().numpy()
                roc_curr, ap_curr = get_roc_score_gae(val_pred, adj_orig, val_edges, val_edges_false)
                print(
                    f"[{dataset}] iter {run_idx + 1}/{n_iter} | epoch {epoch + 1:04d}/{epochs} "
                    f"| val ROC {roc_curr:.4f} | val AP {ap_curr:.4f}"
                )

        model.eval()
        with torch.no_grad():
            recon_array = recon_adj.detach().cpu().numpy()
        roc_score, ap_score = get_roc_score_gae(recon_array, adj_orig, test_edges, test_edges_false)
        roc_scores.append(float(roc_score))
        ap_scores.append(float(ap_score))
        print(
            f"[{dataset}] iter {run_idx + 1}/{n_iter} | "
            f"test ROC {roc_score:.5f} | test AP {ap_score:.5f} | n_diff={n_diff} | alpha={alpha}"
        )

    elapsed = time.time() - start
    return {
        "roc": roc_scores,
        "ap": ap_scores,
        "roc_mean": float(np.mean(roc_scores)),
        "roc_std": float(np.std(roc_scores)),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores)),
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

    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError("test_ratio + val_ratio must be < 1.")

    data = load_data(args.dataset, task="link_prediction", feat_norm=args.feat_norm)
    adj_matrix, features = data

    ratios = _split_ratios(args.test_ratio, args.val_ratio)
    all_logs: List[Dict[str, float]] = []

    for n_diff in args.n_diff_range:
        for alpha in args.alpha_range:
            stats = run_gae_iteration(
                adj_matrix,
                features,
                dataset=args.dataset,
                device=device,
                epochs=args.epochs,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                lr=args.lr,
                dropout=args.dropout,
                test_factor=ratios["test_factor"],
                val_factor=ratios["val_factor"],
                n_diff=n_diff,
                alpha=alpha,
                feature_use=args.feature_use,
                n_iter=args.n_iter,
                seed=args.seed,
                log_interval=args.log_interval,
                quiet=args.results_only,
            )

            summary = {
                "model": "gae",
                "dataset": args.dataset,
                "timestamp": datetime.utcnow().isoformat(),
                "feature_use": args.feature_use,
                "feat_norm": args.feat_norm,
                "epochs": args.epochs,
                "hidden1": args.hidden1,
                "hidden2": args.hidden2,
                "lr": args.lr,
                "dropout": args.dropout,
                "test_ratio": args.test_ratio,
                "val_ratio": args.val_ratio,
                "n_iter": args.n_iter,
                "n_diff": n_diff,
                "alpha": alpha,
                **stats,
            }
            all_logs.append(summary)

            print(
                f"[{args.dataset}] n_diff={n_diff} alpha={alpha} "
                f"| ROC {stats['roc_mean']:.4f}±{stats['roc_std']:.4f} "
                f"| AP {stats['ap_mean']:.4f}±{stats['ap_std']:.4f} "
                f"| elapsed {stats['elapsed_time']:.2f}s"
            )

    if args.log_file:
        append_logs(all_logs, args.log_file)
        print(f"Appended {len(all_logs)} entries to {args.log_file}")


if __name__ == "__main__":
    main()
