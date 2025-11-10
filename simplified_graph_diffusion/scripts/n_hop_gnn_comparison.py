"""Compare k-layer GNN vs single-layer with k-hop diffusion preprocessing."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gcn.model import GraphConvolution  # type: ignore
from gcn.utils import accuracy, load_data, preprocess_graph_diff, split


# -------------------------------
# Experiment configuration
# -------------------------------

DATASETS: Sequence[str] = ("citeseer",)
K_VALUES: Sequence[int] = (6, 7, 8, 9, )
RUNS_PER_SETTING: int = 5
ALPHA_K_LAYER: float = 0.5
ALPHA_SIMDIFF: float = 0.1
DIFFUSION_MODEL_DEPTH: int = 2
EPOCHS: int = 200
LR: float = 0.01
WEIGHT_DECAY: float = 5e-4
DROPOUT: float = 0.5
TRAIN_RATIO: float = 0.5
VAL_RATIO: float = 0.05
NUM_HIDDEN: int = 32
FEATURE_NORMALIZATION: bool = False
BASE_SEED: int = 42
OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "n_hop_gnn_comparison"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StackedGCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: List[GraphConvolution] = []
        if num_layers == 1:
            layers.append(GraphConvolution(nfeat, nclass))
        else:
            layers.append(GraphConvolution(nfeat, nhid))
            for _ in range(num_layers - 2):
                layers.append(GraphConvolution(nhid, nhid))
            layers.append(GraphConvolution(nhid, nclass))
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


@dataclass
class RunStats:
    test_acc: float
    test_loss: float
    elapsed_total: float
    preprocess_time: float
    train_time: float


def train_single_configuration(
    *,
    adj,
    features: torch.Tensor,
    labels: torch.Tensor,
    idx_split: Tuple[List[int], List[int], List[int]],
    num_layers: int,
    diffusion_power: int,
    alpha: float,
    device: torch.device,
    seed: int,
) -> RunStats:
    seed_everything(seed)

    idx_train, idx_val, idx_test = idx_split
    feat_tensor = features.to(device)
    label_tensor = labels.to(device)

    preprocess_start = time.perf_counter()
    adj_processed = preprocess_graph_diff(adj, diffusion_power, alpha).to(device)
    preprocess_time = time.perf_counter() - preprocess_start

    model = StackedGCN(
        nfeat=feat_tensor.shape[1],
        nhid=NUM_HIDDEN,
        nclass=int(label_tensor.max().item()) + 1,
        num_layers=num_layers,
        dropout=DROPOUT,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    idx_train_tensor = torch.LongTensor(idx_train).to(device)
    idx_val_tensor = torch.LongTensor(idx_val).to(device)
    idx_test_tensor = torch.LongTensor(idx_test).to(device)

    train_start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(feat_tensor, adj_processed)
        loss = F.nll_loss(output[idx_train_tensor], label_tensor[idx_train_tensor])
        loss.backward()
        optimizer.step()

    train_time = time.perf_counter() - train_start

    model.eval()
    with torch.no_grad():
        logits = model(feat_tensor, adj_processed)
        test_loss = F.nll_loss(logits[idx_test_tensor], label_tensor[idx_test_tensor])
        test_acc = accuracy(logits[idx_test_tensor], label_tensor[idx_test_tensor])

    return RunStats(
        test_acc=float(test_acc),
        test_loss=float(test_loss),
        elapsed_total=preprocess_time + train_time,
        preprocess_time=preprocess_time,
        train_time=train_time,
    )


def prepare_split(num_nodes: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    seed_everything(seed)
    len_train = int(num_nodes * TRAIN_RATIO)
    len_val = int(len_train * VAL_RATIO)
    len_test = num_nodes - (len_train + len_val)
    if min(len_train, len_val, len_test) <= 0:
        raise ValueError("Invalid split sizes; adjust TRAIN_RATIO/VAL_RATIO.")
    return split(num_nodes, len_train, len_val, len_test)


def run_dataset_experiment(dataset: str, device: torch.device) -> Dict:
    print(f"\n=== Dataset: {dataset} ===")
    adj, features, labels = load_data(dataset, task="classification", feat_norm=FEATURE_NORMALIZATION)
    num_nodes = features.shape[0]

    dataset_results = {
        "k_layer": {},
        "two_layer_diffusion": {},
        "pair_efficiency": {},
    }

    for k in K_VALUES:
        print(f"Running k={k}")
        # Precompute splits to reuse across both model variants for fairness
        splits = [
            prepare_split(num_nodes, seed=BASE_SEED + 1000 * k + run_idx)
            for run_idx in range(RUNS_PER_SETTING)
        ]

        k_layer_runs = []
        two_layer_runs = []
        pair_efficiencies: List[float] = []

        for run_idx, idx_split in enumerate(splits):
            multi_seed = BASE_SEED + 2000 * k + run_idx
            multi_stats = train_single_configuration(
                adj=adj,
                features=features,
                labels=labels,
                idx_split=idx_split,
                num_layers=k,
                diffusion_power=1,
                alpha=ALPHA_K_LAYER,
                device=device,
                seed=multi_seed,
            )
            multi_record = asdict(multi_stats)
            multi_record["run"] = run_idx + 1
            k_layer_runs.append(multi_record)

            single_seed = BASE_SEED + 3000 * k + run_idx
            single_stats = train_single_configuration(
                adj=adj,
                features=features,
                labels=labels,
                idx_split=idx_split,
                num_layers=DIFFUSION_MODEL_DEPTH,
                diffusion_power=k,
                alpha=ALPHA_SIMDIFF,
                device=device,
                seed=single_seed,
            )
            single_record = asdict(single_stats)
            single_record["run"] = run_idx + 1
            two_layer_runs.append(single_record)

            run_efficiency = compute_efficiency_metric(
                baseline_acc=multi_stats.test_acc,
                improved_acc=single_stats.test_acc,
                baseline_time=multi_stats.elapsed_total,
                improved_time=single_stats.elapsed_total,
            )
            pair_efficiencies.append(run_efficiency)

            extra = (
                f", efficiency(Δacc/logΔtime)={run_efficiency:.4f}"
                if not np.isnan(run_efficiency)
                else ""
            )
            print(
                f"  Run {run_idx + 1}/{RUNS_PER_SETTING} | "
                f"k-layer acc={multi_stats.test_acc:.4f} time={multi_stats.elapsed_total:.2f}s, "
                f"diffused 2-layer acc={single_stats.test_acc:.4f} time={single_stats.elapsed_total:.2f}s"
                f"{extra}"
            )

        dataset_results["k_layer"][str(k)] = build_summary(k_layer_runs)
        dataset_results["two_layer_diffusion"][str(k)] = build_summary(two_layer_runs)
        dataset_results["pair_efficiency"][str(k)] = summarize_efficiency(pair_efficiencies)

        k_summary = dataset_results["k_layer"][str(k)]
        two_summary = dataset_results["two_layer_diffusion"][str(k)]
        eff_summary = dataset_results["pair_efficiency"][str(k)]
        k_mean = k_summary["mean"]
        two_mean = two_summary["mean"]

        print(
            f"k-layer GNN   | layers={k} | time={k_mean['elapsed_total']:.2f}±{k_summary['std']['elapsed_total']:.2f}s "
            f"| acc={k_mean['test_acc']:.4f}±{k_summary['std']['test_acc']:.4f}"
        )
        print(
            f"2-layer + S^{k} | time={two_mean['elapsed_total']:.2f}±{two_summary['std']['elapsed_total']:.2f}s "
            f"| acc={two_mean['test_acc']:.4f}±{two_summary['std']['test_acc']:.4f}"
        )
        if not np.isnan(eff_summary["mean"]):
            print(
                f"  Efficiency (ΔAcc/logΔTime): {eff_summary['mean']:.4f}"
                f" ± {eff_summary['std']:.4f}"
            )

    return dataset_results


def build_summary(run_records: List[Dict[str, float]]) -> Dict:
    metrics = ["test_acc", "test_loss", "elapsed_total", "preprocess_time", "train_time"]
    means = {}
    stds = {}
    for key in metrics:
        values = np.array([record[key] for record in run_records], dtype=float)
        means[key] = float(values.mean())
        stds[key] = float(values.std())
    return {
        "runs": run_records,
        "mean": means,
        "std": stds,
        "num_runs": len(run_records),
    }


def summarize_efficiency(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any():
        mean = float("nan")
        std = float("nan")
    else:
        mean = float(arr[mask].mean())
        std = float(arr[mask].std())
    return {"values": values, "mean": mean, "std": std}


def compute_efficiency_metric(*, baseline_acc: float, improved_acc: float, baseline_time: float, improved_time: float) -> float:
    acc_diff = improved_acc - baseline_acc
    if baseline_time <= 0 or improved_time <= 0 or baseline_time == improved_time:
        return float("nan")
    log_ratio = np.log(improved_time / baseline_time)
    if log_ratio == 0:
        return float("nan")
    return acc_diff / log_ratio


def plot_runtimes(dataset: str, dataset_results: Dict[str, Dict[str, Dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(int(k) for k in dataset_results["k_layer"].keys())

    efficiency_vals = []
    for k in ks:
        eff_summary = dataset_results["pair_efficiency"].get(str(k), {})
        mean_val = eff_summary.get("mean", float("nan"))
        if mean_val is not None and not np.isnan(mean_val):
            efficiency_vals.append(mean_val * 100.0)
        else:
            efficiency_vals.append(float("nan"))

    ax.plot(ks, efficiency_vals, marker="o", color="#c62828")
    ax.set_xlabel("k-hop")
    ax.set_ylabel("ΔAcc(%)/logΔTime(s)")
    ax.set_title(f"SimDiff Efficiency Test Results ({dataset}, alpha={ALPHA_SIMDIFF})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{dataset}_acc_time_efficiency_comparison.png")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset in DATASETS:
        dataset_results = run_dataset_experiment(dataset, device)
        plot_runtimes(dataset, dataset_results)
        all_results[dataset] = dataset_results

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"n_hop_gnn_comparison_{timestamp}.json"
    payload = {
        "datasets": list(DATASETS),
        "k_values": list(K_VALUES),
        "runs_per_setting": RUNS_PER_SETTING,
        "alpha_k_layer": ALPHA_K_LAYER,
        "ALPHA_SIMDIFF": ALPHA_SIMDIFF,
        "diffusion_model_depth": DIFFUSION_MODEL_DEPTH,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "num_hidden": NUM_HIDDEN,
        "feature_normalization": FEATURE_NORMALIZATION,
        "results": all_results,
        "created_utc": timestamp,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
