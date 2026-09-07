from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_representation_mlqem_enhanced import (
    QuantumCircuitGraphDataset,
    NOISY_EXPECTATION_INDEX,
)

from gnn_mlqem_enhanced import CircuitGNN


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("mlqem_random_fake_lima_dataset")

TRAIN_PKL = DATA_DIR / "mlqem_random_fake_lima_train.pkl"
TEST_PKL = DATA_DIR / "mlqem_random_fake_lima_test.pkl"

OUT_DIR = Path("models_mlqem_fake_lima_gnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_FEATURE_BACKEND_VARIANT = "fake_lima"
NODE_ANGLE_ENCODING = "none"

TARGET_MODE = "mitigation_delta"
#TARGET_MODE = "ideal_expectation"
# Options:
#   "ideal_expectation" -> model directly predicts ideal <Zi>
#   "mitigation_delta" -> model predicts ideal <Zi> - noisy <Zi>

EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64

VAL_WITHIN_TRAIN = 0.10

EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 0.0

SEED = 34

MODEL_KWARGS = {
    "gnn_hidden": 64,
    "gnn_heads": 4,
    "global_hidden": 64,
    "reg_hidden": 128,
    "num_layers": 7,
    "dropout_rate": 0.2,
}


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset / loaders
# ============================================================

def build_datasets():
    if not TRAIN_PKL.exists():
        raise FileNotFoundError(f"Missing train PKL: {TRAIN_PKL}")
    if not TEST_PKL.exists():
        raise FileNotFoundError(f"Missing test PKL: {TEST_PKL}")

    train_full = QuantumCircuitGraphDataset(
        root=str(OUT_DIR / "pyg_cache_train"),
        pkl_paths=[str(TRAIN_PKL.resolve())],
        node_feature_backend_variant=NODE_FEATURE_BACKEND_VARIANT,
        node_angle_encoding=NODE_ANGLE_ENCODING,
    )

    test_ds = QuantumCircuitGraphDataset(
        root=str(OUT_DIR / "pyg_cache_test"),
        pkl_paths=[str(TEST_PKL.resolve())],
        node_feature_backend_variant=NODE_FEATURE_BACKEND_VARIANT,
        node_angle_encoding=NODE_ANGLE_ENCODING,
    )

    if len(train_full) < 2:
        raise RuntimeError("Train dataset too small.")
    if len(test_ds) < 1:
        raise RuntimeError("Test dataset is empty.")

    generator = torch.Generator().manual_seed(SEED)

    val_len = max(1, int(round(len(train_full) * VAL_WITHIN_TRAIN)))
    val_len = min(len(train_full) - 1, val_len)
    train_len = len(train_full) - val_len

    train_ds, val_ds = torch.utils.data.random_split(
        train_full,
        [train_len, val_len],
        generator=generator,
    )

    return train_ds, val_ds, test_ds, train_full


def make_loaders(train_ds, val_ds, test_ds):
    num_workers = 2 if (torch.get_num_threads() > 2) else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Target handling
# ============================================================

def global_features_2d(batch) -> Tensor:
    g = batch.global_features

    if g.dim() == 1:
        g = g.view(batch.num_graphs, -1)

    return g.float()


def get_noisy_expectation(batch) -> Tensor:
    g = global_features_2d(batch)

    if g.size(1) <= NOISY_EXPECTATION_INDEX:
        raise RuntimeError(
            f"global_features has dim {g.size(1)}, "
            f"but NOISY_EXPECTATION_INDEX={NOISY_EXPECTATION_INDEX}"
        )

    return g[:, NOISY_EXPECTATION_INDEX].view(-1).to(batch.y.device)


def training_target(batch) -> Tensor:
    ideal = batch.y.view(-1)

    if TARGET_MODE == "ideal_expectation":
        return ideal

    if TARGET_MODE == "mitigation_delta":
        noisy = get_noisy_expectation(batch)
        return ideal - noisy

    raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE}")


def prediction_to_ideal(batch, pred_raw: Tensor, clamp_output: bool = True) -> Tensor:
    if TARGET_MODE == "ideal_expectation":
        pred = pred_raw

    elif TARGET_MODE == "mitigation_delta":
        noisy = get_noisy_expectation(batch)
        pred = noisy + pred_raw

    else:
        raise ValueError(f"Unknown TARGET_MODE={TARGET_MODE}")

    if clamp_output:
        pred = pred.clamp(-1.0, 1.0)

    return pred


# ============================================================
# Metrics
# ============================================================

def mse(y: Tensor, pred: Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float(torch.mean((pred - y) ** 2).item())


def rmse(y: Tensor, pred: Tensor) -> float:
    m = mse(y, pred)
    return float(math.sqrt(m)) if math.isfinite(m) else float("nan")


def mae(y: Tensor, pred: Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(pred - y)).item())


def r2(y: Tensor, pred: Tensor) -> float:
    if y.numel() == 0:
        return float("nan")

    ss_res = torch.sum((pred - y) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)

    if ss_tot <= 0:
        return 0.0

    return float((1.0 - ss_res / ss_tot).item())


def pearson(y: Tensor, pred: Tensor) -> float:
    if y.numel() < 2:
        return float("nan")

    a = y.double()
    b = pred.double()

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt(torch.sum(a ** 2) * torch.sum(b ** 2))

    if denom <= 0:
        return 0.0

    return float((torch.sum(a * b) / denom).item())


def compute_scalar_metrics(y: Tensor, pred: Tensor) -> Dict[str, float]:
    return {
        "n": int(y.numel()),
        "mse": mse(y, pred),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "r2": r2(y, pred),
        "pearson": pearson(y, pred),
        "true_mean": float(torch.mean(y).item()) if y.numel() else float("nan"),
        "pred_mean": float(torch.mean(pred).item()) if pred.numel() else float("nan"),
        "true_std": float(torch.std(y, unbiased=True).item()) if y.numel() > 1 else 0.0,
        "pred_std": float(torch.std(pred, unbiased=True).item()) if pred.numel() > 1 else 0.0,
    }


def list_from_batch_attr(batch, name: str, n: int) -> List[Any]:
    value = getattr(batch, name, None)

    if value is None:
        return [None] * n

    if torch.is_tensor(value):
        v = value.detach().cpu().view(-1).tolist()
        return v[:n]

    if isinstance(value, list):
        return value[:n]

    if isinstance(value, tuple):
        return list(value)[:n]

    return [value] * n


# ============================================================
# Training
# ============================================================

def train_model(model: nn.Module, train_loader, val_loader, device: torch.device):
    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_seen = 0

        for batch in train_loader:
            batch = batch.to(device)

            pred_raw = model(batch).view(-1)
            target = training_target(batch)

            mask = torch.isfinite(target) & torch.isfinite(pred_raw)

            if not mask.any():
                continue

            loss = loss_fn(pred_raw[mask], target[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_total += float(loss.detach().item()) * int(mask.sum().item())
            train_seen += int(mask.sum().item())

        train_loss = train_total / max(1, train_seen)

        model.eval()
        val_total = 0.0
        val_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                pred_raw = model(batch).view(-1)
                target = training_target(batch)

                mask = torch.isfinite(target) & torch.isfinite(pred_raw)

                if not mask.any():
                    continue

                loss = loss_fn(pred_raw[mask], target[mask])

                val_total += float(loss.detach().item()) * int(mask.sum().item())
                val_seen += int(mask.sum().item())

        val_loss = val_total / max(1, val_seen)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss + EARLY_STOPPING_MIN_DELTA < best_val

        if improved:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 1 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Train {train_loss:.8e} | "
                f"Val {val_loss:.8e} | "
                f"Best {best_val:.8e} | "
                f"Bad {bad_epochs}"
            )

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val={best_val:.8e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, train_losses, val_losses


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def collect_eval_rows(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    rows: List[Dict[str, Any]] = []

    y_all: List[Tensor] = []
    pred_all: List[Tensor] = []
    noisy_all: List[Tensor] = []

    for batch in loader:
        batch = batch.to(device)

        pred_raw = model(batch).view(-1)
        pred_ideal = prediction_to_ideal(batch, pred_raw, clamp_output=True)

        y_ideal = batch.y.view(-1)
        noisy = get_noisy_expectation(batch)

        mask = torch.isfinite(y_ideal) & torch.isfinite(pred_ideal) & torch.isfinite(noisy)

        if not mask.any():
            continue

        y_cpu = y_ideal[mask].detach().cpu()
        p_cpu = pred_ideal[mask].detach().cpu()
        n_cpu = noisy[mask].detach().cpu()

        y_all.append(y_cpu)
        pred_all.append(p_cpu)
        noisy_all.append(n_cpu)

        mask_cpu = mask.detach().cpu().view(-1).tolist()
        n_graphs = len(mask_cpu)

        circuit_idx_list = list_from_batch_attr(batch, "circuit_idx", n_graphs)
        target_depth_list = list_from_batch_attr(batch, "target_two_qubit_depth", n_graphs)
        actual_depth_list = list_from_batch_attr(batch, "actual_two_qubit_depth", n_graphs)
        observable_qubit_list = list_from_batch_attr(batch, "observable_qubit", n_graphs)

        # Only keep rows where mask is true.
        keep_positions = [i for i, keep in enumerate(mask_cpu) if keep]

        for local_j, original_i in enumerate(keep_positions):
            true_val = float(y_cpu[local_j].item())
            pred_val = float(p_cpu[local_j].item())
            noisy_val = float(n_cpu[local_j].item())

            rows.append({
                "circuit_idx": int(circuit_idx_list[original_i]),
                "target_two_qubit_depth": int(target_depth_list[original_i]),
                "actual_two_qubit_depth": int(actual_depth_list[original_i]),
                "observable_qubit": int(observable_qubit_list[original_i]),
                "ideal_expectation": true_val,
                "noisy_expectation": noisy_val,
                "predicted_ideal_expectation": pred_val,
                "baseline_error": noisy_val - true_val,
                "gnn_error": pred_val - true_val,
                "baseline_abs_error": abs(noisy_val - true_val),
                "gnn_abs_error": abs(pred_val - true_val),
            })

    if not y_all:
        return rows, torch.empty(0), torch.empty(0), torch.empty(0)

    y = torch.cat(y_all, dim=0)
    pred = torch.cat(pred_all, dim=0)
    noisy = torch.cat(noisy_all, dim=0)

    return rows, y, pred, noisy


def compute_l2_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "pred": float(r["predicted_ideal_expectation"]),
        }

    baseline_l2: List[float] = []
    gnn_l2: List[float] = []

    by_depth: Dict[int, Dict[str, List[float]]] = {}

    for key, obs_dict in groups.items():
        target_depth, _circuit_idx = key

        if not all(q in obs_dict for q in [0, 1, 2, 3]):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in [0, 1, 2, 3]], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in [0, 1, 2, 3]], dtype=float)
        pred_vec = np.asarray([obs_dict[q]["pred"] for q in [0, 1, 2, 3]], dtype=float)

        b = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))
        g = float(np.linalg.norm(pred_vec - ideal_vec, ord=2))

        baseline_l2.append(b)
        gnn_l2.append(g)

        by_depth.setdefault(target_depth, {"baseline": [], "gnn": []})
        by_depth[target_depth]["baseline"].append(b)
        by_depth[target_depth]["gnn"].append(g)

    if not baseline_l2:
        return {
            "n_circuits_l2": 0,
            "baseline_l2_mean": float("nan"),
            "gnn_l2_mean": float("nan"),
            "relative_l2_improvement": float("nan"),
        }

    b_arr = np.asarray(baseline_l2, dtype=float)
    g_arr = np.asarray(gnn_l2, dtype=float)

    out: Dict[str, float] = {
        "n_circuits_l2": int(len(baseline_l2)),
        "baseline_l2_mean": float(np.mean(b_arr)),
        "baseline_l2_std": float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0,
        "baseline_l2_median": float(np.median(b_arr)),
        "gnn_l2_mean": float(np.mean(g_arr)),
        "gnn_l2_std": float(np.std(g_arr, ddof=1)) if len(g_arr) > 1 else 0.0,
        "gnn_l2_median": float(np.median(g_arr)),
        "relative_l2_improvement": float((np.mean(b_arr) - np.mean(g_arr)) / np.mean(b_arr)),
    }

    for depth in sorted(by_depth):
        b_d = np.asarray(by_depth[depth]["baseline"], dtype=float)
        g_d = np.asarray(by_depth[depth]["gnn"], dtype=float)

        out[f"depth_{depth}_n"] = int(len(b_d))
        out[f"depth_{depth}_baseline_l2_mean"] = float(np.mean(b_d))
        out[f"depth_{depth}_gnn_l2_mean"] = float(np.mean(g_d))

    return out


def compute_per_observable_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for q in [0, 1, 2, 3]:
        q_rows = [r for r in rows if int(r["observable_qubit"]) == q]

        if not q_rows:
            continue

        y = torch.tensor([r["ideal_expectation"] for r in q_rows], dtype=torch.float)
        pred = torch.tensor([r["predicted_ideal_expectation"] for r in q_rows], dtype=torch.float)
        noisy = torch.tensor([r["noisy_expectation"] for r in q_rows], dtype=torch.float)

        gnn_metrics = compute_scalar_metrics(y, pred)
        baseline_metrics = compute_scalar_metrics(y, noisy)

        out.append({
            "observable": f"Z{q}",
            "n": len(q_rows),
            "gnn_mse": gnn_metrics["mse"],
            "gnn_rmse": gnn_metrics["rmse"],
            "gnn_mae": gnn_metrics["mae"],
            "gnn_r2": gnn_metrics["r2"],
            "baseline_mse": baseline_metrics["mse"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_r2": baseline_metrics["r2"],
        })

    return out


# ============================================================
# Plotting
# ============================================================

def axis_limits(x: Tensor, y: Tensor) -> Tuple[float, float]:
    vals = torch.cat([x.view(-1), y.view(-1)]).numpy()
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -1.0, 1.0

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * max(hi - lo, 1e-6)

    return lo - pad, hi + pad


def save_scatter(
    y: Tensor,
    pred: Tensor,
    metrics: Dict[str, float],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    lo, hi = axis_limits(y, pred)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y.tolist(),
        pred.tolist(),
        s=8,
        alpha=0.35,
        edgecolors="none",
    )

    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

    text = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MSE = {metrics['mse']:.4g}\n"
        f"MAE = {metrics['mae']:.4g}\n"
        f"Pearson = {metrics['pearson']:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.xlabel(r"Ideal $\langle Z_i \rangle$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_l2_histogram(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "pred": float(r["predicted_ideal_expectation"]),
        }

    baseline_l2 = []
    gnn_l2 = []

    for key, obs_dict in groups.items():
        if not all(q in obs_dict for q in [0, 1, 2, 3]):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in [0, 1, 2, 3]], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in [0, 1, 2, 3]], dtype=float)
        pred_vec = np.asarray([obs_dict[q]["pred"] for q in [0, 1, 2, 3]], dtype=float)

        baseline_l2.append(float(np.linalg.norm(noisy_vec - ideal_vec, ord=2)))
        gnn_l2.append(float(np.linalg.norm(pred_vec - ideal_vec, ord=2)))

    plt.figure(figsize=(8, 5))
    plt.hist(baseline_l2, bins=60, alpha=0.55, label="Unmitigated noisy", edgecolor="black")
    plt.hist(gnn_l2, bins=60, alpha=0.55, label="GNN mitigated", edgecolor="black")
    plt.xlabel(r"$L_2$ error on $[\langle Z_0\rangle,\ldots,\langle Z_3\rangle]$")
    plt.ylabel("Number of circuits")
    #plt.title("Paper-style vector error distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_l2_by_depth(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "pred": float(r["predicted_ideal_expectation"]),
        }

    by_depth: Dict[int, Dict[str, List[float]]] = {}

    for key, obs_dict in groups.items():
        target_depth, _ = key

        if not all(q in obs_dict for q in [0, 1, 2, 3]):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in [0, 1, 2, 3]], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in [0, 1, 2, 3]], dtype=float)
        pred_vec = np.asarray([obs_dict[q]["pred"] for q in [0, 1, 2, 3]], dtype=float)

        baseline_l2 = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))
        gnn_l2 = float(np.linalg.norm(pred_vec - ideal_vec, ord=2))

        by_depth.setdefault(target_depth, {"baseline": [], "gnn": []})
        by_depth[target_depth]["baseline"].append(baseline_l2)
        by_depth[target_depth]["gnn"].append(gnn_l2)

    depths = sorted(by_depth.keys())

    baseline_mean = [float(np.mean(by_depth[d]["baseline"])) for d in depths]
    gnn_mean = [float(np.mean(by_depth[d]["gnn"])) for d in depths]

    plt.figure(figsize=(8, 5))
    plt.plot(depths, baseline_mean, marker="o", label="Unmitigated noisy")
    plt.plot(depths, gnn_mean, marker="o", label="GNN mitigated")
    plt.xlabel("Target two-qubit depth")
    plt.ylabel(r"Mean $L_2$ error")
    plt.title(r"Mean $L_2$ mitigation error by depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_loss_curves(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title(f"Training curves ({TARGET_MODE})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_combined_scatter_no_metrics(
    y: Tensor,
    noisy: Tensor,
    pred: Tensor,
    out_path: Path,
) -> None:
    """
    Save a two-panel scatter plot:
        (a) unmitigated noisy vs ideal
        (b) GNN mitigated vs ideal

    No metric box is shown.
    Legend only contains:
        - Generated circuits
        - Ideal prediction
    """
    lo1, hi1 = axis_limits(y, noisy)
    lo2, hi2 = axis_limits(y, pred)

    lo = min(lo1, lo2)
    hi = max(hi1, hi2)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # (a) Unmitigated noisy
    # --------------------------------------------------------

    axes[0].scatter(
        y.tolist(),
        noisy.tolist(),
        s=8,
        alpha=0.35,
        edgecolors="none",
        label="Generated circuits",
    )

    axes[0].plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=2,
        color="red",
        label="Ideal prediction",
    )

    axes[0].set_xlabel(r"Ideal $\langle Z_i \rangle$")
    axes[0].set_ylabel(r"Noisy $\langle Z_i \rangle$")
    axes[0].set_title("(a) Unmitigated noisy values")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_box_aspect(1)

    # --------------------------------------------------------
    # (b) GNN mitigated
    # --------------------------------------------------------

    axes[1].scatter(
        y.tolist(),
        pred.tolist(),
        s=8,
        alpha=0.35,
        edgecolors="none",
        label="Generated circuits",
    )

    axes[1].plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=2,
        color="red",
        label="Ideal prediction",
    )

    axes[1].set_xlabel(r"Ideal $\langle Z_i \rangle$")
    axes[1].set_ylabel(r"GNN mitigated $\langle Z_i \rangle$")
    axes[1].set_title("(b) GNN-mitigated values")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_box_aspect(1)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_all_seeds(SEED)

    print("=" * 80)
    print("ML-QEM FakeLima GNN training")
    print("=" * 80)
    print(f"Train PKL:   {TRAIN_PKL}")
    print(f"Test PKL:    {TEST_PKL}")
    print(f"Target mode: {TARGET_MODE}")
    print(f"Output dir:  {OUT_DIR}")
    print()

    train_ds, val_ds, test_ds, train_full = build_datasets()
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    sample = train_full[0]
    node_dim = int(sample.x.size(-1))
    global_dim = int(sample.global_features.numel())

    print("=" * 80)
    print("Dataset")
    print("=" * 80)
    print(f"Train full graphs: {len(train_full)}")
    print(f"Train graphs:      {len(train_ds)}")
    print(f"Val graphs:        {len(val_ds)}")
    print(f"Test graphs:       {len(test_ds)}")
    print(f"Node dim:          {node_dim}")
    print(f"Global dim:        {global_dim}")
    print(f"Noisy expectation index: {NOISY_EXPECTATION_INDEX}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CircuitGNN(
        node_in_dim=node_dim,
        global_in_dim=global_dim,
        **MODEL_KWARGS,
    ).to(device)

    model, best_val, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
    )

    rows, y_test, pred_test, noisy_test = collect_eval_rows(
        model,
        test_loader,
        device,
    )

    gnn_scalar_metrics = compute_scalar_metrics(y_test, pred_test)
    baseline_scalar_metrics = compute_scalar_metrics(y_test, noisy_test)
    l2_metrics = compute_l2_metrics(rows)
    per_obs_metrics = compute_per_observable_metrics(rows)

    all_metrics = {
        "target_mode": TARGET_MODE,
        "best_val_huber": best_val,
        "gnn_scalar_metrics": gnn_scalar_metrics,
        "baseline_scalar_metrics": baseline_scalar_metrics,
        "l2_metrics": l2_metrics,
        "per_observable_metrics": per_obs_metrics,
    }

    # --------------------------------------------------------
    # Save model and metrics
    # --------------------------------------------------------

    torch.save(model.state_dict(), OUT_DIR / "state_dict.pt")

    torch.save(
        {
            "task": "mlqem_fake_lima_error_mitigation",
            "target_mode": TARGET_MODE,
            "model_state_dict": model.state_dict(),
            "model_kwargs": MODEL_KWARGS,
            "node_dim": node_dim,
            "global_dim": global_dim,
            "node_feature_backend_variant": NODE_FEATURE_BACKEND_VARIANT,
            "node_angle_encoding": NODE_ANGLE_ENCODING,
            "noisy_expectation_index": NOISY_EXPECTATION_INDEX,
            "best_val_huber": best_val,
            "metrics": all_metrics,
        },
        OUT_DIR / "checkpoint.pt",
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with (OUT_DIR / "predictions_test.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "per_observable_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_obs_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(per_obs_metrics)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_scatter(
        y_test,
        noisy_test,
        baseline_scalar_metrics,
        OUT_DIR / "scatter_unmitigated_noisy_vs_ideal.png",
        title="Unmitigated noisy vs ideal expectation",
        ylabel=r"Noisy $\langle Z_i\rangle$",
    )

    save_scatter(
        y_test,
        pred_test,
        gnn_scalar_metrics,
        OUT_DIR / "scatter_gnn_mitigated_vs_ideal.png",
        title="GNN mitigated vs ideal expectation",
        ylabel=r"GNN mitigated $\langle Z_i\rangle$",
    )

    save_combined_scatter_no_metrics(
        y_test,
        noisy_test,
        pred_test,
        OUT_DIR / "scatter_unmitigated_vs_gnn_combined.png",
    )

    save_l2_histogram(
        rows,
        OUT_DIR / "l2_error_distribution_unmitigated_vs_gnn.png",
    )

    save_l2_by_depth(
        rows,
        OUT_DIR / "l2_error_by_depth.png",
    )

    save_loss_curves(
        train_losses,
        val_losses,
        OUT_DIR / "loss_curves.png",
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("Scalar metrics:")
    print(
        f"Unmitigated | "
        f"MSE={baseline_scalar_metrics['mse']:.6g} | "
        f"RMSE={baseline_scalar_metrics['rmse']:.6g} | "
        f"MAE={baseline_scalar_metrics['mae']:.6g} | "
        f"R2={baseline_scalar_metrics['r2']:.4f}"
    )

    print(
        f"GNN         | "
        f"MSE={gnn_scalar_metrics['mse']:.6g} | "
        f"RMSE={gnn_scalar_metrics['rmse']:.6g} | "
        f"MAE={gnn_scalar_metrics['mae']:.6g} | "
        f"R2={gnn_scalar_metrics['r2']:.4f}"
    )

    print("\nPaper-style vector L2 metric:")
    print(
        f"Unmitigated mean L2 = {l2_metrics['baseline_l2_mean']:.6g} "
        f"± {l2_metrics['baseline_l2_std']:.6g}"
    )
    print(
        f"GNN mean L2         = {l2_metrics['gnn_l2_mean']:.6g} "
        f"± {l2_metrics['gnn_l2_std']:.6g}"
    )
    print(
        f"Relative L2 improvement = "
        f"{100.0 * l2_metrics['relative_l2_improvement']:.2f}%"
    )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'checkpoint.pt'}")
    print(f"  {OUT_DIR / 'metrics.json'}")
    print(f"  {OUT_DIR / 'predictions_test.csv'}")
    print(f"  {OUT_DIR / 'scatter_unmitigated_noisy_vs_ideal.png'}")
    print(f"  {OUT_DIR / 'scatter_gnn_mitigated_vs_ideal.png'}")
    print(f"  {OUT_DIR / 'l2_error_distribution_unmitigated_vs_gnn.png'}")
    print(f"  {OUT_DIR / 'l2_error_by_depth.png'}")
    print(f"  {OUT_DIR / 'loss_curves.png'}")
    print(f"  {OUT_DIR / 'scatter_unmitigated_vs_gnn_combined.png'}")


if __name__ == "__main__":
    main()
