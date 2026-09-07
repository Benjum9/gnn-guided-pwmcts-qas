from __future__ import annotations

import argparse
import csv
import hashlib
import os
import json
import math
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_delta import CircuitGNN
from graph_representation import BackendFeatureProvider, qasm_to_pyg_graph


PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PKL = (
    PROJECT_DIR
    / "data_generation_results_constrained_depth_8"
    / "depth_8_without_shallow_cxcx_density_target.pkl"
)
DEFAULT_OUT_DIR = PROJECT_DIR / "models_depth_8_direct_vs_loss_comparison"

TRAIN_GROUP_FRACTION = 0.70
VAL_GROUP_FRACTION = 0.15
TEST_GROUP_FRACTION = 0.15

MODEL_KWARGS = {
    "gnn_hidden": 64,
    "gnn_heads": 4,
    "global_hidden": 64,
    "reg_hidden": 128,
    "num_layers": 7,
    "dropout_rate": 0.2,
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_backend_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("fake", "").replace("_", "").replace("-", "").replace("v2", "")

    if "athens" in s:
        return "athens"
    if "bogota" in s:
        return "bogota"
    if "rome" in s:
        return "rome"
    if "santiago" in s:
        return "santiago"

    return s


def first_existing(meta: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return None


def infer_qasm(meta: Dict[str, Any]) -> Optional[str]:
    qasm = first_existing(
        meta,
        ["qasm_transpiled", "qasm", "qasm_str", "circuit_qasm", "transpiled_qasm"],
    )

    if isinstance(qasm, bytes):
        qasm = qasm.decode("utf-8")

    if isinstance(qasm, str) and qasm.strip():
        return qasm

    return None


def infer_noiseless_fidelity(meta: Dict[str, Any]) -> Optional[float]:
    for key in [
        "oracle_noiseless_fidelity",
        "noiseless_fidelity",
        "fidelity_noiseless",
        "ideal_fidelity",
    ]:
        value = meta.get(key)
        if value is None:
            continue

        try:
            value_float = float(value)
        except Exception:
            continue

        if math.isfinite(value_float):
            return value_float

    return None


def infer_noisy_fidelity(meta: Dict[str, Any], label: Any) -> Optional[float]:
    try:
        value = float(label)
        if math.isfinite(value):
            return value
    except Exception:
        pass

    for key in [
        "noisy_fidelity",
        "fidelity_noisy",
        "backend_noisy_fidelity",
        "target",
        "y",
    ]:
        value = meta.get(key)
        if value is None:
            continue

        try:
            value_float = float(value)
        except Exception:
            continue

        if math.isfinite(value_float):
            return value_float

    return None


def group_id_from_meta(meta: Dict[str, Any], qasm: str, idx: int) -> str:
    if meta.get("group_id") is not None:
        return str(meta["group_id"])

    if meta.get("group") is not None:
        return str(meta["group"])

    if "budget" in meta and "run_idx" in meta:
        return f"budget={meta['budget']}|run={meta['run_idx']}"

    if "circuit_idx" in meta:
        return f"circuit_idx={meta['circuit_idx']}"

    digest = hashlib.md5(qasm.encode("utf-8")).hexdigest()[:12]
    return f"qasm={digest}|idx={idx}"


def load_raw_items(data_pkl: Path, backend_name: str) -> List[Tuple[Dict[str, Any], float]]:
    wanted_backend = normalize_backend_name(backend_name)
    items: List[Tuple[Dict[str, Any], float]] = []

    with data_pkl.open("rb") as f:
        content = pickle.load(f)

    for idx, item in enumerate(content):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            meta = dict(item[0])
            label = item[1]
        elif isinstance(item, dict):
            meta = dict(item)
            label = None
        else:
            continue

        meta_backend = meta.get("backend", meta.get("backend_name"))
        if meta_backend is not None:
            if normalize_backend_name(str(meta_backend)) != wanted_backend:
                continue
        else:
            meta["backend"] = backend_name

        qasm = infer_qasm(meta)
        noiseless = infer_noiseless_fidelity(meta)
        noisy = infer_noisy_fidelity(meta, label)

        if qasm is None or noiseless is None or noisy is None:
            continue

        if not (-1e-9 <= noiseless <= 1.0 + 1e-9 and -1e-9 <= noisy <= 1.0 + 1e-9):
            continue

        noiseless = float(np.clip(noiseless, 0.0, 1.0))
        noisy = float(np.clip(noisy, 0.0, 1.0))

        meta["qasm_transpiled"] = qasm
        meta["oracle_noiseless_fidelity"] = noiseless
        meta["noisy_fidelity"] = noisy
        meta["group_id"] = group_id_from_meta(meta, qasm, idx)

        items.append((meta, noisy))

    if not items:
        raise RuntimeError(f"No usable samples found in {data_pkl}")

    return items


def split_group_ids(
    group_ids: List[str],
    seed: int,
    max_groups: Optional[int],
) -> Tuple[set, set, set]:
    unique_groups = sorted(set(group_ids))

    if len(unique_groups) < 3:
        raise RuntimeError(f"Need at least 3 groups, found {len(unique_groups)}")

    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    if max_groups is not None:
        unique_groups = unique_groups[: min(max_groups, len(unique_groups))]

    n_groups = len(unique_groups)
    n_train = max(1, int(round(TRAIN_GROUP_FRACTION * n_groups)))
    n_val = max(1, int(round(VAL_GROUP_FRACTION * n_groups)))
    n_test = n_groups - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    while n_train + n_val + n_test > n_groups:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    while n_train + n_val + n_test < n_groups:
        n_train += 1

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    return train_groups, val_groups, test_groups


def filter_items_by_group(
    items: List[Tuple[Dict[str, Any], float]],
    allowed_groups: set,
) -> List[Tuple[Dict[str, Any], float]]:
    return [(meta, y) for meta, y in items if str(meta["group_id"]) in allowed_groups]


def build_graphs(
    items: List[Tuple[Dict[str, Any], float]],
    backend_name: str,
    global_feature_variant: str,
) -> List[Data]:
    provider = BackendFeatureProvider(backend_name)
    graphs: List[Data] = []

    for idx, (meta, noisy) in enumerate(items):
        noiseless = float(meta["oracle_noiseless_fidelity"])

        graph, _ = qasm_to_pyg_graph(
            meta["qasm_transpiled"],
            global_feature_variant=global_feature_variant,
            backend_feature_provider=provider,
            noiseless_fidelity=noiseless,
            node_angle_encoding="none",
        )

        graph.y = torch.tensor([float(noisy)], dtype=torch.float)
        graph.noiseless_fidelity = torch.tensor([noiseless], dtype=torch.float)
        graph.group_id = str(meta["group_id"])
        graph.backend_name = backend_name
        graphs.append(graph)

        if (idx + 1) % 10000 == 0:
            print(f"Built {idx + 1}/{len(items)} graphs")

    return graphs


def noiseless_from_batch(batch) -> torch.Tensor:
    noiseless = batch.noiseless_fidelity

    if noiseless.dim() > 1:
        noiseless = noiseless.view(-1)

    return noiseless.to(batch.y.device).float()


def target_from_batch(batch, target_mode: str) -> torch.Tensor:
    noisy = batch.y.view(-1).float()

    if target_mode == "direct":
        return noisy

    if target_mode == "loss":
        return noiseless_from_batch(batch) - noisy

    raise ValueError(f"Unknown target_mode={target_mode}")


def predicted_noisy_from_raw(batch, pred_raw: torch.Tensor, target_mode: str) -> torch.Tensor:
    pred_raw = pred_raw.view(-1)

    if target_mode == "direct":
        return pred_raw.clamp(0.0, 1.0)

    if target_mode == "loss":
        return (noiseless_from_batch(batch) - pred_raw).clamp(0.0, 1.0)

    raise ValueError(f"Unknown target_mode={target_mode}")


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1

        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    spearman = float("nan")
    if len(y_true) > 1:
        true_rank = rankdata_average_ties(y_true)
        pred_rank = rankdata_average_ties(y_pred)
        spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])

    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "bias": float(np.mean(err)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def make_model(node_dim: int, global_dim: int) -> CircuitGNN:
    kwargs = dict(MODEL_KWARGS)
    kwargs["node_in_dim"] = int(node_dim)
    kwargs["global_in_dim"] = int(global_dim)
    return CircuitGNN(**kwargs)


def train_one_model(
    model_name: str,
    target_mode: str,
    train_graphs: List[Data],
    val_graphs: List[Data],
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_all_seeds(seed)

    node_dim = int(train_graphs[0].x.size(-1))
    global_dim = int(train_graphs[0].global_features.numel())
    model = make_model(node_dim=node_dim, global_dim=global_dim).to(device)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    min_delta = 1e-6

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            target = target_from_batch(batch, target_mode)
            mask = torch.isfinite(pred) & torch.isfinite(target)

            if not mask.any():
                continue

            loss = criterion(pred[mask], target[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(float(loss.detach().item()))

        model.eval()
        val_losses: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                target = target_from_batch(batch, target_mode)
                mask = torch.isfinite(pred) & torch.isfinite(target)

                if not mask.any():
                    continue

                loss = criterion(pred[mask], target[mask])
                val_losses.append(float(loss.detach().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "val_huber": val_loss,
            }
        )

        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{model_name}] Epoch {epoch:04d} | "
                f"train_huber={train_loss:.7f} | "
                f"val_huber={val_loss:.7f} | "
                f"best={best_val:.7f} | "
                f"bad={bad_epochs}"
            )

        if bad_epochs >= patience:
            print(f"[{model_name}] Early stopping at epoch {epoch}. best_val={best_val:.7f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "target_mode": target_mode,
        "best_val_huber": best_val,
        "epochs_run": len(history),
        "history": history,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    graphs: List[Data],
    target_mode: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()

    y_all: List[float] = []
    pred_noisy_all: List[float] = []
    raw_all: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        pred_raw = model(batch).view(-1)
        pred_noisy = predicted_noisy_from_raw(batch, pred_raw, target_mode)
        y = batch.y.view(-1).float()

        mask = torch.isfinite(y) & torch.isfinite(pred_noisy)
        if not mask.any():
            continue

        y_all.extend(y[mask].detach().cpu().tolist())
        pred_noisy_all.extend(pred_noisy[mask].detach().cpu().tolist())
        raw_all.extend(pred_raw[mask].detach().cpu().tolist())

    y_np = np.asarray(y_all, dtype=float)
    pred_np = np.asarray(pred_noisy_all, dtype=float)
    raw_np = np.asarray(raw_all, dtype=float)

    return compute_metrics(y_np, pred_np), y_np, pred_np, raw_np


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_huber", "val_huber"])
        writer.writeheader()
        writer.writerows(history)


def save_predictions_csv(
    y_true: np.ndarray,
    direct_pred: np.ndarray,
    direct_raw: np.ndarray,
    loss_pred: np.ndarray,
    loss_raw: np.ndarray,
    path: Path,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "true_noisy_fidelity",
                "direct_pred_noisy_fidelity",
                "direct_raw_output",
                "direct_error",
                "loss_pred_noisy_fidelity",
                "loss_raw_predicted_loss",
                "loss_error",
            ]
        )

        for yt, dp, dr, lp, lr in zip(y_true, direct_pred, direct_raw, loss_pred, loss_raw):
            writer.writerow([yt, dp, dr, dp - yt, lp, lr, lp - yt])


def draw_scatter(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    metrics: Dict[str, float],
) -> None:
    ax.scatter(y_true, y_pred, s=7, alpha=0.35, color="tab:blue", edgecolors="none")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="red", linestyle="--", linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Simulated noisy fidelity")
    ax.set_ylabel("Predicted noisy fidelity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    text = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"Spearman = {metrics['spearman']:.3f}"
    )

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def save_side_by_side_scatter(
    y_true: np.ndarray,
    direct_pred: np.ndarray,
    direct_metrics: Dict[str, float],
    loss_pred: np.ndarray,
    loss_metrics: Dict[str, float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)

    draw_scatter(
        axes[0],
        y_true,
        direct_pred,
        "(a) Direct noisy-fidelity prediction",
        direct_metrics,
    )
    draw_scatter(
        axes[1],
        y_true,
        loss_pred,
        "(b) Noise-loss prediction",
        loss_metrics,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_hexbin(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    metrics: Dict[str, float],
    gridsize: int,
):
    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=gridsize,
        extent=(0.0, 1.0, 0.0, 1.0),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="red", linestyle="--", linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Simulated noisy fidelity")
    ax.set_ylabel("Predicted noisy fidelity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    text = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"Spearman = {metrics['spearman']:.3f}"
    )

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    return hb


def save_side_by_side_hexbin(
    y_true: np.ndarray,
    direct_pred: np.ndarray,
    direct_metrics: Dict[str, float],
    loss_pred: np.ndarray,
    loss_metrics: Dict[str, float],
    path: Path,
    gridsize: int = 55,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    hb0 = draw_hexbin(
        axes[0],
        y_true,
        direct_pred,
        "(a) Direct noisy-fidelity prediction",
        direct_metrics,
        gridsize=gridsize,
    )
    hb1 = draw_hexbin(
        axes[1],
        y_true,
        loss_pred,
        "(b) Noise-loss prediction",
        loss_metrics,
        gridsize=gridsize,
    )

    fig.colorbar(hb0, ax=axes[0], label="Circuits per hexbin")
    fig.colorbar(hb1, ax=axes[1], label="Circuits per hexbin")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train direct noisy-fidelity and residual noise-loss GNNs on the same "
            "depth-10 split, then compare test scatter plots side by side."
        )
    )
    parser.add_argument("--data-pkl", type=Path, default=DEFAULT_DATA_PKL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend-name", default="athens")
    parser.add_argument("--global-feature-variant", default="new_baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--hexbin-gridsize", type=int, default=55)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional quick-test limit on the number of split groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Compare direct noisy-fidelity prediction and noise-loss prediction")
    print("=" * 80)
    print(f"Data PKL:              {args.data_pkl}")
    print(f"Output directory:      {out_dir}")
    print(f"Backend:               {args.backend_name}")
    print(f"Global feature variant:{args.global_feature_variant}")
    print(f"Seed:                  {args.seed}")
    print()

    items = load_raw_items(args.data_pkl, args.backend_name)
    print(f"Loaded samples: {len(items)}")

    groups = [str(meta["group_id"]) for meta, _ in items]
    train_groups, val_groups, test_groups = split_group_ids(groups, args.seed, args.max_groups)

    train_items = filter_items_by_group(items, train_groups)
    val_items = filter_items_by_group(items, val_groups)
    test_items = filter_items_by_group(items, test_groups)

    print(
        "Split sizes | "
        f"train={len(train_items)} | val={len(val_items)} | test={len(test_items)}"
    )

    print("Building train graphs...")
    train_graphs = build_graphs(train_items, args.backend_name, args.global_feature_variant)
    print("Building validation graphs...")
    val_graphs = build_graphs(val_items, args.backend_name, args.global_feature_variant)
    print("Building test graphs...")
    test_graphs = build_graphs(test_items, args.backend_name, args.global_feature_variant)

    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("One split has no graphs. Check the split or --max-groups.")

    node_dim = int(train_graphs[0].x.size(-1))
    global_dim = int(train_graphs[0].global_features.numel())
    print(f"Input dims | node_dim={node_dim} | global_dim={global_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nTraining direct noisy-fidelity model...")
    direct_model, direct_train_info = train_one_model(
        model_name="direct",
        target_mode="direct",
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    print("\nTraining noise-loss model...")
    loss_model, loss_train_info = train_one_model(
        model_name="loss",
        target_mode="loss",
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    direct_metrics, test_y, direct_pred, direct_raw = evaluate_model(
        direct_model,
        test_graphs,
        target_mode="direct",
        device=device,
        batch_size=args.batch_size,
    )
    loss_metrics, loss_y, loss_pred, loss_raw = evaluate_model(
        loss_model,
        test_graphs,
        target_mode="loss",
        device=device,
        batch_size=args.batch_size,
    )

    if len(test_y) != len(loss_y) or not np.allclose(test_y, loss_y):
        raise RuntimeError("The two evaluations did not use the same test labels.")

    save_side_by_side_scatter(
        y_true=test_y,
        direct_pred=direct_pred,
        direct_metrics=direct_metrics,
        loss_pred=loss_pred,
        loss_metrics=loss_metrics,
        path=out_dir / "test_scatter_direct_vs_loss.png",
    )
    save_side_by_side_hexbin(
        y_true=test_y,
        direct_pred=direct_pred,
        direct_metrics=direct_metrics,
        loss_pred=loss_pred,
        loss_metrics=loss_metrics,
        path=out_dir / "test_hexbin_direct_vs_loss.png",
        gridsize=args.hexbin_gridsize,
    )
    save_predictions_csv(
        y_true=test_y,
        direct_pred=direct_pred,
        direct_raw=direct_raw,
        loss_pred=loss_pred,
        loss_raw=loss_raw,
        path=out_dir / "test_predictions_direct_vs_loss.csv",
    )
    save_history_csv(direct_train_info["history"], out_dir / "direct_training_history.csv")
    save_history_csv(loss_train_info["history"], out_dir / "loss_training_history.csv")

    torch.save(direct_model.state_dict(), out_dir / "direct_state.pt")
    torch.save(loss_model.state_dict(), out_dir / "loss_state.pt")

    config = {
        "data_pkl": str(args.data_pkl),
        "backend_name": args.backend_name,
        "global_feature_variant": args.global_feature_variant,
        "node_angle_encoding": "none",
        "seed": args.seed,
        "max_groups": args.max_groups,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_kwargs": {
            **MODEL_KWARGS,
            "node_in_dim": node_dim,
            "global_in_dim": global_dim,
        },
        "split": {
            "train_group_fraction": TRAIN_GROUP_FRACTION,
            "val_group_fraction": VAL_GROUP_FRACTION,
            "test_group_fraction": TEST_GROUP_FRACTION,
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
            "train_samples": len(train_graphs),
            "val_samples": len(val_graphs),
            "test_samples": len(test_graphs),
        },
    }

    metrics = {
        "config": config,
        "direct_train_info": direct_train_info,
        "loss_train_info": loss_train_info,
        "direct_test_metrics": direct_metrics,
        "loss_test_metrics": loss_metrics,
    }

    with (out_dir / "comparison_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    for label, metric in [("direct", direct_metrics), ("loss", loss_metrics)]:
        print(
            f"{label:8s} | "
            f"MSE={metric['mse']:.7g} | "
            f"RMSE={metric['rmse']:.7g} | "
            f"MAE={metric['mae']:.7g} | "
            f"R2={metric['r2']:.4f} | "
            f"Spearman={metric['spearman']:.4f}"
        )

    print("\nSaved:")
    print(f"  {out_dir / 'test_scatter_direct_vs_loss.png'}")
    print(f"  {out_dir / 'test_hexbin_direct_vs_loss.png'}")
    print(f"  {out_dir / 'test_predictions_direct_vs_loss.csv'}")
    print(f"  {out_dir / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()
