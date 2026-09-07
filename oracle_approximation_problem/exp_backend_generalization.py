from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_representation import BackendFeatureProvider, qasm_to_pyg_graph
from gnn_delta import CircuitGNN


BACKEND_ORDER = ["athens", "bogota", "rome", "santiago"]
DEFAULT_TARGET_DEPTH = 10
DEFAULT_DATA_DIR = Path("data_generation_results_constrained_depth_10")
DEFAULT_OUT_ROOT = Path("models_lobo_depth_10_without_shallow_cxcx_delta_new_baseline")

TRAIN_GROUP_FRACTION = 0.70
VAL_GROUP_FRACTION = 0.15
TEST_GROUP_FRACTION = 0.15

DEFAULT_GLOBAL_FEATURE_VARIANT = "enhanced_backend"
NODE_ANGLE_ENCODING = "none"
NOISELESS_FIDELITY_INDEX = 8
SEED = 34

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
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_backend_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("fake", "").replace("-", "_").replace("v2", "")
    s = s.replace("__", "_").strip("_")

    for key in BACKEND_ORDER:
        if key in s:
            return key

    return s


def pkl_paths_for_data_dir(
    data_dir: Path,
    target_depth: int = DEFAULT_TARGET_DEPTH,
    dataset_suffix: str = "",
) -> Dict[str, Path]:
    stem = f"depth_{target_depth}_without_shallow_cxcx"

    return {
        "athens": data_dir / f"{stem}{dataset_suffix}.pkl",
        "bogota": data_dir / f"{stem}_bogota{dataset_suffix}.pkl",
        "rome": data_dir / f"{stem}_rome{dataset_suffix}.pkl",
        "santiago": data_dir / f"{stem}_santiago{dataset_suffix}.pkl",
    }


def first_existing(meta: dict, keys: Sequence[str]):
    for key in keys:
        if key in meta and meta[key] is not None:
            return meta[key]
    return None


def infer_qasm(meta: dict) -> Optional[str]:
    qasm = first_existing(
        meta,
        ["qasm_transpiled", "qasm", "qasm_str", "circuit_qasm", "transpiled_qasm"],
    )

    if isinstance(qasm, bytes):
        qasm = qasm.decode("utf-8")

    if isinstance(qasm, str) and qasm.strip():
        return qasm

    return None


def infer_noiseless_fidelity(meta: dict) -> Optional[float]:
    for key in [
        "oracle_noiseless_fidelity",
        "noiseless_fidelity",
        "fidelity_noiseless",
        "ideal_fidelity",
    ]:
        if key in meta and meta[key] is not None:
            try:
                value = float(meta[key])
            except Exception:
                continue
            if math.isfinite(value):
                return value
    return None


def infer_noisy_fidelity(meta: dict, label) -> Optional[float]:
    try:
        if label is not None:
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
        if key in meta and meta[key] is not None:
            try:
                value = float(meta[key])
            except Exception:
                continue
            if math.isfinite(value):
                return value

    return None


def load_pkl_items(pkl_path: Path, backend_name: str) -> List[Tuple[dict, float]]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing PKL: {pkl_path}")

    with pkl_path.open("rb") as handle:
        content = pickle.load(handle)

    out: List[Tuple[dict, float]] = []

    for item in content:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], dict):
            meta = dict(item[0])
            label = item[1]
        elif isinstance(item, dict):
            meta = dict(item)
            label = None
        else:
            continue

        qasm = infer_qasm(meta)
        noiseless = infer_noiseless_fidelity(meta)
        noisy = infer_noisy_fidelity(meta, label)

        if qasm is None or noiseless is None or noisy is None:
            continue

        if not math.isfinite(noiseless) or not math.isfinite(noisy):
            continue

        if not -1e-9 <= noiseless <= 1.0 + 1e-9:
            continue

        if not -1e-9 <= noisy <= 1.0 + 1e-9:
            continue

        meta["qasm_transpiled"] = qasm
        meta["oracle_noiseless_fidelity"] = float(min(max(noiseless, 0.0), 1.0))
        meta["noisy_fidelity"] = float(min(max(noisy, 0.0), 1.0))
        meta["backend"] = backend_name
        meta["backend_name"] = backend_name

        out.append((meta, meta["noisy_fidelity"]))

    return out


def group_id_from_meta(meta: dict):
    if "budget" in meta and "run_idx" in meta:
        return int(meta["budget"]), int(meta["run_idx"])

    if "group_id" in meta and meta["group_id"] is not None:
        return str(meta["group_id"])

    raise KeyError("Sample is missing group keys. Expected budget/run_idx or group_id.")


def split_group_ids(
    all_group_ids: List[object],
    seed: int,
    max_groups: Optional[int],
) -> Tuple[set, set, set]:
    uniq = list(sorted(set(all_group_ids)))

    if len(uniq) < 3:
        raise ValueError(f"Need at least 3 distinct groups, got {len(uniq)}")

    rng = random.Random(seed)
    rng.shuffle(uniq)

    if max_groups is not None:
        if max_groups < 3:
            raise ValueError("--max-groups must be at least 3.")
        uniq = uniq[: min(max_groups, len(uniq))]

    n = len(uniq)
    n_train = max(1, int(round(TRAIN_GROUP_FRACTION * n)))
    n_val = max(1, int(round(VAL_GROUP_FRACTION * n)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    train_groups = set(uniq[:n_train])
    val_groups = set(uniq[n_train : n_train + n_val])
    test_groups = set(uniq[n_train + n_val :])

    return train_groups, val_groups, test_groups


def filter_items_by_groups(
    items: List[Tuple[dict, float]],
    allowed_groups: set,
) -> List[Tuple[dict, float]]:
    return [(meta, y) for meta, y in items if group_id_from_meta(meta) in allowed_groups]


def build_graphs_from_items(
    items: List[Tuple[dict, float]],
    global_feature_variant: str,
) -> List[Data]:
    graphs: List[Data] = []
    provider_cache: Dict[str, BackendFeatureProvider] = {}

    for idx, (meta, y) in enumerate(items):
        qasm_t = meta.get("qasm_transpiled", None)

        if not isinstance(qasm_t, str):
            continue

        backend_name = normalize_backend_name(str(meta.get("backend", "")))

        if not backend_name:
            raise ValueError("Sample is missing a valid backend name.")

        if backend_name not in provider_cache:
            provider_cache[backend_name] = BackendFeatureProvider(backend_name)

        provider = provider_cache[backend_name]

        graph, _ = qasm_to_pyg_graph(
            qasm_t,
            global_feature_variant=global_feature_variant,
            backend_feature_provider=provider,
            noiseless_fidelity=meta.get("oracle_noiseless_fidelity", None),
            node_angle_encoding=NODE_ANGLE_ENCODING,
        )

        graph.y = torch.tensor([float(y)], dtype=torch.float)
        graph.backend_name = backend_name
        graph.group_id = str(group_id_from_meta(meta))

        graphs.append(graph)

        if (idx + 1) % 10000 == 0:
            print(f"Built {idx + 1}/{len(items)} graphs")

    return graphs


def get_noiseless_fidelity_from_batch(batch) -> torch.Tensor:
    g = batch.global_features

    if g.dim() == 1:
        g = g.view(batch.num_graphs, -1)

    return g[:, NOISELESS_FIDELITY_INDEX].view(-1).to(batch.y.device)


@torch.no_grad()
def collect_delta_predictions(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    ys = []
    preds = []

    for batch in loader:
        batch = batch.to(device)

        pred_loss = model(batch).view(-1)
        y_noisy = batch.y.view(-1)
        f_noiseless = get_noiseless_fidelity_from_batch(batch)
        pred_noisy = (f_noiseless - pred_loss).clamp(0.0, 1.0)

        mask = torch.isfinite(y_noisy) & torch.isfinite(pred_noisy)

        if mask.any():
            ys.append(y_noisy[mask].detach().cpu())
            preds.append(pred_noisy[mask].detach().cpu())

    if not ys:
        return torch.empty(0), torch.empty(0)

    return torch.cat(ys, dim=0), torch.cat(preds, dim=0)


def mse_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float(torch.mean((pred - y) ** 2).item())


def rmse_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    mse = mse_metric(y, pred)
    return float(math.sqrt(mse)) if math.isfinite(mse) else float("nan")


def mae_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(pred - y)).item())


def r2_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    if y.numel() == 0:
        return float("nan")

    y_mean = torch.mean(y)
    ss_res = torch.sum((pred - y) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2)

    if ss_tot <= 0:
        return 0.0

    return float((1.0 - ss_res / ss_tot).item())


def _rankdata_average_ties(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()

    if n == 0:
        return torch.empty(0, dtype=torch.float)

    values = x.detach().cpu().to(torch.float64)
    sorted_vals, sorted_idx = torch.sort(values)
    ranks_sorted = torch.empty(n, dtype=torch.float64)

    i = 0

    while i < n:
        j = i + 1
        while j < n and sorted_vals[j].item() == sorted_vals[i].item():
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = torch.empty(n, dtype=torch.float64)
    ranks[sorted_idx] = ranks_sorted

    return ranks.to(torch.float32)


def pearson_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    if y.numel() == 0 or pred.numel() == 0 or y.numel() != pred.numel():
        return float("nan")

    a = y.to(torch.float64)
    b = pred.to(torch.float64)
    a_centered = a - torch.mean(a)
    b_centered = b - torch.mean(b)
    denom = torch.sqrt(torch.sum(a_centered ** 2) * torch.sum(b_centered ** 2))

    if denom.item() <= 0.0:
        return 0.0

    return float((torch.sum(a_centered * b_centered) / denom).item())


def spearman_metric(y: torch.Tensor, pred: torch.Tensor) -> float:
    if y.numel() == 0 or pred.numel() == 0 or y.numel() != pred.numel():
        return float("nan")

    return pearson_metric(_rankdata_average_ties(y), _rankdata_average_ties(pred))


def compute_all_metrics(y: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    err = pred - y

    return {
        "n": int(y.numel()),
        "mse": mse_metric(y, pred),
        "rmse": rmse_metric(y, pred),
        "mae": mae_metric(y, pred),
        "r2": r2_metric(y, pred),
        "pearson": pearson_metric(y, pred),
        "spearman": spearman_metric(y, pred),
        "mean_error": float(torch.mean(err).item()) if err.numel() else float("nan"),
        "std_error": float(torch.std(err, unbiased=True).item()) if err.numel() > 1 else 0.0,
        "max_abs_error": float(torch.max(torch.abs(err)).item()) if err.numel() else float("nan"),
        "true_mean": float(torch.mean(y).item()) if y.numel() else float("nan"),
        "pred_mean": float(torch.mean(pred).item()) if pred.numel() else float("nan"),
        "true_std": float(torch.std(y, unbiased=True).item()) if y.numel() > 1 else 0.0,
        "pred_std": float(torch.std(pred, unbiased=True).item()) if pred.numel() > 1 else 0.0,
    }


def scatter_on_axis(
    ax,
    y: torch.Tensor,
    pred: torch.Tensor,
    title: str,
    metrics: Dict[str, float],
) -> None:
    ax.scatter(
        y.tolist(),
        pred.tolist(),
        s=7,
        alpha=0.35,
        color="tab:blue",
        edgecolors="none",
    )

    ax.plot([0.0, 1.0], [0.0, 1.0], color="red", linestyle="--", linewidth=2)

    text = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MSE = {metrics['mse']:.4g}\n"
        f"Spearman = {metrics['spearman']:.3f}\n"
        f"MAE = {metrics['mae']:.4g}"
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

    ax.set_xlabel("True noisy fidelity")
    ax.set_ylabel("Predicted noisy fidelity")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect(1)


def save_single_scatter(
    y: torch.Tensor,
    pred: torch.Tensor,
    metrics: Dict[str, float],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter_on_axis(ax, y, pred, title, metrics)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_combined_scatter(
    fold_outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]],
    out_path: Path,
) -> None:
    titles = {
        "athens": "(a) Held-out FakeAthensV2",
        "bogota": "(b) Held-out FakeBogotaV2",
        "rome": "(c) Held-out FakeRomeV2",
        "santiago": "(d) Held-out FakeSantiagoV2",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    axes = axes.flatten()

    for ax, backend_name in zip(axes, BACKEND_ORDER):
        if backend_name not in fold_outputs:
            ax.axis("off")
            continue

        y, pred, metrics = fold_outputs[backend_name]
        scatter_on_axis(ax, y, pred, titles[backend_name], metrics)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_loss_curves(
    train_loss_history: List[float],
    val_loss_history: List[float],
    out_path: Path,
    title: str,
) -> None:
    if not train_loss_history or not val_loss_history:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(train_loss_history, label="Train loss")
    plt.plot(val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def train_lobo_fold(
    heldout_backend: str,
    train_graphs: List[Data],
    val_graphs: List[Data],
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[CircuitGNN, Dict[str, float]]:
    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    model_kwargs = dict(MODEL_KWARGS)
    model_kwargs["node_in_dim"] = int(train_graphs[0].x.size(-1))
    model_kwargs["global_in_dim"] = int(train_graphs[0].global_features.numel())

    model = CircuitGNN(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            batch = batch.to(device)

            pred_loss = model(batch).view(-1)
            y_noisy = batch.y.view(-1)
            f_noiseless = get_noiseless_fidelity_from_batch(batch)
            true_loss = f_noiseless - y_noisy

            mask = torch.isfinite(pred_loss) & torch.isfinite(true_loss)

            if mask.sum() == 0:
                continue

            loss = loss_fn(pred_loss[mask], true_loss[mask])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_losses.append(float(loss.detach().item()))

        model.eval()
        val_losses: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                pred_loss = model(batch).view(-1)
                y_noisy = batch.y.view(-1)
                f_noiseless = get_noiseless_fidelity_from_batch(batch)
                true_loss = f_noiseless - y_noisy

                mask = torch.isfinite(pred_loss) & torch.isfinite(true_loss)

                if mask.sum() == 0:
                    continue

                loss = loss_fn(pred_loss[mask], true_loss[mask])
                val_losses.append(float(loss.detach().item()))

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if val_loss + args.min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep == 1 or ep % 10 == 0:
            print(
                f"[holdout_{heldout_backend}] "
                f"Epoch {ep:4d} | "
                f"train_delta_huber={train_loss:.7f} | "
                f"val_delta_huber={val_loss:.7f} | "
                f"best={best_val:.7f} | "
                f"bad={bad_epochs}"
            )

        if bad_epochs >= args.patience:
            print(f"[holdout_{heldout_backend}] Early stopping at epoch {ep}. best_val={best_val:.7f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "state_dict.pt")

    checkpoint = {
        "task": "leave_one_backend_out_depth_10_delta",
        "heldout_backend": heldout_backend,
        "global_feature_variant": args.global_feature_variant,
        "node_angle_encoding": NODE_ANGLE_ENCODING,
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": SEED,
        "best_val_huber": best_val,
        "target_mode": "delta_loss",
        "delta_definition": "oracle_noiseless_fidelity - noisy_fidelity",
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")

    save_loss_curves(
        train_loss_history,
        val_loss_history,
        out_dir / "loss_curves.png",
        f"Training curves - holdout {heldout_backend}",
    )

    return model, {
        "best_val_huber": best_val,
        "epochs_run": len(train_loss_history),
        "model_kwargs": model_kwargs,
    }


def evaluate_fold(
    model: CircuitGNN,
    test_graphs: List[Data],
    out_dir: Path,
    heldout_backend: str,
    batch_size: int,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    y, pred = collect_delta_predictions(model, loader, device)
    metrics = compute_all_metrics(y, pred)

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    with (out_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["true_noisy_fidelity", "predicted_noisy_fidelity", "error"],
        )
        writer.writeheader()

        for yy, pp in zip(y.tolist(), pred.tolist()):
            writer.writerow(
                {
                    "true_noisy_fidelity": yy,
                    "predicted_noisy_fidelity": pp,
                    "error": pp - yy,
                }
            )

    save_single_scatter(
        y,
        pred,
        metrics,
        out_dir / "scatter.png",
        f"Held-out {heldout_backend}",
    )

    return metrics, y, pred


def parse_backend_list(value: str) -> List[str]:
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in BACKEND_ORDER]

    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown backend(s): {unknown}. Use {BACKEND_ORDER}.")

    if not names:
        raise argparse.ArgumentTypeError("At least one backend is required.")

    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-backend-out generalization for constrained oracle datasets."
    )
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-suffix", type=str, default="")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--folds", type=parse_backend_list, default=list(BACKEND_ORDER))
    parser.add_argument("--global-feature-variant", default=DEFAULT_GLOBAL_FEATURE_VARIANT)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional small subset for smoke tests. Full experiment uses all groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(SEED)
    args.out_root.mkdir(parents=True, exist_ok=True)

    pkl_paths = pkl_paths_for_data_dir(
        args.data_dir,
        args.target_depth,
        args.dataset_suffix,
    )

    print("=" * 80)
    print(f"Depth-{args.target_depth} leave-one-backend-out GNN training")
    print("=" * 80)
    print(f"DATA_DIR:                {args.data_dir}")
    print(f"TARGET_DEPTH:            {args.target_depth}")
    print(f"DATASET_SUFFIX:          {args.dataset_suffix}")
    print(f"OUT_ROOT:                {args.out_root}")
    print(f"GLOBAL_FEATURE_VARIANT:  {args.global_feature_variant}")
    print(f"NODE_ANGLE_ENCODING:     {NODE_ANGLE_ENCODING}")
    print(f"FOLDS:                   {args.folds}")
    print(f"EPOCHS:                  {args.epochs}")
    print(f"PATIENCE:                {args.patience}")
    print(f"BATCH_SIZE:              {args.batch_size}")
    print(f"MAX_GROUPS:              {args.max_groups}")
    print()

    backend_items: Dict[str, List[Tuple[dict, float]]] = {}

    for backend_name in BACKEND_ORDER:
        items = load_pkl_items(pkl_paths[backend_name], backend_name)
        backend_items[backend_name] = items
        print(f"Loaded {backend_name:8s}: {len(items)} items from {pkl_paths[backend_name]}")

    reference_groups = [group_id_from_meta(meta) for meta, _ in backend_items["athens"]]
    train_groups, val_groups, test_groups = split_group_ids(reference_groups, SEED, args.max_groups)

    print()
    print(
        f"Distinct groups | train={len(train_groups)} | "
        f"val={len(val_groups)} | test={len(test_groups)}"
    )

    summary_rows: List[Dict[str, object]] = []
    fold_outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]] = {}

    for heldout_backend in args.folds:
        print("\n" + "=" * 80)
        print(f"LOBO fold: held-out backend = {heldout_backend}")
        print("=" * 80)

        train_backends = [backend for backend in BACKEND_ORDER if backend != heldout_backend]

        train_items: List[Tuple[dict, float]] = []
        val_items: List[Tuple[dict, float]] = []
        test_items: List[Tuple[dict, float]] = []

        for backend_name in train_backends:
            train_items.extend(filter_items_by_groups(backend_items[backend_name], train_groups))
            val_items.extend(filter_items_by_groups(backend_items[backend_name], val_groups))

        test_items.extend(filter_items_by_groups(backend_items[heldout_backend], test_groups))

        print(
            f"Fold {heldout_backend} sample counts | "
            f"train={len(train_items)} | "
            f"val={len(val_items)} | "
            f"test={len(test_items)}"
        )

        print("Building train graphs...")
        train_graphs = build_graphs_from_items(train_items, args.global_feature_variant)

        print("Building validation graphs...")
        val_graphs = build_graphs_from_items(val_items, args.global_feature_variant)

        print("Building test graphs...")
        test_graphs = build_graphs_from_items(test_items, args.global_feature_variant)

        if not train_graphs or not val_graphs or not test_graphs:
            raise RuntimeError(f"Fold {heldout_backend}: one split has no valid graphs.")

        print(
            f"Fold {heldout_backend} graph counts | "
            f"train={len(train_graphs)} | "
            f"val={len(val_graphs)} | "
            f"test={len(test_graphs)} | "
            f"node_dim={train_graphs[0].x.size(-1)} | "
            f"global_dim={train_graphs[0].global_features.numel()}"
        )

        fold_out = args.out_root / f"holdout_{heldout_backend}"

        model, train_info = train_lobo_fold(
            heldout_backend=heldout_backend,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            out_dir=fold_out,
            args=args,
        )

        metrics, y_test, pred_test = evaluate_fold(
            model=model,
            test_graphs=test_graphs,
            out_dir=fold_out,
            heldout_backend=heldout_backend,
            batch_size=args.batch_size,
        )

        fold_outputs[heldout_backend] = (y_test, pred_test, metrics)

        row = {
            "heldout_backend": heldout_backend,
            "train_backends": ",".join(train_backends),
            "global_feature_variant": args.global_feature_variant,
            "node_angle_encoding": NODE_ANGLE_ENCODING,
            "node_dim": int(train_graphs[0].x.size(-1)),
            "global_dim": int(train_graphs[0].global_features.numel()),
            "best_val_huber": float(train_info["best_val_huber"]),
            "epochs_run": int(train_info["epochs_run"]),
            **metrics,
        }
        summary_rows.append(row)

        print(
            f"[holdout_{heldout_backend}] "
            f"MSE={metrics['mse']:.6f} | "
            f"RMSE={metrics['rmse']:.6f} | "
            f"MAE={metrics['mae']:.6f} | "
            f"R2={metrics['r2']:.4f} | "
            f"Pearson={metrics['pearson']:.4f} | "
            f"Spearman={metrics['spearman']:.4f}"
        )

    summary_csv = args.out_root / "lobo_summary.csv"
    fieldnames = [
        "heldout_backend",
        "train_backends",
        "global_feature_variant",
        "node_angle_encoding",
        "node_dim",
        "global_dim",
        "best_val_huber",
        "epochs_run",
        "n",
        "mse",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
        "mean_error",
        "std_error",
        "max_abs_error",
        "true_mean",
        "pred_mean",
        "true_std",
        "pred_std",
    ]

    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.out_root / "lobo_summary.json").open("w") as handle:
        json.dump(summary_rows, handle, indent=2)

    with (args.out_root / "config.json").open("w") as handle:
        json.dump(
            {
                "data_dir": str(args.data_dir),
                "target_depth": args.target_depth,
                "pkl_paths": {k: str(v) for k, v in pkl_paths.items()},
                "out_root": str(args.out_root),
                "folds": args.folds,
                "global_feature_variant": args.global_feature_variant,
                "node_angle_encoding": NODE_ANGLE_ENCODING,
                "target_mode": "delta_loss",
                "delta_definition": "oracle_noiseless_fidelity - noisy_fidelity",
                "reconstruction": "pred_noisy = oracle_noiseless_fidelity - pred_delta",
                "seed": SEED,
                "train_group_fraction": TRAIN_GROUP_FRACTION,
                "val_group_fraction": VAL_GROUP_FRACTION,
                "test_group_fraction": TEST_GROUP_FRACTION,
                "max_groups": args.max_groups,
                "model_kwargs": MODEL_KWARGS,
            },
            handle,
            indent=2,
        )

    combined_scatter = args.out_root / "scatter_lobo_delta.png"
    save_combined_scatter(fold_outputs, combined_scatter)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved LOBO summary to: {summary_csv}")
    print(f"Saved combined scatter to: {combined_scatter}")
    print()

    for row in summary_rows:
        print(
            f"holdout_{row['heldout_backend']:9s} | "
            f"MSE={float(row['mse']):.6f} | "
            f"RMSE={float(row['rmse']):.6f} | "
            f"MAE={float(row['mae']):.6f} | "
            f"R2={float(row['r2']):.4f} | "
            f"Pearson={float(row['pearson']):.4f} | "
            f"Spearman={float(row['spearman']):.4f}"
        )


if __name__ == "__main__":
    main()
