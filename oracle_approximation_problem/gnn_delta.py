from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

try:
    from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
    _AMP_DEVICE_TYPE = "cuda"
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_DEVICE_TYPE = "cuda"


# ------------------------------------------------------------
# Imports graph representation
# ------------------------------------------------------------
try:
    from graph_representation import (
        QuantumCircuitGraphDataset,
        get_node_feature_dim,
        get_global_feature_dim,
    )
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = Path(__file__).resolve().parent
    for p in (project_root, models_dir):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    from graph_representation import (
        QuantumCircuitGraphDataset,
        get_node_feature_dim,
        get_global_feature_dim,
    )


# =========================================
# Data utils
# =========================================

def _cache_root_for_paths(paths: List[str], suffix: str = "") -> str:
    canonical = "|".join(sorted(os.path.abspath(p) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    return os.path.join(os.getcwd(), f"pyg_cache_{digest}{tag}")


def _node_feature_dim(
    node_feature_backend_variant: Optional[str],
    node_angle_encoding: str = "sin_cos",
) -> int:
    """
    Compatible with graph_representation versions with or without node_angle_encoding.
    """
    try:
        return get_node_feature_dim(node_feature_backend_variant, node_angle_encoding)
    except TypeError:
        return get_node_feature_dim(node_feature_backend_variant)


def _make_dataset(
    root: str,
    pkl_paths: List[str],
    global_feature_variant: str,
    node_feature_backend_variant: Optional[str],
    node_angle_encoding: str = "sin_cos",
) -> QuantumCircuitGraphDataset:
    """
    Compatible with graph_representation versions with or without node_angle_encoding.
    """
    try:
        return QuantumCircuitGraphDataset(
            root=root,
            pkl_paths=pkl_paths,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
            node_angle_encoding=node_angle_encoding,
        )
    except TypeError:
        return QuantumCircuitGraphDataset(
            root=root,
            pkl_paths=pkl_paths,
            global_feature_variant=global_feature_variant,
            node_feature_backend_variant=node_feature_backend_variant,
        )


# =========================================
# Logit helpers
# =========================================

def _to_logit(y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)


def _from_logit(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


# =========================================
# Target helpers
# =========================================

TARGET_MODES = {"noisy_fidelity", "noise_loss"}


def _global_features_2d(batch) -> Tensor:
    g_raw = batch.global_features

    if g_raw.dim() == 1:
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            num_graphs = batch.y.view(-1).numel()
        g_raw = g_raw.view(int(num_graphs), -1)

    return g_raw.float()


def _noiseless_fidelity_index(global_dim: int, global_feature_variant: str) -> int:
    """
    For add_noiseless/new_baseline/baseline_backend:
      no-id version:
        [depth, num_param, num_qubits, total_gates, rz, sx, x, cx, noiseless_fidelity, backend5]
        => index 8, dim 14

      older id version:
        [depth, num_param, num_qubits, total_gates, rz, sx, x, cx, id, noiseless_fidelity, backend5]
        => index 9, dim 15

    enhanced_backend starts with the same new_baseline vector, so the noiseless
    fidelity is still stored at index 8.
    """
    v = (global_feature_variant or "").strip().lower()

    if v not in {
        "add_noiseless",
        "noiseless_baseline",
        "new_baseline",
        "baseline_backend",
        "enhanced_backend",
    }:
        raise ValueError(
            "target_mode='noise_loss' requires global features that include noiseless fidelity. "
            "Use global_feature_variant='add_noiseless', 'new_baseline', "
            "'baseline_backend', or 'enhanced_backend'."
        )

    if global_dim == 15:
        return 9

    if global_dim >= 9:
        return 8

    raise ValueError(f"Cannot infer noiseless fidelity index for global_dim={global_dim}")


def _get_noiseless_fidelity_from_batch(batch, global_feature_variant: str) -> Tensor:
    g = _global_features_2d(batch)
    idx = _noiseless_fidelity_index(g.size(1), global_feature_variant)
    return g[:, idx].view(-1).to(batch.y.device)


def _training_target(
    batch,
    target_mode: str,
    global_feature_variant: str,
) -> Tensor:
    """
    Returns target used by the loss.

    target_mode='noisy_fidelity':
        target = F_noisy

    target_mode='noise_loss':
        target = F_noiseless - F_noisy
    """
    tm = (target_mode or "noisy_fidelity").strip().lower()

    if tm not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode={target_mode}. Use one of {TARGET_MODES}")

    y_noisy = batch.y.view(-1)

    if tm == "noisy_fidelity":
        return y_noisy

    f_noiseless = _get_noiseless_fidelity_from_batch(batch, global_feature_variant)
    return f_noiseless - y_noisy


def _prediction_to_noisy_fidelity(
    batch,
    pred_raw: Tensor,
    target_mode: str,
    global_feature_variant: str,
    pred_is_logit: bool = False,
) -> Tensor:
    """
    Converts model output to predicted noisy fidelity for evaluation.

    If target_mode='noisy_fidelity':
        model output predicts F_noisy directly, or logit(F_noisy).

    If target_mode='noise_loss':
        model output predicts L_noise = F_noiseless - F_noisy.
        Then F_noisy_hat = F_noiseless - L_noise_hat.
    """
    tm = (target_mode or "noisy_fidelity").strip().lower()

    if tm == "noisy_fidelity":
        pred = _from_logit(pred_raw) if pred_is_logit else pred_raw
        return pred.clamp(0.0, 1.0)

    if tm == "noise_loss":
        pred_loss = _from_logit(pred_raw) if pred_is_logit else pred_raw
        f_noiseless = _get_noiseless_fidelity_from_batch(batch, global_feature_variant)
        pred_noisy = f_noiseless - pred_loss
        return pred_noisy.clamp(0.0, 1.0)

    raise ValueError(f"Unknown target_mode={target_mode}")


def _check_target_mode(target_mode: str, use_logit_target: bool) -> None:
    tm = (target_mode or "noisy_fidelity").strip().lower()

    if tm not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode={target_mode}. Use one of {TARGET_MODES}")

    if tm == "noise_loss" and use_logit_target:
        raise ValueError(
            "use_logit_target=True is not recommended for target_mode='noise_loss', "
            "because F_noiseless - F_noisy can be zero or occasionally negative. "
            "Use use_logit_target=False."
        )


# =========================================
# Architecture
# =========================================

GNN_HIDDEN = 32
GNN_HEADS = 8
GLOBAL_HIDDEN = 16
REG_HIDDEN = 16
NUM_LAYERS = 5


class GlobalMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        dr = float(dropout_rate) if dropout_rate is not None else 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
        )

    def forward(self, g: Tensor) -> Tensor:
        return self.net(g)


class RegressorHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.0):
        super().__init__()
        dr = float(dropout_rate) if dropout_rate is not None else 0.0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class CircuitGNN(nn.Module):
    """
    GNN that outputs a single scalar per graph.

    Interpretation depends on target_mode:
      - target_mode='noisy_fidelity': output predicts noisy fidelity directly.
      - target_mode='noise_loss': output predicts noise-induced fidelity loss:
            L_noise = F_noiseless - F_noisy.
    """

    def __init__(
        self,
        node_in_dim: int = 13,
        gnn_hidden: int = GNN_HIDDEN,
        gnn_heads: int = GNN_HEADS,
        global_in_dim: int = 8,
        global_hidden: int = GLOBAL_HIDDEN,
        reg_hidden: int = REG_HIDDEN,
        num_layers: int = NUM_LAYERS,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        from torch_geometric.nn import TransformerConv, global_mean_pool

        self.global_mean_pool = global_mean_pool

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = int(num_layers)
        self.gnn_hidden = int(gnn_hidden)
        self.gnn_heads = int(gnn_heads)
        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else 0.0

        convs = []
        convs.append(
            TransformerConv(
                node_in_dim,
                self.gnn_hidden,
                heads=self.gnn_heads,
                dropout=self.dropout_rate,
                beta=False,
            )
        )

        for _ in range(1, self.num_layers):
            convs.append(
                TransformerConv(
                    self.gnn_hidden * self.gnn_heads,
                    self.gnn_hidden,
                    heads=self.gnn_heads,
                    dropout=self.dropout_rate,
                    beta=False,
                )
            )

        self.conv_layers = nn.ModuleList(convs)

        self.global_mlp = GlobalMLP(global_in_dim, global_hidden, dropout_rate=self.dropout_rate)
        concat_dim = self.gnn_hidden * self.gnn_heads + global_hidden
        self.regressor = RegressorHead(concat_dim, reg_hidden, dropout_rate=self.dropout_rate)

    def forward(self, data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if x.size(0) == 0:
            num_graphs = getattr(data, "num_graphs", 1)
            x_pool = torch.zeros(
                (num_graphs, self.gnn_hidden * self.gnn_heads),
                device=x.device,
                dtype=torch.float32,
            )
        else:
            with autocast(device_type="cuda", enabled=False):
                h = x.float()
                for conv in self.conv_layers:
                    h = conv(h, edge_index)
                    h = F.relu(h)
                    if self.dropout_rate > 0.0:
                        h = F.dropout(h, p=self.dropout_rate, training=self.training)
                x_pool = self.global_mean_pool(h, batch)

        g_raw = data.global_features
        if g_raw.dim() == 1:
            num_graphs = getattr(
                data,
                "num_graphs",
                int(batch.max().item()) + 1 if batch.numel() > 0 else 1,
            )
            g_raw = g_raw.view(num_graphs, -1)

        g_feat = self.global_mlp(g_raw.float())

        if x_pool.size(0) < g_feat.size(0):
            pad_rows = g_feat.size(0) - x_pool.size(0)
            pad = torch.zeros((pad_rows, x_pool.size(1)), device=x_pool.device, dtype=x_pool.dtype)
            x_pool = torch.cat([x_pool, pad], dim=0)
        elif x_pool.size(0) > g_feat.size(0):
            pad_rows = x_pool.size(0) - g_feat.size(0)
            pad = torch.zeros((pad_rows, g_feat.size(1)), device=g_feat.device, dtype=g_feat.dtype)
            g_feat = torch.cat([g_feat, pad], dim=0)

        h = torch.cat([x_pool, g_feat], dim=-1)
        out = self.regressor(h)
        return out.view(-1)


# =========================================
# Loss helper
# =========================================

def _make_criterion(loss_type: str) -> nn.Module:
    lt = (loss_type or "huber").strip().lower()

    if lt in ("mse", "mse_loss", "mean_squared_error"):
        return nn.MSELoss()

    return nn.HuberLoss()


def _autocast_ctx(device: torch.device):
    try:
        return autocast(device_type=device.type, enabled=(device.type == "cuda"))
    except TypeError:
        return autocast(enabled=(device.type == "cuda"))


# =========================================
# Loaders
# =========================================

def build_train_test_loaders(
    pkl_paths: List[str],
    train_split: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
) -> Tuple[DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(dataset) < 2:
        raise RuntimeError("Dataset too small to split. Check PKLs.")

    train_len = max(1, int(len(dataset) * train_split))
    train_len = min(len(dataset) - 1, train_len)
    test_len = len(dataset) - train_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_len, test_len], generator=generator)

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


def build_full_loader(
    pkl_paths: List[str],
    batch_size: int = 64,
    global_feature_variant: str = "binned152",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
) -> DataLoader:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PKL paths and formats.")

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)


def build_train_val_test_loaders_two_stage(
    pkl_paths: List[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    generator = torch.Generator().manual_seed(seed)

    primary_train_len = max(1, int(len(dataset) * train_split))
    primary_train_len = min(len(dataset) - 1, primary_train_len)
    test_len = len(dataset) - primary_train_len

    primary_train, test_ds = random_split(dataset, [primary_train_len, test_len], generator=generator)

    val_len = max(1, int(len(primary_train) * val_within_train))
    val_len = min(len(primary_train) - 1, val_len)
    real_train_len = len(primary_train) - val_len

    train_ds, val_ds = random_split(primary_train, [real_train_len, val_len], generator=generator)

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


def build_train_val_test_loaders_two_stage_strat(
    pkl_paths: List[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    global_feature_variant: str = "baseline",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
    split_mode: str = "stratified",
    n_strat_bins: int = 10,
    group_attr: str = "group",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    suffix = f"{global_feature_variant}_backend_{node_feature_backend_variant or 'none'}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(dataset) < 3:
        raise RuntimeError("Dataset too small for train/val/test splitting.")

    def _get_y(i: int) -> float:
        y = getattr(dataset[i], "y", None)
        if y is None:
            return float("nan")
        try:
            return float(y.view(-1)[0].item())
        except Exception:
            return float("nan")

    def _random_indices(n: int):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g).tolist()

        n_primary_train = max(1, int(n * train_split))
        n_primary_train = min(n - 1, n_primary_train)

        primary_train_idx = idx[:n_primary_train]
        test_idx = idx[n_primary_train:]

        n_val = max(1, int(len(primary_train_idx) * val_within_train))
        n_val = min(len(primary_train_idx) - 1, n_val) if len(primary_train_idx) >= 2 else 0

        val_idx = primary_train_idx[:n_val]
        train_idx = primary_train_idx[n_val:]

        return train_idx, val_idx, test_idx

    def _stratified_indices():
        rng = np.random.default_rng(seed)
        y = np.array([_get_y(i) for i in range(len(dataset))], dtype=float)
        finite_mask = np.isfinite(y)

        if finite_mask.sum() < 3:
            return _random_indices(len(dataset))

        y_f = y[finite_mask]
        idx_f = np.where(finite_mask)[0]

        y_min = float(np.min(y_f))
        y_max = float(np.max(y_f))

        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            return _random_indices(len(dataset))

        edges = np.linspace(y_min, y_max, n_strat_bins + 1)
        bin_id = np.digitize(y_f, edges[1:-1], right=False)

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for b in range(n_strat_bins):
            in_bin = np.where(bin_id == b)[0]
            if in_bin.size == 0:
                continue

            bin_indices = idx_f[in_bin]
            rng.shuffle(bin_indices)

            n_total = bin_indices.size
            n_train = int(round(train_split * n_total))

            if n_total >= 2:
                n_train = max(1, min(n_total - 1, n_train))
            else:
                n_train = n_total

            primary_train = bin_indices[:n_train]
            test_part = bin_indices[n_train:]

            n_val = int(round(val_within_train * len(primary_train)))
            if len(primary_train) >= 2:
                n_val = max(1, min(len(primary_train) - 1, n_val))
            else:
                n_val = 0

            val_part = primary_train[:n_val]
            train_part = primary_train[n_val:]

            train_idx.extend(train_part.tolist())
            val_idx.extend(val_part.tolist())
            test_idx.extend(test_part.tolist())

        if len(train_idx) < 1 or len(val_idx) < 1 or len(test_idx) < 1:
            return _random_indices(len(dataset))

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        return train_idx, val_idx, test_idx

    def _group_indices():
        rng = np.random.default_rng(seed)
        groups = []

        for i in range(len(dataset)):
            g = getattr(dataset[i], group_attr, None)
            groups.append("unknown" if g is None else str(g))

        groups = np.asarray(groups)
        uniq = np.unique(groups)

        if uniq.size < 3:
            return _random_indices(len(dataset))

        rng.shuffle(uniq)

        n_groups_train = int(round(train_split * uniq.size))
        n_groups_train = max(1, min(uniq.size - 1, n_groups_train))

        train_groups = set(uniq[:n_groups_train].tolist())
        test_groups = set(uniq[n_groups_train:].tolist())

        primary_train_idx = np.where(np.isin(groups, list(train_groups)))[0]
        test_idx = np.where(np.isin(groups, list(test_groups)))[0]

        uniq_train = np.unique(groups[primary_train_idx])

        if uniq_train.size < 2:
            return _random_indices(len(dataset))

        rng.shuffle(uniq_train)

        n_groups_val = int(round(val_within_train * uniq_train.size))
        n_groups_val = max(1, min(uniq_train.size - 1, n_groups_val))

        val_groups = set(uniq_train[:n_groups_val].tolist())
        train_groups2 = set(uniq_train[n_groups_val:].tolist())

        val_idx = primary_train_idx[np.isin(groups[primary_train_idx], list(val_groups))]
        train_idx = primary_train_idx[np.isin(groups[primary_train_idx], list(train_groups2))]

        if train_idx.size < 1 or val_idx.size < 1 or test_idx.size < 1:
            return _random_indices(len(dataset))

        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    sm = (split_mode or "random").strip().lower()

    if sm == "stratified":
        train_idx, val_idx, test_idx = _stratified_indices()
    elif sm == "group":
        train_idx, val_idx, test_idx = _group_indices()
    else:
        train_idx, val_idx, test_idx = _random_indices(len(dataset))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


# =========================================
# Evaluation in noisy fidelity space
# =========================================

@torch.no_grad()
def evaluate_overall_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_is_logit: bool = False,
    target_mode: str = "noisy_fidelity",
    global_feature_variant: str = "baseline_backend",
) -> float:
    model.eval()

    total_se = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        with _autocast_ctx(device):
            pred_raw = model(batch).view(-1)
            pred_noisy = _prediction_to_noisy_fidelity(
                batch,
                pred_raw,
                target_mode=target_mode,
                global_feature_variant=global_feature_variant,
                pred_is_logit=pred_is_logit,
            )

            y_noisy = batch.y.view(-1)

            mask = torch.isfinite(y_noisy) & torch.isfinite(pred_noisy)
            if mask.sum() == 0:
                continue

            se = torch.sum((pred_noisy[mask] - y_noisy[mask]) ** 2).item()

        total_se += se
        total_n += int(mask.sum().item())

    return total_se / max(1, total_n)


@torch.no_grad()
def evaluate_overall_r2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_is_logit: bool = False,
    target_mode: str = "noisy_fidelity",
    global_feature_variant: str = "baseline_backend",
) -> float:
    model.eval()

    y_all: List[float] = []
    yhat_all: List[float] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        with _autocast_ctx(device):
            pred_raw = model(batch).view(-1)
            pred_noisy = _prediction_to_noisy_fidelity(
                batch,
                pred_raw,
                target_mode=target_mode,
                global_feature_variant=global_feature_variant,
                pred_is_logit=pred_is_logit,
            )

            y_noisy = batch.y.view(-1)

            mask = torch.isfinite(y_noisy) & torch.isfinite(pred_noisy)
            if mask.sum() == 0:
                continue

            y_all.extend(y_noisy[mask].detach().cpu().tolist())
            yhat_all.extend(pred_noisy[mask].detach().cpu().tolist())

    if not y_all:
        return 0.0

    y_mean = float(sum(y_all) / len(y_all))
    ss_res = float(sum((yh - y) ** 2 for yh, y in zip(yhat_all, y_all)))
    ss_tot = float(sum((y - y_mean) ** 2 for y in y_all))

    if ss_tot <= 0.0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


# =========================================
# Training
# =========================================

def train_with_two_stage_split(
    pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    global_feature_variant: str = "baseline_backend",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    loss_type: str = "huber",
    use_logit_target: bool = False,
    logit_eps: float = 1e-4,
    target_mode: str = "noise_loss",
):
    _check_target_mode(target_mode, use_logit_target)

    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        pkl_paths,
        train_split=train_split,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    gdim = get_global_feature_dim(global_feature_variant)
    node_in_dim = _node_feature_dim(node_feature_backend_variant, node_angle_encoding)

    model_args = dict(global_in_dim=gdim, node_in_dim=node_in_dim)

    if model_kwargs:
        model_args.update(model_kwargs)
        model_args.setdefault("global_in_dim", gdim)
        model_args.setdefault("node_in_dim", node_in_dim)

    model = CircuitGNN(**model_args)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(dev)

    criterion = _make_criterion(loss_type)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    epochs_without_improve = 0

    try:
        scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(dev.type == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(dev.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        seen_graphs = 0

        for batch in train_loader:
            batch = batch.to(dev, non_blocking=True)

            if getattr(batch, "y", None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(dev):
                pred_raw = model(batch).view(-1)

                target_raw = _training_target(
                    batch,
                    target_mode=target_mode,
                    global_feature_variant=global_feature_variant,
                )

                mask = torch.isfinite(target_raw)
                if mask.sum() == 0:
                    continue

                if use_logit_target:
                    target_used = _to_logit(target_raw[mask], eps=logit_eps)
                    pred_used = pred_raw[mask]
                else:
                    target_used = target_raw[mask]
                    pred_used = pred_raw[mask]

                loss = criterion(pred_used, target_used)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except Exception:
                pass

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach().item() * batch.num_graphs
            seen_graphs += batch.num_graphs

        model.eval()

        val_loss = 0.0
        val_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)

                with _autocast_ctx(dev):
                    pred_raw = model(batch).view(-1)

                    target_raw = _training_target(
                        batch,
                        target_mode=target_mode,
                        global_feature_variant=global_feature_variant,
                    )

                    mask = torch.isfinite(target_raw)
                    if mask.sum() == 0:
                        continue

                    if use_logit_target:
                        target_used = _to_logit(target_raw[mask], eps=logit_eps)
                        pred_used = pred_raw[mask]
                    else:
                        target_used = target_raw[mask]
                        pred_used = pred_raw[mask]

                    loss = criterion(pred_used, target_used)

                if not torch.isfinite(loss):
                    continue

                val_loss += loss.detach().item() * batch.num_graphs
                val_seen += batch.num_graphs

        train_loss_epoch = total_loss / max(1, seen_graphs)
        val_loss_epoch = val_loss / max(1, val_seen)

        tag = f"{target_mode}{'_logit' if use_logit_target else ''}"
        print(f"Epoch {epoch:03d} | ({tag}) TrainLoss {train_loss_epoch:.6f} | ValLoss {val_loss_epoch:.6f}")

        if val_loss_epoch + early_stopping_min_delta < best_val:
            best_val = val_loss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:03d} (best ValLoss {best_val:.6f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loader, val_loader, test_loader, dev


def train_with_two_stage_split_strat(
    pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
    global_feature_variant: str = "baseline_backend",
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    loss_type: str = "huber",
    split_mode: str = "stratified",
    n_strat_bins: int = 10,
    group_attr: str = "group",
    use_logit_target: bool = False,
    logit_eps: float = 1e-4,
    target_mode: str = "noise_loss",
):
    _check_target_mode(target_mode, use_logit_target)

    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage_strat(
        pkl_paths,
        train_split=train_split,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
        split_mode=split_mode,
        n_strat_bins=n_strat_bins,
        group_attr=group_attr,
    )

    gdim = get_global_feature_dim(global_feature_variant)
    node_in_dim = _node_feature_dim(node_feature_backend_variant, node_angle_encoding)

    model_args = dict(global_in_dim=gdim, node_in_dim=node_in_dim)

    if model_kwargs:
        model_args.update(model_kwargs)
        model_args.setdefault("global_in_dim", gdim)
        model_args.setdefault("node_in_dim", node_in_dim)

    model = CircuitGNN(**model_args)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(dev)

    criterion = _make_criterion(loss_type)
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    epochs_without_improve = 0

    try:
        scaler = GradScaler(device=_AMP_DEVICE_TYPE, enabled=(dev.type == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(dev.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        seen_graphs = 0

        for batch in train_loader:
            batch = batch.to(dev, non_blocking=True)

            if getattr(batch, "y", None) is None:
                raise RuntimeError("Labels 'y' are missing in dataset. Ensure PKLs are labeled.")

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(dev):
                pred_raw = model(batch).view(-1)

                target_raw = _training_target(
                    batch,
                    target_mode=target_mode,
                    global_feature_variant=global_feature_variant,
                )

                mask = torch.isfinite(target_raw)
                if mask.sum() == 0:
                    continue

                if use_logit_target:
                    target_used = _to_logit(target_raw[mask], eps=logit_eps)
                    pred_used = pred_raw[mask]
                else:
                    target_used = target_raw[mask]
                    pred_used = pred_raw[mask]

                loss = criterion(pred_used, target_used)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except Exception:
                pass

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach().item() * batch.num_graphs
            seen_graphs += batch.num_graphs

        model.eval()

        val_loss = 0.0
        val_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)

                with _autocast_ctx(dev):
                    pred_raw = model(batch).view(-1)

                    target_raw = _training_target(
                        batch,
                        target_mode=target_mode,
                        global_feature_variant=global_feature_variant,
                    )

                    mask = torch.isfinite(target_raw)
                    if mask.sum() == 0:
                        continue

                    if use_logit_target:
                        target_used = _to_logit(target_raw[mask], eps=logit_eps)
                        pred_used = pred_raw[mask]
                    else:
                        target_used = target_raw[mask]
                        pred_used = pred_raw[mask]

                    loss = criterion(pred_used, target_used)

                if not torch.isfinite(loss):
                    continue

                val_loss += loss.detach().item() * batch.num_graphs
                val_seen += batch.num_graphs

        train_loss_epoch = total_loss / max(1, seen_graphs)
        val_loss_epoch = val_loss / max(1, val_seen)

        tag = f"{target_mode}{'_logit' if use_logit_target else ''}"
        print(f"Epoch {epoch:03d} | ({tag}) TrainLoss {train_loss_epoch:.6f} | ValLoss {val_loss_epoch:.6f}")

        if val_loss_epoch + early_stopping_min_delta < best_val:
            best_val = val_loss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:03d} (best ValLoss {best_val:.6f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loader, val_loader, test_loader, dev
