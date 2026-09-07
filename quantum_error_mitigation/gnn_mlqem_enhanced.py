# gnn_mlqem_enhanced.py
from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

try:
    from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
    _AMP_DEVICE_TYPE = "cuda"
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_DEVICE_TYPE = "cuda"


# ------------------------------------------------------------
# Imports from ML-QEM graph representation
# ------------------------------------------------------------
try:
    from graph_representation_mlqem_enhanced import (
        QuantumCircuitGraphDataset,
        get_node_feature_dim,
        get_global_feature_dim,
        NOISY_EXPECTATION_INDEX,
    )
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = Path(__file__).resolve().parent
    for p in (project_root, models_dir):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    from graph_representation_mlqem_enhanced import (
        QuantumCircuitGraphDataset,
        get_node_feature_dim,
        get_global_feature_dim,
        NOISY_EXPECTATION_INDEX,
    )


# =========================================
# Data utils
# =========================================

def _cache_root_for_paths(paths: List[str], suffix: str = "") -> str:
    canonical = "|".join(sorted(os.path.abspath(p) for p in paths))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    tag = f"_{suffix}" if suffix else ""
    return os.path.join(os.getcwd(), f"pyg_cache_mlqem_{digest}{tag}")


def _node_feature_dim(
    node_feature_backend_variant: Optional[str],
    node_angle_encoding: str = "none",
) -> int:
    try:
        return get_node_feature_dim(node_feature_backend_variant, node_angle_encoding)
    except TypeError:
        return get_node_feature_dim(node_feature_backend_variant)


def _global_feature_dim() -> int:
    try:
        return int(get_global_feature_dim())
    except TypeError:
        # Defensive fallback for older signatures.
        return int(get_global_feature_dim("enhanced_backend"))


def _make_dataset(
    root: str,
    pkl_paths: List[str],
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
) -> QuantumCircuitGraphDataset:
    try:
        return QuantumCircuitGraphDataset(
            root=root,
            pkl_paths=pkl_paths,
            node_feature_backend_variant=node_feature_backend_variant,
            node_angle_encoding=node_angle_encoding,
        )
    except TypeError:
        return QuantumCircuitGraphDataset(
            root=root,
            pkl_paths=pkl_paths,
        )


# =========================================
# Target helpers for ML-QEM
# =========================================

TARGET_MODES = {"ideal_expectation", "mitigation_delta"}


def _global_features_2d(batch) -> Tensor:
    g_raw = batch.global_features

    if g_raw.dim() == 1:
        num_graphs = getattr(batch, "num_graphs", None)
        if num_graphs is None:
            num_graphs = batch.y.view(-1).numel()
        g_raw = g_raw.view(int(num_graphs), -1)

    return g_raw.float()


def _get_noisy_expectation_from_batch(batch) -> Tensor:
    g = _global_features_2d(batch)
    if g.size(1) <= NOISY_EXPECTATION_INDEX:
        raise ValueError(
            f"global_features has dim={g.size(1)}, but noisy expectation index "
            f"is {NOISY_EXPECTATION_INDEX}. Check graph_representation_mlqem_enhanced.py."
        )
    return g[:, NOISY_EXPECTATION_INDEX].view(-1).to(batch.y.device)


def _training_target(batch, target_mode: str) -> Tensor:
    """
    Returns the target used by the loss.

    target_mode='ideal_expectation':
        target = ideal/noiseless expectation value.

    target_mode='mitigation_delta':
        target = ideal_expectation - noisy_expectation.
    """
    tm = (target_mode or "ideal_expectation").strip().lower()

    if tm not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode={target_mode}. Use one of {TARGET_MODES}")

    y_ideal = batch.y.view(-1)

    if tm == "ideal_expectation":
        return y_ideal

    noisy = _get_noisy_expectation_from_batch(batch)
    return y_ideal - noisy


def _prediction_to_ideal_expectation(
    batch,
    pred_raw: Tensor,
    target_mode: str = "ideal_expectation",
    clamp_output: bool = True,
) -> Tensor:
    """
    Converts model output to predicted ideal/noiseless expectation.

    If target_mode='ideal_expectation':
        model output predicts ideal expectation directly.

    If target_mode='mitigation_delta':
        model output predicts Delta = ideal - noisy.
        Then ideal_hat = noisy + Delta_hat.
    """
    tm = (target_mode or "ideal_expectation").strip().lower()

    if tm == "ideal_expectation":
        pred = pred_raw
    elif tm == "mitigation_delta":
        noisy = _get_noisy_expectation_from_batch(batch)
        pred = noisy + pred_raw
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    if clamp_output:
        pred = pred.clamp(-1.0, 1.0)

    return pred


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
    GNN that outputs one scalar per graph.

    For ML-QEM, the scalar can mean either:
      - ideal/noiseless expectation value directly, or
      - mitigation delta = ideal expectation - noisy expectation.

    No sigmoid/logit is used because expectation values lie in [-1, 1].
    """

    def __init__(
        self,
        node_in_dim: int = 19,
        gnn_hidden: int = GNN_HIDDEN,
        gnn_heads: int = GNN_HEADS,
        global_in_dim: int = 48,
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
# Loss / AMP helpers
# =========================================

def _make_criterion(loss_type: str) -> nn.Module:
    lt = (loss_type or "huber").strip().lower()
    if lt in {"mse", "mse_loss", "mean_squared_error"}:
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

def build_full_loader(
    pkl_paths: List[str],
    batch_size: int = 64,
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
) -> DataLoader:
    suffix = f"backend_{node_feature_backend_variant}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check PKL paths and metadata fields.")

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem)


def build_train_val_test_loaders_from_train_test_pkls(
    train_pkl_paths: List[str],
    test_pkl_paths: List[str],
    val_within_train: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Preferred for the ML-QEM random-circuit dataset, because train/test
    files are generated separately.
    """
    train_root = _cache_root_for_paths(
        train_pkl_paths,
        suffix=f"train_backend_{node_feature_backend_variant}_angle_{node_angle_encoding}",
    )
    test_root = _cache_root_for_paths(
        test_pkl_paths,
        suffix=f"test_backend_{node_feature_backend_variant}_angle_{node_angle_encoding}",
    )

    train_full = _make_dataset(
        root=train_root,
        pkl_paths=train_pkl_paths,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    test_ds = _make_dataset(
        root=test_root,
        pkl_paths=test_pkl_paths,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    if len(train_full) < 2 or len(test_ds) < 1:
        raise RuntimeError("Train/test dataset too small. Check PKLs.")

    generator = torch.Generator().manual_seed(seed)

    val_len = max(1, int(round(len(train_full) * val_within_train)))
    val_len = min(len(train_full) - 1, val_len)
    train_len = len(train_full) - val_len

    train_ds, val_ds = random_split(train_full, [train_len, val_len], generator=generator)

    num_cpus = os.cpu_count() or 0
    default_workers = 2 if num_cpus > 2 else 0
    pin_mem = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=default_workers, pin_memory=pin_mem),
    )


def build_train_val_test_loaders_two_stage(
    pkl_paths: List[str],
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    suffix = f"backend_{node_feature_backend_variant}_angle_{node_angle_encoding}"
    root = _cache_root_for_paths(pkl_paths, suffix=suffix)

    dataset = _make_dataset(
        root=root,
        pkl_paths=pkl_paths,
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


# =========================================
# Evaluation in ideal/noiseless expectation space
# =========================================

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str = "ideal_expectation",
    clamp_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    y_all: List[torch.Tensor] = []
    pred_all: List[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        with _autocast_ctx(device):
            pred_raw = model(batch).view(-1)
            pred_ideal = _prediction_to_ideal_expectation(
                batch,
                pred_raw,
                target_mode=target_mode,
                clamp_output=clamp_output,
            )
            y_ideal = batch.y.view(-1)

            mask = torch.isfinite(y_ideal) & torch.isfinite(pred_ideal)
            if mask.sum() == 0:
                continue

            y_all.append(y_ideal[mask].detach().cpu())
            pred_all.append(pred_ideal[mask].detach().cpu())

    if not y_all:
        return torch.empty(0), torch.empty(0)

    return torch.cat(y_all, dim=0), torch.cat(pred_all, dim=0)


@torch.no_grad()
def evaluate_overall_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str = "ideal_expectation",
    clamp_output: bool = True,
) -> float:
    y, pred = collect_predictions(model, loader, device, target_mode=target_mode, clamp_output=clamp_output)
    if y.numel() == 0:
        return 0.0
    return float(torch.mean((pred - y) ** 2).item())


@torch.no_grad()
def evaluate_overall_r2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str = "ideal_expectation",
    clamp_output: bool = True,
) -> float:
    y, pred = collect_predictions(model, loader, device, target_mode=target_mode, clamp_output=clamp_output)
    if y.numel() == 0:
        return 0.0

    y_mean = torch.mean(y)
    ss_res = torch.sum((pred - y) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2)

    if ss_tot <= 0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


# =========================================
# Training
# =========================================

def train_from_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    device: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    loss_type: str = "huber",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    target_mode: str = "mitigation_delta",
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
):
    tm = (target_mode or "mitigation_delta").strip().lower()
    if tm not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode={target_mode}. Use one of {TARGET_MODES}")

    gdim = _global_feature_dim()
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
                target_raw = _training_target(batch, target_mode=tm)

                mask = torch.isfinite(target_raw) & torch.isfinite(pred_raw)
                if mask.sum() == 0:
                    continue

                loss = criterion(pred_raw[mask], target_raw[mask])

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

            total_loss += loss.detach().item() * int(batch.num_graphs)
            seen_graphs += int(batch.num_graphs)

        model.eval()
        val_loss = 0.0
        val_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev, non_blocking=True)

                with _autocast_ctx(dev):
                    pred_raw = model(batch).view(-1)
                    target_raw = _training_target(batch, target_mode=tm)

                    mask = torch.isfinite(target_raw) & torch.isfinite(pred_raw)
                    if mask.sum() == 0:
                        continue

                    loss = criterion(pred_raw[mask], target_raw[mask])

                if not torch.isfinite(loss):
                    continue

                val_loss += loss.detach().item() * int(batch.num_graphs)
                val_seen += int(batch.num_graphs)

        train_loss_epoch = total_loss / max(1, seen_graphs)
        val_loss_epoch = val_loss / max(1, val_seen)

        print(
            f"Epoch {epoch:03d} | ({tm}) "
            f"TrainLoss {train_loss_epoch:.6f} | ValLoss {val_loss_epoch:.6f}"
        )

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

    return model, dev, {"best_val_loss": float(best_val), "target_mode": tm}


def train_with_train_test_pkls(
    train_pkl_paths: List[str],
    test_pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    loss_type: str = "huber",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    target_mode: str = "mitigation_delta",
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
):
    train_loader, val_loader, test_loader = build_train_val_test_loaders_from_train_test_pkls(
        train_pkl_paths=train_pkl_paths,
        test_pkl_paths=test_pkl_paths,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    model, dev, info = train_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_kwargs=model_kwargs,
        loss_type=loss_type,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        target_mode=target_mode,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    return model, train_loader, val_loader, test_loader, dev, info


def train_with_two_stage_split(
    pkl_paths: List[str],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    train_split: float = 0.8,
    val_within_train: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    loss_type: str = "huber",
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    target_mode: str = "mitigation_delta",
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
):
    train_loader, val_loader, test_loader = build_train_val_test_loaders_two_stage(
        pkl_paths=pkl_paths,
        train_split=train_split,
        val_within_train=val_within_train,
        batch_size=batch_size,
        seed=seed,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    model, dev, info = train_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_kwargs=model_kwargs,
        loss_type=loss_type,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        target_mode=target_mode,
        node_feature_backend_variant=node_feature_backend_variant,
        node_angle_encoding=node_angle_encoding,
    )

    return model, train_loader, val_loader, test_loader, dev, info
