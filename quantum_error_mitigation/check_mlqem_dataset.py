from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("mlqem_random_fake_lima_dataset")

TRAIN_PKL = DATA_DIR / "mlqem_random_fake_lima_train.pkl"
TEST_PKL = DATA_DIR / "mlqem_random_fake_lima_test.pkl"

OUT_DIR = DATA_DIR / "dataset_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUT_DIR / "dataset_check_summary.json"
SUMMARY_CSV = OUT_DIR / "dataset_check_summary.csv"
ROWS_CSV = OUT_DIR / "scalar_samples_rows.csv"

SCATTER_ALL = OUT_DIR / "scatter_noisy_vs_ideal_all.png"
SCATTER_BY_SPLIT = OUT_DIR / "scatter_noisy_vs_ideal_by_split.png"
SCATTER_BY_OBSERVABLE = OUT_DIR / "scatter_noisy_vs_ideal_by_observable.png"
DEPTH_HIST = OUT_DIR / "actual_two_qubit_depth_distribution.png"
ERROR_HIST = OUT_DIR / "noisy_minus_ideal_error_distribution.png"


# ============================================================
# Helpers
# ============================================================

def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def load_scalar_pkl(path: Path) -> List[Tuple[Dict[str, Any], float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PKL: {path}")

    with path.open("rb") as f:
        content = pickle.load(f)

    rows: List[Tuple[Dict[str, Any], float]] = []

    for idx, item in enumerate(content):
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)):
            print(f"Skipping invalid item {idx} in {path}")
            continue

        meta, label = item

        ideal = safe_float(meta.get("ideal_expectation", label))
        noisy = safe_float(meta.get("noisy_expectation", None))

        if not math.isfinite(ideal) or not math.isfinite(noisy):
            print(f"Skipping item {idx} in {path}: invalid ideal/noisy value.")
            continue

        rows.append((dict(meta), ideal))

    return rows


def flatten_records(records: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for meta, label in records:
        ideal = safe_float(meta.get("ideal_expectation", label))
        noisy = safe_float(meta.get("noisy_expectation", None))

        target_depth = int(meta.get("target_two_qubit_depth", -1))
        actual_depth = int(meta.get("actual_two_qubit_depth", -1))
        circuit_idx = int(meta.get("circuit_idx", -1))
        obs_q = int(meta.get("observable_qubit", -1))
        split = str(meta.get("split", "unknown"))
        observable = str(meta.get("observable", f"Z{obs_q}"))

        out.append({
            "split": split,
            "target_two_qubit_depth": target_depth,
            "actual_two_qubit_depth": actual_depth,
            "circuit_idx": circuit_idx,
            "observable": observable,
            "observable_qubit": obs_q,
            "ideal_expectation": ideal,
            "noisy_expectation": noisy,
            "error_noisy_minus_ideal": noisy - ideal,
            "abs_error": abs(noisy - ideal),
            "shots": int(meta.get("shots", -1)),
            "num_qubits": int(meta.get("num_qubits", -1)),
            "depth": int(meta.get("depth", -1)),
            "size": int(meta.get("size", -1)),
            "matched_exactly": bool(meta.get("generation_info", {}).get("matched_exactly", False)),
        })

    return out


def save_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    return mean, std


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def axis_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    vals = np.concatenate([x, y])
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -1.0, 1.0

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * max(hi - lo, 1e-6)

    return lo - pad, hi + pad


# ============================================================
# Summary
# ============================================================

def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    summary["n_scalar_samples"] = len(rows)

    unique_circuits = set(
        (r["split"], r["target_two_qubit_depth"], r["circuit_idx"])
        for r in rows
    )
    summary["n_unique_circuits"] = len(unique_circuits)

    for split in sorted(set(r["split"] for r in rows)):
        split_rows = [r for r in rows if r["split"] == split]

        split_unique_circuits = set(
            (r["target_two_qubit_depth"], r["circuit_idx"])
            for r in split_rows
        )

        summary[f"{split}_n_scalar_samples"] = len(split_rows)
        summary[f"{split}_n_unique_circuits"] = len(split_unique_circuits)

    summary["observables"] = sorted(set(r["observable"] for r in rows))
    summary["target_two_qubit_depths"] = sorted(set(int(r["target_two_qubit_depth"]) for r in rows))
    summary["actual_two_qubit_depths"] = sorted(set(int(r["actual_two_qubit_depth"]) for r in rows))

    ideal_vals = [float(r["ideal_expectation"]) for r in rows]
    noisy_vals = [float(r["noisy_expectation"]) for r in rows]
    err_vals = [float(r["error_noisy_minus_ideal"]) for r in rows]
    abs_err_vals = [float(r["abs_error"]) for r in rows]

    summary["ideal_mean"], summary["ideal_std"] = mean_std(ideal_vals)
    summary["noisy_mean"], summary["noisy_std"] = mean_std(noisy_vals)
    summary["error_mean"], summary["error_std"] = mean_std(err_vals)
    summary["abs_error_mean"], summary["abs_error_std"] = mean_std(abs_err_vals)

    x = np.asarray(ideal_vals, dtype=float)
    y = np.asarray(noisy_vals, dtype=float)
    summary["pearson_noisy_vs_ideal"] = pearson(x, y)

    summary["mse_noisy_vs_ideal"] = float(np.mean((y - x) ** 2))
    summary["mae_noisy_vs_ideal"] = float(np.mean(np.abs(y - x)))

    # Depth table
    depth_counts: Dict[str, Dict[str, int]] = {}

    for r in rows:
        key = f"target_{r['target_two_qubit_depth']}_actual_{r['actual_two_qubit_depth']}"
        split = r["split"]

        if key not in depth_counts:
            depth_counts[key] = {}

        depth_counts[key][split] = depth_counts[key].get(split, 0) + 1

    summary["depth_counts_scalar_samples"] = depth_counts

    # Exact-depth match rate
    matched = [bool(r["matched_exactly"]) for r in rows]
    summary["matched_exactly_fraction"] = float(np.mean(matched)) if matched else float("nan")

    return summary


def save_summary(summary: Dict[str, Any]) -> None:
    with SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    rows = []

    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            rows.append({"key": k, "value": json.dumps(v)})
        else:
            rows.append({"key": k, "value": v})

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value"])
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plots
# ============================================================

def plot_noisy_vs_ideal_all(rows: List[Dict[str, Any]]) -> None:
    x = np.asarray([r["ideal_expectation"] for r in rows], dtype=float)
    y = np.asarray([r["noisy_expectation"] for r in rows], dtype=float)

    lo, hi = axis_limits(x, y)

    mse = float(np.mean((y - x) ** 2))
    mae = float(np.mean(np.abs(y - x)))
    corr = pearson(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=8, alpha=0.35, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

    text = (
        f"n = {len(rows)}\n"
        f"Pearson = {corr:.4f}\n"
        f"MSE = {mse:.4g}\n"
        f"MAE = {mae:.4g}"
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
    plt.ylabel(r"Noisy $\langle Z_i \rangle$")
    plt.title(r"Noisy vs ideal $\langle Z_i \rangle$ — all samples")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCATTER_ALL, dpi=300)
    plt.close()


def plot_noisy_vs_ideal_by_split(rows: List[Dict[str, Any]]) -> None:
    splits = ["train", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    for ax, split in zip(axes, splits):
        split_rows = [r for r in rows if r["split"] == split]

        x = np.asarray([r["ideal_expectation"] for r in split_rows], dtype=float)
        y = np.asarray([r["noisy_expectation"] for r in split_rows], dtype=float)

        lo, hi = axis_limits(x, y)

        ax.scatter(x, y, s=8, alpha=0.35, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

        mse = float(np.mean((y - x) ** 2)) if len(x) else float("nan")
        mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
        corr = pearson(x, y)

        text = (
            f"n = {len(split_rows)}\n"
            f"Pearson = {corr:.4f}\n"
            f"MSE = {mse:.4g}\n"
            f"MAE = {mae:.4g}"
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

        ax.set_xlabel(r"Ideal $\langle Z_i \rangle$")
        ax.set_ylabel(r"Noisy $\langle Z_i \rangle$")
        ax.set_title(f"{split} split")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)

    fig.savefig(SCATTER_BY_SPLIT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_noisy_vs_ideal_by_observable(rows: List[Dict[str, Any]]) -> None:
    observables = sorted(set(r["observable"] for r in rows))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, obs in zip(axes, observables):
        obs_rows = [r for r in rows if r["observable"] == obs]

        x = np.asarray([r["ideal_expectation"] for r in obs_rows], dtype=float)
        y = np.asarray([r["noisy_expectation"] for r in obs_rows], dtype=float)

        lo, hi = axis_limits(x, y)

        ax.scatter(x, y, s=8, alpha=0.35, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

        mse = float(np.mean((y - x) ** 2)) if len(x) else float("nan")
        mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
        corr = pearson(x, y)

        text = (
            f"n = {len(obs_rows)}\n"
            f"Pearson = {corr:.4f}\n"
            f"MSE = {mse:.4g}\n"
            f"MAE = {mae:.4g}"
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

        ax.set_xlabel(r"Ideal $\langle Z_i \rangle$")
        ax.set_ylabel(r"Noisy $\langle Z_i \rangle$")
        ax.set_title(obs)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)

    for ax in axes[len(observables):]:
        ax.axis("off")

    fig.savefig(SCATTER_BY_OBSERVABLE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_distribution(rows: List[Dict[str, Any]]) -> None:
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    train_depths = [r["actual_two_qubit_depth"] for r in train_rows]
    test_depths = [r["actual_two_qubit_depth"] for r in test_rows]

    all_depths = sorted(set(train_depths + test_depths))

    if not all_depths:
        return

    train_counts = [train_depths.count(d) for d in all_depths]
    test_counts = [test_depths.count(d) for d in all_depths]

    x = np.arange(len(all_depths))
    width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train_counts, width=width, label="train")
    plt.bar(x + width / 2, test_counts, width=width, label="test")

    plt.xticks(x, [str(d) for d in all_depths], rotation=45)
    plt.xlabel("Actual two-qubit depth")
    plt.ylabel("Number of scalar samples")
    plt.title("Distribution of actual two-qubit depths")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DEPTH_HIST, dpi=250)
    plt.close()


def plot_error_distribution(rows: List[Dict[str, Any]]) -> None:
    errors = np.asarray([r["error_noisy_minus_ideal"] for r in rows], dtype=float)
    errors = errors[np.isfinite(errors)]

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=80, edgecolor="black", alpha=0.85)

    plt.axvline(float(np.mean(errors)), linestyle="--", linewidth=2, label="mean")
    plt.xlabel(r"Noisy minus ideal $\langle Z_i \rangle$")
    plt.ylabel("Count")
    plt.title(r"Distribution of noisy error: $\langle Z_i\rangle_{\mathrm{noisy}} - \langle Z_i\rangle_{\mathrm{ideal}}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ERROR_HIST, dpi=250)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Checking ML-QEM random FakeLima dataset")
    print("=" * 80)

    train_records = load_scalar_pkl(TRAIN_PKL)
    test_records = load_scalar_pkl(TEST_PKL)

    rows = flatten_records(train_records + test_records)

    if not rows:
        raise RuntimeError("No valid rows found.")

    save_rows_csv(rows, ROWS_CSV)

    summary = compute_summary(rows)
    save_summary(summary)

    plot_noisy_vs_ideal_all(rows)
    plot_noisy_vs_ideal_by_split(rows)
    plot_noisy_vs_ideal_by_observable(rows)
    plot_depth_distribution(rows)
    plot_error_distribution(rows)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Scalar samples:          {summary['n_scalar_samples']}")
    print(f"Unique circuits:         {summary['n_unique_circuits']}")
    print(f"Train scalar samples:    {summary.get('train_n_scalar_samples', 0)}")
    print(f"Train unique circuits:   {summary.get('train_n_unique_circuits', 0)}")
    print(f"Test scalar samples:     {summary.get('test_n_scalar_samples', 0)}")
    print(f"Test unique circuits:    {summary.get('test_n_unique_circuits', 0)}")
    print(f"Observables:             {summary['observables']}")
    print(f"Target 2q depths:        {summary['target_two_qubit_depths']}")
    print(f"Actual 2q depths:        {summary['actual_two_qubit_depths']}")
    print(f"Matched exact fraction:  {summary['matched_exactly_fraction']:.4f}")
    print(f"Pearson noisy vs ideal:  {summary['pearson_noisy_vs_ideal']:.6f}")
    print(f"MSE noisy vs ideal:      {summary['mse_noisy_vs_ideal']:.6g}")
    print(f"MAE noisy vs ideal:      {summary['mae_noisy_vs_ideal']:.6g}")

    print("\nSaved:")
    print(f"  {SUMMARY_JSON}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {ROWS_CSV}")
    print(f"  {SCATTER_ALL}")
    print(f"  {SCATTER_BY_SPLIT}")
    print(f"  {SCATTER_BY_OBSERVABLE}")
    print(f"  {DEPTH_HIST}")
    print(f"  {ERROR_HIST}")


if __name__ == "__main__":
    main()
