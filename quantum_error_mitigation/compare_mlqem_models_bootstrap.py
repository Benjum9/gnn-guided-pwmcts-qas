from __future__ import annotations

import csv
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "ultimate_master_thesis_plot_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


GNN_CSV = BASE_DIR / "models_mlqem_fake_lima_gnn" / "predictions_test.csv"
RF_CSV = BASE_DIR / "models_mlqem_fake_lima_rf_paper_like" / "rf_predictions_test_paper_like.csv"
ZNE_LINEAR_CSV = BASE_DIR / "models_mlqem_fake_lima_zne_linear_1_3" / "zne_linear_13_predictions_test.csv"
ZNE_QUADRATIC_CSV = BASE_DIR / "models_mlqem_fake_lima_zne_quadratic_1_3_5" / "zne_quadratic_predictions_test.csv"

UNMITIGATED_PKL = (
    BASE_DIR
    / "mlqem_random_fake_lima_dataset"
    / "mlqem_random_fake_lima_test_circuit_level.pkl"
)

OUT_DIR = BASE_DIR / "mlqem_model_comparison_bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_OBSERVABLE_QUBITS = 4
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 34

BAR_COLORS = {
    "Unmitigated": "#0072B2",
    "Linear ZNE": "#E69F00",
    "Quadratic ZNE": "#D55E00",
    "RF": "#009E73",
    "GNN": "#CC79A7",
}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#222222",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 10,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


# ============================================================
# CSV HELPERS
# ============================================================

def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")

    return rows


def column_map(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return {normalize_col(c): c for c in rows[0].keys()}


def find_col(
    rows: List[Dict[str, str]],
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    cmap = column_map(rows)

    for cand in candidates:
        key = normalize_col(cand)
        if key in cmap:
            return cmap[key]

    if required:
        raise KeyError(
            "Could not find required column.\n"
            f"Tried: {list(candidates)}\n"
            f"Available columns: {list(rows[0].keys())}"
        )

    return None


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def as_int(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def infer_observable_qubit(row: Dict[str, str], rows: List[Dict[str, str]], fallback_idx: int) -> int:
    obs_q_col = find_col(
        rows,
        ["observable_qubit", "qubit", "q", "observable_index"],
        required=False,
    )

    if obs_q_col is not None:
        return as_int(row.get(obs_q_col), fallback_idx % N_OBSERVABLE_QUBITS)

    obs_col = find_col(rows, ["observable", "pauli", "obs"], required=False)

    if obs_col is not None:
        obs = str(row.get(obs_col, "")).strip().upper()
        if obs.startswith("Z") and len(obs) >= 2:
            try:
                return int(obs[1:])
            except Exception:
                pass

    return fallback_idx % N_OBSERVABLE_QUBITS


def infer_circuit_key(row: Dict[str, str], rows: List[Dict[str, str]], fallback_idx: int) -> Tuple[int, int]:
    depth_col = find_col(
        rows,
        ["target_two_qubit_depth", "two_qubit_depth", "target_depth", "depth"],
        required=False,
    )

    circ_col = find_col(
        rows,
        ["circuit_idx", "sample_idx", "idx", "circuit_id"],
        required=False,
    )

    depth = as_int(row.get(depth_col), -1) if depth_col is not None else -1

    if circ_col is not None:
        circuit_idx = as_int(row.get(circ_col), fallback_idx // N_OBSERVABLE_QUBITS)
    else:
        circuit_idx = fallback_idx // N_OBSERVABLE_QUBITS

    return depth, circuit_idx


# ============================================================
# L2 COMPUTATION FROM CSV
# ============================================================

def rows_to_l2_from_csv(
    path: Path,
    prediction_candidates: Iterable[str],
    method_name: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    rows = read_csv_rows(path)

    ideal_col = find_col(
        rows,
        [
            "ideal_expectation",
            "ideal_exp_value",
            "ideal",
            "ideal_value",
            "true_ideal",
            "y_true",
            "target",
        ],
    )

    pred_col = find_col(rows, prediction_candidates)

    grouped: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for i, row in enumerate(rows):
        key = infer_circuit_key(row, rows, i)
        q = infer_observable_qubit(row, rows, i)

        ideal = as_float(row.get(ideal_col))
        pred = as_float(row.get(pred_col))

        if not (math.isfinite(ideal) and math.isfinite(pred)):
            continue

        grouped.setdefault(key, {})
        grouped[key][q] = {"ideal": ideal, "pred": pred}

    l2_values: List[float] = []
    detail_rows: List[Dict[str, Any]] = []

    for key, obs in grouped.items():
        if not all(q in obs for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)])
        pred_vec = np.asarray([obs[q]["pred"] for q in range(N_OBSERVABLE_QUBITS)])

        l2 = float(np.linalg.norm(pred_vec - ideal_vec, ord=2))
        l2_values.append(l2)

        detail_rows.append({
            "method": method_name,
            "target_two_qubit_depth": key[0],
            "circuit_idx": key[1],
            "l2_error": l2,
        })

    if not l2_values:
        raise RuntimeError(
            f"No complete 4-observable circuits found in {path}.\n"
            f"Available columns: {list(rows[0].keys())}"
        )

    return np.asarray(l2_values, dtype=float), detail_rows


def unmitigated_l2_from_csv(path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows = read_csv_rows(path)

    ideal_col = find_col(
        rows,
        [
            "ideal_expectation",
            "ideal_exp_value",
            "ideal",
            "ideal_value",
            "true_ideal",
            "y_true",
            "target",
        ],
    )

    noisy_col = find_col(
        rows,
        [
            "noisy_expectation",
            "noisy_exp_value",
            "noisy",
            "noisy_value",
            "unmitigated",
            "unmitigated_expectation",
        ],
    )

    grouped: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for i, row in enumerate(rows):
        key = infer_circuit_key(row, rows, i)
        q = infer_observable_qubit(row, rows, i)

        ideal = as_float(row.get(ideal_col))
        noisy = as_float(row.get(noisy_col))

        if not (math.isfinite(ideal) and math.isfinite(noisy)):
            continue

        grouped.setdefault(key, {})
        grouped[key][q] = {"ideal": ideal, "noisy": noisy}

    l2_values: List[float] = []
    detail_rows: List[Dict[str, Any]] = []

    for key, obs in grouped.items():
        if not all(q in obs for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)])
        noisy_vec = np.asarray([obs[q]["noisy"] for q in range(N_OBSERVABLE_QUBITS)])

        l2 = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))
        l2_values.append(l2)

        detail_rows.append({
            "method": "Unmitigated",
            "target_two_qubit_depth": key[0],
            "circuit_idx": key[1],
            "l2_error": l2,
        })

    if not l2_values:
        raise RuntimeError(f"No complete circuits found for unmitigated baseline in {path}")

    return np.asarray(l2_values, dtype=float), detail_rows


# ============================================================
# OPTIONAL PKL UNMITIGATED LOADING
# ============================================================

def extract_vector(meta: Dict[str, Any], candidates: Iterable[str]) -> Optional[np.ndarray]:
    for key in candidates:
        if key in meta and meta[key] is not None:
            try:
                arr = np.asarray(meta[key], dtype=float).reshape(-1)
                if arr.size >= N_OBSERVABLE_QUBITS:
                    return arr[:N_OBSERVABLE_QUBITS]
            except Exception:
                pass
    return None


def unmitigated_l2_from_circuit_level_pkl(path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PKL: {path}")

    with path.open("rb") as f:
        content = pickle.load(f)

    l2_values: List[float] = []
    detail_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(content):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            meta = dict(item[0])
            label = item[1]
        elif isinstance(item, dict):
            meta = dict(item)
            label = None
        else:
            continue

        ideal_vec = extract_vector(
            meta,
            [
                "ideal_expectations",
                "ideal_exp_values",
                "ideal_values",
                "ideal",
                "ideal_expectation_vector",
            ],
        )

        noisy_vec = extract_vector(
            meta,
            [
                "noisy_expectations",
                "noisy_exp_values",
                "noisy_values",
                "noisy",
                "noisy_expectation_vector",
            ],
        )

        if ideal_vec is None and label is not None:
            try:
                arr = np.asarray(label, dtype=float).reshape(-1)
                if arr.size >= N_OBSERVABLE_QUBITS:
                    ideal_vec = arr[:N_OBSERVABLE_QUBITS]
            except Exception:
                pass

        if ideal_vec is None or noisy_vec is None:
            continue

        if not (np.all(np.isfinite(ideal_vec)) and np.all(np.isfinite(noisy_vec))):
            continue

        l2 = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))

        depth = as_int(meta.get("target_two_qubit_depth", meta.get("target_depth", -1)))
        circuit_idx = as_int(meta.get("circuit_idx", meta.get("sample_idx", i)))

        l2_values.append(l2)
        detail_rows.append({
            "method": "Unmitigated",
            "target_two_qubit_depth": depth,
            "circuit_idx": circuit_idx,
            "l2_error": l2,
        })

    if not l2_values:
        raise RuntimeError("Could not parse circuit-level PKL as ideal/noisy vectors.")

    return np.asarray(l2_values, dtype=float), detail_rows


# ============================================================
# BOOTSTRAP 95% CI
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No finite values for bootstrap.")

    rng = np.random.default_rng(seed)
    n = values.size

    boot_means = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(values[sample_idx]))

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "n_circuits": int(n),
        "mean": mean,
        "std_across_circuits": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "err_low": mean - ci_low,
        "err_high": ci_high - mean,
    }


# ============================================================
# SAVING / PLOTTING
# ============================================================

def save_summary_csv(summary: Dict[str, Dict[str, float]], path: Path) -> None:
    baseline = summary["Unmitigated"]["mean"]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "n_circuits",
            "mean_l2",
            "std_l2_across_circuits",
            "ci95_low",
            "ci95_high",
            "err_low",
            "err_high",
            "relative_improvement_percent",
        ])

        for method in ["Unmitigated", "Linear ZNE", "Quadratic ZNE", "RF", "GNN"]:
            s = summary[method]
            improvement = 0.0 if method == "Unmitigated" else (baseline - s["mean"]) / baseline * 100.0

            writer.writerow([
                method,
                int(s["n_circuits"]),
                s["mean"],
                s["std_across_circuits"],
                s["ci95_low"],
                s["ci95_high"],
                s["err_low"],
                s["err_high"],
                improvement,
            ])


def save_per_circuit_l2_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "target_two_qubit_depth", "circuit_idx", "l2_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_l2_bar(summary: Dict[str, Dict[str, float]], out_png: Path, out_pdf: Path) -> None:
    configure_matplotlib()
    methods = ["Unmitigated", "Linear ZNE", "Quadratic ZNE", "RF", "GNN"]

    means = np.asarray([summary[m]["mean"] for m in methods])
    yerr = np.asarray([
        [summary[m]["err_low"] for m in methods],
        [summary[m]["err_high"] for m in methods],
    ])

    baseline = means[0]
    improvements = (baseline - means) / baseline * 100.0
    improvements[0] = 0.0

    colors = [BAR_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        error_kw={
            "ecolor": "#222222",
            "elinewidth": 1.0,
            "capthick": 1.0,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(r"Mean $L_2$ error")
    ax.grid(True, axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = float(np.max(means + yerr[1]) * 1.26)
    ax.set_ylim(0.0, ymax)
    offset = ymax * 0.035

    for i, (bar, mean, improvement) in enumerate(zip(bars, means, improvements)):
        if i == 0:
            label = f"{mean:.3f}"
        else:
            label = f"{mean:.3f}\n-{improvement:.1f}%"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + yerr[1, i] + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )

    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_relative_improvement(summary: Dict[str, Dict[str, float]], out_png: Path, out_pdf: Path) -> None:
    configure_matplotlib()
    methods = ["Linear ZNE", "Quadratic ZNE", "RF", "GNN"]
    baseline = summary["Unmitigated"]["mean"]

    improvements = np.asarray([
        (baseline - summary[m]["mean"]) / baseline * 100.0
        for m in methods
    ])

    colors = [BAR_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        improvements,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Relative improvement vs. unmitigated (%)")
    ax.grid(True, axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = float(np.max(improvements) * 1.20)
    ax.set_ylim(0.0, ymax)

    for bar, improvement in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            improvement + ymax * 0.025,
            f"{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )

    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("ML-QEM model comparison with 95% bootstrap CI")
    print("=" * 80)

    all_detail_rows: List[Dict[str, Any]] = []

    # Unmitigated
    try:
        unmit_l2, detail = unmitigated_l2_from_circuit_level_pkl(UNMITIGATED_PKL)
        print(f"Unmitigated loaded from circuit-level PKL: n={len(unmit_l2)}")
    except Exception as exc:
        print(f"Could not parse circuit-level PKL: {exc}")
        print("Falling back to unmitigated values from a ZNE CSV.")
        fallback_csv = ZNE_LINEAR_CSV if ZNE_LINEAR_CSV.exists() else ZNE_QUADRATIC_CSV
        unmit_l2, detail = unmitigated_l2_from_csv(fallback_csv)

    all_detail_rows.extend(detail)

    # Paper-style linear ZNE with scale factors {1, 3}
    linear_zne_l2, detail = rows_to_l2_from_csv(
        ZNE_LINEAR_CSV,
        prediction_candidates=[
            "zne_expectation",
            "zne_linear_13_expectation",
            "zne",
            "zne_value",
            "pred_zne",
            "mitigated_expectation",
        ],
        method_name="Linear ZNE",
    )
    all_detail_rows.extend(detail)

    # Extended quadratic ZNE with scale factors {1, 3, 5}
    quadratic_zne_l2, detail = rows_to_l2_from_csv(
        ZNE_QUADRATIC_CSV,
        prediction_candidates=[
            "zne_quadratic_expectation",
            "zne_expectation",
            "zne",
            "zne_value",
            "pred_zne",
            "mitigated_expectation",
        ],
        method_name="Quadratic ZNE",
    )
    all_detail_rows.extend(detail)

    # RF
    rf_l2, detail = rows_to_l2_from_csv(
        RF_CSV,
        prediction_candidates=[
            "rf_predicted_ideal_expectation",
            "rf_expectation",
            "rf_prediction",
            "rf_pred",
            "rf_mitigated_expectation",
            "pred_rf",
            "pred_mitigated",
            "mitigated_expectation",
            "pred_ideal",
            "predicted_ideal",
            "predicted_expectation",
            "prediction",
            "y_pred",
        ],
        method_name="RF",
    )
    all_detail_rows.extend(detail)

    # GNN

    gnn_l2, detail = rows_to_l2_from_csv(
        GNN_CSV,
        prediction_candidates=[
            "predicted_ideal_expectation",
            "gnn_predicted_ideal_expectation",
            "gnn_expectation",
            "gnn_prediction",
            "gnn_pred",
            "pred_gnn",
            "pred_mitigated",
            "mitigated_expectation",
            "pred_ideal",
            "predicted_ideal",
            "predicted_expectation",
            "prediction",
            "y_pred",
        ],
        method_name="GNN",
    )

    all_detail_rows.extend(detail)

    l2_by_method = {
        "Unmitigated": unmit_l2,
        "Linear ZNE": linear_zne_l2,
        "Quadratic ZNE": quadratic_zne_l2,
        "RF": rf_l2,
        "GNN": gnn_l2,
    }

    summary = {
        method: bootstrap_mean_ci(values)
        for method, values in l2_by_method.items()
    }

    save_summary_csv(summary, OUT_DIR / "mlqem_l2_comparison_bootstrap_summary.csv")
    save_per_circuit_l2_csv(all_detail_rows, OUT_DIR / "mlqem_per_circuit_l2_errors.csv")

    with (OUT_DIR / "mlqem_l2_comparison_bootstrap_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_l2_bar(
        summary,
        OUT_DIR / "mlqem_l2_comparison_bootstrap_ci.png",
        OUT_DIR / "mlqem_l2_comparison_bootstrap_ci.pdf",
    )

    plot_relative_improvement(
        summary,
        OUT_DIR / "mlqem_relative_improvement.png",
        OUT_DIR / "mlqem_relative_improvement.pdf",
    )

    baseline = summary["Unmitigated"]["mean"]

    print("\nResults:")
    for method in ["Unmitigated", "Linear ZNE", "Quadratic ZNE", "RF", "GNN"]:
        s = summary[method]
        improvement = 0.0 if method == "Unmitigated" else (baseline - s["mean"]) / baseline * 100.0

        print(
            f"{method:12s} | "
            f"mean L2={s['mean']:.6f} | "
            f"95% CI=[{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | "
            f"std={s['std_across_circuits']:.6f} | "
            f"improvement={improvement:.2f}% | "
            f"n={int(s['n_circuits'])}"
        )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'mlqem_l2_comparison_bootstrap_summary.csv'}")
    print(f"  {OUT_DIR / 'mlqem_l2_comparison_bootstrap_summary.json'}")
    print(f"  {OUT_DIR / 'mlqem_per_circuit_l2_errors.csv'}")
    print(f"  {OUT_DIR / 'mlqem_l2_comparison_bootstrap_ci.png'}")
    print(f"  {OUT_DIR / 'mlqem_l2_comparison_bootstrap_ci.pdf'}")
    print(f"  {OUT_DIR / 'mlqem_relative_improvement.png'}")
    print(f"  {OUT_DIR / 'mlqem_relative_improvement.pdf'}")


if __name__ == "__main__":
    main()
