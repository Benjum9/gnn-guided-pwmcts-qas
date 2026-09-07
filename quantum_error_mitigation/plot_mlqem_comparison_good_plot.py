from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "ultimate_master_thesis_plot_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "mlqem_model_comparison_bootstrap" / "mlqem_l2_comparison_bootstrap_summary.csv"
OUT_DIR = BASE_DIR / "mlqem_model_comparison_bootstrap"

METHODS = ["Unmitigated", "Linear ZNE", "Quadratic ZNE", "RF", "GNN"]
MITIGATED_METHODS = ["Linear ZNE", "Quadratic ZNE", "RF", "GNN"]

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


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            method = str(row["method"])
            summary[method] = {
                "n_circuits": float(row["n_circuits"]),
                "mean": float(row["mean_l2"]),
                "std_across_circuits": float(row["std_l2_across_circuits"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "err_low": float(row["err_low"]),
                "err_high": float(row["err_high"]),
                "relative_improvement_percent": float(row["relative_improvement_percent"]),
            }

    missing = [method for method in METHODS if method not in summary]
    if missing:
        raise ValueError(f"Missing methods in summary CSV: {missing}")

    return summary


def plot_l2_bar(summary: Dict[str, Dict[str, float]]) -> None:
    configure_matplotlib()

    means = np.asarray([summary[m]["mean"] for m in METHODS], dtype=float)
    yerr = np.asarray([
        [summary[m]["err_low"] for m in METHODS],
        [summary[m]["err_high"] for m in METHODS],
    ])
    baseline = means[0]
    improvements = (baseline - means) / baseline * 100.0
    improvements[0] = 0.0

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    x = np.arange(len(METHODS))

    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=4,
        color=[BAR_COLORS[m] for m in METHODS],
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
    ax.set_xticklabels(METHODS)
    ax.set_ylabel(r"Mean $L_2$ error")
    ax.grid(True, axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = float(np.max(means + yerr[1]) * 1.26)
    ax.set_ylim(0.0, ymax)
    offset = ymax * 0.035

    for i, (bar, mean, improvement) in enumerate(zip(bars, means, improvements)):
        label = f"{mean:.3f}" if i == 0 else f"{mean:.3f}\n-{improvement:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + yerr[1, i] + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )

    fig.savefig(OUT_DIR / "mlqem_l2_comparison_good_plot.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "mlqem_l2_comparison_good_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_relative_improvement(summary: Dict[str, Dict[str, float]]) -> None:
    configure_matplotlib()

    improvements = np.asarray([
        summary[m]["relative_improvement_percent"]
        for m in MITIGATED_METHODS
    ])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    x = np.arange(len(MITIGATED_METHODS))

    bars = ax.bar(
        x,
        improvements,
        color=[BAR_COLORS[m] for m in MITIGATED_METHODS],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MITIGATED_METHODS)
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

    fig.savefig(OUT_DIR / "mlqem_relative_improvement_good_plot.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "mlqem_relative_improvement_good_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary(INPUT_CSV)
    plot_l2_bar(summary)
    plot_relative_improvement(summary)

    print(f"Saved plot: {OUT_DIR / 'mlqem_l2_comparison_good_plot.png'}")
    print(f"Saved plot: {OUT_DIR / 'mlqem_l2_comparison_good_plot.pdf'}")
    print(f"Saved plot: {OUT_DIR / 'mlqem_relative_improvement_good_plot.png'}")
    print(f"Saved plot: {OUT_DIR / 'mlqem_relative_improvement_good_plot.pdf'}")


if __name__ == "__main__":
    main()
