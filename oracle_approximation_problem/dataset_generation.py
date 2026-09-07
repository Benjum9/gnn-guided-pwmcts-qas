from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit

try:
    from qiskit import qasm2
except Exception:
    qasm2 = None

from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeAthensV2

from generate_constrained_target_circuit import (
    append_initial_hadamard_layers,
    make_constrained_target_circuit,
)
from random_circuit_generation import get_directed_coupling_edges
from mcts import make_root_node_from_qc, mcts



TARGET_DEPTH = 8
TARGET_SEED = 43
DEFAULT_OUT_DIR = Path(f"data_generation_results_constrained_depth_{TARGET_DEPTH}")
DEFAULT_BUDGETS = [50, 100, 500, 2000, 8000, 16000]
DEFAULT_N_RUNS = 5000

MAX_DEPTH = 20
CHOICES = {"a": 50, "d": 10, "s": 20, "c": 20, "p": 0}
CRITERIA = "average_value"
UCB_VALUE = 0.4
PW_C = 1.0
PW_ALPHA = 0.3
ROLLOUT_STEPS = 0


backend = FakeAthensV2()
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(method="density_matrix", noise_model=noise_model)

n_qubits = backend.num_qubits
coupling_edges = get_directed_coupling_edges(backend, n_qubits)


def parse_budgets(value: str) -> List[int]:
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("At least one budget is required.")
    return budgets


def circuit_to_qasm(qc: QuantumCircuit) -> str:
    if qasm2 is not None:
        return qasm2.dumps(qc)
    if hasattr(qc, "qasm"):
        return qc.qasm()
    raise RuntimeError("No QASM exporter available for this Qiskit version.")


def make_hadamard_root_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    append_initial_hadamard_layers(qc)
    return qc


def ideal_statevector(qc: QuantumCircuit) -> Statevector:
    return Statevector.from_instruction(qc)


def noisy_density_matrix(qc: QuantumCircuit) -> DensityMatrix:
    qc_dm = qc.copy()
    qc_dm.save_density_matrix()
    result = noisy_sim.run(qc_dm).result()
    return DensityMatrix(result.data(0)["density_matrix"])


def final_noisy_fidelity(candidate_qc: QuantumCircuit, target_dm: DensityMatrix) -> float:
    cand_dm = noisy_density_matrix(candidate_qc)
    return float(state_fidelity(cand_dm, target_dm))


def make_target_circuit(target_seed: int) -> QuantumCircuit:
    return make_constrained_target_circuit(
        target_depth=TARGET_DEPTH,
        seed=target_seed,
        backend=backend,
        num_qubits=n_qubits,
    )


def noiseless_fidelity_evaluator_factory(target_qc: QuantumCircuit):
    target_sv = ideal_statevector(target_qc)

    def evaluate_candidate(candidate_qc: QuantumCircuit) -> float:
        cand_sv = ideal_statevector(candidate_qc)
        return float(state_fidelity(cand_sv, target_sv))

    return evaluate_candidate


def save_budget_distribution_plots(rows: List[Dict], budgets: List[int], out_dir: Path) -> None:
    if not rows:
        return

    values_by_budget = {
        budget: np.asarray(
            [row["final_noisy_fidelity"] for row in rows if row["budget"] == budget],
            dtype=float,
        )
        for budget in budgets
    }

    bins = np.linspace(0.0, 1.0, 51)

    n_cols = 3
    n_rows = int(np.ceil(len(budgets) / n_cols))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budgets)))

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    ):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10.5, 3.0 * n_rows),
            sharex=True,
            constrained_layout=True,
        )
        axes_arr = np.asarray(axes).reshape(-1)

        for idx, budget in enumerate(budgets):
            ax = axes_arr[idx]
            values = values_by_budget[budget]
            ax.hist(
                values,
                bins=bins,
                color=colors[idx],
                edgecolor="#262626",
                linewidth=0.35,
                alpha=0.88,
            )
            ax.axvline(float(np.mean(values)), color="#111111", linestyle="--", linewidth=1.0)
            ax.set_title(f"Budget {budget}")
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, axis="y", color="#D5D8DF", linewidth=0.7, alpha=0.75)
            ax.grid(True, axis="x", color="#E7E9EE", linewidth=0.55, alpha=0.45)
            ax.set_axisbelow(True)
            ax.text(
                0.04,
                0.93,
                f"n={values.size}\nmean={np.mean(values):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                bbox={
                    "boxstyle": "round,pad=0.22,rounding_size=0.08",
                    "facecolor": "white",
                    "edgecolor": "#D4D7DE",
                    "linewidth": 0.6,
                    "alpha": 0.88,
                },
            )

        for ax in axes_arr[len(budgets) :]:
            ax.axis("off")

        for ax in axes_arr[-n_cols:]:
            ax.set_xlabel("Final noisy fidelity")
        for ax in axes_arr[::n_cols]:
            ax.set_ylabel("Count")

        grid_png = out_dir / "final_noisy_fidelity_histograms_per_budget.png"
        grid_pdf = out_dir / "final_noisy_fidelity_histograms_per_budget.pdf"
        fig.savefig(grid_png, dpi=300)
        fig.savefig(grid_pdf)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        for idx, budget in enumerate(budgets):
            values = values_by_budget[budget]
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                density=True,
                linewidth=1.6,
                color=colors[idx],
                label=f"Budget {budget}",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Final noisy fidelity")
        ax.set_ylabel("Density")
        ax.grid(True, color="#D5D8DF", linewidth=0.7, alpha=0.75)
        ax.legend(frameon=False, fontsize=8.5)

        overlay_png = out_dir / "final_noisy_fidelity_histogram_all_budgets.png"
        overlay_pdf = out_dir / "final_noisy_fidelity_histogram_all_budgets.pdf"
        fig.savefig(overlay_png, dpi=300)
        fig.savefig(overlay_pdf)
        plt.close(fig)

    print(f"Saved histogram grid to:    {grid_png}")
    print(f"Saved histogram overlay to: {overlay_png}")


def run_experiment(out_dir: Path, budgets: List[int], n_runs: int, target_seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = [100 + i for i in range(n_runs)]

    target_qc = make_target_circuit(target_seed)
    target_dm = noisy_density_matrix(target_qc)
    oracle_evaluator = noiseless_fidelity_evaluator_factory(target_qc)

    target_qasm_path = out_dir / f"target_depth_{TARGET_DEPTH}_seed_{target_seed}.qasm"
    target_qasm_path.write_text(circuit_to_qasm(target_qc))

    print("=" * 80)
    print("Constrained target circuit")
    print("=" * 80)
    print(f"target depth: {target_qc.depth()}")
    print(f"target size:  {target_qc.size()}")
    print(f"saved qasm:   {target_qasm_path}")
    print(target_qc)
    print()

    all_rows: List[Dict] = []
    budget_timings: Dict[int, float] = {}

    for budget in budgets:
        print("=" * 80)
        print(f"Budget = {budget}")
        print("=" * 80)

        budget_start_time = time.perf_counter()
        budget_pkl_data: List = []

        for run_idx, seed in enumerate(run_seeds):
            run_start_time = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)

            root_qc = make_hadamard_root_circuit(n_qubits)
            root = make_root_node_from_qc(
                quantum_circuit=root_qc,
                max_depth=MAX_DEPTH,
                coupling_map=coupling_edges,
                directed_coupling=True,
            )

            result = mcts(
                root=root,
                budget=budget,
                evaluation_function=oracle_evaluator,
                criteria=CRITERIA,
                rollout_type="classic",
                roll_out_steps=ROLLOUT_STEPS,
                simulation=False,
                branches=False,
                choices=CHOICES,
                epsilon=None,
                stop_deterministic=False,
                ucb_value=UCB_VALUE,
                pw_C=PW_C,
                pw_alpha=PW_ALPHA,
                verbose=False,
                evaluation_mode="oracle",
                gnn_value_function=None,
                expansion_mode="random",
                gnn_rank_function=None,
                gnn_candidate_pool_size=8,
                ignore_stop_in_guided=True,
                prefer_unvisited_by_prior=True,
                coupling_map=coupling_edges,
                directed_coupling=True,
            )

            final_qc = result["best_overall_qc"]
            noiseless_fid = float(result["best_overall_value"])
            noisy_fid = final_noisy_fidelity(final_qc, target_dm)
            run_elapsed = time.perf_counter() - run_start_time

            row = {
                "budget": budget,
                "run_idx": run_idx,
                "seed": seed,
                "final_noisy_fidelity": noisy_fid,
                "best_overall_value_noiseless": noiseless_fid,
                "oracle_eval_calls": int(result["oracle_eval_calls"]),
                "final_depth": int(final_qc.depth()),
                "final_size": int(final_qc.size()),
                "run_time_seconds": run_elapsed,
            }
            all_rows.append(row)

            budget_pkl_data.append(
                (
                    {
                        "budget": budget,
                        "run_idx": run_idx,
                        "seed": seed,
                        "qasm_transpiled": circuit_to_qasm(final_qc),
                        "target_qasm_transpiled": circuit_to_qasm(target_qc),
                        "oracle_noiseless_fidelity": noiseless_fid,
                        "noisy_fidelity": noisy_fid,
                        "target_depth": TARGET_DEPTH,
                        "target_seed": target_seed,
                        "root_starts_with_hadamard_block": True,
                    },
                    noisy_fid,
                )
            )

            if (run_idx + 1) % 50 == 0 or (run_idx + 1) == n_runs:
                print(
                    f"run {run_idx + 1}/{n_runs} | "
                    f"seed={seed} | "
                    f"final noisy fidelity={noisy_fid:.6f} | "
                    f"final depth={final_qc.depth()} | "
                    f"final size={final_qc.size()} | "
                    f"run time={run_elapsed:.2f}s"
                )

        pkl_path = out_dir / f"budget_{budget}_circuits.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(budget_pkl_data, f)

        budget_elapsed = time.perf_counter() - budget_start_time
        budget_timings[budget] = budget_elapsed
        print(f"Saved PKL to: {pkl_path}")
        print(
            f"Budget {budget} runtime: {budget_elapsed:.2f}s "
            f"({budget_elapsed / max(n_runs, 1):.3f}s/run)"
        )
        print()

    runs_csv = out_dir / "all_runs_final_noisy_fidelity.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget",
                "run_idx",
                "seed",
                "final_noisy_fidelity",
                "best_overall_value_noiseless",
                "oracle_eval_calls",
                "final_depth",
                "final_size",
                "run_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows: List[Dict] = []
    for budget in budgets:
        vals = [row["final_noisy_fidelity"] for row in all_rows if row["budget"] == budget]
        summary_rows.append(
            {
                "budget": budget,
                "n_runs": len(vals),
                "mean_final_noisy_fidelity": float(np.mean(vals)),
                "std_final_noisy_fidelity": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_final_noisy_fidelity": float(np.min(vals)),
                "max_final_noisy_fidelity": float(np.max(vals)),
                "budget_runtime_seconds": float(budget_timings.get(budget, 0.0)),
                "mean_runtime_per_run_seconds": float(
                    budget_timings.get(budget, 0.0) / max(len(vals), 1)
                ),
            }
        )

    summary_csv = out_dir / "budget_summary_final_noisy_fidelity.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "budget",
                "n_runs",
                "mean_final_noisy_fidelity",
                "std_final_noisy_fidelity",
                "min_final_noisy_fidelity",
                "max_final_noisy_fidelity",
                "budget_runtime_seconds",
                "mean_runtime_per_run_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_budget_distribution_plots(all_rows, budgets, out_dir)

    print(f"Saved all runs to: {runs_csv}")
    print(f"Saved summary to:  {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PWMCTS data for a constrained depth-5 target circuit."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--budgets", type=parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--target-seed", type=int, default=TARGET_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        out_dir=args.out_dir,
        budgets=args.budgets,
        n_runs=args.n_runs,
        target_seed=args.target_seed,
    )


if __name__ == "__main__":
    main()


