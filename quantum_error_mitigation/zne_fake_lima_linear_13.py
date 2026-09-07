from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from qiskit import QuantumCircuit

import zne_fake_lima_quadratic_135 as base


DATA_DIR = Path("mlqem_random_fake_lima_dataset")
TEST_PKL = DATA_DIR / "mlqem_random_fake_lima_test.pkl"

OUT_DIR = Path("models_mlqem_fake_lima_zne_linear_1_3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = 10_000
SEED = 34
CLIP_ZNE_TO_PHYSICAL_RANGE = False


def save_predictions_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "row_idx",
        "target_two_qubit_depth",
        "circuit_idx",
        "observable_qubit",
        "observable",
        "ideal_expectation",
        "noisy_expectation",
        "folded_scale3_expectation",
        "zne_linear_13_expectation",
        "zne_expectation",
        "unmitigated_error",
        "zne_linear_13_error",
        "unmitigated_abs_error",
        "zne_linear_13_abs_error",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    base.set_all_seeds(SEED)

    print("=" * 80)
    print("Paper-style ZNE on reproduced FakeLima random-circuit test set")
    print("=" * 80)
    print(f"Test PKL:        {TEST_PKL}")
    print(f"Output dir:      {OUT_DIR}")
    print(f"Shots:           {SHOTS}")
    print("ZNE method:      two-qubit digital folding, scale factors {1, 3}, linear extrapolation")
    print(f"Clip ZNE:        {CLIP_ZNE_TO_PHYSICAL_RANGE}")
    print()

    rows = base.load_test_rows(TEST_PKL)
    print(f"Loaded scalar test rows: {len(rows)}")

    circuit_groups: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["target_two_qubit_depth"]),
            int(row["circuit_idx"]),
            str(row["qasm"]),
        )
        circuit_groups.setdefault(key, []).append(row)

    print(f"Unique circuits to fold/simulate: {len(circuit_groups)}")

    simulator = base.make_noisy_simulator()
    processed_rows: List[Dict[str, Any]] = []

    for idx, ((target_depth, circuit_idx, qasm_str), group_rows) in enumerate(circuit_groups.items()):
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        folded_qc = base.fold_two_qubit_gates_odd_scale(qc, 3)
        folded_scale3 = base.run_noisy_expectations(
            folded_qc,
            simulator=simulator,
            shots=SHOTS,
            seed_simulator=SEED + 30_000 + idx,
        )

        for row in group_rows:
            q = int(row["observable_qubit"])

            noisy_1 = float(row["noisy_expectation"])
            noisy_3 = float(folded_scale3[q])

            zne_linear_13 = base.linear_zne_from_scales({
                1.0: noisy_1,
                3.0: noisy_3,
            })

            if CLIP_ZNE_TO_PHYSICAL_RANGE:
                zne_linear_13 = float(np.clip(zne_linear_13, -1.0, 1.0))

            ideal = float(row["ideal_expectation"])
            updated = dict(row)
            updated["folded_scale3_expectation"] = noisy_3
            updated["zne_linear_13_expectation"] = float(zne_linear_13)
            updated["zne_expectation"] = float(zne_linear_13)
            updated["unmitigated_error"] = noisy_1 - ideal
            updated["zne_linear_13_error"] = float(zne_linear_13 - ideal)
            updated["unmitigated_abs_error"] = abs(updated["unmitigated_error"])
            updated["zne_linear_13_abs_error"] = abs(updated["zne_linear_13_error"])
            processed_rows.append(updated)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(circuit_groups):
            print(
                f"Processed {idx + 1}/{len(circuit_groups)} circuits | "
                f"last depth={target_depth} | circuit_idx={circuit_idx}"
            )

    processed_rows = sorted(
        processed_rows,
        key=lambda row: (
            int(row["target_two_qubit_depth"]),
            int(row["circuit_idx"]),
            int(row["observable_qubit"]),
        ),
    )

    ideal = np.asarray([row["ideal_expectation"] for row in processed_rows], dtype=float)
    noisy = np.asarray([row["noisy_expectation"] for row in processed_rows], dtype=float)
    zne_linear_13 = np.asarray(
        [row["zne_linear_13_expectation"] for row in processed_rows],
        dtype=float,
    )

    unmitigated_scalar = base.scalar_metrics(ideal, noisy)
    zne_linear_13_scalar = base.scalar_metrics(ideal, zne_linear_13)

    l2_linear_13 = base.compute_l2_metrics(
        processed_rows,
        pred_key="zne_linear_13_expectation",
        label="zne_linear_13",
    )
    per_obs_linear_13 = base.per_observable_metrics(
        processed_rows,
        pred_key="zne_linear_13_expectation",
        label="zne_linear_13",
    )

    metrics = {
        "method": "zne_two_qubit_digital_folding_linear_extrapolation",
        "scale_factors": [1.0, 3.0],
        "shots": SHOTS,
        "clip_zne_to_physical_range": CLIP_ZNE_TO_PHYSICAL_RANGE,
        "test_pkl": str(TEST_PKL),
        "n_scalar_rows": len(processed_rows),
        "n_unique_circuits": len(circuit_groups),
        "unmitigated_scalar_metrics": unmitigated_scalar,
        "zne_linear_13_scalar_metrics": zne_linear_13_scalar,
        "l2_metrics_linear_13": l2_linear_13,
        "per_observable_metrics_linear_13": per_obs_linear_13,
    }

    save_predictions_csv(processed_rows, OUT_DIR / "zne_linear_13_predictions_test.csv")

    with (OUT_DIR / "zne_linear_13_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (OUT_DIR / "per_observable_metrics_linear_13.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_obs_linear_13[0].keys()))
        writer.writeheader()
        writer.writerows(per_obs_linear_13)

    base.save_scatter(
        ideal,
        noisy,
        OUT_DIR / "scatter_unmitigated_noisy_vs_ideal.png",
        title="Unmitigated noisy vs ideal expectation",
        ylabel=r"Noisy $\langle Z_i\rangle$",
    )

    base.save_scatter(
        ideal,
        zne_linear_13,
        OUT_DIR / "scatter_zne_linear_13_vs_ideal.png",
        title="Linear ZNE with scales 1,3 vs ideal expectation",
        ylabel=r"Linear ZNE $\langle Z_i\rangle$",
    )

    base.save_l2_histogram(
        processed_rows,
        pred_key="zne_linear_13_expectation",
        out_path=OUT_DIR / "l2_error_distribution_unmitigated_vs_zne_linear_13.png",
        pred_label="Linear ZNE",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("Scalar metrics:")
    print(
        f"Unmitigated   | "
        f"MSE={unmitigated_scalar['mse']:.6g} | "
        f"RMSE={unmitigated_scalar['rmse']:.6g} | "
        f"MAE={unmitigated_scalar['mae']:.6g} | "
        f"R2={unmitigated_scalar['r2']:.4f}"
    )
    print(
        f"ZNE linear13  | "
        f"MSE={zne_linear_13_scalar['mse']:.6g} | "
        f"RMSE={zne_linear_13_scalar['rmse']:.6g} | "
        f"MAE={zne_linear_13_scalar['mae']:.6g} | "
        f"R2={zne_linear_13_scalar['r2']:.4f}"
    )

    print("\nPaper-style vector L2 metric:")
    print(
        f"Unmitigated mean L2 = {l2_linear_13['unmitigated_l2_mean']:.6g} "
        f"+/- {l2_linear_13['unmitigated_l2_std']:.6g}"
    )
    print(
        f"ZNE linear13 mean L2 = {l2_linear_13['zne_linear_13_l2_mean']:.6g} "
        f"+/- {l2_linear_13['zne_linear_13_l2_std']:.6g}"
    )
    print(
        f"ZNE linear13 relative L2 improvement = "
        f"{100.0 * l2_linear_13['zne_linear_13_relative_l2_improvement']:.2f}%"
    )

    print("\nPer observable, linear ZNE:")
    for row in per_obs_linear_13:
        print(
            f"{row['observable']:>2s} | "
            f"unmitigated_MAE={row['unmitigated_mae']:.6g} | "
            f"zne_linear_13_MAE={row['zne_linear_13_mae']:.6g} | "
            f"unmitigated_R2={row['unmitigated_r2']:.4f} | "
            f"zne_linear_13_R2={row['zne_linear_13_r2']:.4f}"
        )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'zne_linear_13_predictions_test.csv'}")
    print(f"  {OUT_DIR / 'zne_linear_13_metrics.json'}")
    print(f"  {OUT_DIR / 'per_observable_metrics_linear_13.csv'}")
    print(f"  {OUT_DIR / 'scatter_zne_linear_13_vs_ideal.png'}")
    print(f"  {OUT_DIR / 'l2_error_distribution_unmitigated_vs_zne_linear_13.png'}")


if __name__ == "__main__":
    main()
