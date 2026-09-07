from __future__ import annotations

import csv
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ============================================================
# FakeLima backend helper
# ============================================================

def make_fake_lima_backend():
    """
    Supports several Qiskit versions.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeLimaV2
        return FakeLimaV2()
    except Exception:
        pass

    try:
        from qiskit_ibm_runtime.fake_provider import FakeLima
        return FakeLima()
    except Exception:
        pass

    try:
        from qiskit.providers.fake_provider import FakeLimaV2
        return FakeLimaV2()
    except Exception:
        pass

    try:
        from qiskit.providers.fake_provider import FakeLima
        return FakeLima()
    except Exception:
        pass

    raise ImportError(
        "Could not import FakeLima/FakeLimaV2 from qiskit_ibm_runtime.fake_provider "
        "or qiskit.providers.fake_provider."
    )


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("mlqem_random_fake_lima_dataset")
TEST_PKL = DATA_DIR / "mlqem_random_fake_lima_test.pkl"

OUT_DIR = Path("models_mlqem_fake_lima_zne_quadratic_1_3_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = 10_000
SEED = 34

# Extended ZNE setup:
#   digital folding on two-qubit gates
#   noise scale factors {1, 3, 5}
#   quadratic extrapolation to zero noise
SCALE_FACTORS = [1.0, 3.0, 5.0]
FOLDED_SCALE_FACTORS = [3, 5]

# Do not clip by default, because ZNE can extrapolate outside [-1, 1].
CLIP_ZNE_TO_PHYSICAL_RANGE = False

N_OBSERVABLE_QUBITS = 4


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Backend / simulator
# ============================================================

def make_noisy_simulator() -> AerSimulator:
    backend = make_fake_lima_backend()
    return AerSimulator.from_backend(backend)


# ============================================================
# Raw PKL loading
# ============================================================

def get_first_existing_key(d: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def qasm_from_meta(meta: Dict[str, Any]) -> str:
    qasm_str = get_first_existing_key(
        meta,
        [
            "qasm_transpiled",
            "qasm",
            "qasm_str",
            "circuit_qasm",
            "transpiled_qasm",
        ],
    )

    if isinstance(qasm_str, bytes):
        qasm_str = qasm_str.decode("utf-8")

    if isinstance(qasm_str, str) and qasm_str.strip():
        return qasm_str

    circ = get_first_existing_key(meta, ["circuit", "quantum_circuit", "qc"])
    if isinstance(circ, QuantumCircuit):
        try:
            from qiskit import qasm2
            return qasm2.dumps(circ)
        except Exception:
            if hasattr(circ, "qasm"):
                return circ.qasm()

    raise KeyError(
        "Could not find a QASM string or QuantumCircuit in the PKL item. "
        f"Available keys: {sorted(meta.keys())}"
    )


def infer_observable_qubit(meta: Dict[str, Any]) -> int:
    if "observable_qubit" in meta:
        return int(meta["observable_qubit"])

    obs = meta.get("observable", None)
    if isinstance(obs, bytes):
        obs = obs.decode("utf-8")

    if isinstance(obs, str):
        obs_clean = obs.strip().upper()
        if obs_clean.startswith("Z") and len(obs_clean) >= 2:
            return int(obs_clean[1:])

    raise KeyError(
        "Could not infer observable qubit. Expected 'observable_qubit' "
        "or an observable string like 'Z0'."
    )


def infer_noisy_expectation(meta: Dict[str, Any]) -> float:
    for key in [
        "noisy_expectation",
        "noisy_exp_value",
        "noisy",
        "noisy_value",
    ]:
        if key in meta:
            val = meta[key]
            arr = np.asarray(val, dtype=float).reshape(-1)

            if arr.size == 1:
                return float(arr[0])

            q = infer_observable_qubit(meta)
            return float(arr[q])

    for key in ["noisy_exp_values", "noisy_exp_val"]:
        if key in meta:
            q = infer_observable_qubit(meta)
            vals = np.asarray(meta[key], dtype=float).reshape(-1)
            return float(vals[q])

    raise KeyError("Could not find noisy expectation in PKL item.")


def infer_ideal_expectation(meta: Dict[str, Any], label: Any) -> float:
    try:
        if label is not None:
            arr = np.asarray(label, dtype=float).reshape(-1)
            if arr.size == 1:
                return float(arr[0])

            q = infer_observable_qubit(meta)
            return float(arr[q])
    except Exception:
        pass

    for key in [
        "ideal_expectation",
        "ideal_exp_value",
        "ideal",
        "ideal_value",
        "ideal_exp_val",
    ]:
        if key in meta:
            val = meta[key]
            arr = np.asarray(val, dtype=float).reshape(-1)

            if arr.size == 1:
                return float(arr[0])

            q = infer_observable_qubit(meta)
            return float(arr[q])

    raise KeyError("Could not find ideal expectation / label in PKL item.")


def infer_circuit_idx(meta: Dict[str, Any], fallback_index: int) -> int:
    for key in ["circuit_idx", "sample_idx", "idx"]:
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass

    # Fallback for scalar datasets ordered as 4 observables per circuit.
    return fallback_index // N_OBSERVABLE_QUBITS


def infer_target_depth(meta: Dict[str, Any]) -> int:
    for key in [
        "target_two_qubit_depth",
        "target_depth",
        "two_qubit_depth",
        "depth",
    ]:
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass
    return -1


def load_test_rows(pkl_path: Path) -> List[Dict[str, Any]]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing test PKL: {pkl_path}")

    with pkl_path.open("rb") as f:
        content = pickle.load(f)

    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(content):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            meta = item[0]
            label = item[1]
        elif isinstance(item, dict):
            meta = item
            label = None
        else:
            print(f"Skipping item {i}: unsupported type {type(item)}")
            continue

        try:
            q = infer_observable_qubit(meta)
            qasm_str = qasm_from_meta(meta)
            noisy = infer_noisy_expectation(meta)
            ideal = infer_ideal_expectation(meta, label)
            circuit_idx = infer_circuit_idx(meta, i)
            target_depth = infer_target_depth(meta)
        except Exception as exc:
            print(f"Skipping item {i}: {exc}")
            continue

        rows.append({
            "row_idx": i,
            "qasm": qasm_str,
            "observable_qubit": int(q),
            "observable": f"Z{q}",
            "noisy_expectation": float(noisy),
            "ideal_expectation": float(ideal),
            "circuit_idx": int(circuit_idx),
            "target_two_qubit_depth": int(target_depth),
        })

    if not rows:
        raise RuntimeError(f"No usable rows loaded from {pkl_path}")

    return rows


# ============================================================
# Two-qubit digital folding
# ============================================================

def fold_two_qubit_gates_odd_scale(qc: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Digital folding on two-qubit gates for odd integer scale factors.

    For each two-qubit unitary U:
        scale 1: U
        scale 3: U U^\dagger U
        scale 5: U U^\dagger U U^\dagger U

    Measurements, barriers, resets, and single-qubit gates are left unchanged.
    """
    if scale_factor < 1 or scale_factor % 2 != 1:
        raise ValueError("scale_factor must be a positive odd integer, e.g. 1, 3, 5.")

    if scale_factor == 1:
        return qc.copy()

    folded = QuantumCircuit(qc.num_qubits, qc.num_clbits)

    try:
        folded.global_phase = qc.global_phase
    except Exception:
        pass

    for instr in qc.data:
        try:
            op = instr.operation
            qargs = instr.qubits
            cargs = instr.clbits
        except AttributeError:
            op, qargs, cargs = instr

        q_indices = [qc.find_bit(q).index for q in qargs]
        c_indices = [qc.find_bit(c).index for c in cargs]

        op_name = op.name.lower()

        if op_name in {"measure", "barrier", "reset"}:
            folded.append(op.copy(), q_indices, c_indices)
            continue

        if getattr(op, "num_qubits", 0) == 2:
            for rep in range(scale_factor):
                if rep % 2 == 0:
                    folded.append(op.copy(), q_indices, c_indices)
                else:
                    try:
                        folded.append(op.inverse(), q_indices, c_indices)
                    except Exception:
                        # Common two-qubit gates are self-inverse.
                        if op_name in {"cx", "cz", "swap"}:
                            folded.append(op.copy(), q_indices, c_indices)
                        else:
                            raise
        else:
            folded.append(op.copy(), q_indices, c_indices)

    return folded


# ============================================================
# Measurement / expectation values
# ============================================================

def measurement_map_qubit_to_clbit(qc: QuantumCircuit) -> Dict[int, int]:
    mapping: Dict[int, int] = {}

    for instr in qc.data:
        try:
            op = instr.operation
            qargs = instr.qubits
            cargs = instr.clbits
        except AttributeError:
            op, qargs, cargs = instr

        if op.name.lower() != "measure":
            continue

        if not qargs or not cargs:
            continue

        q_idx = qc.find_bit(qargs[0]).index
        c_idx = qc.find_bit(cargs[0]).index

        mapping[q_idx] = c_idx

    return mapping


def circuit_has_measurements(qc: QuantumCircuit) -> bool:
    for instr in qc.data:
        try:
            name = instr.operation.name.lower()
        except AttributeError:
            name = instr[0].name.lower()
        if name == "measure":
            return True
    return False


def ensure_all_qubits_measured(qc: QuantumCircuit) -> QuantumCircuit:
    if circuit_has_measurements(qc):
        return qc

    out = qc.copy()
    out.measure_all()
    return out


def z_expectations_from_counts(
    counts: Dict[str, int],
    qubit_to_clbit: Dict[int, int],
    n_observable_qubits: int = N_OBSERVABLE_QUBITS,
) -> Dict[int, float]:
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("Empty counts dictionary.")

    exps: Dict[int, float] = {}

    for q in range(n_observable_qubits):
        if q not in qubit_to_clbit:
            raise KeyError(
                f"Qubit {q} is not measured. Measurement map: {qubit_to_clbit}"
            )

        c = qubit_to_clbit[q]
        acc = 0.0

        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")

            if c >= len(bits):
                raise RuntimeError(
                    f"Classical bit index {c} incompatible with bitstring {bitstring}"
                )

            bit = bits[-1 - c]
            z = 1.0 if bit == "0" else -1.0
            acc += z * count

        exps[q] = float(acc / total)

    return exps


def run_noisy_expectations(
    qc: QuantumCircuit,
    simulator: AerSimulator,
    shots: int,
    seed_simulator: int,
) -> Dict[int, float]:
    """
    Run a circuit and return <Z0>,...,<Z3>.
    """
    qc = ensure_all_qubits_measured(qc)
    q_to_c = measurement_map_qubit_to_clbit(qc)

    result = simulator.run(
        qc,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()

    counts = result.get_counts(0)

    return z_expectations_from_counts(
        counts,
        qubit_to_clbit=q_to_c,
        n_observable_qubits=N_OBSERVABLE_QUBITS,
    )


# ============================================================
# ZNE extrapolation
# ============================================================

def quadratic_zne_from_scales(scale_values: Dict[float, float]) -> float:
    """
    Fit y(lambda) = a lambda^2 + b lambda + c using scales {1,3,5},
    then return c = y(0).

    With exactly three points, this polynomial interpolates the three values.
    """
    xs = np.asarray(sorted(scale_values.keys()), dtype=float)
    ys = np.asarray([scale_values[x] for x in xs], dtype=float)

    if xs.size < 3:
        raise ValueError("Quadratic extrapolation needs at least 3 scale factors.")

    coeffs = np.polyfit(xs, ys, deg=2)  # [a, b, c]
    intercept = float(coeffs[-1])
    return intercept


def linear_zne_from_scales(scale_values: Dict[float, float]) -> float:
    """
    Optional reference: linear least-squares extrapolation to lambda=0.
    Uses all available scale factors.
    """
    xs = np.asarray(sorted(scale_values.keys()), dtype=float)
    ys = np.asarray([scale_values[x] for x in xs], dtype=float)

    coeffs = np.polyfit(xs, ys, deg=1)  # [a, b]
    return float(coeffs[-1])


# ============================================================
# Metrics
# ============================================================

def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true

    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    return {
        "n": int(y_true.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "mean_error": float(np.mean(err)),
        "std_error": float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
        "max_abs_error": float(np.max(np.abs(err))) if err.size else float("nan"),
    }


def compute_l2_metrics(rows: List[Dict[str, Any]], pred_key: str, label: str) -> Dict[str, float]:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "pred": float(r[pred_key]),
        }

    noisy_l2: List[float] = []
    pred_l2: List[float] = []

    by_depth: Dict[int, Dict[str, List[float]]] = {}

    for key, obs_dict in groups.items():
        target_depth, _circuit_idx = key

        if not all(q in obs_dict for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        pred_vec = np.asarray([obs_dict[q]["pred"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)

        n = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))
        p = float(np.linalg.norm(pred_vec - ideal_vec, ord=2))

        noisy_l2.append(n)
        pred_l2.append(p)

        by_depth.setdefault(target_depth, {"noisy": [], "pred": []})
        by_depth[target_depth]["noisy"].append(n)
        by_depth[target_depth]["pred"].append(p)

    noisy_arr = np.asarray(noisy_l2, dtype=float)
    pred_arr = np.asarray(pred_l2, dtype=float)

    out: Dict[str, float] = {
        "n_circuits_l2": int(noisy_arr.size),
        "unmitigated_l2_mean": float(np.mean(noisy_arr)),
        "unmitigated_l2_std": float(np.std(noisy_arr, ddof=1)) if noisy_arr.size > 1 else 0.0,
        f"{label}_l2_mean": float(np.mean(pred_arr)),
        f"{label}_l2_std": float(np.std(pred_arr, ddof=1)) if pred_arr.size > 1 else 0.0,
        f"{label}_relative_l2_improvement": float((np.mean(noisy_arr) - np.mean(pred_arr)) / np.mean(noisy_arr)),
    }

    for depth in sorted(by_depth):
        n_d = np.asarray(by_depth[depth]["noisy"], dtype=float)
        p_d = np.asarray(by_depth[depth]["pred"], dtype=float)

        out[f"depth_{depth}_n"] = int(n_d.size)
        out[f"depth_{depth}_unmitigated_l2_mean"] = float(np.mean(n_d))
        out[f"depth_{depth}_{label}_l2_mean"] = float(np.mean(p_d))

    return out


def per_observable_metrics(rows: List[Dict[str, Any]], pred_key: str, label: str) -> List[Dict[str, Any]]:
    out = []

    for q in range(N_OBSERVABLE_QUBITS):
        q_rows = [r for r in rows if int(r["observable_qubit"]) == q]

        y = np.asarray([r["ideal_expectation"] for r in q_rows], dtype=float)
        noisy = np.asarray([r["noisy_expectation"] for r in q_rows], dtype=float)
        pred = np.asarray([r[pred_key] for r in q_rows], dtype=float)

        noisy_m = scalar_metrics(y, noisy)
        pred_m = scalar_metrics(y, pred)

        out.append({
            "observable": f"Z{q}",
            "n": len(q_rows),
            "unmitigated_mae": noisy_m["mae"],
            f"{label}_mae": pred_m["mae"],
            "unmitigated_mse": noisy_m["mse"],
            f"{label}_mse": pred_m["mse"],
            "unmitigated_r2": noisy_m["r2"],
            f"{label}_r2": pred_m["r2"],
        })

    return out


# ============================================================
# Saving and plotting
# ============================================================

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
        "folded_scale5_expectation",
        "zne_quadratic_expectation",
        "zne_linear_135_expectation",
        "unmitigated_error",
        "zne_quadratic_error",
        "zne_linear_135_error",
        "unmitigated_abs_error",
        "zne_quadratic_abs_error",
        "zne_linear_135_abs_error",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def save_scatter(y_true: np.ndarray, pred: np.ndarray, out_path: Path, title: str, ylabel: str) -> None:
    vals = np.concatenate([y_true, pred])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * max(hi - lo, 1e-6)

    lo -= pad
    hi += pad

    metrics = scalar_metrics(y_true, pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, pred, s=8, alpha=0.35, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel(r"Ideal $\langle Z_i\rangle$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True, alpha=0.3)

    txt = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MSE = {metrics['mse']:.4g}\n"
        f"MAE = {metrics['mae']:.4g}\n"
        f"Pearson = {metrics['pearson']:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_l2_histogram(rows: List[Dict[str, Any]], pred_key: str, out_path: Path, pred_label: str) -> None:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "pred": float(r[pred_key]),
        }

    noisy_l2 = []
    pred_l2 = []

    for _key, obs_dict in groups.items():
        if not all(q in obs_dict for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        pred_vec = np.asarray([obs_dict[q]["pred"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)

        noisy_l2.append(float(np.linalg.norm(noisy_vec - ideal_vec, ord=2)))
        pred_l2.append(float(np.linalg.norm(pred_vec - ideal_vec, ord=2)))

    plt.figure(figsize=(8, 5))
    plt.hist(noisy_l2, bins=60, alpha=0.55, label="Unmitigated noisy", edgecolor="black")
    plt.hist(pred_l2, bins=60, alpha=0.55, label=pred_label, edgecolor="black")
    plt.xlabel(r"$L_2$ error on $[\langle Z_0\rangle,\ldots,\langle Z_3\rangle]$")
    plt.ylabel("Number of circuits")
    plt.title("Paper-style vector error distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Main ZNE routine
# ============================================================

def main() -> None:
    set_all_seeds(SEED)

    print("=" * 80)
    print("Extended ZNE on reproduced FakeLima random-circuit test set")
    print("=" * 80)
    print(f"Test PKL:        {TEST_PKL}")
    print(f"Output dir:      {OUT_DIR}")
    print(f"Shots:           {SHOTS}")
    print("ZNE method:      two-qubit digital folding, scale factors {1, 3, 5}, quadratic extrapolation")
    print(f"Clip ZNE:        {CLIP_ZNE_TO_PHYSICAL_RANGE}")
    print()

    rows = load_test_rows(TEST_PKL)
    print(f"Loaded scalar test rows: {len(rows)}")

    circuit_groups: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}

    for r in rows:
        key = (
            int(r["target_two_qubit_depth"]),
            int(r["circuit_idx"]),
            str(r["qasm"]),
        )
        circuit_groups.setdefault(key, [])
        circuit_groups[key].append(r)

    print(f"Unique circuits to fold/simulate: {len(circuit_groups)}")

    simulator = make_noisy_simulator()
    processed_rows: List[Dict[str, Any]] = []

    for idx, ((target_depth, circuit_idx, qasm_str), group_rows) in enumerate(circuit_groups.items()):
        qc = QuantumCircuit.from_qasm_str(qasm_str)

        folded_exps_by_scale: Dict[int, Dict[int, float]] = {}

        for scale in FOLDED_SCALE_FACTORS:
            folded_qc = fold_two_qubit_gates_odd_scale(qc, scale)

            folded_exps_by_scale[scale] = run_noisy_expectations(
                folded_qc,
                simulator=simulator,
                shots=SHOTS,
                seed_simulator=SEED + 10_000 * scale + idx,
            )

        for r in group_rows:
            q = int(r["observable_qubit"])

            noisy_1 = float(r["noisy_expectation"])
            noisy_3 = float(folded_exps_by_scale[3][q])
            noisy_5 = float(folded_exps_by_scale[5][q])

            scale_values = {
                1.0: noisy_1,
                3.0: noisy_3,
                5.0: noisy_5,
            }

            zne_quad = quadratic_zne_from_scales(scale_values)
            zne_linear_135 = linear_zne_from_scales(scale_values)

            if CLIP_ZNE_TO_PHYSICAL_RANGE:
                zne_quad = float(np.clip(zne_quad, -1.0, 1.0))
                zne_linear_135 = float(np.clip(zne_linear_135, -1.0, 1.0))

            rr = dict(r)
            rr["folded_scale3_expectation"] = float(noisy_3)
            rr["folded_scale5_expectation"] = float(noisy_5)
            rr["zne_quadratic_expectation"] = float(zne_quad)
            rr["zne_linear_135_expectation"] = float(zne_linear_135)

            rr["unmitigated_error"] = float(noisy_1 - float(r["ideal_expectation"]))
            rr["zne_quadratic_error"] = float(zne_quad - float(r["ideal_expectation"]))
            rr["zne_linear_135_error"] = float(zne_linear_135 - float(r["ideal_expectation"]))

            rr["unmitigated_abs_error"] = abs(rr["unmitigated_error"])
            rr["zne_quadratic_abs_error"] = abs(rr["zne_quadratic_error"])
            rr["zne_linear_135_abs_error"] = abs(rr["zne_linear_135_error"])

            processed_rows.append(rr)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(circuit_groups):
            print(
                f"Processed {idx + 1}/{len(circuit_groups)} circuits | "
                f"last depth={target_depth} | circuit_idx={circuit_idx}"
            )

    processed_rows = sorted(
        processed_rows,
        key=lambda r: (
            int(r["target_two_qubit_depth"]),
            int(r["circuit_idx"]),
            int(r["observable_qubit"]),
        ),
    )

    y = np.asarray([r["ideal_expectation"] for r in processed_rows], dtype=float)
    noisy = np.asarray([r["noisy_expectation"] for r in processed_rows], dtype=float)
    zne_quad = np.asarray([r["zne_quadratic_expectation"] for r in processed_rows], dtype=float)
    zne_linear_135 = np.asarray([r["zne_linear_135_expectation"] for r in processed_rows], dtype=float)

    unmitigated_scalar = scalar_metrics(y, noisy)
    zne_quad_scalar = scalar_metrics(y, zne_quad)
    zne_linear_135_scalar = scalar_metrics(y, zne_linear_135)

    l2_quad = compute_l2_metrics(
        processed_rows,
        pred_key="zne_quadratic_expectation",
        label="zne_quadratic",
    )
    l2_linear_135 = compute_l2_metrics(
        processed_rows,
        pred_key="zne_linear_135_expectation",
        label="zne_linear_135",
    )

    per_obs_quad = per_observable_metrics(
        processed_rows,
        pred_key="zne_quadratic_expectation",
        label="zne_quadratic",
    )
    per_obs_linear_135 = per_observable_metrics(
        processed_rows,
        pred_key="zne_linear_135_expectation",
        label="zne_linear_135",
    )

    metrics = {
        "method": "zne_two_qubit_digital_folding_quadratic_extrapolation",
        "scale_factors": SCALE_FACTORS,
        "shots": SHOTS,
        "clip_zne_to_physical_range": CLIP_ZNE_TO_PHYSICAL_RANGE,
        "test_pkl": str(TEST_PKL),
        "n_scalar_rows": len(processed_rows),
        "n_unique_circuits": len(circuit_groups),
        "unmitigated_scalar_metrics": unmitigated_scalar,
        "zne_quadratic_scalar_metrics": zne_quad_scalar,
        "zne_linear_135_scalar_metrics": zne_linear_135_scalar,
        "l2_metrics_quadratic": l2_quad,
        "l2_metrics_linear_135": l2_linear_135,
        "per_observable_metrics_quadratic": per_obs_quad,
        "per_observable_metrics_linear_135": per_obs_linear_135,
    }

    save_predictions_csv(processed_rows, OUT_DIR / "zne_quadratic_predictions_test.csv")

    with (OUT_DIR / "zne_quadratic_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (OUT_DIR / "per_observable_metrics_quadratic.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_obs_quad[0].keys()))
        writer.writeheader()
        writer.writerows(per_obs_quad)

    with (OUT_DIR / "per_observable_metrics_linear_135.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_obs_linear_135[0].keys()))
        writer.writeheader()
        writer.writerows(per_obs_linear_135)

    save_scatter(
        y,
        noisy,
        OUT_DIR / "scatter_unmitigated_noisy_vs_ideal.png",
        title="Unmitigated noisy vs ideal expectation",
        ylabel=r"Noisy $\langle Z_i\rangle$",
    )

    save_scatter(
        y,
        zne_quad,
        OUT_DIR / "scatter_zne_quadratic_vs_ideal.png",
        title="Quadratic ZNE vs ideal expectation",
        ylabel=r"Quadratic ZNE $\langle Z_i\rangle$",
    )

    save_scatter(
        y,
        zne_linear_135,
        OUT_DIR / "scatter_zne_linear_135_vs_ideal.png",
        title="Linear ZNE with scales 1,3,5 vs ideal expectation",
        ylabel=r"Linear ZNE $\langle Z_i\rangle$",
    )

    save_l2_histogram(
        processed_rows,
        pred_key="zne_quadratic_expectation",
        out_path=OUT_DIR / "l2_error_distribution_unmitigated_vs_zne_quadratic.png",
        pred_label="Quadratic ZNE",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("Scalar metrics:")
    print(
        f"Unmitigated    | "
        f"MSE={unmitigated_scalar['mse']:.6g} | "
        f"RMSE={unmitigated_scalar['rmse']:.6g} | "
        f"MAE={unmitigated_scalar['mae']:.6g} | "
        f"R2={unmitigated_scalar['r2']:.4f}"
    )
    print(
        f"ZNE quadratic  | "
        f"MSE={zne_quad_scalar['mse']:.6g} | "
        f"RMSE={zne_quad_scalar['rmse']:.6g} | "
        f"MAE={zne_quad_scalar['mae']:.6g} | "
        f"R2={zne_quad_scalar['r2']:.4f}"
    )
    print(
        f"ZNE linear135  | "
        f"MSE={zne_linear_135_scalar['mse']:.6g} | "
        f"RMSE={zne_linear_135_scalar['rmse']:.6g} | "
        f"MAE={zne_linear_135_scalar['mae']:.6g} | "
        f"R2={zne_linear_135_scalar['r2']:.4f}"
    )

    print("\nPaper-style vector L2 metric:")
    print(
        f"Unmitigated mean L2 = {l2_quad['unmitigated_l2_mean']:.6g} "
        f"± {l2_quad['unmitigated_l2_std']:.6g}"
    )
    print(
        f"ZNE quadratic mean L2 = {l2_quad['zne_quadratic_l2_mean']:.6g} "
        f"± {l2_quad['zne_quadratic_l2_std']:.6g}"
    )
    print(
        f"ZNE quadratic relative L2 improvement = "
        f"{100.0 * l2_quad['zne_quadratic_relative_l2_improvement']:.2f}%"
    )
    print(
        f"ZNE linear135 mean L2 = {l2_linear_135['zne_linear_135_l2_mean']:.6g} "
        f"± {l2_linear_135['zne_linear_135_l2_std']:.6g}"
    )
    print(
        f"ZNE linear135 relative L2 improvement = "
        f"{100.0 * l2_linear_135['zne_linear_135_relative_l2_improvement']:.2f}%"
    )

    print("\nPer observable, quadratic ZNE:")
    for row in per_obs_quad:
        print(
            f"{row['observable']:>2s} | "
            f"unmitigated_MAE={row['unmitigated_mae']:.6g} | "
            f"zne_quadratic_MAE={row['zne_quadratic_mae']:.6g} | "
            f"unmitigated_R2={row['unmitigated_r2']:.4f} | "
            f"zne_quadratic_R2={row['zne_quadratic_r2']:.4f}"
        )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'zne_quadratic_predictions_test.csv'}")
    print(f"  {OUT_DIR / 'zne_quadratic_metrics.json'}")
    print(f"  {OUT_DIR / 'per_observable_metrics_quadratic.csv'}")
    print(f"  {OUT_DIR / 'scatter_zne_quadratic_vs_ideal.png'}")
    print(f"  {OUT_DIR / 'l2_error_distribution_unmitigated_vs_zne_quadratic.png'}")


if __name__ == "__main__":
    main()
