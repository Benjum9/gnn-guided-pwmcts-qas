from __future__ import annotations

import csv
import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qiskit import QuantumCircuit, transpile
try:
    from qiskit import qasm2
except Exception:
    qasm2 = None

from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


# ============================================================
# FakeLima import compatibility
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
        "Could not import FakeLima/FakeLimaV2. "
        "Check your Qiskit installation and fake-provider package."
    )


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("mlqem_random_fake_lima_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 4

# Paper-style depths: two-qubit-gate depth up to 18, step 2.
TWO_QUBIT_DEPTHS = list(range(2, 20, 2))

N_TRAIN_PER_DEPTH = 500
N_TEST_PER_DEPTH = 200

# For a fast test, uncomment:
# N_TRAIN_PER_DEPTH = 20
# N_TEST_PER_DEPTH = 10

SHOTS = 10_000

MAX_REJECTION_ATTEMPTS_PER_CIRCUIT = 5000

OPTIMIZATION_LEVEL = 1

RANDOM_SEED = 1234

OBSERVABLE_QUBITS = [0, 1, 2, 3]

BACKEND_NAME = "fake_lima"

TRAIN_PKL = OUT_DIR / "mlqem_random_fake_lima_train.pkl"
TEST_PKL = OUT_DIR / "mlqem_random_fake_lima_test.pkl"

TRAIN_CIRCUIT_PKL = OUT_DIR / "mlqem_random_fake_lima_train_circuit_level.pkl"
TEST_CIRCUIT_PKL = OUT_DIR / "mlqem_random_fake_lima_test_circuit_level.pkl"

SUMMARY_JSON = OUT_DIR / "dataset_summary.json"
SUMMARY_CSV = OUT_DIR / "dataset_summary.csv"


# ============================================================
# Backend / simulator
# ============================================================

backend = make_fake_lima_backend()
noise_model = NoiseModel.from_backend(backend)

# Shot-based noisy simulator, as in the paper's simulated noisy data.
noisy_sim = AerSimulator(
    noise_model=noise_model,
    basis_gates=noise_model.basis_gates,
)


# ============================================================
# Helpers
# ============================================================

def circuit_to_qasm(qc: QuantumCircuit) -> str:
    if qasm2 is not None:
        return qasm2.dumps(qc)

    if hasattr(qc, "qasm"):
        return qc.qasm()

    raise RuntimeError("No QASM exporter available for this Qiskit version.")


def strip_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Make sure the circuit used for Statevector has no measurements.
    """
    qc2 = qc.copy()
    try:
        qc2.remove_final_measurements(inplace=True)
    except Exception:
        pass
    return qc2


def add_all_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Measure qubit i into classical bit i.
    """
    qc_m = qc.copy()
    qc_m.measure_all()
    return qc_m


def get_backend_basis_gates(backend) -> Optional[List[str]]:
    try:
        if getattr(backend, "target", None) is not None:
            return list(backend.target.operation_names)
    except Exception:
        pass

    try:
        return list(backend.configuration().basis_gates)
    except Exception:
        return None


def get_backend_coupling_map(backend):
    try:
        if getattr(backend, "coupling_map", None) is not None:
            return backend.coupling_map
    except Exception:
        pass

    try:
        return backend.configuration().coupling_map
    except Exception:
        return None


def transpile_to_backend(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Transpile to FakeLima/FakeLimaV2 but keep only the first 4 qubits logically.
    """
    return transpile(
        qc,
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=RANDOM_SEED,
    )


def two_qubit_depth(qc: QuantumCircuit) -> int:
    """
    Compute CX/two-qubit depth manually.

    It counts the maximum number of sequential two-qubit operations along any qubit line.
    This is robust and independent of Qiskit's full circuit depth.
    """
    wire_depth = [0 for _ in range(qc.num_qubits)]

    for instr in getattr(qc, "data", []):
        qargs = instr.qubits
        if len(qargs) != 2:
            continue

        q0 = qc.find_bit(qargs[0]).index
        q1 = qc.find_bit(qargs[1]).index

        new_depth = max(wire_depth[q0], wire_depth[q1]) + 1
        wire_depth[q0] = new_depth
        wire_depth[q1] = new_depth

    return max(wire_depth) if wire_depth else 0


def count_ops_safe(qc: QuantumCircuit) -> Dict[str, int]:
    counts = {k: int(v) for k, v in qc.count_ops().items()}
    return counts


def ideal_z_expectations(qc: QuantumCircuit, observable_qubits: List[int]) -> Dict[int, float]:
    """
    Exact noiseless <Z_i> from statevector.

    Qiskit basis convention:
        qubit q corresponds to bit (basis_index >> q) & 1.
    """
    qc_no_meas = strip_final_measurements(qc)
    sv = Statevector.from_instruction(qc_no_meas)

    probs = np.asarray(np.abs(sv.data) ** 2, dtype=float)
    n = qc_no_meas.num_qubits

    out: Dict[int, float] = {}

    for q in observable_qubits:
        exp = 0.0
        for idx, p in enumerate(probs):
            bit = (idx >> q) & 1
            z_val = 1.0 if bit == 0 else -1.0
            exp += z_val * float(p)

        out[q] = float(np.real(exp))

    return out


def noisy_z_expectations_from_counts(
    qc: QuantumCircuit,
    observable_qubits: List[int],
    shots: int,
    seed_simulator: int,
) -> Dict[int, float]:
    """
    Shot-based noisy <Z_i> from measured counts.

    With measure_all(), qubit i is measured into classical bit i.
    Qiskit count bitstrings are displayed as c[n-1]...c[0],
    so the bit for qubit q is bitstring[-1-q].
    """
    qc_m = add_all_measurements(strip_final_measurements(qc))

    sim = noisy_sim

    result = sim.run(
        qc_m,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()

    counts = result.get_counts()

    totals = {q: 0.0 for q in observable_qubits}
    n_shots = 0

    for bitstring_raw, count in counts.items():
        # Sometimes Qiskit includes spaces for multiple classical registers.
        bitstring = bitstring_raw.replace(" ", "")
        n_shots += int(count)

        for q in observable_qubits:
            bit_char = bitstring[-1 - q]
            z_val = 1.0 if bit_char == "0" else -1.0
            totals[q] += z_val * int(count)

    if n_shots <= 0:
        raise RuntimeError("No shots returned by simulator.")

    return {q: float(totals[q] / n_shots) for q in observable_qubits}


def make_random_transpiled_circuit_with_target_2q_depth(
    target_2q_depth: int,
    rng: random.Random,
    seed_base: int,
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Rejection-sample Qiskit random circuits until the transpiled circuit
    has the requested two-qubit depth.

    This is the closest simple way to reproduce the paper's depth-controlled setup.
    """
    best_qc: Optional[QuantumCircuit] = None
    best_depth_diff = 10**9
    best_actual_depth = -1

    for attempt in range(1, MAX_REJECTION_ATTEMPTS_PER_CIRCUIT + 1):
        # Heuristic: random_circuit depth is not the same as two-qubit depth.
        # Use a range around the target and reject after transpilation.
        raw_depth = max(2, int(rng.randint(target_2q_depth, 3 * target_2q_depth + 4)))
        seed = seed_base + attempt

        qc_raw = random_circuit(
            num_qubits=N_QUBITS,
            depth=raw_depth,
            max_operands=2,
            measure=False,
            seed=seed,
        )

        qc_t = transpile_to_backend(qc_raw)

        # Reduce to circuits acting on exactly 4 qubits if transpiler keeps ancillas out.
        # For fake 5-qubit Lima, transpiled circuit may have 5 qubits. That is okay
        # for the graph later, but the observable task uses the first four logical qubits.
        actual_2q_depth = two_qubit_depth(qc_t)
        diff = abs(actual_2q_depth - target_2q_depth)

        if diff < best_depth_diff:
            best_depth_diff = diff
            best_qc = qc_t
            best_actual_depth = actual_2q_depth

        if actual_2q_depth == target_2q_depth:
            return qc_t, {
                "matched_exactly": True,
                "attempts": attempt,
                "target_two_qubit_depth": target_2q_depth,
                "actual_two_qubit_depth": actual_2q_depth,
                "raw_depth": raw_depth,
            }

    if best_qc is None:
        raise RuntimeError("Failed to generate any candidate circuit.")

    print(
        f"Warning: exact depth {target_2q_depth} not found after "
        f"{MAX_REJECTION_ATTEMPTS_PER_CIRCUIT} attempts. "
        f"Using closest depth {best_actual_depth}."
    )

    return best_qc, {
        "matched_exactly": False,
        "attempts": MAX_REJECTION_ATTEMPTS_PER_CIRCUIT,
        "target_two_qubit_depth": target_2q_depth,
        "actual_two_qubit_depth": best_actual_depth,
        "raw_depth": None,
    }


def make_scalar_samples_for_circuit(
    qc: QuantumCircuit,
    split: str,
    target_2q_depth: int,
    circuit_idx: int,
    seed: int,
    generation_info: Dict[str, Any],
) -> Tuple[List[Tuple[Dict[str, Any], float]], Tuple[Dict[str, Any], Dict[str, float]]]:
    """
    Returns:
        scalar_samples: one item per observable Zi
        circuit_level_item: one item with all four expectations as vectors/dicts
    """
    ideal = ideal_z_expectations(qc, OBSERVABLE_QUBITS)
    noisy = noisy_z_expectations_from_counts(
        qc,
        OBSERVABLE_QUBITS,
        shots=SHOTS,
        seed_simulator=seed,
    )

    qasm_t = circuit_to_qasm(strip_final_measurements(qc))
    ops = count_ops_safe(qc)
    actual_2q_depth = two_qubit_depth(qc)

    scalar_samples: List[Tuple[Dict[str, Any], float]] = []

    for q in OBSERVABLE_QUBITS:
        meta = {
            "problem_name": "mlqem_random_circuits",
            "backend": BACKEND_NAME,
            "split": split,
            "target_two_qubit_depth": int(target_2q_depth),
            "actual_two_qubit_depth": int(actual_2q_depth),
            "circuit_idx": int(circuit_idx),
            "seed": int(seed),
            "observable": f"Z{q}",
            "observable_type": "single_qubit_Z",
            "observable_qubit": int(q),
            "qasm_transpiled": qasm_t,
            "noisy_expectation": float(noisy[q]),
            "ideal_expectation": float(ideal[q]),
            "shots": int(SHOTS),
            "num_qubits": int(qc.num_qubits),
            "depth": int(qc.depth()),
            "size": int(qc.size()),
            "count_ops": ops,
            "generation_info": generation_info,
        }

        # Label = ideal/noiseless expectation value.
        scalar_samples.append((meta, float(ideal[q])))

    circuit_meta = {
        "problem_name": "mlqem_random_circuits",
        "backend": BACKEND_NAME,
        "split": split,
        "target_two_qubit_depth": int(target_2q_depth),
        "actual_two_qubit_depth": int(actual_2q_depth),
        "circuit_idx": int(circuit_idx),
        "seed": int(seed),
        "qasm_transpiled": qasm_t,
        "ideal_z_expectations": {f"Z{q}": float(ideal[q]) for q in OBSERVABLE_QUBITS},
        "noisy_z_expectations": {f"Z{q}": float(noisy[q]) for q in OBSERVABLE_QUBITS},
        "shots": int(SHOTS),
        "num_qubits": int(qc.num_qubits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "count_ops": ops,
        "generation_info": generation_info,
    }

    circuit_level_item = (circuit_meta, circuit_meta["ideal_z_expectations"])

    return scalar_samples, circuit_level_item


def generate_split(
    split: str,
    n_per_depth: int,
    seed_offset: int,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Any], List[Dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED + seed_offset)

    scalar_dataset: List[Tuple[Dict[str, Any], float]] = []
    circuit_level_dataset: List[Any] = []
    rows: List[Dict[str, Any]] = []

    global_circuit_idx = 0

    for depth in TWO_QUBIT_DEPTHS:
        print("=" * 80)
        print(f"Generating split={split} | target two-qubit depth={depth}")
        print("=" * 80)

        t0 = time.perf_counter()

        for i in range(n_per_depth):
            seed = RANDOM_SEED + seed_offset + depth * 100_000 + i

            qc_t, gen_info = make_random_transpiled_circuit_with_target_2q_depth(
                target_2q_depth=depth,
                rng=rng,
                seed_base=seed,
            )

            samples, circuit_item = make_scalar_samples_for_circuit(
                qc=qc_t,
                split=split,
                target_2q_depth=depth,
                circuit_idx=global_circuit_idx,
                seed=seed,
                generation_info=gen_info,
            )

            scalar_dataset.extend(samples)
            circuit_level_dataset.append(circuit_item)

            actual_depth = int(circuit_item[0]["actual_two_qubit_depth"])

            rows.append({
                "split": split,
                "target_two_qubit_depth": depth,
                "actual_two_qubit_depth": actual_depth,
                "circuit_idx": global_circuit_idx,
                "seed": seed,
                "matched_exactly": bool(gen_info["matched_exactly"]),
                "attempts": int(gen_info["attempts"]),
                "num_qubits": int(qc_t.num_qubits),
                "depth": int(qc_t.depth()),
                "size": int(qc_t.size()),
                "num_scalar_samples": len(samples),
            })

            global_circuit_idx += 1

            if (i + 1) % 50 == 0 or (i + 1) == n_per_depth:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {i + 1:04d}/{n_per_depth} circuits | "
                    f"scalar samples={len(scalar_dataset)} | "
                    f"elapsed={elapsed:.1f}s"
                )

    return scalar_dataset, circuit_level_dataset, rows


def save_pkl(path: Path, data: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def save_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 80)
    print("ML-QEM random-circuit FakeLima dataset generation")
    print("=" * 80)
    print(f"Backend: {backend}")
    print(f"Noise model basis gates: {noise_model.basis_gates}")
    print(f"Qubits: {N_QUBITS}")
    print(f"Target two-qubit depths: {TWO_QUBIT_DEPTHS}")
    print(f"Train circuits per depth: {N_TRAIN_PER_DEPTH}")
    print(f"Test circuits per depth:  {N_TEST_PER_DEPTH}")
    print(f"Shots: {SHOTS}")
    print(f"Output dir: {OUT_DIR}")
    print()

    train_scalar, train_circuits, train_rows = generate_split(
        split="train",
        n_per_depth=N_TRAIN_PER_DEPTH,
        seed_offset=0,
    )

    test_scalar, test_circuits, test_rows = generate_split(
        split="test",
        n_per_depth=N_TEST_PER_DEPTH,
        seed_offset=10_000_000,
    )

    save_pkl(TRAIN_PKL, train_scalar)
    save_pkl(TEST_PKL, test_scalar)

    save_pkl(TRAIN_CIRCUIT_PKL, train_circuits)
    save_pkl(TEST_CIRCUIT_PKL, test_circuits)

    rows_csv = OUT_DIR / "generation_rows.csv"
    save_rows_csv(rows_csv, train_rows + test_rows)

    summary = {
        "backend": BACKEND_NAME,
        "n_qubits_requested": N_QUBITS,
        "shots": SHOTS,
        "two_qubit_depths": TWO_QUBIT_DEPTHS,
        "n_train_per_depth": N_TRAIN_PER_DEPTH,
        "n_test_per_depth": N_TEST_PER_DEPTH,
        "num_train_circuits": len(train_circuits),
        "num_test_circuits": len(test_circuits),
        "num_train_scalar_samples": len(train_scalar),
        "num_test_scalar_samples": len(test_scalar),
        "observables": [f"Z{q}" for q in OBSERVABLE_QUBITS],
        "train_pkl": str(TRAIN_PKL),
        "test_pkl": str(TEST_PKL),
        "train_circuit_level_pkl": str(TRAIN_CIRCUIT_PKL),
        "test_circuit_level_pkl": str(TEST_CIRCUIT_PKL),
        "rows_csv": str(rows_csv),
    }

    with SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    summary_csv_rows = [
        {"key": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v}
        for k, v in summary.items()
    ]
    save_rows_csv(SUMMARY_CSV, summary_csv_rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved train scalar PKL to:        {TRAIN_PKL}")
    print(f"Saved test scalar PKL to:         {TEST_PKL}")
    print(f"Saved train circuit-level PKL to: {TRAIN_CIRCUIT_PKL}")
    print(f"Saved test circuit-level PKL to:  {TEST_CIRCUIT_PKL}")
    print(f"Saved rows CSV to:                {rows_csv}")
    print(f"Saved summary JSON to:            {SUMMARY_JSON}")
    print()
    print(f"Train scalar samples: {len(train_scalar)}")
    print(f"Test scalar samples:  {len(test_scalar)}")


if __name__ == "__main__":
    main()
