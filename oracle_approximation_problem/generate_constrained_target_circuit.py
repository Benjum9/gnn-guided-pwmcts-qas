from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

from qiskit import QuantumCircuit

try:
    from qiskit import qasm2
except Exception:
    qasm2 = None

from qiskit_ibm_runtime.fake_provider import FakeAthensV2

from random_circuit_generation import get_directed_coupling_edges


DEFAULT_DEPTH = 8
DEFAULT_SEED = 43
DEFAULT_NUM_QUBITS = 5
DEFAULT_OUT_DIR = Path("generated_target_circuits")

ONE_QUBIT_GATES = ("rz", "sx", "x")


def qubit_index(qc: QuantumCircuit, qubit) -> int:
    return qc.find_bit(qubit).index


def circuit_to_qasm(qc: QuantumCircuit) -> str:
    if qasm2 is not None:
        return qasm2.dumps(qc)
    if hasattr(qc, "qasm"):
        return qc.qasm()
    raise RuntimeError("No QASM exporter available for this Qiskit version.")


def append_initial_hadamard_layers(qc: QuantumCircuit) -> None:
    for q in range(qc.num_qubits):
        qc.rz(math.pi / 2.0, q)
    for q in range(qc.num_qubits):
        qc.sx(q)
    for q in range(qc.num_qubits):
        qc.rz(math.pi / 2.0, q)


def is_valid_one_qubit_gate(last_gate_on_qubit: List[Optional[Tuple]], gate: str, q: int) -> bool:
    previous = last_gate_on_qubit[q]
    return previous is None or previous[0] != gate


def is_valid_cx(
    last_gate_on_qubit: List[Optional[Tuple]],
    non_empty_qubit: List[bool],
    control: int,
    target: int,
) -> bool:
    if not non_empty_qubit[control]:
        return False
    signature = ("cx", control, target)
    return last_gate_on_qubit[control] != signature and last_gate_on_qubit[target] != signature


def append_gate(
    qc: QuantumCircuit,
    last_gate_on_qubit: List[Optional[Tuple]],
    non_empty_qubit: List[bool],
    gate: str,
    qubits: Tuple[int, ...],
    rng: random.Random,
    theta: Optional[float] = None,
) -> None:
    if gate == "rz":
        q = qubits[0]
        if theta is None:
            theta = rng.uniform(0.0, 2.0 * math.pi)
        qc.rz(theta, q)
        last_gate_on_qubit[q] = ("rz", q)
        non_empty_qubit[q] = True
    elif gate == "sx":
        q = qubits[0]
        qc.sx(q)
        last_gate_on_qubit[q] = ("sx", q)
        non_empty_qubit[q] = True
    elif gate == "x":
        q = qubits[0]
        qc.x(q)
        last_gate_on_qubit[q] = ("x", q)
        non_empty_qubit[q] = True
    elif gate == "cx":
        control, target = qubits
        qc.cx(control, target)
        signature = ("cx", control, target)
        last_gate_on_qubit[control] = signature
        last_gate_on_qubit[target] = signature
        non_empty_qubit[control] = True
        non_empty_qubit[target] = True
    else:
        raise ValueError(f"Unexpected gate: {gate}")


def valid_candidates(
    *,
    num_qubits: int,
    legal_edges: List[Tuple[int, int]],
    last_gate_on_qubit: List[Optional[Tuple]],
    non_empty_qubit: List[bool],
) -> List[Tuple[str, Tuple[int, ...]]]:
    candidates: List[Tuple[str, Tuple[int, ...]]] = []

    for gate in ONE_QUBIT_GATES:
        for q in range(num_qubits):
            if is_valid_one_qubit_gate(last_gate_on_qubit, gate, q):
                candidates.append((gate, (q,)))

    for control, target in legal_edges:
        if is_valid_cx(last_gate_on_qubit, non_empty_qubit, control, target):
            candidates.append(("cx", (control, target)))

    return candidates


def make_constrained_target_circuit(
    *,
    target_depth: int,
    seed: int = DEFAULT_SEED,
    backend=None,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    max_attempts: int = 10000,
) -> QuantumCircuit:
    if target_depth < 3:
        raise ValueError("target_depth must be at least 3 because the initial H decomposition has depth 3.")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    if backend is None:
        backend = FakeAthensV2()

    rng = random.Random(seed)
    legal_edges = get_directed_coupling_edges(backend, num_qubits)

    qc = QuantumCircuit(num_qubits)
    append_initial_hadamard_layers(qc)

    last_gate_on_qubit: List[Optional[Tuple]] = [("rz", q) for q in range(num_qubits)]
    non_empty_qubit = [True for _ in range(num_qubits)]

    attempts = 0
    while qc.depth() < target_depth:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not reach target depth={target_depth} after {max_attempts} attempts. "
                f"Current depth={qc.depth()}."
            )

        candidates = valid_candidates(
            num_qubits=num_qubits,
            legal_edges=legal_edges,
            last_gate_on_qubit=last_gate_on_qubit,
            non_empty_qubit=non_empty_qubit,
        )
        if not candidates:
            raise RuntimeError("No valid gate candidate remains under the current constraints.")

        gate, qubits = rng.choice(candidates)
        theta = rng.uniform(0.0, 2.0 * math.pi) if gate == "rz" else None

        trial_qc = qc.copy()
        append_gate(
            trial_qc,
            last_gate_on_qubit.copy(),
            non_empty_qubit.copy(),
            gate,
            qubits,
            rng,
            theta=theta,
        )

        if trial_qc.depth() <= target_depth:
            append_gate(qc, last_gate_on_qubit, non_empty_qubit, gate, qubits, rng, theta=theta)

    validate_constraints(qc, target_depth)
    return qc


def validate_constraints(qc: QuantumCircuit, target_depth: int) -> None:
    if qc.depth() != target_depth:
        raise AssertionError(f"Expected depth {target_depth}, got {qc.depth()}.")

    expected_initial = ["rz", "sx", "rz"]
    for q in range(qc.num_qubits):
        wire_ops = []
        for instruction in qc.data:
            operation = instruction.operation
            qubits = [qubit_index(qc, qb) for qb in instruction.qubits]
            if q in qubits:
                wire_ops.append(operation.name)
        if wire_ops[:3] != expected_initial:
            raise AssertionError(f"Qubit {q} does not start with rz, sx, rz. Found {wire_ops[:3]}.")

    last_gate_on_qubit: List[Optional[Tuple]] = [None for _ in range(qc.num_qubits)]
    non_empty_qubit = [False for _ in range(qc.num_qubits)]

    for instruction in qc.data:
        operation = instruction.operation
        name = operation.name
        qubits = tuple(qubit_index(qc, qb) for qb in instruction.qubits)

        if name in ONE_QUBIT_GATES:
            q = qubits[0]
            if last_gate_on_qubit[q] is not None and last_gate_on_qubit[q][0] == name:
                raise AssertionError(f"Repeated sequential {name} gate on q{q}.")
            last_gate_on_qubit[q] = (name, q)
            non_empty_qubit[q] = True
        elif name == "cx":
            control, target = qubits
            if not non_empty_qubit[control]:
                raise AssertionError(f"CX control q{control} is empty.")
            signature = ("cx", control, target)
            if last_gate_on_qubit[control] == signature or last_gate_on_qubit[target] == signature:
                raise AssertionError(f"Repeated sequential cx on q{control}, q{target}.")
            last_gate_on_qubit[control] = signature
            last_gate_on_qubit[target] = signature
            non_empty_qubit[control] = True
            non_empty_qubit[target] = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a constrained random target circuit.")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = FakeAthensV2()
    qc = make_constrained_target_circuit(
        target_depth=args.depth,
        seed=args.seed,
        backend=backend,
        num_qubits=args.num_qubits,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qasm_path = args.out_dir / f"target_depth_{args.depth}_seed_{args.seed}.qasm"
    qasm_path.write_text(circuit_to_qasm(qc))

    print("=" * 80)
    print("Constrained target circuit")
    print("=" * 80)
    print(f"depth:      {qc.depth()}")
    print(f"size:       {qc.size()}")
    print(f"num_qubits: {qc.num_qubits}")
    print(f"saved qasm: {qasm_path}")
    print()
    print(qc)


if __name__ == "__main__":
    main()
