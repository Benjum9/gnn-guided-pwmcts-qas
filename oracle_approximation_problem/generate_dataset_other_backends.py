from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBogotaV2, FakeRomeV2, FakeSantiagoV2


DEFAULT_INPUT_PKL = (
    Path("data_generation_results_constrained_depth_9")
    / "depth_9_without_shallow_cxcx.pkl"
)
DEFAULT_OUTPUT_DIR = Path("data_generation_results_constrained_depth_9")

BACKENDS = {
    "bogota": FakeBogotaV2,
    "rome": FakeRomeV2,
    "santiago": FakeSantiagoV2,
}


def parse_backend_names(value: str) -> List[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in BACKENDS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown backend(s): {unknown}. Choose from {sorted(BACKENDS)}.")
    if not names:
        raise argparse.ArgumentTypeError("At least one backend is required.")
    return names


def extract_meta_and_label(item: Any) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], dict):
        try:
            label = float(item[1])
        except Exception:
            label = None
        return item[0], label

    if isinstance(item, dict):
        return item, None

    return None, None


def load_items(path: Path) -> List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input PKL not found: {path}")

    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {path}, got {type(data)}")

    return data


def load_common_target_statevector(items: List[Any]) -> Statevector:
    target_qasm = None

    for item in items:
        meta, _ = extract_meta_and_label(item)
        if not isinstance(meta, dict):
            continue
        qasm = meta.get("target_qasm_transpiled") or meta.get("target_qasm")
        if isinstance(qasm, bytes):
            qasm = qasm.decode("utf-8")
        if isinstance(qasm, str) and qasm.strip():
            target_qasm = qasm
            break

    if target_qasm is None:
        raise KeyError("Could not find target_qasm_transpiled or target_qasm in the input PKL.")

    for idx, item in enumerate(items):
        meta, _ = extract_meta_and_label(item)
        if not isinstance(meta, dict):
            continue
        qasm = meta.get("target_qasm_transpiled") or meta.get("target_qasm")
        if isinstance(qasm, bytes):
            qasm = qasm.decode("utf-8")
        if isinstance(qasm, str) and qasm.strip() and qasm != target_qasm:
            raise ValueError(f"Inconsistent target QASM found at item {idx}.")

    return Statevector.from_instruction(QuantumCircuit.from_qasm_str(target_qasm))


def make_noisy_simulator(fake_backend_cls) -> AerSimulator:
    backend = fake_backend_cls()
    noise_model = NoiseModel.from_backend(backend)
    return AerSimulator(method="density_matrix", noise_model=noise_model)


def noisy_density_matrix(qc: QuantumCircuit, sim: AerSimulator) -> DensityMatrix:
    qc_dm = qc.copy()
    qc_dm.save_density_matrix()
    result = sim.run(qc_dm).result()
    data0 = result.data(0)
    if "density_matrix" not in data0:
        raise RuntimeError(f"Aer result missing density_matrix. Keys: {list(data0.keys())}")
    return DensityMatrix(data0["density_matrix"])


def output_path_for_backend(output_dir: Path, input_pkl: Path, backend_name: str) -> Path:
    return output_dir / f"{input_pkl.stem}_{backend_name}.pkl"


def convert_for_backend(
    *,
    items: List[Any],
    target_sv: Statevector,
    backend_name: str,
    output_path: Path,
    limit: Optional[int],
) -> None:
    print("=" * 80)
    print(f"Processing backend: {backend_name}")
    print("=" * 80)

    sim = make_noisy_simulator(BACKENDS[backend_name])
    converted: List[Any] = []
    skipped = 0
    start_time = time.perf_counter()

    items_to_process = items if limit is None else items[:limit]

    for idx, item in enumerate(items_to_process):
        meta, _old_label = extract_meta_and_label(item)
        if not isinstance(meta, dict):
            skipped += 1
            continue

        qasm = meta.get("qasm_transpiled") or meta.get("qasm")
        if isinstance(qasm, bytes):
            qasm = qasm.decode("utf-8")
        if not isinstance(qasm, str) or not qasm.strip():
            skipped += 1
            continue

        try:
            candidate_qc = QuantumCircuit.from_qasm_str(qasm)
            candidate_dm = noisy_density_matrix(candidate_qc, sim)
            noisy_fidelity = float(state_fidelity(candidate_dm, target_sv))
        except Exception as exc:
            skipped += 1
            if skipped <= 10 or skipped % 100 == 0:
                print(f"[WARN] Skipping item {idx}: {type(exc).__name__}: {exc}")
            continue

        new_meta = dict(meta)
        previous_backend = new_meta.get("backend")
        new_meta["source_backend"] = previous_backend if previous_backend is not None else "athens"
        new_meta["backend"] = backend_name
        new_meta["backend_name"] = backend_name
        new_meta["noisy_fidelity"] = noisy_fidelity
        new_meta["label_backend"] = backend_name
        new_meta["label_generation"] = "candidate_noisy_density_matrix_vs_ideal_target_statevector"

        converted.append((new_meta, noisy_fidelity))

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(items_to_process):
            elapsed = time.perf_counter() - start_time
            print(
                f"{backend_name}: processed {idx + 1}/{len(items_to_process)} | "
                f"kept={len(converted)} | skipped={skipped} | elapsed={elapsed:.1f}s"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(converted, handle)

    elapsed = time.perf_counter() - start_time
    print(f"Saved: {output_path}")
    print(f"Items written: {len(converted)}")
    print(f"Items skipped: {skipped}")
    print(f"Runtime: {elapsed:.2f}s")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute depth-10 constrained oracle dataset labels on Bogota, Rome, and Santiago."
    )
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backends", type=parse_backend_names, default=list(BACKENDS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_pkl = args.input_pkl
    output_dir = args.output_dir

    items = load_items(input_pkl)
    target_sv = load_common_target_statevector(items)

    print(f"Loaded input PKL: {input_pkl}")
    print(f"Items: {len(items)}")
    print(f"Output directory: {output_dir}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")
    print()

    for backend_name in args.backends:
        convert_for_backend(
            items=items,
            target_sv=target_sv,
            backend_name=backend_name,
            output_path=output_path_for_backend(output_dir, input_pkl, backend_name),
            limit=args.limit,
        )

    print("Done.")


if __name__ == "__main__":
    main()
