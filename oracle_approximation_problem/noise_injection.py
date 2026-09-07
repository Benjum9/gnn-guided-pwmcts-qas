from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeAthensV2


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data_generation_results_constrained_depth_9")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PKLS = [
    DATA_DIR / "budget_50_circuits.pkl",
    DATA_DIR / "budget_100_circuits.pkl",
    DATA_DIR / "budget_500_circuits.pkl",
    DATA_DIR / "budget_2000_circuits.pkl",
    DATA_DIR / "budget_8000_circuits.pkl",
    DATA_DIR / "budget_16000_circuits.pkl",
]
OUTPUT_PKL = DATA_DIR / "depth_9_without_shallow_cxcx.pkl"

AUGMENT_ALL = True          # keep all originals; if True, add one augmented copy per original
RANDOM_SEED = 42

N_PAIRS_MIN = 1
N_PAIRS_MAX = 16


# ============================================================
# Backend / simulator
# ============================================================

backend = FakeAthensV2()
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(method="density_matrix", noise_model=noise_model)
coupling_edges = list(backend.coupling_map.get_edges())


# ============================================================
# Helpers
# ============================================================

def extract_meta_and_label(item: Any) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
        meta = item[0]
        try:
            label = float(item[1])
        except Exception:
            label = None
        return meta, label
    if isinstance(item, dict):
        return item, None
    return None, None


def load_items(pkl_path: Path) -> List[Tuple[dict, float]]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing PKL: {pkl_path}")
    with pkl_path.open("rb") as f:
        content = pickle.load(f)

    out: List[Tuple[dict, float]] = []
    for item in content:
        meta, label = extract_meta_and_label(item)
        if isinstance(meta, dict) and label is not None:
            out.append((meta, label))
    return out


def load_all_items(pkl_paths: List[Path]) -> List[Tuple[dict, float]]:
    all_items: List[Tuple[dict, float]] = []
    for pkl_path in pkl_paths:
        items = load_items(pkl_path)
        print(f"Loaded {len(items)} items from {pkl_path}")
        all_items.extend(items)
    return all_items


def load_common_target_qc(items: List[Tuple[dict, float]]) -> QuantumCircuit:
    target_qasm = None
    for meta, _ in items:
        qasm = meta.get("target_qasm_transpiled")
        if isinstance(qasm, str) and qasm.strip():
            target_qasm = qasm
            break

    if target_qasm is None:
        raise RuntimeError("Could not find target_qasm_transpiled in input PKL")

    for idx, (meta, _) in enumerate(items):
        qasm = meta.get("target_qasm_transpiled")
        if qasm is None:
            continue
        if qasm != target_qasm:
            raise ValueError(f"Inconsistent target_qasm_transpiled found at item {idx}")

    return QuantumCircuit.from_qasm_str(target_qasm)


def noisy_density_matrix(qc: QuantumCircuit) -> DensityMatrix:
    qc2 = qc.copy()
    qc2.save_density_matrix()
    result = noisy_sim.run(qc2).result()
    data0 = result.data(0)
    if "density_matrix" not in data0:
        raise RuntimeError(f"Aer result missing density_matrix. Keys: {list(data0.keys())}")
    return DensityMatrix(data0["density_matrix"])


def noisy_fidelity_from_target_dm(candidate_qc: QuantumCircuit, target_dm: DensityMatrix) -> float:
    cand_dm = noisy_density_matrix(candidate_qc)
    return float(state_fidelity(cand_dm, target_dm))


def insert_cxcx_pairs(qc: QuantumCircuit, n_pairs: int, rng: random.Random) -> QuantumCircuit:
    out = qc.copy()
    if not coupling_edges:
        raise RuntimeError("No legal coupling-map edges available.")

    for _ in range(n_pairs):
        a, b = rng.choice(coupling_edges)
        out.cx(a, b)
        out.cx(a, b)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    items = load_all_items(INPUT_PKLS)
    target_qc = load_common_target_qc(items)
    target_dm = noisy_density_matrix(target_qc)

    augmented_items: List[Tuple[dict, float]] = []
    skipped_augmented = 0

    for idx, (meta, label) in enumerate(items):
        # Keep original
        original_meta = dict(meta)
        original_meta["accepted_source"] = original_meta.get("accepted_source", "original")
        augmented_items.append((original_meta, float(label)))

        if not AUGMENT_ALL:
            continue

        qasm = meta.get("qasm_transpiled")
        if not isinstance(qasm, str) or not qasm.strip():
            skipped_augmented += 1
            continue

        try:
            base_qc = QuantumCircuit.from_qasm_str(qasm)

            n_pairs = rng.randint(1, 16)

            aug_qc = insert_cxcx_pairs(base_qc, n_pairs=n_pairs, rng=rng)

            # This may fail if SVD does not converge
            new_noisy_fid = noisy_fidelity_from_target_dm(aug_qc, target_dm)

            aug_meta = dict(meta)

            try:
                from qiskit import qasm2
                aug_meta["qasm_transpiled"] = qasm2.dumps(aug_qc)
            except Exception:
                aug_meta["qasm_transpiled"] = aug_qc.qasm() if hasattr(aug_qc, "qasm") else qasm

            aug_meta["noisy_fidelity"] = new_noisy_fid
            aug_meta["accepted_source"] = "cxcx_augmented"
            aug_meta["n_inserted_cxcx_pairs"] = int(n_pairs)
            aug_meta["final_depth"] = int(aug_qc.depth())
            aug_meta["final_size"] = int(aug_qc.size())

            augmented_items.append((aug_meta, float(new_noisy_fid)))

        except Exception as e:
            skipped_augmented += 1
            print(f"Warning: skipped augmented circuit {idx} because of error: {e}")
            continue

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(items):
            print(f"Processed {idx + 1}/{len(items)} originals")

    with OUTPUT_PKL.open("wb") as f:
        pickle.dump(augmented_items, f)

    print()
    print(f"Input items:              {len(items)}")
    print(f"Output items:             {len(augmented_items)}")
    print(f"Skipped augmented items:  {skipped_augmented}")
    print(f"Saved augmented PKL to:   {OUTPUT_PKL}")

if __name__ == "__main__":
    main()
