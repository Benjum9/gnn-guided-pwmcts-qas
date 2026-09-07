import os
import re
import pickle
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
from torch import Tensor

try:
    from torch_geometric.data import Data, InMemoryDataset
except Exception as exc:
    raise ImportError(
        "torch_geometric is required for graph representation. Please install 'torch-geometric'."
    ) from exc

from qiskit import QuantumCircuit


# ============================================================
# Encoding configuration
# ============================================================

NODE_TYPES_BASE: List[str] = [
    "input",
    "measurement",
    "rx",
    "ry",
    "rz",
    "cx",
    "h",
    "x",
]

# Generated Fake*V2 circuits use {rz, sx, x, cx}.
NODE_TYPES_BACKEND_V2: List[str] = [
    "input",
    "measurement",
    "rz",
    "sx",
    "x",
    "cx",
]

# Shared 5-qubit directed coupling map for selected Fake*V2 backends.
# This fixes the dimension of per-edge CX features.
DEFAULT_SHARED_COUPLING_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 0),
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 4), (4, 3),
]

BACKEND_NUM_QUBITS = 5

# For this thesis, all circuits are generated on 5-qubit fake backends.
# The qubit mask therefore has dimension 5, not 25.
QUBIT_MASK_DIM: int = BACKEND_NUM_QUBITS


def get_node_types(node_feature_backend_variant: Optional[str]) -> List[str]:
    if node_feature_backend_variant:
        return NODE_TYPES_BACKEND_V2

    try:
        variant_env = str(os.environ.get("GNN_NODE_TYPES_VARIANT", "")).strip().lower()
        if variant_env == "classification":
            return [t for t in NODE_TYPES_BASE if t != "x"]
    except Exception:
        pass

    return NODE_TYPES_BASE


NODE_FEATURE_BACKEND_EXTRA_DIM: int = 7
ANGLE_ENCODING_VARIANTS = {"none", "sin_cos"}


def get_angle_feature_dim(node_angle_encoding: str = "sin_cos") -> int:
    enc = (node_angle_encoding or "sin_cos").strip().lower()

    if enc == "none":
        return 0

    if enc == "sin_cos":
        return 2

    raise ValueError(f"Unknown node_angle_encoding: {node_angle_encoding}")


def get_node_feature_dim(
    node_feature_backend_variant: Optional[str],
    node_angle_encoding: str = "sin_cos",
) -> int:
    node_types = get_node_types(node_feature_backend_variant)

    dim = len(node_types) + QUBIT_MASK_DIM + get_angle_feature_dim(node_angle_encoding)

    if node_feature_backend_variant:
        dim += NODE_FEATURE_BACKEND_EXTRA_DIM

    return dim


GLOBAL_FEATURE_VARIANTS = {
    "baseline",
    "old_baseline",
    "new_baseline",
    "baseline_backend",
    "enhanced_backend",
    "binned152",
}


def get_global_feature_dim(variant: str) -> int:
    v = (variant or "old_baseline").strip().lower()

    if v == "binned152":
        return 152

    if v in {"baseline", "old_baseline"}:
        return 8

    if v in {"new_baseline", "baseline_backend"}:
        return 14

    if v == "enhanced_backend":
        # new_baseline = 14
        # accumulated noise features = 7
        # per-qubit depth = 5
        # per-qubit gate count = 5
        # per-qubit CX participation = 5
        # per-edge CX counts = 8
        # per-edge CX error sums = 8
        return 14 + 7 + 5 + 5 + 5 + 8 + 8

    raise ValueError(f"Unknown global_feature_variant: {variant}")


def _one_hot(index: int, size: int) -> Tensor:
    vector = torch.zeros(size, dtype=torch.float)
    vector[index] = 1.0
    return vector


def _encode_node_feature(
    node_type: str,
    num_qubits: int,
    active_qubits: Sequence[int],
    angle_feat: Optional[Tensor] = None,
    backend_feat: Optional[Tensor] = None,
    node_feature_backend_variant: Optional[str] = None,
    node_angle_encoding: str = "sin_cos",
) -> Tensor:
    vocab = get_node_types(node_feature_backend_variant)

    if node_type not in vocab:
        raise ValueError(f"Unsupported node type: {node_type}")

    type_one_hot = _one_hot(vocab.index(node_type), len(vocab))

    if num_qubits > QUBIT_MASK_DIM:
        raise ValueError(
            f"Qubit count {num_qubits} exceeds mask capacity {QUBIT_MASK_DIM}. "
            "This representation is configured for 5-qubit backends."
        )

    qubit_mask = torch.zeros(QUBIT_MASK_DIM, dtype=torch.float)

    for q in active_qubits:
        if q < 0 or q >= QUBIT_MASK_DIM:
            raise ValueError(f"Qubit index {q} out of range for {QUBIT_MASK_DIM}-qubit mask")
        qubit_mask[q] = 1.0

    parts: List[Tensor] = [type_one_hot, qubit_mask]

    enc = (node_angle_encoding or "sin_cos").strip().lower()

    if enc == "sin_cos":
        if angle_feat is None:
            angle_feat = torch.zeros(2, dtype=torch.float)
        else:
            angle_feat = angle_feat.to(dtype=torch.float)
            if angle_feat.numel() != 2:
                raise ValueError(f"angle_feat must have dim 2, got {angle_feat.numel()}")

        parts.append(angle_feat)

    elif enc == "none":
        pass

    else:
        raise ValueError(f"Unknown node_angle_encoding: {node_angle_encoding}")

    if backend_feat is not None:
        backend_feat = backend_feat.to(dtype=torch.float)
        parts.append(backend_feat)

    return torch.cat(parts, dim=0)


# ============================================================
# Backend features
# ============================================================

class BackendFeatureProvider:
    """
    Provides per-node and global backend features for selected Fake*V2 backends.

    Per-node backend feature:
        [q0_T1, q0_T2, q1_T1, q1_T2, gate_error, q0_readout_error, q1_readout_error]

    Global backend means:
        [mean_T1, mean_T2, mean_readout_error, mean_1q_gate_error, mean_cx_gate_error]
    """

    def __init__(self, backend_name: str = "backend"):
        self.backend_name = (backend_name or "backend").strip().lower()
        self._backend = None
        self._properties = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from qiskit_ibm_runtime.fake_provider import (
                FakeAthensV2,
                FakeRomeV2,
                FakeSantiagoV2,
                FakeBogotaV2,
            )
        except Exception:
            self._backend = None
            self._properties = None
            return

        name = self.backend_name

        mapping = {
            "fakeathensv2": FakeAthensV2,
            "fake_athensv2": FakeAthensV2,
            "athensv2": FakeAthensV2,
            "athens": FakeAthensV2,

            "fakeromev2": FakeRomeV2,
            "fake_romev2": FakeRomeV2,
            "romev2": FakeRomeV2,
            "rome": FakeRomeV2,

            "fakesantiagov2": FakeSantiagoV2,
            "fake_santiagov2": FakeSantiagoV2,
            "santiagov2": FakeSantiagoV2,
            "santiago": FakeSantiagoV2,

            "fakebogotav2": FakeBogotaV2,
            "fake_bogotav2": FakeBogotaV2,
            "bogotav2": FakeBogotaV2,
            "bogota": FakeBogotaV2,
        }

        cls = mapping.get(name, None)

        if cls is None and name in ("backend", "fake_backend", "default"):
            cls = FakeAthensV2

        if cls is None:
            self._backend = None
            self._properties = None
            return

        try:
            self._backend = cls()
            self._properties = getattr(self._backend, "properties", lambda: None)()
        except Exception:
            self._backend = None
            self._properties = None

    def coupling_edges(self) -> List[Tuple[int, int]]:
        try:
            if self._backend is not None and getattr(self._backend, "coupling_map", None) is not None:
                edges = list(self._backend.coupling_map.get_edges())
                if edges:
                    return [(int(a), int(b)) for a, b in edges]
        except Exception:
            pass
        return list(DEFAULT_SHARED_COUPLING_EDGES)

    def _get_qubit_val(self, qubit: int, prop_name: str) -> float:
        try:
            qubits = getattr(self._properties, "qubits", None)

            if qubits is None:
                return 0.0

            entries = qubits[int(qubit)]

            for nd in entries:
                if getattr(nd, "name", "").lower() == prop_name.lower():
                    val = getattr(nd, "value", None)
                    return float(val) if val is not None else 0.0

            if prop_name.lower() == "readout_error":
                p01 = self._get_qubit_val(qubit, "prob_meas1_prep0")
                p10 = self._get_qubit_val(qubit, "prob_meas0_prep1")

                if p01 > 0.0 or p10 > 0.0:
                    return float((p01 + p10) / 2.0)

        except Exception:
            return 0.0

        return 0.0

    @staticmethod
    def _safe_mean(xs: List[float]) -> float:
        vals = [float(v) for v in xs if isinstance(v, (float, int)) and math.isfinite(float(v))]
        return float(sum(vals) / len(vals)) if vals else 0.0

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        try:
            if x is None or not math.isfinite(x):
                return 0.0
            return float(min(max(x, lo), hi))
        except Exception:
            return 0.0

    @staticmethod
    def _log1p_pos(x: float) -> float:
        return float(math.log1p(x)) if x > 0.0 else 0.0

    def raw_readout_error(self, qubit: int) -> float:
        return float(self._get_qubit_val(int(qubit), "readout_error"))

    def raw_gate_error(self, gate_name: str, qubits: Sequence[int]) -> float:
        return float(self._get_gate_error(gate_name, qubits))

    def _get_target_gate_error(self, gate_name: str, qubits: Sequence[int]) -> Optional[float]:
        try:
            target = getattr(self._backend, "target", None)

            if target is None:
                return None

            gate_name_l = (gate_name or "").lower()
            operation_names = set(getattr(target, "operation_names", []) or [])

            if gate_name_l not in operation_names:
                return None

            props_by_qargs = target[gate_name_l]
            qargs = tuple(int(q) for q in qubits)
            candidates = [qargs]

            if len(qargs) == 2:
                candidates.append((qargs[1], qargs[0]))

            for candidate in candidates:
                if candidate not in props_by_qargs:
                    continue

                err = getattr(props_by_qargs[candidate], "error", None)

                if err is None:
                    continue

                err_f = float(err)

                if math.isfinite(err_f):
                    return err_f

        except Exception:
            return None

        return None

    def _target_gate_errors(self, gate_names: Sequence[str], arity: int) -> List[float]:
        values: List[float] = []

        try:
            target = getattr(self._backend, "target", None)

            if target is None:
                return values

            operation_names = set(getattr(target, "operation_names", []) or [])

            for gate_name in gate_names:
                gate_name_l = gate_name.lower()

                if gate_name_l not in operation_names:
                    continue

                for qargs, props in target[gate_name_l].items():
                    if len(tuple(qargs)) != int(arity):
                        continue

                    err = getattr(props, "error", None)

                    if err is None:
                        continue

                    err_f = float(err)

                    if math.isfinite(err_f):
                        values.append(err_f)

        except Exception:
            return values

        return values

    def global_means(self, num_qubits: int) -> Tensor:
        if num_qubits <= 0 or self._properties is None:
            return torch.zeros(5, dtype=torch.float)

        t1_vals: List[float] = []
        t2_vals: List[float] = []
        ro_vals: List[float] = []

        for q in range(int(num_qubits)):
            t1_vals.append(float(self._get_qubit_val(q, "T1")))
            t2_vals.append(float(self._get_qubit_val(q, "T2")))
            ro_vals.append(float(self._get_qubit_val(q, "readout_error")))

        mean_t1 = self._safe_mean(t1_vals)
        mean_t2 = self._safe_mean(t2_vals)
        mean_ro = self._safe_mean(ro_vals)

        mean_1q_err = 0.0
        mean_cx_err = 0.0

        try:
            oneq_errs: List[float] = self._target_gate_errors(["sx", "x", "rz"], arity=1)
            cx_errs: List[float] = self._target_gate_errors(["cx"], arity=2)

            gates = getattr(self._properties, "gates", []) or []

            def extract_gate_error(g) -> Optional[float]:
                for par in getattr(g, "parameters", []):
                    if getattr(par, "name", "").lower() == "gate_error":
                        val = getattr(par, "value", None)
                        try:
                            return float(val) if val is not None else None
                        except Exception:
                            return None
                return None

            oneq_names = {"sx", "x", "rz"}

            if not oneq_errs or not cx_errs:
                for g in gates:
                    try:
                        gname = getattr(g, "name", "").lower()
                        gqubits = list(getattr(g, "qubits", []))
                        err = extract_gate_error(g)

                        if err is None or not math.isfinite(err):
                            continue

                        if not cx_errs and gname == "cx" and len(gqubits) == 2:
                            cx_errs.append(float(err))
                        elif not oneq_errs and gname in oneq_names and len(gqubits) == 1:
                            oneq_errs.append(float(err))

                    except Exception:
                        continue

            mean_1q_err = self._safe_mean(oneq_errs)
            mean_cx_err = self._safe_mean(cx_errs)

        except Exception:
            mean_1q_err = 0.0
            mean_cx_err = 0.0

        mean_t1 = self._log1p_pos(self._clip(mean_t1, 0.0, 1e6))
        mean_t2 = self._log1p_pos(self._clip(mean_t2, 0.0, 1e6))
        mean_ro = self._log1p_pos(self._clip(mean_ro, 0.0, 0.5))
        mean_1q_err = self._log1p_pos(self._clip(mean_1q_err, 0.0, 0.5))
        mean_cx_err = self._log1p_pos(self._clip(mean_cx_err, 0.0, 0.5))

        return torch.tensor(
            [mean_t1, mean_t2, mean_ro, mean_1q_err, mean_cx_err],
            dtype=torch.float,
        )

    def _get_gate_error(self, gate_name: str, qubits: Sequence[int]) -> float:
        try:
            gate_name_l = (gate_name or "").lower()
            target_err = self._get_target_gate_error(gate_name_l, qubits)

            if target_err is not None:
                return float(target_err)

            props = self._properties

            if props is None:
                return 0.0

            gates = getattr(props, "gates", [])

            if not gates:
                return 0.0

            def extract_gate_error(g) -> Optional[float]:
                for par in getattr(g, "parameters", []):
                    if getattr(par, "name", "").lower() == "gate_error":
                        val = getattr(par, "value", None)
                        try:
                            return float(val) if val is not None else None
                        except Exception:
                            return None
                return None

            if len(qubits) == 2:
                q0, q1 = int(qubits[0]), int(qubits[1])

                for qargs in ([q0, q1], [q1, q0]):
                    for g in gates:
                        try:
                            if getattr(g, "name", "").lower() != "cx":
                                continue

                            g_qubits = list(getattr(g, "qubits", []))

                            if g_qubits == list(qargs):
                                err = extract_gate_error(g)
                                if err is not None:
                                    return float(err)

                        except Exception:
                            continue

                vals: List[float] = []

                for g in gates:
                    try:
                        g_qubits = list(getattr(g, "qubits", []))

                        if len(g_qubits) != 2:
                            continue

                        if getattr(g, "name", "").lower() != "cx":
                            continue

                        err = extract_gate_error(g)

                        if err is not None:
                            vals.append(float(err))

                    except Exception:
                        continue

                if vals:
                    return float(sum(vals) / len(vals))

                return 0.0

            q0 = int(qubits[0])
            oneq_candidates = ["sx", "x", "rz", "id", "u", "u3"]

            if gate_name_l in oneq_candidates:
                oneq_candidates = [gate_name_l] + [g for g in oneq_candidates if g != gate_name_l]

            for cand in oneq_candidates:
                for g in gates:
                    try:
                        if getattr(g, "name", "").lower() != cand:
                            continue

                        g_qubits = list(getattr(g, "qubits", []))

                        if g_qubits == [q0]:
                            err = extract_gate_error(g)
                            if err is not None:
                                return float(err)

                    except Exception:
                        continue

            vals: List[float] = []

            for g in gates:
                try:
                    g_qubits = list(getattr(g, "qubits", []))

                    if len(g_qubits) != 1:
                        continue

                    err = extract_gate_error(g)

                    if err is not None:
                        vals.append(float(err))

                except Exception:
                    continue

            if vals:
                return float(sum(vals) / len(vals))

            return 0.0

        except Exception:
            return 0.0

    def features_for(self, gate_name: str, qubits: Sequence[int]) -> Tensor:
        if not qubits:
            return torch.zeros(NODE_FEATURE_BACKEND_EXTRA_DIM, dtype=torch.float)

        q0 = int(qubits[0])
        q1 = int(qubits[1]) if len(qubits) > 1 else None

        q0_t1 = self._get_qubit_val(q0, "T1")
        q0_t2 = self._get_qubit_val(q0, "T2")
        q1_t1 = self._get_qubit_val(q1, "T1") if q1 is not None else 0.0
        q1_t2 = self._get_qubit_val(q1, "T2") if q1 is not None else 0.0

        gate_err = self._get_gate_error(gate_name, [q0] + ([q1] if q1 is not None else []))

        q0_ro = self._get_qubit_val(q0, "readout_error")
        q1_ro = self._get_qubit_val(q1, "readout_error") if q1 is not None else 0.0

        q0_t1 = self._log1p_pos(self._clip(q0_t1, 0.0, 1e6))
        q0_t2 = self._log1p_pos(self._clip(q0_t2, 0.0, 1e6))
        q1_t1 = self._log1p_pos(self._clip(q1_t1, 0.0, 1e6))
        q1_t2 = self._log1p_pos(self._clip(q1_t2, 0.0, 1e6))

        gate_err = self._log1p_pos(self._clip(gate_err, 0.0, 0.5))
        q0_ro = self._log1p_pos(self._clip(q0_ro, 0.0, 0.5))
        q1_ro = self._log1p_pos(self._clip(q1_ro, 0.0, 0.5))

        return torch.tensor(
            [q0_t1, q0_t2, q1_t1, q1_t2, gate_err, q0_ro, q1_ro],
            dtype=torch.float,
        )


# ============================================================
# Global features
# ============================================================

def _count_gates_qiskit(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"rx": 0, "ry": 0, "rz": 0, "cx": 0, "h": 0}

    for instr in circuit.data:
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1

    return counts


def _count_gates_backend_v2(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"rz": 0, "sx": 0, "x": 0, "cx": 0}

    for instr in getattr(circuit, "data", []):
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1

    return counts


def _safe_float_param(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _global_features_old_baseline(circuit: QuantumCircuit, num_qubits: int) -> Tensor:
    gate_counts = _count_gates_backend_v2(circuit)
    num_param = gate_counts["rz"]
    total_gates = sum(gate_counts.values())
    depth = float(circuit.depth()) if hasattr(circuit, "depth") else float(total_gates)

    return torch.tensor(
        [
            depth,
            float(num_param),
            float(num_qubits),
            float(total_gates),
            float(gate_counts["rz"]),
            float(gate_counts["sx"]),
            float(gate_counts["x"]),
            float(gate_counts["cx"]),
        ],
        dtype=torch.float,
    )


def _global_features_new_baseline(
    circuit: QuantumCircuit,
    num_qubits: int,
    backend_feature_provider: Optional[BackendFeatureProvider],
    noiseless_fidelity: Optional[float] = None,
) -> Tensor:
    old = _global_features_old_baseline(circuit, num_qubits)

    if noiseless_fidelity is None:
        nf_feat = 0.0
    else:
        try:
            nf = float(noiseless_fidelity)
            nf_feat = nf if math.isfinite(nf) else 0.0
        except Exception:
            nf_feat = 0.0

    backend_means = (
        backend_feature_provider.global_means(num_qubits)
        if backend_feature_provider is not None
        else torch.zeros(5, dtype=torch.float)
    )

    return torch.cat([old, torch.tensor([nf_feat], dtype=torch.float), backend_means], dim=0)


def _compute_enhanced_backend_features(
    circuit: QuantumCircuit,
    num_qubits: int,
    backend_feature_provider: Optional[BackendFeatureProvider],
) -> Tensor:
    if backend_feature_provider is None:
        return torch.zeros(7 + 5 + 5 + 5 + 8 + 8, dtype=torch.float)

    n_fixed = BACKEND_NUM_QUBITS
    edge_list = DEFAULT_SHARED_COUPLING_EDGES
    edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}

    per_qubit_depth = [0.0 for _ in range(n_fixed)]
    per_qubit_gate_count = [0.0 for _ in range(n_fixed)]
    per_qubit_cx_participation = [0.0 for _ in range(n_fixed)]

    wire_depth = [0 for _ in range(max(num_qubits, n_fixed))]

    per_edge_cx_count = [0.0 for _ in edge_list]
    per_edge_cx_error_sum = [0.0 for _ in edge_list]

    gate_errors: List[float] = []
    oneq_error_sum = 0.0
    cx_error_sum = 0.0
    total_gate_error_sum = 0.0
    max_gate_error = 0.0
    log_success = 0.0

    used_qubits = set()

    def _q_index(q) -> int:
        if hasattr(q, "index") and isinstance(getattr(q, "index"), int):
            return int(getattr(q, "index"))
        if hasattr(q, "_index") and isinstance(getattr(q, "_index"), int):
            return int(getattr(q, "_index"))
        match = re.search(r"(,\s*)(\d+)(\))$", str(q))
        if match:
            return int(match.group(2))
        raise AttributeError("Unable to extract qubit index")

    for instr in getattr(circuit, "data", []):
        name = instr.operation.name.lower()
        if name == "barrier":
            continue

        if name in {"rz", "sx", "x"}:
            q = _q_index(instr.qubits[0])
            used_qubits.add(q)

            if q < n_fixed:
                per_qubit_gate_count[q] += 1.0

            wire_depth[q] += 1

            err = backend_feature_provider.raw_gate_error(name, [q])
            gate_errors.append(err)
            oneq_error_sum += err
            total_gate_error_sum += err
            max_gate_error = max(max_gate_error, err)
            log_success += math.log(max(1.0 - err, 1e-12))

        elif name == "cx":
            q0 = _q_index(instr.qubits[0])
            q1 = _q_index(instr.qubits[1])
            used_qubits.add(q0)
            used_qubits.add(q1)

            if q0 < n_fixed:
                per_qubit_gate_count[q0] += 1.0
                per_qubit_cx_participation[q0] += 1.0
            if q1 < n_fixed:
                per_qubit_gate_count[q1] += 1.0
                per_qubit_cx_participation[q1] += 1.0

            new_depth = max(wire_depth[q0], wire_depth[q1]) + 1
            wire_depth[q0] = new_depth
            wire_depth[q1] = new_depth

            err = backend_feature_provider.raw_gate_error("cx", [q0, q1])
            gate_errors.append(err)
            cx_error_sum += err
            total_gate_error_sum += err
            max_gate_error = max(max_gate_error, err)
            log_success += math.log(max(1.0 - err, 1e-12))

            edge = (q0, q1)
            if edge in edge_to_idx:
                idx = edge_to_idx[edge]
                per_edge_cx_count[idx] += 1.0
                per_edge_cx_error_sum[idx] += err

    for q in range(min(n_fixed, len(wire_depth))):
        per_qubit_depth[q] = float(wire_depth[q])

    mean_gate_error = float(sum(gate_errors) / len(gate_errors)) if gate_errors else 0.0
    expected_error_proxy = float(1.0 - math.exp(log_success)) if gate_errors else 0.0

    used_ro_sum = 0.0
    for q in used_qubits:
        if q < n_fixed:
            used_ro_sum += backend_feature_provider.raw_readout_error(q)

    accumulated = torch.tensor(
        [
            total_gate_error_sum,
            oneq_error_sum,
            cx_error_sum,
            mean_gate_error,
            max_gate_error,
            expected_error_proxy,
            used_ro_sum,
        ],
        dtype=torch.float,
    )

    return torch.cat(
        [
            accumulated,
            torch.tensor(per_qubit_depth, dtype=torch.float),
            torch.tensor(per_qubit_gate_count, dtype=torch.float),
            torch.tensor(per_qubit_cx_participation, dtype=torch.float),
            torch.tensor(per_edge_cx_count, dtype=torch.float),
            torch.tensor(per_edge_cx_error_sum, dtype=torch.float),
        ],
        dim=0,
    )


def _global_features_enhanced_backend(
    circuit: QuantumCircuit,
    num_qubits: int,
    backend_feature_provider: Optional[BackendFeatureProvider],
    noiseless_fidelity: Optional[float] = None,
) -> Tensor:
    new_base = _global_features_new_baseline(
        circuit,
        num_qubits,
        backend_feature_provider,
        noiseless_fidelity=noiseless_fidelity,
    )
    enhanced = _compute_enhanced_backend_features(
        circuit,
        num_qubits,
        backend_feature_provider,
    )
    return torch.cat([new_base, enhanced], dim=0)


def _global_features_binned152(circuit: QuantumCircuit) -> Tensor:
    num_bins = 50
    bin_width = 2 * math.pi / num_bins

    def angle_to_bin(angle: float) -> int:
        a = angle % (2 * math.pi)
        idx = int(a // bin_width)
        return min(idx, num_bins - 1)

    rx_bins = torch.zeros(num_bins, dtype=torch.float)
    ry_bins = torch.zeros(num_bins, dtype=torch.float)
    rz_bins = torch.zeros(num_bins, dtype=torch.float)

    h_count = 0.0
    cx_count = 0.0

    for instr in getattr(circuit, "data", []):
        name = instr.operation.name.lower()

        if name in {"rx", "ry", "rz"}:
            params = getattr(instr.operation, "params", [])
            if not params:
                continue
            val = _safe_float_param(params[0])
            if val is None:
                continue
            b = angle_to_bin(val)
            if name == "rx":
                rx_bins[b] += 1.0
            elif name == "ry":
                ry_bins[b] += 1.0
            else:
                rz_bins[b] += 1.0

        elif name == "h":
            h_count += 1.0

        elif name == "cx":
            cx_count += 1.0

    return torch.cat(
        [rx_bins, ry_bins, rz_bins, torch.tensor([h_count, cx_count], dtype=torch.float)],
        dim=0,
    )


def _angle_sin_cos_from_instr(op_name: str, instr) -> Tensor:
    if op_name != "rz":
        return torch.zeros(2, dtype=torch.float)

    params = getattr(instr.operation, "params", [])
    if not params:
        return torch.zeros(2, dtype=torch.float)

    theta = _safe_float_param(params[0])
    if theta is None or not math.isfinite(theta):
        return torch.zeros(2, dtype=torch.float)

    return torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float)


# ============================================================
# QASM to PyG graph
# ============================================================

def qasm_to_pyg_graph(
    qasm_str: str,
    num_qubits_hint: Optional[int] = None,
    global_feature_variant: str = "new_baseline",
    backend_feature_provider: Optional[BackendFeatureProvider] = None,
    noiseless_fidelity: Optional[float] = None,
    node_angle_encoding: str = "sin_cos",
) -> Tuple[Data, Dict[str, int]]:

    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    num_qubits = len(circuit.qubits)

    if num_qubits == 0:
        match = re.search(r"qreg\s+\w+\[(\d+)\];", qasm_str)
        if match:
            num_qubits = int(match.group(1))
        elif num_qubits_hint is not None:
            num_qubits = int(num_qubits_hint)

    x_features: List[Tensor] = []
    edge_src: List[int] = []
    edge_dst: List[int] = []

    last_node_for_qubit: List[int] = []
    node_backend_variant = getattr(backend_feature_provider, "backend_name", None)
    vocab = get_node_types(node_backend_variant)

    for q in range(num_qubits):
        idx = len(x_features)
        backend_feat = None
        if backend_feature_provider is not None:
            backend_feat = backend_feature_provider.features_for("input", [q])

        x_features.append(
            _encode_node_feature(
                "input",
                num_qubits,
                [q],
                angle_feat=None,
                backend_feat=backend_feat,
                node_feature_backend_variant=node_backend_variant,
                node_angle_encoding=node_angle_encoding,
            )
        )
        last_node_for_qubit.append(idx)

    def _q_index(q) -> int:
        if hasattr(q, "index") and isinstance(getattr(q, "index"), int):
            return int(getattr(q, "index"))
        if hasattr(q, "_index") and isinstance(getattr(q, "_index"), int):
            return int(getattr(q, "_index"))
        match = re.search(r"(,\s*)(\d+)(\))$", str(q))
        if match:
            return int(match.group(2))
        raise AttributeError("Unable to extract qubit index from Qiskit Qubit object")

    for instr in getattr(circuit, "data", []):
        op_name = instr.operation.name.lower()

        if op_name == "barrier":
            continue

        if op_name in {"rz", "sx", "x"}:
            if op_name not in vocab:
                continue

            qubit = _q_index(instr.qubits[0])
            node_idx = len(x_features)

            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for(op_name, [qubit])

            angle_feat = (
                _angle_sin_cos_from_instr(op_name, instr)
                if (node_angle_encoding or "sin_cos").strip().lower() == "sin_cos"
                else None
            )

            x_features.append(
                _encode_node_feature(
                    op_name,
                    num_qubits,
                    [qubit],
                    angle_feat=angle_feat,
                    backend_feat=backend_feat,
                    node_feature_backend_variant=node_backend_variant,
                    node_angle_encoding=node_angle_encoding,
                )
            )

            edge_src.append(last_node_for_qubit[qubit])
            edge_dst.append(node_idx)
            last_node_for_qubit[qubit] = node_idx

        elif op_name == "cx":
            control = _q_index(instr.qubits[0])
            target = _q_index(instr.qubits[1])
            node_idx = len(x_features)

            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for(op_name, [control, target])

            x_features.append(
                _encode_node_feature(
                    op_name,
                    num_qubits,
                    [control, target],
                    angle_feat=None,
                    backend_feat=backend_feat,
                    node_feature_backend_variant=node_backend_variant,
                    node_angle_encoding=node_angle_encoding,
                )
            )

            edge_src.extend([last_node_for_qubit[control], last_node_for_qubit[target]])
            edge_dst.extend([node_idx, node_idx])

            last_node_for_qubit[control] = node_idx
            last_node_for_qubit[target] = node_idx

        else:
            continue

    if num_qubits > 0:
        for q in range(num_qubits):
            node_idx = len(x_features)

            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for("measurement", [q])

            x_features.append(
                _encode_node_feature(
                    "measurement",
                    num_qubits,
                    [q],
                    angle_feat=None,
                    backend_feat=backend_feat,
                    node_feature_backend_variant=node_backend_variant,
                    node_angle_encoding=node_angle_encoding,
                )
            )

            edge_src.append(last_node_for_qubit[q])
            edge_dst.append(node_idx)
            last_node_for_qubit[q] = node_idx

    if not x_features:
        vocab_len = len(get_node_types(node_backend_variant))
        node_dim = (
            vocab_len
            + QUBIT_MASK_DIM
            + get_angle_feature_dim(node_angle_encoding)
            + (NODE_FEATURE_BACKEND_EXTRA_DIM if backend_feature_provider is not None else 0)
        )
        x = torch.zeros((0, node_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.stack(x_features, dim=0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    v = (global_feature_variant or "new_baseline").strip().lower()

    if v in {"baseline", "old_baseline"}:
        global_feat = _global_features_old_baseline(circuit, num_qubits)
        meta_counts = _count_gates_backend_v2(circuit)

    elif v in {"new_baseline", "baseline_backend"}:
        global_feat = _global_features_new_baseline(
            circuit,
            num_qubits,
            backend_feature_provider,
            noiseless_fidelity=noiseless_fidelity,
        )
        meta_counts = _count_gates_backend_v2(circuit)

    elif v == "enhanced_backend":
        global_feat = _global_features_enhanced_backend(
            circuit,
            num_qubits,
            backend_feature_provider,
            noiseless_fidelity=noiseless_fidelity,
        )
        meta_counts = _count_gates_backend_v2(circuit)

    elif v == "binned152":
        global_feat = _global_features_binned152(circuit)
        meta_counts = _count_gates_qiskit(circuit)

    else:
        raise ValueError(f"Unknown global_feature_variant: {global_feature_variant}")

    global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

    data = Data(x=x, edge_index=edge_index)
    data.num_qubits = num_qubits
    data.global_features = global_feat

    return data, meta_counts


# ============================================================
# Dataset
# ============================================================

class QuantumCircuitGraphDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        pkl_paths: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        global_feature_variant: str = "new_baseline",
        node_feature_backend_variant: Optional[str] = None,
        node_angle_encoding: str = "sin_cos",
    ):
        self.pkl_paths = pkl_paths
        self.global_feature_variant = global_feature_variant
        self.node_feature_backend_variant = node_feature_backend_variant
        self.node_angle_encoding = node_angle_encoding

        super().__init__(root, transform, pre_transform)

        processed_path = self.processed_paths[0]

        try:
            self.data, self.slices = torch.load(processed_path, weights_only=False)
        except Exception:
            try:
                from torch.serialization import add_safe_globals, safe_globals
                try:
                    from torch_geometric.data import Data
                    from torch_geometric.data.data import DataEdgeAttr
                    add_safe_globals([Data, DataEdgeAttr])
                except Exception:
                    pass
                with safe_globals([]):
                    self.data, self.slices = torch.load(processed_path)
            except Exception:
                try:
                    if os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    pass
                self.process()
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        gdim = get_global_feature_dim(self.global_feature_variant)
        node_dim = get_node_feature_dim(
            self.node_feature_backend_variant,
            self.node_angle_encoding,
        )

        backend_tag = self.node_feature_backend_variant or "none"
        angle_tag = self.node_angle_encoding or "sin_cos"

        return [
            (
                f"graphs.node{node_dim}.gfeat{gdim}."
                f"{self.global_feature_variant}."
                f"backend_{backend_tag}."
                f"angle_{angle_tag}."
                f"qasm_transpiled.feature_compare.mask5.dataset.pt"
            )
        ]

    def download(self):
        return

    def _iter_items_from_pkls(self):
        if not self.pkl_paths:
            raise ValueError("pkl_paths must be provided with absolute paths to .pkl files")

        for pkl_path in self.pkl_paths:
            if not os.path.isabs(pkl_path):
                raise ValueError("Please provide absolute paths for reliability in tooling")

            with open(pkl_path, "rb") as f:
                content = pickle.load(f)

            for item in content:
                if isinstance(item, tuple) and len(item) == 2:
                    circ_info, label = item
                    yield circ_info, float(label)

                elif isinstance(item, dict):
                    yield item, None

                else:
                    continue

    def process(self):
        data_list: List[Data] = []

        backend_provider = None
        if self.node_feature_backend_variant:
            try:
                backend_provider = BackendFeatureProvider(self.node_feature_backend_variant)
            except Exception:
                backend_provider = None

        for circ_info, label in self._iter_items_from_pkls():
            if not isinstance(circ_info, dict):
                continue

            qasm_t = circ_info.get("qasm_transpiled", circ_info.get("qasm", None))

            if not isinstance(qasm_t, str):
                continue

            graph, _ = qasm_to_pyg_graph(
                qasm_t,
                global_feature_variant=self.global_feature_variant,
                backend_feature_provider=backend_provider,
                noiseless_fidelity=circ_info.get("oracle_noiseless_fidelity", None),
                node_angle_encoding=self.node_angle_encoding,
            )

            if label is not None:
                val = float(label)
                if not math.isfinite(val):
                    continue
                graph.y = torch.tensor([val], dtype=torch.float)

            data_list.append(graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def encode_single_qasm(
    qasm_transpiled_str: str,
    label: Optional[float] = None,
    global_feature_variant: str = "new_baseline",
    node_feature_backend_variant: Optional[str] = None,
    noiseless_fidelity: Optional[float] = None,
    node_angle_encoding: str = "sin_cos",
) -> Data:
    provider = BackendFeatureProvider(node_feature_backend_variant) if node_feature_backend_variant else None

    data, _ = qasm_to_pyg_graph(
        qasm_transpiled_str,
        global_feature_variant=global_feature_variant,
        backend_feature_provider=provider,
        noiseless_fidelity=noiseless_fidelity,
        node_angle_encoding=node_angle_encoding,
    )

    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)

    return data
