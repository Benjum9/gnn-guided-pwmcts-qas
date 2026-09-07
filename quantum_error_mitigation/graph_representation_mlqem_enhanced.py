from __future__ import annotations

import os
import re
import pickle
import math
from typing import Dict, List, Optional, Sequence, Tuple

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

# Error-mitigation dataset is generated on FakeLima-style 5-qubit backends.
BACKEND_NUM_QUBITS = 5
QUBIT_MASK_DIM = BACKEND_NUM_QUBITS
OBSERVABLE_MASK_DIM = BACKEND_NUM_QUBITS

# Backend-native IBM basis. Include 'id' because some FakeLima transpiled
# circuits may contain identity/id operations depending on Qiskit version.
NODE_TYPES_BACKEND: List[str] = [
    "input",
    "measurement",
    "id",
    "rz",
    "sx",
    "x",
    "cx",
]

# Fallback generic vocabulary if no backend provider is used.
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

# Shared 5-qubit directed coupling map used by Lima-like 5-qubit IBM fake backends.
# This fixes the dimension/order of the per-edge CX features.
DEFAULT_SHARED_COUPLING_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 0),
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 4), (4, 3),
]

NODE_FEATURE_BACKEND_EXTRA_DIM = 7
ANGLE_ENCODING_VARIANTS = {"none", "sin_cos"}

# Global feature layout:
#   circuit statistics                       9
#     [depth, #param, #qubits, #gates, #id, #rz, #sx, #x, #cx]
#   noisy expectation value                  1
#   observable-qubit one-hot mask             5
#   backend mean calibration features         5
#   accumulated circuit-specific noise        3
#     [sum_1q_error, sum_cx_error, used_readout_error_sum]
#   per-qubit depth                           5
#   per-qubit gate count                      5
#   per-edge CX count                         8
#   per-edge CX error sum                     8
GLOBAL_FEATURE_DIM = 9 + 1 + OBSERVABLE_MASK_DIM + 5 + 3 + 5 + 5 + 8 + 8

# Index of noisy expectation value inside global_features.
NOISY_EXPECTATION_INDEX = 9


def get_node_types(node_feature_backend_variant: Optional[str] = "backend") -> List[str]:
    if node_feature_backend_variant:
        return NODE_TYPES_BACKEND
    return NODE_TYPES_BASE


def get_angle_feature_dim(node_angle_encoding: str = "none") -> int:
    enc = (node_angle_encoding or "none").strip().lower()
    if enc == "none":
        return 0
    if enc == "sin_cos":
        return 2
    raise ValueError(f"Unknown node_angle_encoding: {node_angle_encoding}")


def get_node_feature_dim(
    node_feature_backend_variant: Optional[str] = "backend",
    node_angle_encoding: str = "none",
) -> int:
    dim = len(get_node_types(node_feature_backend_variant))
    dim += QUBIT_MASK_DIM
    dim += get_angle_feature_dim(node_angle_encoding)
    if node_feature_backend_variant:
        dim += NODE_FEATURE_BACKEND_EXTRA_DIM
    return dim


def get_global_feature_dim() -> int:
    return GLOBAL_FEATURE_DIM


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
    node_feature_backend_variant: Optional[str] = "backend",
    node_angle_encoding: str = "none",
) -> Tensor:
    vocab = get_node_types(node_feature_backend_variant)

    if node_type not in vocab:
        raise ValueError(f"Unsupported node type: {node_type}. Vocabulary: {vocab}")

    type_one_hot = _one_hot(vocab.index(node_type), len(vocab))

    if num_qubits > QUBIT_MASK_DIM:
        raise ValueError(
            f"Qubit count {num_qubits} exceeds mask capacity {QUBIT_MASK_DIM}. "
            "This ML-QEM representation is configured for 5-qubit FakeLima-style backends."
        )

    qubit_mask = torch.zeros(QUBIT_MASK_DIM, dtype=torch.float)
    for q in active_qubits:
        if q < 0 or q >= QUBIT_MASK_DIM:
            raise ValueError(f"Qubit index {q} out of range for {QUBIT_MASK_DIM}-qubit mask")
        qubit_mask[q] = 1.0

    parts: List[Tensor] = [type_one_hot, qubit_mask]

    enc = (node_angle_encoding or "none").strip().lower()
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
        if backend_feat.numel() != NODE_FEATURE_BACKEND_EXTRA_DIM:
            raise ValueError(
                f"backend_feat must have dim {NODE_FEATURE_BACKEND_EXTRA_DIM}, got {backend_feat.numel()}"
            )
        parts.append(backend_feat)

    return torch.cat(parts, dim=0)


# ============================================================
# Backend features
# ============================================================

class BackendFeatureProvider:
    """
    Provides per-node and global backend features.

    Per-node backend feature:
        [q0_T1, q0_T2, q1_T1, q1_T2, gate_error, q0_readout_error, q1_readout_error]

    Backend mean features:
        [mean_T1, mean_T2, mean_readout_error, mean_1q_gate_error, mean_cx_gate_error]
    """

    def __init__(self, backend_name: str = "fake_lima"):
        self.backend_name = (backend_name or "fake_lima").strip().lower()
        self._backend = None
        self._properties = None
        self._init_backend()

    def _init_backend(self) -> None:
        cls = None

        # qiskit-ibm-runtime fake provider variants
        try:
            from qiskit_ibm_runtime.fake_provider import FakeLimaV2  # type: ignore
            if self.backend_name in {"fake_lima", "fakelima", "fake_lima_v2", "fakelimav2", "lima", "limav2"}:
                cls = FakeLimaV2
        except Exception:
            pass

        if cls is None:
            try:
                from qiskit_ibm_runtime.fake_provider import FakeLima  # type: ignore
                if self.backend_name in {"fake_lima", "fakelima", "lima"}:
                    cls = FakeLima
            except Exception:
                pass

        # Older qiskit.providers.fake_provider variants
        if cls is None:
            try:
                from qiskit.providers.fake_provider import FakeLimaV2  # type: ignore
                if self.backend_name in {"fake_lima", "fakelima", "fake_lima_v2", "fakelimav2", "lima", "limav2"}:
                    cls = FakeLimaV2
            except Exception:
                pass

        if cls is None:
            try:
                from qiskit.providers.fake_provider import FakeLima  # type: ignore
                if self.backend_name in {"fake_lima", "fakelima", "lima"}:
                    cls = FakeLima
            except Exception:
                pass

        # Optional support for the other 5-qubit backends used earlier.
        if cls is None:
            try:
                from qiskit_ibm_runtime.fake_provider import (  # type: ignore
                    FakeAthensV2,
                    FakeRomeV2,
                    FakeSantiagoV2,
                    FakeBogotaV2,
                )
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
                cls = mapping.get(self.backend_name, None)
            except Exception:
                pass

        if cls is None:
            self._backend = None
            self._properties = None
            return

        try:
            self._backend = cls()
            props_fn = getattr(self._backend, "properties", None)
            self._properties = props_fn() if callable(props_fn) else None
        except Exception:
            self._backend = None
            self._properties = None

    def coupling_edges(self) -> List[Tuple[int, int]]:
        try:
            if self._backend is not None and getattr(self._backend, "coupling_map", None) is not None:
                edges = list(self._backend.coupling_map.get_edges())
                if edges:
                    edges_int = [(int(a), int(b)) for a, b in edges]
                    # Keep only 5-qubit edges and the fixed order if possible.
                    edge_set = set(edges_int)
                    if all(e in edge_set for e in DEFAULT_SHARED_COUPLING_EDGES):
                        return list(DEFAULT_SHARED_COUPLING_EDGES)
                    return [e for e in edges_int if e[0] < BACKEND_NUM_QUBITS and e[1] < BACKEND_NUM_QUBITS]
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
            if x is None or not math.isfinite(float(x)):
                return 0.0
            return float(min(max(float(x), lo), hi))
        except Exception:
            return 0.0

    @staticmethod
    def _log1p_pos(x: float) -> float:
        return float(math.log1p(x)) if x > 0.0 else 0.0

    def _extract_gate_error_from_target(self, gate_name: str, qubits: Sequence[int]) -> Optional[float]:
        try:
            target = getattr(self._backend, "target", None)
            if target is None:
                return None
            gate_name = (gate_name or "").lower()
            qtuple = tuple(int(q) for q in qubits)
            if gate_name not in getattr(target, "operation_names", []):
                return None
            props = target[gate_name].get(qtuple, None)
            if props is None:
                return None
            err = getattr(props, "error", None)
            if err is None:
                return None
            return float(err)
        except Exception:
            return None

    def _extract_gate_error_from_properties(self, gate_name: str, qubits: Sequence[int]) -> Optional[float]:
        try:
            props = self._properties
            if props is None:
                return None
            gates = getattr(props, "gates", []) or []
            qlist = [int(q) for q in qubits]
            gate_name_l = (gate_name or "").lower()

            for g in gates:
                try:
                    if getattr(g, "name", "").lower() != gate_name_l:
                        continue
                    if list(getattr(g, "qubits", [])) != qlist:
                        continue
                    for par in getattr(g, "parameters", []):
                        if getattr(par, "name", "").lower() == "gate_error":
                            val = getattr(par, "value", None)
                            return float(val) if val is not None else None
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _get_gate_error(self, gate_name: str, qubits: Sequence[int]) -> float:
        if not qubits:
            return 0.0

        gate_name_l = (gate_name or "").lower()
        qlist = [int(q) for q in qubits]

        err = self._extract_gate_error_from_target(gate_name_l, qlist)
        if err is not None:
            return float(err)

        err = self._extract_gate_error_from_properties(gate_name_l, qlist)
        if err is not None:
            return float(err)

        # For CX fallback, try reverse direction and then average all CX errors.
        if len(qlist) == 2:
            qrev = [qlist[1], qlist[0]]
            err = self._extract_gate_error_from_target(gate_name_l, qrev)
            if err is not None:
                return float(err)
            err = self._extract_gate_error_from_properties(gate_name_l, qrev)
            if err is not None:
                return float(err)

        return 0.0

    def raw_readout_error(self, qubit: int) -> float:
        return float(self._get_qubit_val(int(qubit), "readout_error"))

    def raw_gate_error(self, gate_name: str, qubits: Sequence[int]) -> float:
        return float(self._get_gate_error(gate_name, qubits))

    def global_means(self, num_qubits: int) -> Tensor:
        if num_qubits <= 0:
            return torch.zeros(5, dtype=torch.float)

        t1_vals: List[float] = []
        t2_vals: List[float] = []
        ro_vals: List[float] = []

        for q in range(min(int(num_qubits), BACKEND_NUM_QUBITS)):
            t1_vals.append(float(self._get_qubit_val(q, "T1")))
            t2_vals.append(float(self._get_qubit_val(q, "T2")))
            ro_vals.append(float(self._get_qubit_val(q, "readout_error")))

        mean_t1 = self._safe_mean(t1_vals)
        mean_t2 = self._safe_mean(t2_vals)
        mean_ro = self._safe_mean(ro_vals)

        oneq_errs: List[float] = []
        cx_errs: List[float] = []

        for q in range(min(int(num_qubits), BACKEND_NUM_QUBITS)):
            for gate_name in ["id", "rz", "sx", "x"]:
                err = self._get_gate_error(gate_name, [q])
                if math.isfinite(err) and err > 0.0:
                    oneq_errs.append(float(err))

        for q0, q1 in self.coupling_edges():
            err = self._get_gate_error("cx", [q0, q1])
            if math.isfinite(err) and err > 0.0:
                cx_errs.append(float(err))

        mean_1q_err = self._safe_mean(oneq_errs)
        mean_cx_err = self._safe_mean(cx_errs)

        return torch.tensor(
            [
                self._log1p_pos(self._clip(mean_t1, 0.0, 1e6)),
                self._log1p_pos(self._clip(mean_t2, 0.0, 1e6)),
                self._log1p_pos(self._clip(mean_ro, 0.0, 0.5)),
                self._log1p_pos(self._clip(mean_1q_err, 0.0, 0.5)),
                self._log1p_pos(self._clip(mean_cx_err, 0.0, 0.5)),
            ],
            dtype=torch.float,
        )

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

        return torch.tensor(
            [
                self._log1p_pos(self._clip(q0_t1, 0.0, 1e6)),
                self._log1p_pos(self._clip(q0_t2, 0.0, 1e6)),
                self._log1p_pos(self._clip(q1_t1, 0.0, 1e6)),
                self._log1p_pos(self._clip(q1_t2, 0.0, 1e6)),
                self._log1p_pos(self._clip(gate_err, 0.0, 0.5)),
                self._log1p_pos(self._clip(q0_ro, 0.0, 0.5)),
                self._log1p_pos(self._clip(q1_ro, 0.0, 0.5)),
            ],
            dtype=torch.float,
        )


# ============================================================
# Global features
# ============================================================

def _count_gates_backend(circuit: QuantumCircuit) -> Dict[str, int]:
    counts = {"id": 0, "rz": 0, "sx": 0, "x": 0, "cx": 0}
    for instr in getattr(circuit, "data", []):
        name = instr.operation.name.lower()
        if name in counts:
            counts[name] += 1
    return counts


def _safe_float_param(val) -> Optional[float]:
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _circuit_statistics(circuit: QuantumCircuit, num_qubits: int) -> Tensor:
    gate_counts = _count_gates_backend(circuit)
    num_param = gate_counts["rz"]
    total_gates = sum(gate_counts.values())
    depth = float(circuit.depth()) if hasattr(circuit, "depth") else float(total_gates)

    return torch.tensor(
        [
            depth,
            float(num_param),
            float(num_qubits),
            float(total_gates),
            float(gate_counts["id"]),
            float(gate_counts["rz"]),
            float(gate_counts["sx"]),
            float(gate_counts["x"]),
            float(gate_counts["cx"]),
        ],
        dtype=torch.float,
    )


def _observable_mask(observable_qubit: int) -> Tensor:
    mask = torch.zeros(OBSERVABLE_MASK_DIM, dtype=torch.float)
    q = int(observable_qubit)
    if 0 <= q < OBSERVABLE_MASK_DIM:
        mask[q] = 1.0
    else:
        raise ValueError(f"observable_qubit={q} out of range for dim {OBSERVABLE_MASK_DIM}")
    return mask


def _compute_enhanced_backend_features(
    circuit: QuantumCircuit,
    num_qubits: int,
    backend_feature_provider: Optional[BackendFeatureProvider],
) -> Tensor:
    if backend_feature_provider is None:
        return torch.zeros(5 + 3 + 5 + 5 + 8 + 8, dtype=torch.float)

    backend_means = backend_feature_provider.global_means(num_qubits)

    n_fixed = BACKEND_NUM_QUBITS
    edge_list = DEFAULT_SHARED_COUPLING_EDGES
    edge_to_idx = {edge: i for i, edge in enumerate(edge_list)}

    per_qubit_depth = [0.0 for _ in range(n_fixed)]
    per_qubit_gate_count = [0.0 for _ in range(n_fixed)]
    wire_depth = [0 for _ in range(max(num_qubits, n_fixed))]

    per_edge_cx_count = [0.0 for _ in edge_list]
    per_edge_cx_error_sum = [0.0 for _ in edge_list]

    oneq_error_sum = 0.0
    cx_error_sum = 0.0
    used_qubits = set()

    def _q_index(q) -> int:
        try:
            return circuit.find_bit(q).index
        except Exception:
            pass
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

        if name in {"id", "rz", "sx", "x"}:
            q = _q_index(instr.qubits[0])
            used_qubits.add(q)

            if q < n_fixed:
                per_qubit_gate_count[q] += 1.0

            wire_depth[q] += 1

            err = backend_feature_provider.raw_gate_error(name, [q])
            oneq_error_sum += err

        elif name == "cx":
            q0 = _q_index(instr.qubits[0])
            q1 = _q_index(instr.qubits[1])
            used_qubits.add(q0)
            used_qubits.add(q1)

            if q0 < n_fixed:
                per_qubit_gate_count[q0] += 1.0
            if q1 < n_fixed:
                per_qubit_gate_count[q1] += 1.0

            new_depth = max(wire_depth[q0], wire_depth[q1]) + 1
            wire_depth[q0] = new_depth
            wire_depth[q1] = new_depth

            err = backend_feature_provider.raw_gate_error("cx", [q0, q1])
            cx_error_sum += err

            edge = (q0, q1)
            if edge in edge_to_idx:
                idx = edge_to_idx[edge]
                per_edge_cx_count[idx] += 1.0
                per_edge_cx_error_sum[idx] += err

    for q in range(min(n_fixed, len(wire_depth))):
        per_qubit_depth[q] = float(wire_depth[q])

    used_ro_sum = 0.0
    for q in used_qubits:
        if q < n_fixed:
            used_ro_sum += backend_feature_provider.raw_readout_error(q)

    accumulated = torch.tensor(
        [oneq_error_sum, cx_error_sum, used_ro_sum],
        dtype=torch.float,
    )

    return torch.cat(
        [
            backend_means,
            accumulated,
            torch.tensor(per_qubit_depth, dtype=torch.float),
            torch.tensor(per_qubit_gate_count, dtype=torch.float),
            torch.tensor(per_edge_cx_count, dtype=torch.float),
            torch.tensor(per_edge_cx_error_sum, dtype=torch.float),
        ],
        dim=0,
    )


def _global_features_enhanced_backend(
    circuit: QuantumCircuit,
    num_qubits: int,
    backend_feature_provider: Optional[BackendFeatureProvider],
    noisy_expectation: float,
    observable_qubit: int,
) -> Tensor:
    stats = _circuit_statistics(circuit, num_qubits)
    noisy = torch.tensor([float(noisy_expectation)], dtype=torch.float)
    obs = _observable_mask(int(observable_qubit))
    backend = _compute_enhanced_backend_features(circuit, num_qubits, backend_feature_provider)
    return torch.cat([stats, noisy, obs, backend], dim=0)


def _angle_sin_cos_from_instr(op_name: str, instr) -> Tensor:
    if op_name != "rz":
        return torch.zeros(2, dtype=torch.float)

    params = getattr(instr.operation, "params", [])
    if not params:
        return torch.zeros(2, dtype=torch.float)

    theta = _safe_float_param(params[0])
    if theta is None:
        return torch.zeros(2, dtype=torch.float)

    return torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float)


# ============================================================
# QASM to PyG graph
# ============================================================

def qasm_to_pyg_graph(
    qasm_str: str,
    num_qubits_hint: Optional[int] = None,
    backend_feature_provider: Optional[BackendFeatureProvider] = None,
    noisy_expectation: Optional[float] = None,
    observable_qubit: Optional[int] = None,
    node_angle_encoding: str = "none",
) -> Tuple[Data, Dict[str, int]]:
    """
    Convert one ML-QEM sample to a PyG graph.

    Input features include noisy_expectation and observable_qubit.
    Target y is set by the dataset class to ideal_expectation.
    """
    if noisy_expectation is None:
        raise ValueError("ML-QEM graph requires noisy_expectation as an input feature.")
    if observable_qubit is None:
        raise ValueError("ML-QEM graph requires observable_qubit as an input feature.")

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
    node_backend_variant = getattr(backend_feature_provider, "backend_name", "backend")
    vocab = get_node_types(node_backend_variant)

    for q in range(num_qubits):
        backend_feat = None
        if backend_feature_provider is not None:
            backend_feat = backend_feature_provider.features_for("input", [q])

        idx = len(x_features)
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
        try:
            return circuit.find_bit(q).index
        except Exception:
            pass
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

        if op_name in {"barrier", "measure"}:
            continue

        if op_name in {"id", "rz", "sx", "x"}:
            if op_name not in vocab:
                continue

            q = _q_index(instr.qubits[0])
            node_idx = len(x_features)

            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for(op_name, [q])

            angle_feat = (
                _angle_sin_cos_from_instr(op_name, instr)
                if (node_angle_encoding or "none").strip().lower() == "sin_cos"
                else None
            )

            x_features.append(
                _encode_node_feature(
                    op_name,
                    num_qubits,
                    [q],
                    angle_feat=angle_feat,
                    backend_feat=backend_feat,
                    node_feature_backend_variant=node_backend_variant,
                    node_angle_encoding=node_angle_encoding,
                )
            )

            edge_src.append(last_node_for_qubit[q])
            edge_dst.append(node_idx)
            last_node_for_qubit[q] = node_idx

        elif op_name == "cx":
            q0 = _q_index(instr.qubits[0])
            q1 = _q_index(instr.qubits[1])
            node_idx = len(x_features)

            backend_feat = None
            if backend_feature_provider is not None:
                backend_feat = backend_feature_provider.features_for("cx", [q0, q1])

            x_features.append(
                _encode_node_feature(
                    "cx",
                    num_qubits,
                    [q0, q1],
                    angle_feat=None,
                    backend_feat=backend_feat,
                    node_feature_backend_variant=node_backend_variant,
                    node_angle_encoding=node_angle_encoding,
                )
            )

            edge_src.extend([last_node_for_qubit[q0], last_node_for_qubit[q1]])
            edge_dst.extend([node_idx, node_idx])
            last_node_for_qubit[q0] = node_idx
            last_node_for_qubit[q1] = node_idx

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
        node_dim = get_node_feature_dim(node_backend_variant, node_angle_encoding)
        x = torch.zeros((0, node_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        x = torch.stack(x_features, dim=0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    global_feat = _global_features_enhanced_backend(
        circuit,
        num_qubits,
        backend_feature_provider,
        noisy_expectation=_safe_float(noisy_expectation),
        observable_qubit=int(observable_qubit),
    )
    global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

    if global_feat.numel() != GLOBAL_FEATURE_DIM:
        raise RuntimeError(
            f"Global feature dimension mismatch: got {global_feat.numel()}, expected {GLOBAL_FEATURE_DIM}"
        )

    data = Data(x=x, edge_index=edge_index)
    data.num_qubits = num_qubits
    data.global_features = global_feat
    data.noisy_expectation = torch.tensor([_safe_float(noisy_expectation)], dtype=torch.float)
    data.observable_qubit = int(observable_qubit)

    return data, _count_gates_backend(circuit)


# ============================================================
# Dataset
# ============================================================

class QuantumCircuitGraphDataset(InMemoryDataset):
    """
    ML-QEM error-mitigation graph dataset.

    Expected PKL items:
        (circ_info, ideal_expectation)

    Required circ_info fields:
        qasm_transpiled
        noisy_expectation
        observable_qubit

    Target:
        y = ideal_expectation

    The noiseless/ideal value is NOT used as an input feature.
    """

    def __init__(
        self,
        root: str,
        pkl_paths: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        node_feature_backend_variant: str = "fake_lima",
        node_angle_encoding: str = "none",
    ):
        self.pkl_paths = pkl_paths
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
        node_dim = get_node_feature_dim(
            self.node_feature_backend_variant,
            self.node_angle_encoding,
        )
        angle_tag = self.node_angle_encoding or "none"
        backend_tag = self.node_feature_backend_variant or "none"
        return [
            (
                f"graphs.mlqem.enhanced_only."
                f"node{node_dim}.gfeat{GLOBAL_FEATURE_DIM}."
                f"backend_{backend_tag}.angle_{angle_tag}.dataset.pt"
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
                    # Fallback if label is stored in metadata.
                    label = item.get("ideal_expectation", None)
                    if label is None:
                        continue
                    yield item, float(label)
                else:
                    continue

    def process(self):
        data_list: List[Data] = []

        backend_provider = BackendFeatureProvider(self.node_feature_backend_variant)

        for circ_info, label in self._iter_items_from_pkls():
            if not isinstance(circ_info, dict):
                continue

            qasm_t = circ_info.get("qasm_transpiled", circ_info.get("qasm", None))
            if not isinstance(qasm_t, str):
                continue

            if "noisy_expectation" not in circ_info:
                raise KeyError("ML-QEM sample missing required field 'noisy_expectation'")
            if "observable_qubit" not in circ_info:
                raise KeyError("ML-QEM sample missing required field 'observable_qubit'")

            graph, _ = qasm_to_pyg_graph(
                qasm_t,
                backend_feature_provider=backend_provider,
                noisy_expectation=circ_info.get("noisy_expectation"),
                observable_qubit=int(circ_info.get("observable_qubit")),
                node_angle_encoding=self.node_angle_encoding,
            )

            val = float(label)
            if not math.isfinite(val):
                continue

            graph.y = torch.tensor([val], dtype=torch.float)

            # Metadata used later for regrouping Z0..Z3 predictions by circuit.
            graph.split = str(circ_info.get("split", ""))
            graph.target_two_qubit_depth = int(circ_info.get("target_two_qubit_depth", -1))
            graph.actual_two_qubit_depth = int(circ_info.get("actual_two_qubit_depth", -1))
            graph.circuit_idx = int(circ_info.get("circuit_idx", -1))
            graph.seed = int(circ_info.get("seed", -1))
            graph.observable = str(circ_info.get("observable", f"Z{graph.observable_qubit}"))
            graph.ideal_expectation = float(circ_info.get("ideal_expectation", val))
            graph.backend_name = str(circ_info.get("backend", self.node_feature_backend_variant))

            data_list.append(graph)

        if len(data_list) == 0:
            raise RuntimeError("No graphs were produced. Check PKL paths and required metadata fields.")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def encode_single_qasm(
    qasm_transpiled_str: str,
    label: Optional[float] = None,
    noisy_expectation: Optional[float] = None,
    observable_qubit: Optional[int] = None,
    node_feature_backend_variant: str = "fake_lima",
    node_angle_encoding: str = "none",
) -> Data:
    provider = BackendFeatureProvider(node_feature_backend_variant)

    data, _ = qasm_to_pyg_graph(
        qasm_transpiled_str,
        backend_feature_provider=provider,
        noisy_expectation=noisy_expectation,
        observable_qubit=observable_qubit,
        node_angle_encoding=node_angle_encoding,
    )

    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)

    return data
