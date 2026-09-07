import random
import math
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, Tuple, Union, Optional, List, Iterable, Set

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    RYGate,
    RXGate,
    RZGate,
    HGate,
    CXGate,
    SXGate,
    XGate,
    IGate,
)
from qiskit.quantum_info import Operator


class Circuit:
    def __init__(self, variable_qubits: int, ancilla_qubits: int, initialization: str = None):
        """
        Initializes the quantum circuit for the target problem.

        :param variable_qubits: Number of qubits necessary to encode the problem variables
        :param ancilla_qubits:  Number of ancilla qubits
        :param initialization: Initialization method for qubits
        """
        variable_qubits = QuantumRegister(variable_qubits, name="v")
        ancilla_qubits = QuantumRegister(ancilla_qubits, name="a")
        qc = QuantumCircuit(variable_qubits, ancilla_qubits)

        if initialization in ("h", "equal_superposition", "hadamard"):
            qc.h([qubits for qubits in qc.qubits])

        self.circuit = qc
        self.is_nisq: Optional[bool] = None

    def building_state(self, quantum_circuit: QuantumCircuit) -> "Circuit":
        self.circuit = quantum_circuit
        return self

    def nisq_control(self, max_depth: int) -> bool:
        self.is_nisq = self.circuit.depth() < max_depth
        return self.is_nisq

    def evaluation(self, evaluation_function: Callable[[QuantumCircuit], float]) -> float:
        return evaluation_function(self.circuit)

    def get_legal_action(
        self,
        gate_set: "GateSet",
        max_depth: int,
        prob_choice: Dict[str, float],
        stop: bool,
    ) -> Tuple[Callable, str]:
        """
        Determines a legal action for modifying the circuit.
        Returns (callable_action, action_str).
        """
        # IMPORTANT: copy, do not mutate caller's dict in place
        local_prob_choice = dict(prob_choice)

        if stop and "p" in local_prob_choice:
            local_prob_choice["p"] = 0

        # Recompute NISQ flag if needed
        self.nisq_control(max_depth)
        if not self.is_nisq:
            if "a" in local_prob_choice:
                local_prob_choice["a"] = 0
            if "d" in local_prob_choice:
                local_prob_choice["d"] = max(local_prob_choice.get("d", 0), 50)

        keys = list(local_prob_choice.keys())
        probabilities = np.array(list(local_prob_choice.values()), dtype=float)

        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("All action probabilities are zero in get_legal_action().")

        probabilities = probabilities / total

        action_str = np.random.choice(keys, p=probabilities)
        action = actions_on_circuit(action_chosen=action_str, gate_set=gate_set)

        if action and callable(action):
            return action, action_str
        raise NotImplementedError("Action not implemented or callable")


class GateSet:
    """
    Defines the gate pool available for actions.

    Set gate_type='backend' to use the transpiled backend basis
    {rz, sx, x, cx} (and optional id).

    coupling_map:
        Iterable of directed edges (control, target), e.g. backend.coupling_map.get_edges()

    directed_coupling:
        - True  => keep only the directed edges provided
        - False => treat each edge as undirected and include reverse edges too
    """

    def __init__(
        self,
        gate_type: str = "backend",
        coupling_map: Optional[Iterable[Tuple[int, int]]] = None,
        directed_coupling: bool = True,
        include_id: bool = False,
    ):
        self.gate_type = gate_type
        self.directed_coupling = directed_coupling
        self.coupling_map: Set[Tuple[int, int]] = self._normalize_coupling_map(coupling_map, directed_coupling)

        if self.gate_type == "discrete":
            gates = ["s", "cx", "h", "t"]
        elif self.gate_type == "continuous":
            gates = ["cx", "ry", "rx", "rz"]
        elif self.gate_type == "backend":
            gates = ["cx", "rz", "sx", "x"]
            if include_id:
                gates.append("id")
        else:
            raise NotImplementedError(f"Unknown gate_type: {self.gate_type}")

        self.pool = gates

    @staticmethod
    def _normalize_coupling_map(
        coupling_map: Optional[Iterable[Tuple[int, int]]],
        directed_coupling: bool,
    ) -> Set[Tuple[int, int]]:
        if coupling_map is None:
            return set()

        edges: Set[Tuple[int, int]] = set()
        for a, b in coupling_map:
            a_i, b_i = int(a), int(b)
            edges.add((a_i, b_i))
            if not directed_coupling:
                edges.add((b_i, a_i))
        return edges

    def legal_cx_edges(self, n_qubits: int) -> List[Tuple[int, int]]:
        """
        Returns coupling-map edges restricted to the currently available qubits.
        """
        if not self.coupling_map:
            return []
        return [(a, b) for (a, b) in self.coupling_map if a < n_qubits and b < n_qubits and a != b]

    def sample_cx_edge(self, n_qubits: int) -> Optional[Tuple[int, int]]:
        edges = self.legal_cx_edges(n_qubits)
        if not edges:
            return None
        return random.choice(edges)

    def backend_one_qubit_pool(self) -> List[str]:
        return [g for g in self.pool if g != "cx"]



def actions_on_circuit(action_chosen: str, gate_set: GateSet) -> Callable[[QuantumCircuit], Union[QuantumCircuit, None]]:
    """
    Returns a function that applies the chosen action to a Qiskit QuantumCircuit.
    """

    def add_gate(quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Pick a random gate from the GateSet pool and apply it.

        Backend mode:
        - cx uses only legal coupling-map edges
        - rz uses an angle
        - sx/x/id are fixed 1q gates
        """
        qc = quantum_circuit.copy()
        n_qubits = qc.num_qubits
        if n_qubits < 1:
            return qc

        q0 = random.randrange(n_qubits)
        angle = 2 * math.pi * random.random()

        if gate_set.gate_type == "backend":
            # If no legal CX exists, do not sample CX
            allowed_pool = list(gate_set.pool)
            if gate_set.sample_cx_edge(n_qubits) is None:
                allowed_pool = gate_set.backend_one_qubit_pool()
            if not allowed_pool:
                return qc

            choice = random.choice(allowed_pool)

            if choice == "cx":
                edge = gate_set.sample_cx_edge(n_qubits)
                if edge is None:
                    return qc
                qc.cx(edge[0], edge[1])
                return qc
            if choice == "rz":
                qc.rz(angle, q0)
                return qc
            if choice == "sx":
                qc.sx(q0)
                return qc
            if choice == "x":
                qc.x(q0)
                return qc
            if choice == "id":
                qc.id(q0)
                return qc

            return qc

        # Non-backend pools
        if n_qubits >= 2:
            qubits2 = random.sample(list(range(n_qubits)), k=2)
        else:
            qubits2 = [0, 0]

        choice = random.choice(gate_set.pool)

        if choice == "cx":
            if n_qubits >= 2:
                qc.cx(qubits2[0], qubits2[1])
            return qc

        gate_map = {
            "ry": RYGate(angle),
            "rx": RXGate(angle),
            "rz": RZGate(angle),
            "h": HGate(),
            "t": lambda: qc.t(q0),
            "s": lambda: qc.s(q0),
        }
        gate = gate_map.get(choice)
        if gate is None:
            return qc
        if callable(gate):
            gate()
        else:
            qc.append(gate, [q0])
        return qc

    def delete_gate(quantum_circuit: QuantumCircuit) -> Union[QuantumCircuit, None]:
        """
        Removes a random gate from the circuit.
        Kept intentionally conservative for shallow circuits.
        """
        qc = quantum_circuit.copy()
        if len(qc.data) < 4:
            return None
        position = random.randint(0, len(qc.data) - 2)
        qc.data.remove(qc.data[position])
        return qc

    def swap(quantum_circuit: QuantumCircuit) -> Union[QuantumCircuit, None]:
        """
        Swap one random gate in the circuit with a randomly chosen gate from the current GateSet pool.

        In backend mode, replacement CX gates are sampled only from the legal coupling map.
        """
        angle = random.random() * 2 * math.pi
        if len(quantum_circuit.data) <= 1:
            return None

        n_qubits = quantum_circuit.num_qubits
        n_clbits = quantum_circuit.num_clbits
        qc_out = QuantumCircuit(n_qubits, n_clbits)

        position = random.randint(0, len(quantum_circuit.data) - 2)
        gate_to_remove = quantum_circuit.data[position]

        # Choose replacement gate
        replacement_pool = list(gate_set.pool)
        if gate_set.gate_type == "backend" and gate_set.sample_cx_edge(n_qubits) is None:
            replacement_pool = gate_set.backend_one_qubit_pool()
        if not replacement_pool:
            return None

        gate_to_add_str = random.choice(replacement_pool)
        gate_to_add = get_gate(gate_to_add_str, angle=angle)
        if gate_to_add is None:
            return None

        two_qubit_gate = (gate_to_add_str == "cx")

        def unpack(item):
            if hasattr(item, "operation"):
                return item.operation, list(item.qubits), list(item.clbits)
            op, qargs, cargs = item
            return op, list(qargs), list(cargs)

        op_remove, qargs_remove, _cargs_remove = unpack(gate_to_remove)
        removed_arity = len(qargs_remove)
        new_arity = 2 if two_qubit_gate else 1
        delta = removed_arity - new_arity  # +1: 2q->1q, -1: 1q->2q

        cx_edge = None
        if two_qubit_gate:
            if gate_set.gate_type == "backend":
                cx_edge = gate_set.sample_cx_edge(n_qubits)
                if cx_edge is None:
                    return None

        for pos, item in enumerate(quantum_circuit.data):
            op, qargs, cargs = unpack(item)
            q_idxs = [quantum_circuit.find_bit(q).index for q in qargs]
            c_idxs = [quantum_circuit.find_bit(c).index for c in cargs] if cargs else []

            if pos == position:
                op = gate_to_add
                if delta == 1:
                    # 2q -> 1q
                    q_idxs = [q_idxs[0]]
                    c_idxs = []
                elif delta == -1:
                    # 1q -> 2q
                    if two_qubit_gate:
                        if cx_edge is not None:
                            q_idxs = [cx_edge[0], cx_edge[1]]
                        else:
                            first = q_idxs[0]
                            second = random.choice([i for i in range(n_qubits) if i != first])
                            q_idxs = [first, second]
                    c_idxs = []
                else:
                    # same arity replacement
                    if two_qubit_gate and cx_edge is not None:
                        q_idxs = [cx_edge[0], cx_edge[1]]
                    elif not two_qubit_gate:
                        q_idxs = [q_idxs[0]]
                    c_idxs = []

            qc_out.append(
                op,
                [qc_out.qubits[i] for i in q_idxs],
                [qc_out.clbits[i] for i in c_idxs] if c_idxs else [],
            )

        return qc_out

    def change(quantum_circuit: QuantumCircuit) -> Union[QuantumCircuit, None]:
        """
        Change the parameter (angle) of a randomly chosen parameterized gate.

        The new value is:
            theta_i <- theta_i + epsilon
        where
            epsilon ~ N(0, delta_phi)

        For backend basis, this targets RZ only.
        """
        delta_phi = 0.2

        qc = quantum_circuit.copy()
        if len(qc.data) == 0:
            return None

        def unpack(item):
            if hasattr(item, "operation"):
                return item.operation, list(item.qubits), list(item.clbits)
            op, qargs, cargs = item
            return op, list(qargs), list(cargs)

        candidates: List[int] = []
        for i, item in enumerate(qc.data):
            op, _qargs, _cargs = unpack(item)
            name = getattr(op, "name", "").lower()
            params = getattr(op, "params", [])
            if len(params) == 1:
                if gate_set.gate_type == "backend":
                    if name == "rz":
                        candidates.append(i)
                else:
                    candidates.append(i)

        if not candidates:
            return None

        pos = random.choice(candidates)

        # In-place edit when possible
        try:
            if hasattr(qc.data[pos], "operation"):
                op = qc.data[pos].operation
                old = float(op.params[0])
                op.params[0] = old + float(np.random.normal(0.0, delta_phi))
                qc.data[pos].operation = op
                return qc
        except Exception:
            pass

        # Fallback: rebuild
        n_qubits = qc.num_qubits
        n_clbits = qc.num_clbits
        qc_out = QuantumCircuit(n_qubits, n_clbits)

        for i, item in enumerate(qc.data):
            op, qargs, cargs = unpack(item)
            q_idxs = [qc.find_bit(q).index for q in qargs]
            c_idxs = [qc.find_bit(c).index for c in cargs] if cargs else []

            if i == pos and getattr(op, "params", None) and len(op.params) == 1:
                try:
                    new_angle = float(op.params[0]) + float(np.random.normal(0.0, delta_phi))
                    name = getattr(op, "name", "").lower()
                    if name == "rz":
                        op = RZGate(new_angle)
                    elif name == "rx":
                        op = RXGate(new_angle)
                    elif name == "ry":
                        op = RYGate(new_angle)
                except Exception:
                    pass

            qc_out.append(
                op,
                [qc_out.qubits[iq] for iq in q_idxs],
                [qc_out.clbits[ic] for ic in c_idxs] if c_idxs else [],
            )

        return qc_out

    def stop() -> str:
        return "stop"

    actions = {"a": add_gate, "d": delete_gate, "s": swap, "c": change, "p": stop}
    return actions.get(action_chosen, None)


def get_gate(gate_str: str, angle: float = None):
    """
    Returns the Qiskit gate object corresponding to the given gate string.
    Supports backend basis gates sx/x/id in addition to rx/ry/rz/cx/h.
    """
    gate_map = {
        "h": HGate(),
        "cx": CXGate(),
        "rx": RXGate(angle) if angle is not None else RXGate(0.0),
        "ry": RYGate(angle) if angle is not None else RYGate(0.0),
        "rz": RZGate(angle) if angle is not None else RZGate(0.0),
        "sx": SXGate(),
        "x": XGate(),
        "id": IGate(),
    }
    return gate_map.get(gate_str)



def check_equivalence(qc1, qc2):
    """True if the two input circuits are equivalent (same matrix)."""
    op1 = Operator(qc1)
    op2 = Operator(qc2)
    return op1.equiv(op2)



# ------------------------------
# SIMULATION STRATEGY: NGRAMS
# ------------------------------
def extract_gate_sequence_by_qubit(circuit: QuantumCircuit):
    qubit_gates = defaultdict(list)

    for instruction in circuit.data:
        if hasattr(instruction, "operation"):
            gate = instruction.operation.name
            qubits = instruction.qubits
        else:
            gate = instruction[0].name
            qubits = instruction[1]

        for qubit in qubits:
            qubit_gates[circuit.find_bit(qubit).index].append(gate)

    for q in range(len(circuit.qubits)):
        if q not in qubit_gates:
            qubit_gates[q] = []
    return qubit_gates


def build_ngrams(gate_sequence, n):
    return [tuple(gate_sequence[i : i + n]) for i in range(len(gate_sequence) - n + 1)]


def update_ngrams(circuit: QuantumCircuit, n: int, qubit_ngrams_counter):
    qubit_gates = extract_gate_sequence_by_qubit(circuit)
    n_qubits = len(circuit.qubits)

    for qubit in range(n_qubits):
        gate_sequence = qubit_gates[qubit]
        ngrams = build_ngrams(gate_sequence, n)
        qubit_ngrams_counter[qubit].update(ngrams)

    return qubit_ngrams_counter


def update_conditional_results(qubit_ngrams_counter, quality_results_by_qubit, quality_score: float):
    for qubit, ngrams_counter in qubit_ngrams_counter.items():
        for ngram, count in ngrams_counter.items():
            if ngram not in quality_results_by_qubit[qubit]:
                quality_results_by_qubit[qubit][ngram] = (quality_score, 1)
            else:
                counter = quality_results_by_qubit[qubit][ngram][1]
                average_quality = (quality_results_by_qubit[qubit][ngram][0] + quality_score) / counter
                quality_results_by_qubit[qubit][ngram] = (average_quality, counter + 1)

    return quality_results_by_qubit
