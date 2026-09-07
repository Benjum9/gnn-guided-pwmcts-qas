from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qiskit import QuantumCircuit

try:
    from qiskit import qasm2
except Exception:
    qasm2 = None

from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from generate_constrained_target_circuit import append_initial_hadamard_layers
from random_circuit_generation import get_directed_coupling_edges
from mcts import make_root_node_from_qc, mcts
from graph_representation import BackendFeatureProvider, qasm_to_pyg_graph
from gnn_delta import CircuitGNN


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPTH = 8
DEFAULT_OUT_DIR = PROJECT_DIR / "mcts_6configs_constrained_depth_8"
DEFAULT_TARGET_QASM = (
    PROJECT_DIR
    / "data_generation_results_constrained_depth_8"
    / "target_depth_8_seed_43.qasm"
)
DEFAULT_TARGET_SOURCE_PKL = (
    PROJECT_DIR
    / "data_generation_results_constrained_depth_8"
    / "depth_8_without_shallow_cxcx.pkl"
)
DEFAULT_MODEL_CHECKPOINT = (
    PROJECT_DIR
    / "models_constrained_depth_8_without_shallow_cxcx_delta_new_baseline"
    / "checkpoint.pt"
)

DEFAULT_BUDGETS = [50, 100, 200, 500, 1000, 2000, 8000]
DEFAULT_N_RUNS = 10
DEFAULT_MAX_DEPTH = 20
DEFAULT_CHOICES = {"a": 50, "d": 10, "s": 20, "c": 20, "p": 0}
DEFAULT_BACKEND_NAME = "athens"
DEFAULT_GLOBAL_FEATURE_VARIANT = "new_baseline"

NODE_ANGLE_ENCODING = "none"
NOISELESS_FIDELITY_INDEX = 8
GNN_BATCH_SIZE = 64
GNN_CANDIDATE_POOL_SIZE = 8

MODEL_KWARGS_DEFAULT = {
    "gnn_hidden": 64,
    "gnn_heads": 4,
    "global_hidden": 64,
    "reg_hidden": 128,
    "num_layers": 7,
    "dropout_rate": 0.2,
}

CONFIGS = [
    {
        "name": "noiseless_random",
        "use_guided": False,
        "use_surrogate": False,
        "use_noiseless": True,
        "use_noisy": False,
    },
    {
        "name": "noiseless_guided",
        "use_guided": True,
        "use_surrogate": False,
        "use_noiseless": True,
        "use_noisy": False,
    },
    {
        "name": "noisy_random",
        "use_guided": False,
        "use_surrogate": False,
        "use_noiseless": False,
        "use_noisy": True,
    },
    {
        "name": "noisy_guided",
        "use_guided": True,
        "use_surrogate": False,
        "use_noiseless": False,
        "use_noisy": True,
    },
    {
        "name": "gnn_random",
        "use_guided": False,
        "use_surrogate": True,
        "use_noiseless": False,
        "use_noisy": False,
    },
    {
        "name": "gnn_guided",
        "use_guided": True,
        "use_surrogate": True,
        "use_noiseless": False,
        "use_noisy": False,
    },
]


def parse_int_list(value: str) -> List[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def circuit_to_qasm(qc: QuantumCircuit) -> str:
    if qasm2 is not None:
        return qasm2.dumps(qc)
    if hasattr(qc, "qasm"):
        return qc.qasm()
    raise RuntimeError("No QASM exporter available for this Qiskit version.")


def ideal_statevector(qc: QuantumCircuit) -> Statevector:
    return Statevector.from_instruction(qc)


def make_root_circuit(num_qubits: int, root_hadamard: bool) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    if root_hadamard:
        append_initial_hadamard_layers(qc)

    return qc


def load_target_from_pkl(pkl_path: Path) -> QuantumCircuit:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing target source PKL: {pkl_path}")

    with pkl_path.open("rb") as handle:
        content = pickle.load(handle)

    for item in content:
        meta = None
        if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], dict):
            meta = item[0]
        elif isinstance(item, dict):
            meta = item

        if meta is None:
            continue

        target_qasm = meta.get("target_qasm_transpiled")
        if isinstance(target_qasm, str) and target_qasm.strip():
            return QuantumCircuit.from_qasm_str(target_qasm)

    raise KeyError(f"Could not find target_qasm_transpiled in {pkl_path}")


def load_target_circuit(target_qasm: Optional[Path], target_source_pkl: Path) -> QuantumCircuit:
    if target_qasm is not None and target_qasm.exists():
        return QuantumCircuit.from_qasm_str(target_qasm.read_text())

    return load_target_from_pkl(target_source_pkl)


def stabilize_density_matrix(dm: DensityMatrix) -> DensityMatrix:
    arr = np.asarray(dm.data, dtype=np.complex128)
    arr = 0.5 * (arr + arr.conj().T)

    evals, evecs = np.linalg.eigh(arr)
    evals = np.clip(evals.real, 0.0, None)

    if evals.sum() <= 0:
        dim = arr.shape[0]
        arr = np.eye(dim, dtype=np.complex128) / dim
    else:
        arr = evecs @ np.diag(evals) @ evecs.conj().T
        arr = arr / np.trace(arr)

    arr = 0.5 * (arr + arr.conj().T)
    return DensityMatrix(arr)


def noisy_density_matrix(qc: QuantumCircuit, noisy_sim: AerSimulator) -> DensityMatrix:
    qc_dm = qc.copy()
    qc_dm.save_density_matrix()
    result = noisy_sim.run(qc_dm).result()
    data0 = result.data(0)

    if "density_matrix" not in data0:
        raise RuntimeError(f"Aer result missing density_matrix. Keys: {list(data0.keys())}")

    return stabilize_density_matrix(DensityMatrix(data0["density_matrix"]))


def noiseless_fidelity(candidate_qc: QuantumCircuit, target_qc: QuantumCircuit) -> float:
    candidate_sv = ideal_statevector(candidate_qc)
    target_sv = ideal_statevector(target_qc)
    return float(state_fidelity(candidate_sv, target_sv))


def make_noisy_fidelity_function(
    target_qc: QuantumCircuit,
    noisy_sim: AerSimulator,
    target_mode: str,
) -> Callable[[QuantumCircuit], float]:
    mode = (target_mode or "ideal_target").strip().lower()

    if mode == "ideal_target":
        target_state = ideal_statevector(target_qc)

        def evaluate(candidate_qc: QuantumCircuit) -> float:
            candidate_dm = noisy_density_matrix(candidate_qc, noisy_sim)
            return float(state_fidelity(candidate_dm, target_state))

        return evaluate

    if mode == "noisy_target":
        target_dm = noisy_density_matrix(target_qc, noisy_sim)

        def evaluate(candidate_qc: QuantumCircuit) -> float:
            candidate_dm = noisy_density_matrix(candidate_qc, noisy_sim)
            return float(state_fidelity(candidate_dm, target_dm))

        return evaluate

    raise ValueError("--noisy-target-mode must be 'ideal_target' or 'noisy_target'.")


def load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GNN checkpoint: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        return {"model_state_dict": payload, "model_kwargs": {}}

    return payload


def model_kwargs_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("model_kwargs"), dict):
        return dict(payload["model_kwargs"])

    config = payload.get("config", {})
    if isinstance(config, dict) and isinstance(config.get("model_kwargs"), dict):
        return dict(config["model_kwargs"])

    return dict(MODEL_KWARGS_DEFAULT)


def make_model(
    payload: Dict[str, Any],
    sample_graph,
    device: torch.device,
) -> CircuitGNN:
    model_kwargs = model_kwargs_from_payload(payload)
    model_kwargs["node_in_dim"] = int(sample_graph.x.size(-1))
    model_kwargs["global_in_dim"] = int(sample_graph.global_features.numel())

    state_dict = payload.get("model_state_dict", payload.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict or state_dict.")

    model = CircuitGNN(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_graph_for_prediction(
    qc: QuantumCircuit,
    provider: BackendFeatureProvider,
    target_qc: QuantumCircuit,
    global_feature_variant: str,
) -> Tuple[Any, float]:
    qasm = circuit_to_qasm(qc)
    noiseless = noiseless_fidelity(qc, target_qc)

    graph, _ = qasm_to_pyg_graph(
        qasm,
        global_feature_variant=global_feature_variant,
        backend_feature_provider=provider,
        noiseless_fidelity=noiseless,
        node_angle_encoding=NODE_ANGLE_ENCODING,
    )

    return graph, noiseless


def load_gnn_model(
    checkpoint_path: Path,
    provider: BackendFeatureProvider,
    target_qc: QuantumCircuit,
    global_feature_variant: str,
    device: torch.device,
    sample_qc: QuantumCircuit,
) -> CircuitGNN:
    payload = load_checkpoint_payload(checkpoint_path)
    sample_graph, _ = build_graph_for_prediction(
        sample_qc,
        provider,
        target_qc,
        global_feature_variant,
    )

    model = make_model(payload, sample_graph, device)

    print("=" * 80)
    print("Loaded GNN delta model")
    print("=" * 80)
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Node dim:        {sample_graph.x.size(-1)}")
    print(f"Global dim:      {sample_graph.global_features.numel()}")
    print(f"Global features: {global_feature_variant}")
    print()

    return model


def make_gnn_predictors(
    model: CircuitGNN,
    provider: BackendFeatureProvider,
    target_qc: QuantumCircuit,
    global_feature_variant: str,
    device: torch.device,
) -> Tuple[Callable[[QuantumCircuit], float], Callable[[List[QuantumCircuit]], np.ndarray]]:
    prediction_cache: Dict[str, float] = {}

    @torch.no_grad()
    def predict_many(qcs: List[QuantumCircuit]) -> np.ndarray:
        keys = [circuit_to_qasm(qc) for qc in qcs]
        missing_keys: List[str] = []
        missing_graphs = []
        missing_noiseless: List[float] = []

        for qc, key in zip(qcs, keys):
            if key in prediction_cache:
                continue

            graph, noiseless = build_graph_for_prediction(
                qc,
                provider,
                target_qc,
                global_feature_variant,
            )
            missing_keys.append(key)
            missing_graphs.append(graph)
            missing_noiseless.append(float(noiseless))

        if missing_graphs:
            loader = DataLoader(missing_graphs, batch_size=GNN_BATCH_SIZE, shuffle=False)
            cursor = 0

            for batch in loader:
                batch = batch.to(device)
                pred_losses = model(batch).view(-1).detach().cpu().tolist()

                for pred_loss in pred_losses:
                    pred_noisy = missing_noiseless[cursor] - float(pred_loss)
                    prediction_cache[missing_keys[cursor]] = float(np.clip(pred_noisy, 0.0, 1.0))
                    cursor += 1

        return np.asarray([prediction_cache[key] for key in keys], dtype=float)

    def predict_one(qc: QuantumCircuit) -> float:
        return float(predict_many([qc])[0])

    return predict_one, predict_many


def make_evaluation_function(
    cfg: Dict[str, Any],
    target_qc: QuantumCircuit,
    noisy_fidelity_function: Callable[[QuantumCircuit], float],
) -> Tuple[str, Optional[Callable[[QuantumCircuit], float]], str]:
    if cfg["use_surrogate"]:
        return "gnn", None, "gnn_delta_predicted_noisy_fidelity"

    if cfg["use_noiseless"]:
        def evaluate(qc: QuantumCircuit) -> float:
            return noiseless_fidelity(qc, target_qc)

        return "oracle", evaluate, "noiseless_fidelity"

    if cfg["use_noisy"]:
        return "oracle", noisy_fidelity_function, "simulated_noisy_fidelity"

    raise RuntimeError(f"Invalid configuration: {cfg}")


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    for cfg in CONFIGS:
        for budget in sorted({int(row["budget"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["config_name"] == cfg["name"] and int(row["budget"]) == budget
            ]

            if not selected:
                continue

            def mean_std(key: str) -> Tuple[float, float]:
                vals = np.asarray([float(row[key]) for row in selected], dtype=float)
                return (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                )

            mean_eval, std_eval = mean_std("best_eval_value")
            mean_noiseless, std_noiseless = mean_std("best_true_noiseless")
            mean_noisy, std_noisy = mean_std("best_true_noisy")
            mean_pred_noisy, std_pred_noisy = mean_std("best_predicted_noisy_fidelity")
            mean_runtime, std_runtime = mean_std("run_time_seconds")

            summary_rows.append(
                {
                    "config_name": cfg["name"],
                    "budget": budget,
                    "n_runs": len(selected),
                    "mean_best_eval_value": mean_eval,
                    "std_best_eval_value": std_eval,
                    "mean_best_true_noiseless": mean_noiseless,
                    "std_best_true_noiseless": std_noiseless,
                    "mean_best_true_noisy": mean_noisy,
                    "std_best_true_noisy": std_noisy,
                    "mean_best_predicted_noisy_fidelity": mean_pred_noisy,
                    "std_best_predicted_noisy_fidelity": std_pred_noisy,
                    "mean_run_time_seconds": mean_runtime,
                    "std_run_time_seconds": std_runtime,
                }
            )

    return summary_rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_convergence_plot(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = {
        "noiseless_random": "Noiseless random",
        "noiseless_guided": "Noiseless guided",
        "noisy_random": "Noisy random",
        "noisy_guided": "Noisy guided",
        "gnn_random": "GNN random",
        "gnn_guided": "GNN guided",
    }

    colors = {
        "noiseless_random": "#0072B2",
        "noiseless_guided": "#E69F00",
        "noisy_random": "#009E73",
        "noisy_guided": "#D55E00",
        "gnn_random": "#CC79A7",
        "gnn_guided": "#5F4B8B",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for cfg in CONFIGS:
        name = cfg["name"]
        rows = sorted(
            [row for row in summary_rows if row["config_name"] == name],
            key=lambda row: int(row["budget"]),
        )

        if not rows:
            continue

        x = [int(row["budget"]) for row in rows]
        y = [float(row["mean_best_true_noisy"]) for row in rows]
        yerr = [float(row["std_best_true_noisy"]) for row in rows]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2,
            label=labels.get(name, name),
            color=colors.get(name),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("MCTS budget")
    ax.set_ylabel("Mean simulated noisy fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_config(args: argparse.Namespace, out_path: Path) -> None:
    config = {
        "depth": args.depth,
        "backend_name": args.backend_name,
        "target_qasm": str(args.target_qasm) if args.target_qasm else None,
        "target_source_pkl": str(args.target_source_pkl),
        "model_checkpoint": str(args.model_checkpoint),
        "out_dir": str(args.out_dir),
        "budgets": args.budgets,
        "n_runs": args.n_runs,
        "max_depth": args.max_depth,
        "root_hadamard": args.root_hadamard,
        "global_feature_variant": args.global_feature_variant,
        "noisy_target_mode": args.noisy_target_mode,
        "choices": DEFAULT_CHOICES,
        "criteria": args.criteria,
        "ucb_value": args.ucb_value,
        "pw_c": args.pw_c,
        "pw_alpha": args.pw_alpha,
        "gnn_candidate_pool_size": args.gnn_candidate_pool_size,
        "configs": CONFIGS,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(config, handle, indent=2)


def run_experiment(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    provider = BackendFeatureProvider(args.backend_name)
    backend = getattr(provider, "_backend", None)

    if backend is None:
        raise RuntimeError(f"Could not create backend for name: {args.backend_name}")

    noise_model = NoiseModel.from_backend(backend)
    noisy_sim = AerSimulator(method="density_matrix", noise_model=noise_model)

    n_qubits = int(backend.num_qubits)
    coupling_edges = get_directed_coupling_edges(backend, n_qubits)
    target_qc = load_target_circuit(args.target_qasm, args.target_source_pkl)
    target_qasm = circuit_to_qasm(target_qc)
    noisy_fidelity_function = make_noisy_fidelity_function(
        target_qc=target_qc,
        noisy_sim=noisy_sim,
        target_mode=args.noisy_target_mode,
    )

    sample_qc = make_root_circuit(n_qubits, args.root_hadamard)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gnn_model(
        checkpoint_path=args.model_checkpoint,
        provider=provider,
        target_qc=target_qc,
        global_feature_variant=args.global_feature_variant,
        device=device,
        sample_qc=sample_qc,
    )
    gnn_value_function, gnn_rank_function = make_gnn_predictors(
        model=model,
        provider=provider,
        target_qc=target_qc,
        global_feature_variant=args.global_feature_variant,
        device=device,
    )

    save_config(args, args.out_dir / "experiment_config.json")

    print("=" * 80)
    print("PWMCTS 6-configuration GNN integration experiment")
    print("=" * 80)
    print(f"Depth:             {args.depth}")
    print(f"Backend:           {args.backend_name}")
    print(f"Target depth:      {target_qc.depth()}")
    print(f"Target size:       {target_qc.size()}")
    print(f"Root starts with H: {args.root_hadamard}")
    print(f"Noisy target mode: {args.noisy_target_mode}")
    print(f"Budgets:           {args.budgets}")
    print(f"N runs:            {args.n_runs}")
    print()
    print(target_qc)
    print()

    all_rows: List[Dict[str, Any]] = []
    final_circuit_records: List[Tuple[Dict[str, Any], float]] = []
    run_seeds = [100 + idx for idx in range(args.n_runs)]

    for cfg_idx, cfg in enumerate(CONFIGS, start=1):
        print("=" * 80)
        print(f"Config {cfg_idx}/{len(CONFIGS)}: {cfg['name']}")
        print("=" * 80)

        expansion_mode = "gnn_guided" if cfg["use_guided"] else "random"
        evaluation_mode, evaluation_function, eval_name = make_evaluation_function(
            cfg,
            target_qc,
            noisy_fidelity_function,
        )

        for budget in args.budgets:
            print(f"Budget {budget}")

            for run_idx, seed in enumerate(run_seeds):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                run_start = time.perf_counter()
                root_qc = make_root_circuit(n_qubits, args.root_hadamard)
                root = make_root_node_from_qc(
                    quantum_circuit=root_qc,
                    max_depth=args.max_depth,
                    coupling_map=coupling_edges,
                    directed_coupling=True,
                )

                result = mcts(
                    root=root,
                    budget=int(budget),
                    evaluation_function=evaluation_function,
                    criteria=args.criteria,
                    rollout_type="classic",
                    roll_out_steps=0,
                    simulation=False,
                    branches=False,
                    choices=DEFAULT_CHOICES,
                    epsilon=None,
                    stop_deterministic=False,
                    ucb_value=args.ucb_value,
                    pw_C=args.pw_c,
                    pw_alpha=args.pw_alpha,
                    verbose=False,
                    evaluation_mode=evaluation_mode,
                    gnn_value_function=gnn_value_function if cfg["use_surrogate"] else None,
                    expansion_mode=expansion_mode,
                    gnn_rank_function=gnn_rank_function if cfg["use_guided"] else None,
                    gnn_candidate_pool_size=args.gnn_candidate_pool_size,
                    ignore_stop_in_guided=True,
                    prefer_unvisited_by_prior=True,
                    coupling_map=coupling_edges,
                    directed_coupling=True,
                )
                run_time = time.perf_counter() - run_start

                best_qc = result["best_overall_qc"]
                best_eval_value = float(result["best_overall_value"])
                best_true_noiseless = noiseless_fidelity(best_qc, target_qc)
                best_true_noisy = noisy_fidelity_function(best_qc)
                best_pred_noisy = float(gnn_value_function(best_qc))
                best_qasm = circuit_to_qasm(best_qc)

                row = {
                    "config_name": cfg["name"],
                    "budget": int(budget),
                    "run_idx": int(run_idx),
                    "seed": int(seed),
                    "use_guided": bool(cfg["use_guided"]),
                    "use_surrogate": bool(cfg["use_surrogate"]),
                    "use_noiseless": bool(cfg["use_noiseless"]),
                    "use_noisy": bool(cfg["use_noisy"]),
                    "evaluation_mode": evaluation_mode,
                    "expansion_mode": expansion_mode,
                    "eval_name": eval_name,
                    "best_eval_value": best_eval_value,
                    "best_true_noiseless": best_true_noiseless,
                    "best_true_noisy": best_true_noisy,
                    "best_predicted_noisy_fidelity": best_pred_noisy,
                    "best_depth": int(best_qc.depth()),
                    "best_size": int(best_qc.size()),
                    "oracle_eval_calls": int(result["oracle_eval_calls"]),
                    "gnn_eval_calls": int(result["gnn_eval_calls"]),
                    "gnn_rank_batches": int(result["gnn_rank_batches"]),
                    "gnn_rank_candidates_scored": int(result["gnn_rank_candidates_scored"]),
                    "run_time_seconds": float(run_time),
                }
                all_rows.append(row)

                final_meta = {
                    **row,
                    "qasm_transpiled": best_qasm,
                    "target_qasm_transpiled": target_qasm,
                    "oracle_noiseless_fidelity": best_true_noiseless,
                    "noisy_fidelity": best_true_noisy,
                    "predicted_noisy_fidelity": best_pred_noisy,
                    "backend": args.backend_name,
                    "global_feature_variant": args.global_feature_variant,
                    "node_angle_encoding": NODE_ANGLE_ENCODING,
                    "source": "mcts_final_circuit",
                }
                final_circuit_records.append((final_meta, float(best_true_noisy)))

                print(
                    f"  run {run_idx + 1:02d}/{args.n_runs} | "
                    f"eval={best_eval_value:.6f} | "
                    f"sim_noisy={best_true_noisy:.6f} | "
                    f"pred_noisy={best_pred_noisy:.6f} | "
                    f"time={run_time:.2f}s"
                )

                write_csv(all_rows, args.out_dir / "mcts_6configs_runs.csv")

    if not all_rows:
        raise RuntimeError("No experiment rows were produced.")

    final_circuits_pkl = args.out_dir / "mcts_final_circuits_all_configs.pkl"
    with final_circuits_pkl.open("wb") as handle:
        pickle.dump(final_circuit_records, handle)

    summary_rows = summarize_rows(all_rows)
    write_csv(summary_rows, args.out_dir / "mcts_6configs_summary.csv")
    save_convergence_plot(summary_rows, args.out_dir / "mcts_6configs_simulated_noisy_vs_budget.png")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for row in summary_rows:
        print(
            f"{row['config_name']:18s} | "
            f"budget={int(row['budget']):5d} | "
            f"sim_noisy={float(row['mean_best_true_noisy']):.6f} "
            f"+/- {float(row['std_best_true_noisy']):.6f} | "
            f"time={float(row['mean_run_time_seconds']):.2f}s"
        )

    print()
    print(f"Saved runs to:           {args.out_dir / 'mcts_6configs_runs.csv'}")
    print(f"Saved final circuits to: {final_circuits_pkl}")
    print(f"Saved summary to:        {args.out_dir / 'mcts_6configs_summary.csv'}")
    print(f"Saved plot to:           {args.out_dir / 'mcts_6configs_simulated_noisy_vs_budget.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six PWMCTS/GNN integration configurations on a constrained target."
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--target-qasm", type=Path, default=DEFAULT_TARGET_QASM)
    parser.add_argument("--target-source-pkl", type=Path, default=DEFAULT_TARGET_SOURCE_PKL)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--budgets", type=parse_int_list, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--global-feature-variant", default=DEFAULT_GLOBAL_FEATURE_VARIANT)
    parser.add_argument(
        "--noisy-target-mode",
        choices=["ideal_target", "noisy_target"],
        default="ideal_target",
        help="ideal_target matches the constrained data-generation scripts.",
    )
    parser.add_argument("--criteria", default="average_value")
    parser.add_argument("--ucb-value", type=float, default=0.4)
    parser.add_argument("--pw-c", type=float, default=1.0)
    parser.add_argument("--pw-alpha", type=float, default=0.3)
    parser.add_argument("--gnn-candidate-pool-size", type=int, default=GNN_CANDIDATE_POOL_SIZE)
    parser.add_argument(
        "--empty-root",
        action="store_false",
        dest="root_hadamard",
        help="Use an empty root circuit instead of the initial Hadamard decomposition.",
    )
    parser.set_defaults(root_hadamard=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
