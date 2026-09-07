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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("mlqem_random_fake_lima_dataset")

TRAIN_PKL = DATA_DIR / "mlqem_random_fake_lima_train.pkl"
TEST_PKL = DATA_DIR / "mlqem_random_fake_lima_test.pkl"

OUT_DIR = Path("models_mlqem_fake_lima_rf_paper_like")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper RF setup:
#   - one RF regressor per observable
#   - direct prediction of the mitigated / ideal expectation value
RF_PER_OBSERVABLE = True
TARGET_MODE = "ideal_expectation"

# Paper RF hyperparameters:
#   - 100 tree estimators for each observable
#   - CART-style regression trees
#   - mean-squared-error reduction for regression splits
#   - at least 2 samples required to split an internal node
#   - one feature considered when looking for the best split
RF_KWARGS = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "bootstrap": True,
    "random_state": 34,
    "n_jobs": -1,
}

SEED = 34

# Paper-like tabular features:
#   X = [
#       noisy expectation,
#       counts of native non-parameterized gates,
#       counts of native parameterized gates in angle bins,
#       Pauli observable encoding
#   ]
#
# For FakeLima / IBM-style circuits, native gates are usually:
#   id, rz, sx, x, cx
#
# RZ is the native parameterized gate, so it is represented by binned angle counts.
# The paper does not specify the exact bin edges; we use 8 uniform bins on [-pi, pi].
NON_PARAMETERIZED_NATIVE_GATES = ["id", "sx", "x", "cx"]
RZ_ANGLE_BINS = np.linspace(-np.pi, np.pi, 9)  # 8 bins
N_OBSERVABLE_QUBITS = 4


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# QASM / circuit helpers
# ============================================================

def normalize_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def get_first_existing_key(d: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def circuit_from_meta(meta: Dict[str, Any]) -> QuantumCircuit:
    """
    Recover a QuantumCircuit from the raw PKL metadata.

    Expected preferred keys:
        qasm_transpiled, qasm, qasm_str, circuit_qasm, transpiled_qasm

    Also supports a direct QuantumCircuit object under:
        circuit, quantum_circuit, qc
    """
    circ = get_first_existing_key(meta, ["circuit", "quantum_circuit", "qc"])
    if isinstance(circ, QuantumCircuit):
        return circ

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
        return QuantumCircuit.from_qasm_str(qasm_str)

    available = sorted(meta.keys())
    raise KeyError(
        "Could not find a circuit or QASM string in the raw PKL item. "
        f"Available keys: {available}"
    )


def count_nonparam_native_gates_and_rz_bins(qc: QuantumCircuit) -> np.ndarray:
    """
    Paper-like circuit-level feature encoding.

    The paper states that circuit-level features are native gate counts, with
    parameterized native gates counted in binned angles.

    Here:
      - id, sx, x, cx are counted directly;
      - rz(theta) is counted in angle bins.
    """
    counts = {name: 0.0 for name in NON_PARAMETERIZED_NATIVE_GATES}
    rz_hist = np.zeros(len(RZ_ANGLE_BINS) - 1, dtype=float)

    for instr in qc.data:
        try:
            op = instr.operation
        except AttributeError:
            op, _qargs, _cargs = instr

        name = op.name.lower()

        if name in counts:
            counts[name] += 1.0

        if name == "rz":
            params = getattr(op, "params", [])
            if params:
                try:
                    theta = normalize_angle(float(params[0]))
                    bin_idx = int(np.digitize([theta], RZ_ANGLE_BINS, right=False)[0] - 1)
                    bin_idx = max(0, min(bin_idx, len(rz_hist) - 1))
                    rz_hist[bin_idx] += 1.0
                except Exception:
                    # If a parameter cannot be converted to float, ignore it.
                    pass

    gate_counts = np.asarray(
        [counts[name] for name in NON_PARAMETERIZED_NATIVE_GATES],
        dtype=float,
    )

    return np.concatenate([gate_counts, rz_hist])


def sparse_pauli_observable_encoding(observable_qubit: int) -> np.ndarray:
    """
    Sparse Pauli observable representation for the random-circuit experiment.

    Observables are Z0, Z1, Z2, Z3, so we encode them as a one-hot vector:
        Z0 -> [1, 0, 0, 0]
        Z1 -> [0, 1, 0, 0]
        Z2 -> [0, 0, 1, 0]
        Z3 -> [0, 0, 0, 1]
    """
    enc = np.zeros(N_OBSERVABLE_QUBITS, dtype=float)

    if 0 <= observable_qubit < N_OBSERVABLE_QUBITS:
        enc[observable_qubit] = 1.0
    else:
        raise ValueError(f"Invalid observable_qubit={observable_qubit}")

    return enc


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
        "Could not infer observable_qubit. Expected key 'observable_qubit' "
        "or observable string like 'Z0'."
    )


def infer_noisy_expectation(meta: Dict[str, Any]) -> float:
    for key in [
        "noisy_expectation",
        "noisy_exp_value",
        "noisy",
        "noisy_value",
    ]:
        if key in meta:
            return float(meta[key])

    # Some raw data formats store a vector of noisy expectations.
    # In that case the observable_qubit selects the scalar.
    if "noisy_exp_values" in meta:
        q = infer_observable_qubit(meta)
        vals = np.asarray(meta["noisy_exp_values"], dtype=float).reshape(-1)
        return float(vals[q])

    if "noisy_exp_val" in meta:
        q = infer_observable_qubit(meta)
        vals = np.asarray(meta["noisy_exp_val"], dtype=float).reshape(-1)
        return float(vals[q])

    raise KeyError("Could not find noisy expectation in raw PKL item.")


def infer_ideal_expectation(meta: Dict[str, Any], label: Any) -> float:
    # Preferred case for our reproduced dataset:
    # item = (meta, label), where label is already the scalar ideal expectation.
    try:
        if label is not None:
            return float(label)
    except Exception:
        pass

    for key in [
        "ideal_expectation",
        "ideal_exp_value",
        "ideal",
        "ideal_value",
    ]:
        if key in meta:
            val = meta[key]
            arr = np.asarray(val)

            if arr.size == 1:
                return float(arr.reshape(-1)[0])

            # Vector case: select observable qubit.
            q = infer_observable_qubit(meta)
            return float(arr.reshape(-1)[q])

    if "ideal_exp_val" in meta:
        q = infer_observable_qubit(meta)
        vals = np.asarray(meta["ideal_exp_val"], dtype=float).reshape(-1)
        return float(vals[q])

    raise KeyError("Could not find ideal expectation / label in raw PKL item.")


def make_paper_like_features(
    meta: Dict[str, Any],
    noisy: float,
    observable_qubit: int,
) -> np.ndarray:
    """
    Build the paper-like RF feature vector:
        [noisy expectation,
         native non-parameterized gate counts,
         RZ binned angle counts,
         observable one-hot encoding]

    No GNN graph node features, edge features, backend summaries, or extra
    handcrafted global features are appended.
    """
    qc = circuit_from_meta(meta)

    circuit_features = count_nonparam_native_gates_and_rz_bins(qc)
    observable_features = sparse_pauli_observable_encoding(observable_qubit)

    return np.concatenate([
        np.asarray([float(noisy)], dtype=float),
        circuit_features,
        observable_features,
    ])


# ============================================================
# Raw PKL loading
# ============================================================

def load_raw_pkl_rows(pkl_path: Path, split_name: str) -> List[Dict[str, Any]]:
    """
    Load the raw PKL directly instead of using the PyG graph dataset.

    This is necessary for the paper-like RF because the processed graph dataset
    may not preserve QASM / QuantumCircuit. RF does not need the graph; it only
    needs tabular features.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing PKL: {pkl_path}")

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

        observable_qubit = infer_observable_qubit(meta)
        noisy = infer_noisy_expectation(meta)
        ideal = infer_ideal_expectation(meta, label)

        X = make_paper_like_features(
            meta=meta,
            noisy=noisy,
            observable_qubit=observable_qubit,
        )

        circuit_idx = int(meta.get("circuit_idx", meta.get("sample_idx", i // N_OBSERVABLE_QUBITS)))
        target_depth = int(meta.get("target_two_qubit_depth", meta.get("target_depth", meta.get("depth", -1))))
        actual_depth = int(meta.get("actual_two_qubit_depth", meta.get("actual_depth", target_depth)))

        rows.append({
            "idx": i,
            "split": str(meta.get("split", split_name)),
            "X": X,
            "target": float(ideal),
            "ideal_expectation": float(ideal),
            "noisy_expectation": float(noisy),
            "observable_qubit": int(observable_qubit),
            "observable": str(meta.get("observable", f"Z{observable_qubit}")),
            "circuit_idx": circuit_idx,
            "target_two_qubit_depth": target_depth,
            "actual_two_qubit_depth": actual_depth,
        })

    if not rows:
        raise RuntimeError(f"No usable rows loaded from {pkl_path}")

    return rows


def rows_to_X_y(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray([r["X"] for r in rows], dtype=float)
    y = np.asarray([r["target"] for r in rows], dtype=float)
    return X, y


# ============================================================
# Train RF
# ============================================================

def train_rf_models(train_rows: List[Dict[str, Any]]):
    if RF_PER_OBSERVABLE:
        models: Dict[int, RandomForestRegressor] = {}

        for q in range(N_OBSERVABLE_QUBITS):
            q_rows = [r for r in train_rows if int(r["observable_qubit"]) == q]

            if not q_rows:
                raise RuntimeError(f"No training rows found for observable Z{q}")

            X, y = rows_to_X_y(q_rows)

            model = RandomForestRegressor(**RF_KWARGS)
            model.fit(X, y)

            models[q] = model

            print(
                f"Trained RF_Z{q}: n={len(q_rows)} | "
                f"X_dim={X.shape[1]} | "
                f"n_estimators={RF_KWARGS['n_estimators']} | "
                f"max_features={RF_KWARGS['max_features']}"
            )

        return models

    X, y = rows_to_X_y(train_rows)

    model = RandomForestRegressor(**RF_KWARGS)
    model.fit(X, y)

    print(f"Trained single RF: n={len(train_rows)} | X_dim={X.shape[1]}")
    return model


def predict_rf(models, row: Dict[str, Any]) -> float:
    X = np.asarray(row["X"], dtype=float).reshape(1, -1)

    if RF_PER_OBSERVABLE:
        q = int(row["observable_qubit"])
        pred_ideal = float(models[q].predict(X)[0])
    else:
        pred_ideal = float(models.predict(X)[0])

    # Pauli expectation values are physically bounded.
    return float(np.clip(pred_ideal, -1.0, 1.0))


# ============================================================
# Metrics
# ============================================================

def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def compute_l2_metrics(pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in pred_rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "rf": float(r["rf_predicted_ideal_expectation"]),
        }

    baseline_l2: List[float] = []
    rf_l2: List[float] = []

    by_depth: Dict[int, Dict[str, List[float]]] = {}

    for key, obs_dict in groups.items():
        target_depth, _circuit_idx = key

        if not all(q in obs_dict for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        rf_vec = np.asarray([obs_dict[q]["rf"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)

        b = float(np.linalg.norm(noisy_vec - ideal_vec, ord=2))
        r = float(np.linalg.norm(rf_vec - ideal_vec, ord=2))

        baseline_l2.append(b)
        rf_l2.append(r)

        by_depth.setdefault(target_depth, {"baseline": [], "rf": []})
        by_depth[target_depth]["baseline"].append(b)
        by_depth[target_depth]["rf"].append(r)

    b_arr = np.asarray(baseline_l2, dtype=float)
    r_arr = np.asarray(rf_l2, dtype=float)

    out: Dict[str, float] = {
        "n_circuits_l2": int(len(b_arr)),
        "baseline_l2_mean": float(np.mean(b_arr)),
        "baseline_l2_std": float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0,
        "rf_l2_mean": float(np.mean(r_arr)),
        "rf_l2_std": float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0,
        "relative_l2_improvement": float((np.mean(b_arr) - np.mean(r_arr)) / np.mean(b_arr)),
    }

    for depth in sorted(by_depth):
        b_d = np.asarray(by_depth[depth]["baseline"], dtype=float)
        r_d = np.asarray(by_depth[depth]["rf"], dtype=float)

        out[f"depth_{depth}_n"] = int(len(b_d))
        out[f"depth_{depth}_baseline_l2_mean"] = float(np.mean(b_d))
        out[f"depth_{depth}_rf_l2_mean"] = float(np.mean(r_d))

    return out


def per_observable_metrics(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for q in range(N_OBSERVABLE_QUBITS):
        q_rows = [r for r in pred_rows if int(r["observable_qubit"]) == q]

        y = np.asarray([r["ideal_expectation"] for r in q_rows], dtype=float)
        noisy = np.asarray([r["noisy_expectation"] for r in q_rows], dtype=float)
        rf = np.asarray([r["rf_predicted_ideal_expectation"] for r in q_rows], dtype=float)

        b = scalar_metrics(y, noisy)
        r = scalar_metrics(y, rf)

        out.append({
            "observable": f"Z{q}",
            "n": len(q_rows),
            "baseline_mae": b["mae"],
            "rf_mae": r["mae"],
            "baseline_mse": b["mse"],
            "rf_mse": r["mse"],
            "baseline_r2": b["r2"],
            "rf_r2": r["r2"],
        })

    return out


# ============================================================
# Save / plots
# ============================================================

def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    clean_rows = []

    for r in rows:
        rr = {
            k: v for k, v in r.items()
            if k != "X"
        }
        clean_rows.append(rr)

    fieldnames = sorted(set().union(*(r.keys() for r in clean_rows)))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


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


def save_l2_histogram(pred_rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

    for r in pred_rows:
        key = (int(r["target_two_qubit_depth"]), int(r["circuit_idx"]))
        q = int(r["observable_qubit"])

        groups.setdefault(key, {})
        groups[key][q] = {
            "ideal": float(r["ideal_expectation"]),
            "noisy": float(r["noisy_expectation"]),
            "rf": float(r["rf_predicted_ideal_expectation"]),
        }

    baseline_l2 = []
    rf_l2 = []

    for key, obs_dict in groups.items():
        if not all(q in obs_dict for q in range(N_OBSERVABLE_QUBITS)):
            continue

        ideal_vec = np.asarray([obs_dict[q]["ideal"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        noisy_vec = np.asarray([obs_dict[q]["noisy"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)
        rf_vec = np.asarray([obs_dict[q]["rf"] for q in range(N_OBSERVABLE_QUBITS)], dtype=float)

        baseline_l2.append(float(np.linalg.norm(noisy_vec - ideal_vec, ord=2)))
        rf_l2.append(float(np.linalg.norm(rf_vec - ideal_vec, ord=2)))

    plt.figure(figsize=(8, 5))
    plt.hist(baseline_l2, bins=60, alpha=0.55, label="Unmitigated noisy", edgecolor="black")
    plt.hist(rf_l2, bins=60, alpha=0.55, label="RF mitigated", edgecolor="black")
    plt.xlabel(r"$L_2$ error on $[\langle Z_0\rangle,\ldots,\langle Z_3\rangle]$")
    plt.ylabel("Number of circuits")
    plt.title("Paper-style vector error distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_all_seeds(SEED)

    print("=" * 80)
    print("ML-QEM FakeLima RF baseline: paper-like model and features")
    print("=" * 80)
    print(f"Train PKL:      {TRAIN_PKL}")
    print(f"Test PKL:       {TEST_PKL}")
    print(f"RF per obs:     {RF_PER_OBSERVABLE}")
    print(f"Target mode:    {TARGET_MODE}")
    print(f"RF kwargs:      {RF_KWARGS}")
    print(f"Output dir:     {OUT_DIR}")
    print(f"Feature set:    noisy + gate counts + RZ angle bins + observable one-hot")
    print(f"RZ bins:        {RZ_ANGLE_BINS.tolist()}")
    print()

    train_rows = load_raw_pkl_rows(TRAIN_PKL, split_name="train")
    test_rows = load_raw_pkl_rows(TEST_PKL, split_name="test")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    X_train_noisy = np.asarray([[r["noisy_expectation"]] for r in train_rows], dtype=float)
    y_train = np.asarray([r["ideal_expectation"] for r in train_rows], dtype=float)

    X_test_noisy = np.asarray([[r["noisy_expectation"]] for r in test_rows], dtype=float)
    y_test = np.asarray([r["ideal_expectation"] for r in test_rows], dtype=float)
    noisy_test = np.asarray([r["noisy_expectation"] for r in test_rows], dtype=float)

    rf_noisy_only = RandomForestRegressor(
        n_estimators=100,
        random_state=34,
        n_jobs=-1,
    )

    rf_noisy_only.fit(X_train_noisy, y_train)
    pred = np.clip(rf_noisy_only.predict(X_test_noisy), -1.0, 1.0)

    print("Noisy-only sanity check:")
    print("Baseline MAE:", mean_absolute_error(y_test, noisy_test))
    print("RF noisy-only MAE:", mean_absolute_error(y_test, pred))

    print("=" * 80)
    print("Dataset")
    print("=" * 80)
    print(f"Train samples: {len(train_rows)}")
    print(f"Test samples:  {len(test_rows)}")
    print(f"Feature dim:   {len(train_rows[0]['X'])}")
    print()

    models = train_rf_models(train_rows)

    pred_rows: List[Dict[str, Any]] = []

    for r in test_rows:
        pred = predict_rf(models, r)

        rr = dict(r)
        rr["rf_predicted_ideal_expectation"] = float(pred)
        rr["baseline_error"] = float(r["noisy_expectation"] - r["ideal_expectation"])
        rr["rf_error"] = float(pred - r["ideal_expectation"])
        rr["baseline_abs_error"] = abs(rr["baseline_error"])
        rr["rf_abs_error"] = abs(rr["rf_error"])

        pred_rows.append(rr)

    y = np.asarray([r["ideal_expectation"] for r in pred_rows], dtype=float)
    noisy = np.asarray([r["noisy_expectation"] for r in pred_rows], dtype=float)
    rf = np.asarray([r["rf_predicted_ideal_expectation"] for r in pred_rows], dtype=float)

    baseline_scalar = scalar_metrics(y, noisy)
    rf_scalar = scalar_metrics(y, rf)
    l2 = compute_l2_metrics(pred_rows)
    per_obs = per_observable_metrics(pred_rows)

    metrics = {
        "target_mode": TARGET_MODE,
        "rf_per_observable": RF_PER_OBSERVABLE,
        "rf_kwargs": RF_KWARGS,
        "feature_description": {
            "features": [
                "noisy_expectation",
                "native_non_parameterized_gate_counts_id_sx_x_cx",
                "rz_angle_binned_counts",
                "single_qubit_Z_observable_one_hot",
            ],
            "native_non_parameterized_gates": NON_PARAMETERIZED_NATIVE_GATES,
            "rz_angle_bins": RZ_ANGLE_BINS.tolist(),
            "extra_gnn_global_features_used": False,
        },
        "baseline_scalar_metrics": baseline_scalar,
        "rf_scalar_metrics": rf_scalar,
        "l2_metrics": l2,
        "per_observable_metrics": per_obs,
    }

    save_pickle(models, OUT_DIR / "rf_models_paper_like.pkl")
    save_csv(pred_rows, OUT_DIR / "rf_predictions_test_paper_like.csv")

    with (OUT_DIR / "rf_metrics_paper_like.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (OUT_DIR / "per_observable_metrics_paper_like.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_obs[0].keys()))
        writer.writeheader()
        writer.writerows(per_obs)

    save_scatter(
        y,
        noisy,
        OUT_DIR / "scatter_unmitigated_noisy_vs_ideal.png",
        title="Unmitigated noisy vs ideal expectation",
        ylabel=r"Noisy $\langle Z_i\rangle$",
    )

    save_scatter(
        y,
        rf,
        OUT_DIR / "scatter_rf_mitigated_vs_ideal_paper_like.png",
        title="Paper-like RF mitigated vs ideal expectation",
        ylabel=r"RF mitigated $\langle Z_i\rangle$",
    )

    save_l2_histogram(
        pred_rows,
        OUT_DIR / "l2_error_distribution_unmitigated_vs_rf_paper_like.png",
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("Scalar metrics:")
    print(
        f"Unmitigated | "
        f"MSE={baseline_scalar['mse']:.6g} | "
        f"RMSE={baseline_scalar['rmse']:.6g} | "
        f"MAE={baseline_scalar['mae']:.6g} | "
        f"R2={baseline_scalar['r2']:.4f}"
    )

    print(
        f"RF          | "
        f"MSE={rf_scalar['mse']:.6g} | "
        f"RMSE={rf_scalar['rmse']:.6g} | "
        f"MAE={rf_scalar['mae']:.6g} | "
        f"R2={rf_scalar['r2']:.4f}"
    )

    print("\nPaper-style vector L2 metric:")
    print(
        f"Unmitigated mean L2 = {l2['baseline_l2_mean']:.6g} "
        f"± {l2['baseline_l2_std']:.6g}"
    )
    print(
        f"RF mean L2          = {l2['rf_l2_mean']:.6g} "
        f"± {l2['rf_l2_std']:.6g}"
    )
    print(
        f"Relative L2 improvement = "
        f"{100.0 * l2['relative_l2_improvement']:.2f}%"
    )

    print("\nPer observable:")
    for row in per_obs:
        print(
            f"{row['observable']:>2s} | "
            f"baseline_MAE={row['baseline_mae']:.6g} | "
            f"rf_MAE={row['rf_mae']:.6g} | "
            f"baseline_R2={row['baseline_r2']:.4f} | "
            f"rf_R2={row['rf_r2']:.4f}"
        )

    print("\nSaved:")
    print(f"  {OUT_DIR / 'rf_models_paper_like.pkl'}")
    print(f"  {OUT_DIR / 'rf_predictions_test_paper_like.csv'}")
    print(f"  {OUT_DIR / 'rf_metrics_paper_like.json'}")
    print(f"  {OUT_DIR / 'scatter_rf_mitigated_vs_ideal_paper_like.png'}")
    print(f"  {OUT_DIR / 'l2_error_distribution_unmitigated_vs_rf_paper_like.png'}")


if __name__ == "__main__":
    main()
