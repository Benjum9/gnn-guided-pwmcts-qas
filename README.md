# GNN-Guided PWMCTS for Noise-Aware Quantum Architecture Search

This repository contains the code used for the experiments in the paper on GNN-guided progressive-widening Monte Carlo tree search for noise-aware quantum architecture search and quantum error mitigation.

The project studies how graph neural networks can be used as surrogate models to accelerate quantum circuit search under realistic hardware noise. Candidate quantum circuits are represented as graphs, trained with backend-aware features, and integrated into MCTS to reduce the cost of repeated noisy simulations.

## Repository Structure

```text
oracle_approximation_problem/
quantum_error_mitigation/
```

## Oracle Approximation and QAS

The `oracle_approximation_problem` folder contains the scripts for the quantum architecture search experiments based on oracle approximation.

Main components include:

- constrained target-circuit generation
- PWMCTS dataset generation
- noisy-fidelity relabeling and backend-specific dataset generation
- graph representation of quantum circuits
- GNN models for predicting noise-induced fidelity loss
- direct prediction versus loss prediction experiments
- leave-one-backend-out generalization experiments
- new-target generalization experiments
- MCTS integration experiments comparing noiseless, simulated-noisy, and GNN-surrogate search modes

In these experiments, MCTS searches for candidate circuits using different evaluation strategies. The GNN is trained to approximate the effect of hardware noise and can then be used inside the search to reduce the number of expensive noisy simulations.

## Quantum Error Mitigation

The `quantum_error_mitigation` folder contains the scripts for the machine-learning quantum error mitigation experiments.

Main components include:

- ML-QEM dataset generation on fake IBM backends
- dataset checking and diagnostic plotting
- graph representation of quantum circuits
- GNN-based mitigation models
- random forest mitigation baselines
- zero-noise extrapolation baselines
- bootstrap confidence-interval comparison scripts

The goal of these experiments is to compare unmitigated noisy estimates, zero-noise extrapolation, random forest mitigation, and GNN-based mitigation.

## Main Scripts

### Oracle Approximation

```text
generate_constrained_target_circuit.py
```

Utilities for generating constrained random target circuits.

```text
data_generation.py
data_generation_constrained_target_depth_9.py
data_generation_constrained_target_depth_10.py
```

Generate PWMCTS datasets for constrained target circuits.

```text
noise_injection.py
```

Augments generated circuit datasets by injecting additional CNOT gates and recomputing fidelities.

```text
generate_other_backend.py
```

Generates backend-specific datasets by recomputing noisy fidelities on additional fake IBM backends.

```text
graph_representation.py
graph_representation_no_clean.py
```

Convert quantum circuits into graph representations for GNN training.

```text
gnn_delta.py
gnn_delta_no_clean.py
```

GNN models for predicting noise-induced fidelity loss.

```text
backend_generalization_depth_10.py
backend_generalization_depth_10_no_clean.py
```

Run leave-one-backend-out generalization experiments.

```text
compare_direct_vs_loss.py
```

Compare direct noisy-fidelity prediction with noise-loss prediction.

```text
experiment_6_configs_constrained_depth_8.py
```

Compare MCTS configurations using noiseless, simulated-noisy, and GNN-surrogate evaluation.

```text
data_generation_new_target_depth_8.py
target_transfer_finetune_depth_8_new_target.py
```

Generate and evaluate generalization to a new target circuit.

### Quantum Error Mitigation

```text
data_generation_mlqem_fake_lima.py
```

Generate the ML-QEM dataset on the FakeLima backend.

```text
check_mlqem_dataset.py
```

Check and visualize the generated ML-QEM dataset.

```text
graph_representation_mlqem_enhanced.py
```

Build graph representations for ML-QEM circuits.

```text
gnn_mlqem_enhanced.py
```

Define the enhanced GNN model for ML-QEM.

```text
train_mlqem_fake_lima_gnn.py
train_mlqem_fake_lima_rf.py
```

Train the GNN and random forest mitigation models.

```text
zne_fake_lima_linear_13.py
zne_fake_lima_quadratic_135.py
```

Run zero-noise extrapolation baselines.

```text
compare_mlqem_models_bootstrap.py
plot_mlqem_comparison_good_plot.py
```

Compare mitigation methods using bootstrap confidence intervals and generate final plots.

## Dependencies

The main dependencies are:

```text
numpy
pandas
matplotlib
scipy
qiskit
qiskit-aer
qiskit-ibm-runtime
torch
torch-geometric
scikit-learn
```

The experiments were developed with Python 3 and Qiskit fake backends. Exact versions may need to be adjusted depending on the local installation.

## Data and Outputs

Large generated datasets, trained models, logs, and intermediate result folders are not included directly in this repository.

Typical omitted files include:

```text
*.pkl
*.pt
*.log
*.png
*.pdf
data_generation_results_*/
models_*/
mlqem_random_fake_lima_dataset/
mlqem_model_comparison_bootstrap/
```

The scripts can be used to regenerate the datasets and experimental outputs.

## Noisy Fidelity Convention

For the oracle approximation experiments, noisy fidelities are computed with density-matrix simulation. The noisy candidate circuit output is compared with the noisy target circuit output:

```text
F(rho_candidate_noisy, rho_target_noisy)
```

This convention is used to keep the labels consistent with the simulated backend noise model.

## Citation

If you use this code, please cite the associated paper.
