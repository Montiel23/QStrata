# Q72S — Quantum Metrics Smoke

**Slice ID**: Q72S-QUANTUM-METRICS-SMOKE  
**Campaign**: pure_quantum_readout_smoke  
**Status**: BLOCKED  
**Depends on**: Q67S, Q68S, Q68DS  
**Estimated runtime**: LOW (< 5 min; analysis only, no training)  
**Date planned**: After Q67S, Q68S, Q68DS complete  

---

## 1. Objective

Compute available quantum metrics from the smoke run outputs of Q67S, Q68S, Q68DS
and from the Q62A saved checkpoints. No re-training. Provide a quantitative snapshot
of quantum circuit behavior to inform the Q74S gate decision.

---

## 2. Inputs

| Artifact | Path | Notes |
|---|---|---|
| Q67S smoke results | `workspace/experiments/Q67S/results/q67s_smoke_results.json` | DV-QNN per-sample probs |
| Q67S smoke metrics | `workspace/experiments/Q67S/results/q67s_smoke_metrics.csv` | |
| Q68S homodyne stats | `workspace/experiments/Q68S/results/q68s_homodyne_stats.csv` | CV X-quadrature |
| Q68DS dual homodyne stats | `workspace/experiments/Q68DS/results/q68ds_dual_homodyne_stats.csv` | |
| Q62A DV-QNN checkpoints | `workspace/experiments/Q62A/checkpoints/dv_qnn_seed42.pt` | trained weights |
| Q62A CV-QNN checkpoints | `workspace/experiments/Q62A/checkpoints/cv_qnn_seed42.pt` | trained weights |

---

## 3. Metrics to Compute

### DV-QNN Metrics (from Q67S outputs)

| Metric | Formula | Source |
|--------|---------|--------|
| n_qubits | 4 | config |
| circuit_depth | 1 | config |
| hilbert_space_dim | 2^4 = 16 | config |
| n_quantum_params | 24 | config |
| n_classical_params | 550 (proj+readout from Q57) | config |
| quantum_classical_ratio | 24/550 = 0.044 | config |
| compression_ratio | 128/4 = 32:1 | config |
| measurement_entropy_mean | mean(-sum(p*log2(p+eps))) over val set | Q67S probs |
| measurement_entropy_class0 | entropy conditioned on class=0 | Q67S probs + labels |
| measurement_entropy_class1 | entropy conditioned on class=1 | Q67S probs + labels |
| prob_margin_mean | mean(max p_class1_states - max p_class0_states) | Q67S probs |
| state_purity | 1.0 (always pure for unitary from vacuum) | analytic |
| class_conditioned_fidelity_proxy | cosine similarity between mean class-0 and class-1 prob vectors | Q67S probs |

### CV-QNN Metrics (from Q68S/Q68DS outputs and Q62A checkpoint)

| Metric | Formula | Source |
|--------|---------|--------|
| n_modes | 2 | config |
| circuit_depth | 1 | config |
| n_quantum_params | 10 | config |
| n_classical_params | 522 (encoder+readout from Q58) | config |
| quantum_classical_ratio | 10/522 = 0.019 | config |
| compression_ratio | 128/4 = 32:1 | config |
| homodyne_X_separation | |mean_X_class1 - mean_X_class0| | Q68S stats |
| homodyne_X_snr | (mean_X_class1-mean_X_class0)^2 / (var_X_class1+var_X_class0) | Q68S stats |
| dual_homodyne_separation | ||mean_features_class1 - mean_features_class0||_2 | Q68DS stats |
| squeezing_post_training_max | max(tanh(squeezing_raw)*1.5) from Q62A checkpoint | Q62A checkpoint |
| squeezing_post_training_mean | mean(tanh(squeezing_raw)*1.5) from Q62A checkpoint | Q62A checkpoint |
| squeezing_cap_utilization | squeezing_post_training_max / 1.5 | derived |
| state_purity | always 1.0 for deterministic Gaussian sim from vacuum | analytic |
| detector_efficiency_placeholder | 1.0 (ideal detector) | placeholder |

---

## 4. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q72S/results/q72s_quantum_metrics.csv` | All metrics in one table |
| `workspace/experiments/Q72S/results/q72s_dv_state_stats.json` | DV-QNN state statistics |
| `workspace/experiments/Q72S/results/q72s_cv_state_stats.json` | CV-QNN state statistics |
| `workspace/experiments/Q72S/reports/q72s_quantum_metrics_smoke_report.md` | Analysis |
| `reports/q72s_quantum_metrics_smoke.md` | Publication copy |

---

## 5. Pass Criteria

- [ ] All metrics in section 3 computed and logged
- [ ] Squeezing values extracted from Q62A checkpoint without error
- [ ] Measurement entropy per class computed for DV-QNN
- [ ] Homodyne SNR computed for CV-QNN
- [ ] State purity documented as analytically 1.0 for both (deterministic sim)
- [ ] quantum_classical_ratio documented for both models
- [ ] No re-training executed
- [ ] No external quantum frameworks
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
