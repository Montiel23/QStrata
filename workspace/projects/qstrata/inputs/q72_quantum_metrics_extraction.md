# Q72 — Quantum Metrics Extraction

**Slice ID**: Q72-QUANTUM-METRICS-EXTRACTION  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q69, Q70, Q71  
**Estimated runtime**: MEDIUM (< 30 min; inference + analysis; no training)  

---

## 1. Objective

Re-run inference on Q62A checkpoints (hybrid models) and Q69/Q70/Q71 models with
extended quantum state logging. Extract class-conditioned quantum state statistics,
measurement entropy, homodyne SNR, post-training squeezing, and gradient norm proxies.
No re-training.

---

## 2. Inputs

| Artifact | Path |
|---|---|
| Q62A DV-QNN checkpoints | `workspace/experiments/Q62A/checkpoints/dv_qnn_seed{42,7,123}.pt` |
| Q62A CV-QNN checkpoints | `workspace/experiments/Q62A/checkpoints/cv_qnn_seed{42,7,123}.pt` |
| Q69 DV pure readout models | trained during Q69 |
| Q70 CV single homodyne models | trained during Q70 |
| Q49 embeddings | `workspace/experiments/Q49/embeddings/` |

---

## 3. Metrics to Extract

### DV-QNN (hybrid Q57 and pure Q69)

| Metric | Method |
|--------|--------|
| state_purity | 1.0 (analytic for pure unitary circuit) |
| class_conditioned_fidelity | `|<ψ_c0|ψ_c1>|²`; mean class states from val set |
| trace_distance_proxy | `0.5 * ||p_c0 - p_c1||_1` (L1 on probability vectors) |
| measurement_entropy_per_class | `-sum(p*log2(p+eps))` per class, val set |
| probability_margin_per_class | `max(p_class1_states) - max(p_class0_states)` |
| gradient_norm_theta | `||d_loss/d_theta||_2` — from one training step on val set |
| gradient_norm_proj | `||d_loss/d_proj||_2` |

### CV-QNN (hybrid Q58 and pure Q70/Q71)

| Metric | Method |
|--------|--------|
| state_purity | 1.0 (analytic for deterministic Gaussian sim) |
| mean_displacement_separation | `||E[mu_out | c=1] - E[mu_out | c=0]||_2` |
| homodyne_X_snr | `(mean_X_c1 - mean_X_c0)^2 / (var_X_c1 + var_X_c0)` |
| covariance_trace_by_class | `Tr(Sigma_c)` averaged per class |
| squeezing_post_training | `tanh(squeezing_raw)*squeezing_cap` per mode, from checkpoint |
| squeezing_cap_utilization | max squeezing / squeezing_cap |
| gradient_norm_ansatz | `||d_loss/d_ansatz_params||_2` — one training step on val set |

---

## 4. Figures to Generate

| Figure | Content |
|--------|---------|
| `dv_measurement_entropy_by_class.svg` | Overlapping histograms of measurement entropy for class 0 vs 1 |
| `cv_homodyne_class_separation.svg` | Box plots of X_mode0 by class for all CV-QNN variants |
| `cv_displacement_phase_space.svg` | 2D scatter (X_mode0, P_mode0) colored by class |

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q72/results/dv_qnn_quantum_metrics.csv` | DV metrics table |
| `workspace/experiments/Q72/results/cv_qnn_quantum_metrics.csv` | CV metrics table |
| `workspace/experiments/Q72/results/dv_class_separation.json` | DV state separation stats |
| `workspace/experiments/Q72/results/cv_class_separation.json` | CV displacement stats |
| `workspace/experiments/Q72/figures/*.svg` | 3 figures |
| `workspace/experiments/Q72/reports/q72_quantum_metrics_extraction_report.md` | |
| `reports/q72_quantum_metrics_extraction.md` | |

---

## 6. Pass Criteria

- [ ] class_conditioned_fidelity_proxy computed for DV-QNN (hybrid and pure)
- [ ] mean_displacement_separation computed for CV-QNN (hybrid and pure)
- [ ] homodyne_X_snr computed for all CV variants
- [ ] measurement_entropy per class computed for DV variants
- [ ] squeezing_post_training extracted from Q62A checkpoints
- [ ] gradient_norm_theta logged for DV; gradient_norm_ansatz for CV
- [ ] All 3 figures generated
- [ ] No re-training
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
