# Q70 — CV-QNN Single Homodyne Readout Full Benchmark

**Slice ID**: Q70-CV-QNN-HOMODYNE-READOUT-FULL-BENCHMARK  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q74S (gate: READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN)  
**Estimated runtime**: HIGH (3 seeds × 2 datasets × 4 epochs; ~10–20 min total; CV sim faster than DV)  
**Date planned**: After Q74S gate passes  

---

## 1. Objective

Full 3-seed benchmark of CV-QNN with single homodyne readout (best variant from Q68S).
Run on both VinDr-SpineXR and BUU-LSPINE.
Compare to Q58 CV-QNN hybrid (AUROC 0.9534) and Q69 DV-QNN pure readout.

---

## 2. Architecture

Use the single homodyne variant that achieved highest smoke AUROC in Q68S.

Architecture:
- Encoder Linear(128→4) [trainable, 516 params]
- GaussianVariationalAnsatz (n_modes=2, depth=1, squeezing_cap=1.5)
- Ansatz params [trainable, 10 params]
- Homodyne X measurement: mu[::2] = (X_mode0, X_mode1)
- Single threshold on X_mode0 [trainable, 1 param max] or fixed at 0
- No Linear(2→2) classical readout

Total trainable: 526–527 params (vs 532 in Q58 hybrid).

---

## 3. Protocol

Same as Q69 — seeds [42,7,123], epochs=4, lr=1e-3, wd=1e-4, batch_size=8.

---

## 4. Quantum State Logging (required for all 3 seeds)

For each (seed, dataset), log from val set:
- Per-sample `(X_mode0, X_mode1, class_label)` → save to `homodyne_stats.csv`
- Class-conditioned homodyne SNR: `(mean_X_c1 - mean_X_c0)^2 / (var_X_c1 + var_X_c0)`
- Post-training squeezing values: `tanh(squeezing_raw) * squeezing_cap` per mode

Generate figure: `homodyne_class_separation.svg` — overlapping histograms of X_mode0 by class.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q70/results/vindr_metrics.csv` | Per-seed VinDr results |
| `workspace/experiments/Q70/results/buu_lspine_metrics.csv` | Per-seed BUU results |
| `workspace/experiments/Q70/results/metrics.csv` | Combined |
| `workspace/experiments/Q70/results/y_true_y_score.csv` | Predictions |
| `workspace/experiments/Q70/results/homodyne_stats.csv` | Per-sample X-quadrature + class |
| `workspace/experiments/Q70/figures/homodyne_class_separation.svg` | Histogram by class |
| `workspace/experiments/Q70/reports/q70_cv_qnn_homodyne_readout_benchmark_report.md` | |
| `reports/q70_cv_qnn_homodyne_readout_full_benchmark.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] 3 seeds × 2 datasets = 6 runs completed
- [ ] CI95 AUROC computed for VinDr
- [ ] Test set evaluation completed
- [ ] y_true/y_score/y_pred saved
- [ ] homodyne_X_snr computed and logged
- [ ] homodyne_class_separation SVG generated
- [ ] Delta AUROC vs Q58 hybrid and Q69 DV pure documented
- [ ] n_quantum_params = 10 confirmed
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
