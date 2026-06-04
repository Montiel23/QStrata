# Q71 — CV-QNN Dual Homodyne Readout Full Benchmark

**Slice ID**: Q71-CV-QNN-DUAL-HOMODYNE-READOUT-FULL-BENCHMARK  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q74S (gate: READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN)  
**Estimated runtime**: HIGH (same as Q70)  

---

## 1. Objective

Full 3-seed benchmark of CV-QNN dual homodyne readout using `(X_mode0, X_mode1)`.
Compare to Q70 single homodyne to test whether using both modes improves classification.

---

## 2. Architecture

Use the dual homodyne variant that achieved highest smoke AUROC in Q68DS.

Architecture:
- Encoder Linear(128→4) [trainable, 516 params]
- GaussianVariationalAnsatz (n_modes=2, depth=1, squeezing_cap=1.5) [trainable, 10 params]
- Dual homodyne measurement: (X_mode0, X_mode1) = mu[::2]
- Centroid readout (4 params) or fixed linear combination (0 params); report which
- No Linear(2→2)

Total trainable: 526–530 params.

---

## 3. Protocol

Same as Q70. Seeds [42,7,123], epochs=4.

---

## 4. Required Logging

Per-sample `(X_mode0, X_mode1, class_label)` → save to `dual_homodyne_stats.csv`.
Figure: 2D scatter `(X_mode0, X_mode1)` colored by class — `dual_homodyne_class_scatter.svg`.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q71/results/vindr_metrics.csv` | |
| `workspace/experiments/Q71/results/buu_lspine_metrics.csv` | |
| `workspace/experiments/Q71/results/metrics.csv` | |
| `workspace/experiments/Q71/results/y_true_y_score.csv` | |
| `workspace/experiments/Q71/results/dual_homodyne_stats.csv` | |
| `workspace/experiments/Q71/figures/dual_homodyne_class_scatter.svg` | |
| `workspace/experiments/Q71/reports/q71_cv_qnn_dual_homodyne_readout_benchmark_report.md` | |
| `reports/q71_cv_qnn_dual_homodyne_readout_full_benchmark.md` | |

---

## 6. Pass Criteria

- [ ] 3 seeds × 2 datasets completed
- [ ] CI95 AUROC computed
- [ ] Test set evaluation completed
- [ ] y_true/y_score/y_pred saved
- [ ] dual_homodyne_separation computed
- [ ] 2D scatter SVG generated
- [ ] Delta AUROC vs Q70 single homodyne documented
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
