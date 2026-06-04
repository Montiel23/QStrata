# Q69 — DV-QNN Pure Readout Full Benchmark

**Slice ID**: Q69-DV-QNN-PURE-READOUT-FULL-BENCHMARK  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q74S (gate: READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN)  
**Estimated runtime**: HIGH (3 seeds × 2 datasets × 4 epochs; ~15–45 min total)  
**Date planned**: After Q74S gate passes  

---

## 1. Objective

Full 3-seed benchmark of the best-performing DV-QNN pure Born-rule readout variant
selected from Q67S smoke results. Run on both VinDr-SpineXR and BUU-LSPINE.
Compare to Q57 DV-QNN hybrid baseline (AUROC 0.8842 on VinDr).

---

## 2. Architecture

Use the pure readout variant that achieved highest smoke AUROC in Q67S.
If tied, prefer in order: Variant C (expectation value) > Variant B (top-k) > Variant A (parity).

Architecture:
- Encoder Linear(128→4) [trainable, 516 params]
- medical_ansatz (4 qubits, depth=1, alpha=0.1)
- theta parameters [trainable, 24 params]
- Selected pure readout (no Linear(16→2))
- Optional calibration: if required, ≤ 2 params; report explicitly

Total trainable: 540–542 params (vs 574 in Q57 hybrid).

---

## 3. Protocol

| Parameter | Value |
|-----------|-------|
| Seeds | [42, 7, 123] |
| Epochs | 4 |
| LR | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Datasets | VinDr-SpineXR, BUU-LSPINE |
| Device | CPU (quantum simulator) |

Train/val splits: same as Q57/Q58 (VinDr: 6712/1677; BUU: 1000/200 per class).
Test evaluation: VinDr-SpineXR test set (2077 samples) — report separately.

---

## 4. Required Outputs per Run

- `y_true`, `y_score`, `y_pred` saved as CSV for every (seed, dataset) combination
- AUROC, AUPRC, accuracy, precision, recall, F1, sensitivity, specificity, PPV, NPV
- Confusion matrix (2×2)
- Runtime (s)
- n_trainable_params, n_quantum_params, readout_type

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q69/results/vindr_metrics.csv` | Per-seed VinDr results |
| `workspace/experiments/Q69/results/buu_lspine_metrics.csv` | Per-seed BUU results |
| `workspace/experiments/Q69/results/metrics.csv` | All datasets combined |
| `workspace/experiments/Q69/results/y_true_y_score.csv` | Predictions for ROC/PR analysis |
| `workspace/experiments/Q69/reports/q69_dv_qnn_pure_readout_benchmark_report.md` | Full report |
| `reports/q69_dv_qnn_pure_readout_full_benchmark.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] 3 seeds × 2 datasets = 6 runs completed
- [ ] CI95 AUROC computed for VinDr-SpineXR
- [ ] Test set evaluation on VinDr completed
- [ ] y_true/y_score/y_pred saved
- [ ] n_quantum_params = 24 confirmed
- [ ] Delta AUROC vs Q57 hybrid documented
- [ ] No external quantum frameworks
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
