# Q74S — Hybrid vs Pure Smoke Comparison

**Slice ID**: Q74S-HYBRID-VS-PURE-SMOKE-COMPARISON  
**Campaign**: pure_quantum_readout_smoke  
**Status**: BLOCKED  
**Depends on**: Q72S  
**Estimated runtime**: LOW (< 5 min; analysis only)  
**Date planned**: After Q72S completes  
**Gate**: Emits READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN or NOT_READY_WITH_BLOCKERS  

---

## 1. Objective

Compare the smoke-run pure quantum readout results against the full-benchmark hybrid baselines.
Determine whether the pure readout approach is viable and whether the full campaign should proceed.

---

## 2. Inputs

| Artifact | Path | Notes |
|---|---|---|
| Q56 CNN baseline | `workspace/experiments/Q56/results/vindr_metrics.csv` | AUROC 0.9731 |
| Q57 DV-QNN hybrid | `workspace/experiments/Q57/results/vindr_metrics.csv` | AUROC 0.8842 |
| Q58 CV-QNN hybrid | `workspace/experiments/Q58/results/vindr_metrics.csv` | AUROC 0.9534 |
| Q67S DV pure smoke | `workspace/experiments/Q67S/results/q67s_smoke_metrics.csv` | |
| Q68S CV single homodyne smoke | `workspace/experiments/Q68S/results/q68s_smoke_results.json` | |
| Q68DS CV dual homodyne smoke | `workspace/experiments/Q68DS/results/q68ds_smoke_results.json` | |
| Q72S quantum metrics | `workspace/experiments/Q72S/results/q72s_quantum_metrics.csv` | |

---

## 3. Comparison Table

Generate a unified comparison table with the following columns:

`model, campaign, auroc, f1, accuracy, n_trainable_params, n_quantum_params, quantum_classical_ratio, runtime_s, readout_type, dataset, notes`

Models to include:
1. Classical CNN (Q56) — hybrid baseline
2. DV-QNN hybrid (Q57) — hybrid baseline
3. CV-QNN hybrid (Q58) — hybrid baseline
4. DV-QNN parity readout (Q67S) — pure smoke
5. DV-QNN top-k readout (Q67S) — pure smoke
6. DV-QNN expectation value readout (Q67S) — pure smoke
7. CV-QNN single homodyne threshold (Q68S) — pure smoke
8. CV-QNN homodyne difference (Q68S) — pure smoke
9. CV-QNN dual homodyne centroid (Q68DS) — pure smoke
10. CV-QNN dual homodyne linear (Q68DS) — pure smoke

---

## 4. Gate Decision Logic

Emit `READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN` if ALL of:
- [ ] All smoke slices (Q67S, Q68S, Q68DS) passed without hard failures
- [ ] At least 1 DV-QNN pure variant produced AUROC > 0.55 (above random)
- [ ] At least 1 CV-QNN pure variant produced AUROC > 0.55
- [ ] Quantum state logging completed (Q72S)
- [ ] No critical blocking issues identified

Emit `NOT_READY_WITH_BLOCKERS` if ANY of:
- Hard failure in any smoke slice
- All pure readout variants produce AUROC ≤ 0.50 (indistinguishable from random)
- Quantum state logging failed to produce required metrics

Report specific blockers if NOT_READY.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q74S/results/q74s_comparison_table.csv` | Unified comparison table |
| `workspace/experiments/Q74S/reports/q74s_comparison_report.md` | Analysis + gate decision |
| `reports/q74s_hybrid_vs_pure_smoke_comparison.md` | Publication copy |

The report must end with exactly one of:
```
GATE DECISION: READY_FOR_FULL_PURE_QUANTUM_READOUT_CAMPAIGN
```
or
```
GATE DECISION: NOT_READY_WITH_BLOCKERS
BLOCKERS: <list>
```

---

## 6. Pass Criteria

- [ ] Comparison table includes all 10 model variants
- [ ] Delta AUROC (pure - hybrid) computed for DV and CV separately
- [ ] Quantum/classical param ratio documented for all variants
- [ ] Gate decision clearly stated at end of report
- [ ] No training executed
- [ ] No external quantum frameworks
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
