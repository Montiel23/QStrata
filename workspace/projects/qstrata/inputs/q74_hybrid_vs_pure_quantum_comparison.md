# Q74 — Hybrid vs Pure Quantum Comparison

**Slice ID**: Q74-HYBRID-VS-PURE-QUANTUM-COMPARISON  
**Campaign**: pure_quantum_readout_full  
**Status**: BLOCKED  
**Depends on**: Q73  
**Estimated runtime**: LOW (analysis and figure generation)  

---

## 1. Objective

Aggregate results from all quantum model variants (Q56–Q71) and produce a unified
publication-quality comparison including tables, figures, and LaTeX export.

---

## 2. Models to Compare

| # | Model | Campaign | AUROC source |
|---|-------|----------|-------------|
| 1 | Classical CNN | Q56 | vindr_metrics.csv |
| 2 | DV-QNN hybrid | Q57 | vindr_metrics.csv |
| 3 | CV-QNN hybrid | Q58 | vindr_metrics.csv |
| 4 | DV-QNN pure (best variant) | Q69 | vindr_metrics.csv |
| 5 | CV-QNN single homodyne | Q70 | vindr_metrics.csv |
| 6 | CV-QNN dual homodyne | Q71 | vindr_metrics.csv |

---

## 3. Required Figures

| Figure | Description |
|--------|-------------|
| `auroc_vs_quantum_param_ratio.svg` | Scatter: x=quantum/classical param ratio, y=AUROC; labeled by model |
| `hybrid_vs_pure_bar_comparison.svg` | Bar chart: AUROC for all 6 models on VinDr-SpineXR (with CI95 error bars) |
| `roc_curves_all_models.svg` | ROC curves for all 6 models on VinDr test set |

---

## 4. Required LaTeX Table

`comparison_table.tex`:

```
\begin{table}
Model | AUROC | F1 | Acc | Params | Q-Params | Q/C ratio | Runtime
```

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q74/results/comparison_table.csv` | |
| `workspace/experiments/Q74/figures/auroc_vs_quantum_param_ratio.svg` | |
| `workspace/experiments/Q74/figures/hybrid_vs_pure_bar_comparison.svg` | |
| `workspace/experiments/Q74/figures/roc_curves_all_models.svg` | |
| `workspace/experiments/Q74/tables/comparison_table.tex` | |
| `workspace/experiments/Q74/reports/q74_hybrid_vs_pure_quantum_comparison_report.md` | |
| `reports/q74_hybrid_vs_pure_quantum_comparison.md` | |

---

## 6. Pass Criteria

- [ ] All 6 models in comparison table
- [ ] CI95 AUROC from 3-seed benchmarks
- [ ] quantum_param_ratio documented for all models
- [ ] All 3 figures + LaTeX table generated
- [ ] Quantum advantage assessment stated explicitly
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
