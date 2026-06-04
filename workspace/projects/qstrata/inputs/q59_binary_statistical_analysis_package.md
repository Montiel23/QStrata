# Q59 — Binary Statistical Analysis Package

**Slice ID**: Q59-BINARY-STATISTICAL-ANALYSIS-PACKAGE  
**Campaign**: binary_classical_publication_package  
**Status**: BLOCKED  
**Depends on**: Q58 (BLOCKED on Q57)  
**Estimated runtime**: LOW (10–20 min)  
**Date planned**: After Q58 completes  

---

## 1. Objective

Produce the full statistical analysis package for the binary classification results, covering:

1. Bootstrap 95% confidence intervals for AUROC and F1 on key models
2. Paired statistical significance tests across seeds
3. Publication-ready ROC curve comparison figure
4. Publication-ready Precision-Recall curve comparison figure

This package feeds directly into Q60 (paper export) and is the statistical backbone of the paper.

---

## 2. Inputs

| Source | Path | Notes |
|---|---|---|
| Q56 leaderboard | `workspace/experiments/Q56/leaderboards/q56_discrete_benchmark_leaderboard.csv` | Discrete classifier results |
| Q57 leaderboard | `workspace/experiments/Q57/leaderboards/q57_continuous_benchmark_leaderboard.csv` | Continuous classifier results |
| Q58 cross-dataset table | `workspace/experiments/Q58/leaderboards/q58_cross_dataset_leaderboard.csv` | Unified comparison |
| Q49 test embeddings | `workspace/experiments/Q49/embeddings/test_embeddings.npy` | For ROC/PR curve computation |
| Q49 test labels | `workspace/experiments/Q49/embeddings/test_labels.npy` | Ground truth |

---

## 3. Statistical Protocol

### 3.1 Bootstrap Confidence Intervals

For each model with ≥ 3 seeds, compute:
- `n_bootstrap = 1000`
- `confidence_level = 0.95`
- Method: percentile bootstrap (not BCa — simpler and sufficient at n_seeds=3)
- Applied to: AUROC and F1

For models with a single seed (Q17, Q21, Q22, Q27 from Phase 1), compute:
- Test-set bootstrap CI by resampling the test predictions 1000× with replacement
- `n_bootstrap = 1000`, `confidence_level = 0.95`

### 3.2 Statistical Significance Tests

Compute pairwise Wilcoxon signed-rank tests (non-parametric; appropriate for n=3 seeds) for
the following pairs:

| Comparison | Hypothesis |
|---|---|
| Q56 winner vs Q57 winner | Continuous > Discrete on AUROC |
| Q56 winner vs Q46B head baseline | Frozen head vs linear probe |
| Q57 winner vs Q46B head baseline | Continuous ML vs trained head |
| Q56 winner vs Q17 baseline | ML probe vs end-to-end classical |

Report: test statistic, p-value (two-tailed), significant (p < 0.05 flag).  
Note: With n=3 seeds, p-values will rarely be significant — report without overclaiming.

### 3.3 ROC Curve Figure

Generate ROC curves for the following models, using mean test-split predictions across seeds:

| Model | Color |
|---|---|
| Q17 Classical CNN | gray (dashed) |
| Q46B MobileNetV3-Large (classical head) | black (solid) |
| Q56 winner (discrete) | blue (solid) |
| Q57 winner (continuous) | orange (solid) |

Layout: 8×8 inches, AUC annotated in legend, random classifier diagonal (gray dashed).

### 3.4 Precision-Recall Curve Figure

Same model set as ROC curves. PR curves are preferred for imbalanced datasets; include for
completeness even though the VinDr-SpineXR binary split is near-balanced (~49% positive).

---

## 4. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q59/tables/auroc_confidence_intervals.csv` | AUROC CIs for all key models |
| `workspace/experiments/Q59/tables/f1_confidence_intervals.csv` | F1 CIs for all key models |
| `workspace/experiments/Q59/figures/roc_curves_comparison.png` | ROC curves, 300 DPI |
| `workspace/experiments/Q59/figures/pr_curves_comparison.png` | PR curves, 300 DPI |
| `workspace/experiments/Q59/reports/q59_statistical_analysis_report.md` | Full statistical narrative |

---

## 5. Confidence Interval Table Format

CSV columns: `model_id, metric, mean, ci95_lo, ci95_hi, n_seeds, ci_method, source_slice`

---

## 6. Pass Criteria

- [ ] All key models have bootstrap CIs computed (AUROC and F1)
- [ ] Wilcoxon tests run for all 4 specified pairs
- [ ] ROC and PR figures at 300 DPI with labeled curves
- [ ] Statistical limitations noted (n=3 seeds; low power)
- [ ] No source code modified; no git commit made
