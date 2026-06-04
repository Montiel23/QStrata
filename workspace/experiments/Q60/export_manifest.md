# Q60 Export Manifest

Generated: 2026-06-03  
Slice: Q60-PAPER-ARTIFACTS-EXPORT  
Source experiments: Q56 (Classical CNN), Q57 (DV-QNN), Q58 (CV-QNN), Q59 (Statistical Analysis)  
Datasets: VinDr-SpineXR (real DICOM), BUU-LSPINE (synthetic)

---

## Artifacts

| File | Type | Description |
|------|------|-------------|
| `figures/main_comparison_figure.svg` | figure | Five-panel benchmark comparison (AUROC, runtime, generalization gap, parameter counts) |
| `figures/dataset_overview.svg` | figure | Side-by-side dataset statistics for VinDr-SpineXR and BUU-LSPINE |
| `tables/main_results_table.tex` | table | Primary results: AUROC ± std, F1, ΔAUROC vs CNN, runtime across all models and datasets |
| `tables/architecture_table.tex` | table | Architecture details: backbone, head/circuit type, simulator, layers, activation |
| `tables/parameter_table.tex` | table | Trainable/total/frozen parameter counts and AUROC-per-1k-param efficiency |
| `tables/experiment_setup_table.tex` | table | Unified training protocol, dataset settings, quantum circuit hyperparameters |
| `overleaf_bundle/main.tex` | latex-root | Top-level Overleaf document (\input-ing all tables + including all figures) |
| `overleaf_bundle/figures/main_comparison_figure.svg` | figure (bundle copy) | Same as figures/main_comparison_figure.svg |
| `overleaf_bundle/figures/dataset_overview.svg` | figure (bundle copy) | Same as figures/dataset_overview.svg |
| `overleaf_bundle/tables/main_results_table.tex` | table (bundle copy) | Same as tables/main_results_table.tex |
| `overleaf_bundle/tables/architecture_table.tex` | table (bundle copy) | Same as tables/architecture_table.tex |
| `overleaf_bundle/tables/parameter_table.tex` | table (bundle copy) | Same as tables/parameter_table.tex |
| `overleaf_bundle/tables/experiment_setup_table.tex` | table (bundle copy) | Same as tables/experiment_setup_table.tex |
| `scripts/run_q60_paper_artifacts_export.py` | script | Export pipeline script (assembles bundle, writes manifest) |

---

## Key Metrics (frozen from Q56–Q59)

| Model | VinDr AUROC | VinDr F1 | BUU AUROC | Runtime (s) | Trainable params |
|-------|-------------|----------|-----------|-------------|-----------------|
| Classical CNN (Q56) | **0.9731 ± 0.0018** | **0.9153** | 1.0000 | **0.57** | 2,250 |
| DV-QNN (Q57)        | 0.8842 ± 0.0008 | 0.7888 | 1.0000 | 263.50 | 574 |
| CV-QNN (Q58)        | 0.9534 ± 0.0010 | 0.8791 | 1.0000 | 56.20 | **532** |

AUROC CI₉₅: CNN [0.9685, 0.9776] · DV-QNN [0.8822, 0.8863] · CV-QNN [0.9510, 0.9558]

---

## Scientific Highlights

1. **Classical CNN dominates on real data**: 0.9731 VinDr AUROC — highest AUROC, fastest runtime (0.57s), smallest generalization gap (2.7pp).
2. **CV-QNN is the best quantum option**: 0.9534 AUROC (98.0% of CNN) with 76.4% fewer trainable parameters (532 vs. 2,250). 4.7× faster than DV-QNN.
3. **DV-QNN underperforms**: 0.8842 AUROC (8.9pp behind CNN), largest generalization gap (11.6pp), slowest inference (263.5s = 462× CNN).
4. **BUU-LSPINE saturated**: All three models achieve AUROC 1.0000 — synthetic dataset is too easy to discriminate between architectures.
5. **Quantum advantage not yet present**: At this scale (frozen backbone, shallow circuits), CV-QNN approaches but does not surpass classical CNN.

---

## Validation

```bash
test -f workspace/experiments/Q60/export_manifest.md && echo "PASS: Q60 export manifest exists"
```

---

## Checklist

- [x] Main results LaTeX table generated (`tables/main_results_table.tex`)
- [x] Architecture comparison table generated (`tables/architecture_table.tex`)
- [x] Parameter efficiency table generated (`tables/parameter_table.tex`)
- [x] Experiment setup table generated (`tables/experiment_setup_table.tex`)
- [x] SVG figures exported (`figures/main_comparison_figure.svg`, `figures/dataset_overview.svg`)
- [x] Overleaf bundle assembled (`overleaf_bundle/` with main.tex, figures/, tables/)
- [x] Export manifest lists all artifacts (this file)
- [x] No source code modified (Q56–Q59 artifacts untouched)
- [x] No git commit made
