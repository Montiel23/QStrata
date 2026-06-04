# Q60 — Paper Tables and Figures Export Package

**Slice ID**: Q60-PAPER-TABLES-FIGURES-EXPORT-PACKAGE  
**Campaign**: binary_classical_publication_package  
**Status**: BLOCKED  
**Depends on**: Q59 (BLOCKED on Q58)  
**Estimated runtime**: LOW (5–15 min)  
**Date planned**: After Q59 completes  

---

## 1. Objective

Export all publication-ready tables (LaTeX format) and figures (PDF/PNG at 300 DPI) from Q55–Q59.
Applies a unified visual style and produces the complete artifact package for manuscript submission.

---

## 2. Inputs

| Source | Artifact |
|---|---|
| Q55 | All 5 figures (PNG) |
| Q56 | Discrete benchmark leaderboard CSV |
| Q57 | Continuous benchmark leaderboard CSV |
| Q58 | Cross-dataset comparison CSV |
| Q59 | CI tables (CSV), ROC/PR figures (PNG) |

---

## 3. Tables to Export (LaTeX)

### Table 1 — Dataset Statistics

```latex
% table1_dataset_statistics.tex
% Columns: Split | N | Class 0 (No Finding) | Class 1 (Any Pathology) | Positive Rate
```

Rows: train, val, test, total. Values from Q49 embedding export report.

### Table 2 — Binary Classical Benchmark Leaderboard

```latex
% table2_binary_classical_leaderboard.tex
% Columns: Rank | Classifier | Family | Test AUROC (mean ± std) | Test F1 | Params | Seeds
```

Rows: top 5 from Q56 + top 5 from Q57 (ordered by test AUROC mean). Include Q46B classical head
as the "trained head" reference row. Highlight best row per family.

### Table 3 — Cross-Dataset Comparison

```latex
% table3_cross_dataset_comparison.tex
% Columns: Model | Dataset | Type | AUROC | F1 | Params | Source
```

Rows: all models from Q58 unified table, grouped by dataset (VinDr, then PneumoniaMNIST).

---

## 4. Figures to Export (PDF + PNG at 300 DPI)

| Figure | Source | Description |
|---|---|---|
| figure1_embedding_umap.pdf | Q55 UMAP train figure | Train embedding space colored by class |
| figure2_roc_curves.pdf | Q59 ROC figure | Multi-model ROC comparison |
| figure3_pr_curves.pdf | Q59 PR figure | Multi-model PR comparison |
| figure4_sample_grid.pdf | Q55 sample grid figure | CLAHE sample images |

---

## 5. Visual Style Requirements

- **Color palette**: Use colorblind-safe palette (Okabe-Ito or viridis-compatible)
  - Primary positive class: orange (#E69F00)
  - Primary negative class: blue (#56B4E9)
  - Reference/baseline: gray (#999999)
  - Winner/highlight: black (#000000)
- **Font**: Sans-serif; axis labels 12pt, tick labels 10pt, legend 10pt
- **Line width**: 2pt for main curves, 1pt for reference curves
- **Marker size**: 6pt
- **Figure DPI**: 300 DPI for PNG; vector PDF preferred
- **Margins**: tight_layout() applied; 0.1 inch padding

---

## 6. LaTeX Requirements

- Tables use `booktabs` package (`\toprule`, `\midrule`, `\bottomrule`)
- Numeric formatting: AUROC/F1 to 4 decimal places; param counts with comma separator
- ± formatting: `$0.9873 \pm 0.0004$`
- Bold best value per column: `\textbf{0.9873}`
- Caption and label included in each `.tex` file

---

## 7. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q60/paper/table1_dataset_statistics.tex` | Dataset statistics table |
| `workspace/experiments/Q60/paper/table2_binary_classical_leaderboard.tex` | Benchmark leaderboard table |
| `workspace/experiments/Q60/paper/table3_cross_dataset_comparison.tex` | Cross-dataset comparison table |
| `workspace/experiments/Q60/paper/figure1_embedding_umap.pdf` | UMAP figure (vector PDF) |
| `workspace/experiments/Q60/paper/figure2_roc_curves.pdf` | ROC curves (vector PDF) |
| `workspace/experiments/Q60/paper/figure3_pr_curves.pdf` | PR curves (vector PDF) |
| `workspace/experiments/Q60/paper/figure4_sample_grid.pdf` | Sample image grid (vector PDF) |
| `workspace/experiments/Q60/reports/q60_paper_export_report.md` | Export manifest and style guide |

---

## 8. Pass Criteria

- [ ] All 3 LaTeX tables produced with booktabs formatting
- [ ] All 4 PDF figures produced (vector, not rasterized)
- [ ] Visual style requirements applied consistently across all figures
- [ ] Export report includes file manifest with checksums (MD5)
- [ ] No source code modified; no git commit made
