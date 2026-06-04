# Q59 Statistical Analysis Report
## Classical CNN vs DV-QNN vs CV-QNN — Three-Way Binary Classification Comparison

**Slice ID:** Q59-CLASSICAL-VS-DVQNN-VS-CVQNN-STATISTICAL-ANALYSIS  
**Date:** 2026-06-03  
**Source experiments:** Q56 (Classical CNN), Q57 (DV-QNN), Q58 (CV-QNN)  
**Seeds:** [42, 7, 123] — all experiments use identical seeds, datasets, and backbone  
**Datasets:** VinDr-SpineXR (real DICOM, 6,712 train / 1,677 val), BUU-LSPINE (synthetic, 1,000 train / 200 val)  
**CI95 method:** Student t-distribution, n=3, t-critical=4.303 (two-tailed 95%)

---

## 1. Architecture Summary

| Component | Classical CNN (Q56) | DV-QNN (Q57) | CV-QNN (Q58) |
|-----------|--------------------|--------------| -------------|
| Backbone | MobileNetV3-Large (frozen) | MobileNetV3-Large (frozen) | MobileNetV3-Large (frozen) |
| Projection | Linear(960→128, frozen) | Linear(960→128, frozen) | Linear(960→128, frozen) |
| Head type | Q34A MLP (128→16→8→2) | DV circuit: Linear(128→4) + 4-qubit ansatz + Linear(16→2) | CV circuit: Linear(128→4) + 2-mode Gaussian ansatz + Linear(2→2) |
| Quantum framework | None | Custom qcore DV simulator | Custom qcore Gaussian CV simulator |
| Trainable params | **2,250** | **574** | **532** |
| Total params | 5,608,290 | 5,606,614 | 5,606,572 |
| Backbone params | 5,483,032 | 5,483,032 | 5,483,032 |
| Circuit qubits/modes | — | 4 qubits, depth=1 | 2 modes, depth=1 |

---

## 2. VinDr-SpineXR Results (Real DICOM Data)

| Architecture | Seed 42 AUROC | Seed 7 AUROC | Seed 123 AUROC | Mean AUROC | Std | CI95 AUROC | Mean F1 | CI95 F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| Classical CNN | 0.97504 | 0.97269 | 0.97146 | **0.97306** | 0.001804 | [0.9685, 0.9776] | **0.91530** | [0.9009, 0.9297] | **0.57** |
| DV-QNN | 0.88331 | 0.88443 | 0.88491 | 0.88421 | 0.000804 | [0.8822, 0.8863] | 0.78878 | [0.7663, 0.8113] | 263.50 |
| CV-QNN | 0.95244 | 0.95438 | 0.95335 | 0.95339 | 0.000971 | [0.9510, 0.9558] | 0.87909 | [0.8635, 0.8947] | 56.20 |

**Winner: Classical CNN** — highest AUROC (0.9731) and F1 (0.9153) at lowest runtime.

### VinDr Pairwise AUROC Gaps
- CNN − DV-QNN: **+0.08885** (CNN outperforms by 8.9pp)
- CNN − CV-QNN: **+0.01967** (CNN outperforms by 2.0pp)
- CV-QNN − DV-QNN: **+0.06918** (CV-QNN outperforms DV-QNN by 6.9pp)

---

## 3. BUU-LSPINE Results (Synthetic Procedural Data)

| Architecture | Seed 42 AUROC | Seed 7 AUROC | Seed 123 AUROC | Mean AUROC | Std | CI95 AUROC | Mean F1 | CI95 F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| Classical CNN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0 | [1.0, 1.0] | **1.0000** | [1.0, 1.0] | **0.08** |
| DV-QNN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0 | [1.0, 1.0] | 0.98983 | [0.9560, 1.0235] | 39.41 |
| CV-QNN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.0 | [1.0, 1.0] | **1.0000** | [1.0, 1.0] | 8.49 |

**Winner: Three-way tie on AUROC.** All architectures achieve perfect discrimination on the synthetic dataset. Classical CNN is fastest; CV-QNN F1=1.0 matches CNN (DV-QNN F1=0.9898 slightly lower).

---

## 4. Cross-Dataset Generalization Analysis

The generalization gap is defined as `AUROC(BUU) − AUROC(VinDr)`. A larger positive gap indicates greater overfitting to the simpler synthetic dataset, i.e., weaker generalization to real data.

| Architecture | VinDr AUROC | BUU AUROC | Gen Gap (↓ better) | Interpretation |
|---|---|---|---|---|
| Classical CNN | 0.97306 | 1.0000 | **+0.02694** | Excellent; small synthetic ceiling effect |
| DV-QNN | 0.88421 | 1.0000 | +0.11579 | Large gap — DV circuit struggles on real data complexity |
| CV-QNN | 0.95339 | 1.0000 | +0.04661 | Moderate gap; CV-QNN generalizes well relative to DV |

**Key finding:** CNN has the smallest generalization gap (2.7pp), confirming it is the most robust architecture across data domains. DV-QNN shows the largest gap (11.6pp), indicating the discrete qubit circuit is sensitive to real-world noise and class imbalance in VinDr. CV-QNN's gap (4.7pp) is nearly half that of DV-QNN, suggesting continuous-variable Gaussian simulation adapts better to real medical imaging features.

---

## 5. Parameter Efficiency Analysis

| Architecture | Trainable Params | VinDr AUROC | AUROC per Param (×10⁻⁵) | Runtime/Param (ms) |
|---|---|---|---|---|
| Classical CNN | 2,250 | 0.97306 | 432.5 | 0.25 |
| DV-QNN | 574 | 0.88421 | 1540.4 | 459.1 |
| CV-QNN | 532 | 0.95339 | 1792.1 | 105.6 |

While quantum heads use far fewer trainable parameters (74–76% fewer than CNN), the AUROC per parameter metric does not compensate for lower absolute performance on real data. CV-QNN achieves 98.0% of CNN AUROC with only 23.6% of CNN's trainable parameter count — the most favorable quantum tradeoff.

---

## 6. Runtime Comparison

| Architecture | VinDr Runtime | BUU Runtime | VinDr × CNN | BUU × CNN |
|---|---|---|---|---|
| Classical CNN | 0.57 s | 0.08 s | 1.0× | 1.0× |
| DV-QNN | 263.50 s | 39.41 s | 462× slower | 493× slower |
| CV-QNN | 56.20 s | 8.49 s | 99× slower | 106× slower |
| **CV-QNN vs DV-QNN** | | | **4.7× faster** | **4.6× faster** |

CV-QNN is approximately 4.7× faster than DV-QNN on VinDr inference due to Gaussian symplectic simulation scaling better than state-vector simulation at small mode counts. Both remain far slower than the classical MLP head.

---

## 7. Reference Baseline Comparison (vs Q46 Winner)

Q46 winner (MobileNetV3-Large full pipeline, VinDr AUROC): **0.9873**

| Architecture | VinDr AUROC | Δ vs Q46 Winner |
|---|---|---|
| Classical CNN (Q56) | 0.97306 | −0.01427 (−1.4pp) |
| CV-QNN (Q58) | 0.95339 | −0.03394 (−3.4pp) |
| DV-QNN (Q57) | 0.88421 | −0.10312 (−10.3pp) |

Classical CNN closes 98.6% of the gap to the Q46 full-pipeline winner. CV-QNN closes 96.6%. DV-QNN closes only 89.7%.

---

## 8. Overall Winner and Justification

**Overall winner: Classical CNN (Q56)**

Classical CNN achieves the highest AUROC (0.9731), highest F1 (0.9153), lowest cross-dataset generalization gap (+0.027), and fastest runtime (0.57s) on the clinically relevant VinDr-SpineXR real DICOM dataset. It wins on every primary metric.

**Best quantum architecture: CV-QNN (Q58)**

Among quantum approaches, CV-QNN is the clear winner: it achieves 98.0% of CNN AUROC (0.9534 vs 0.9731) with only 532 trainable parameters — 76% fewer than CNN — and runs 4.7× faster than DV-QNN. The Gaussian continuous-variable simulation generalizes better to real-world data than the discrete-variable circuit.

**DV-QNN limitations:**

DV-QNN (0.8842 VinDr AUROC) lags behind both alternatives by a significant margin and has the largest generalization gap (11.6pp). With its 4-qubit depth-1 circuit, the 16-dimensional Hilbert space is insufficient to capture the complexity of real spinal pathology features without deeper circuits or more qubits.

---

## 9. Validation Checklist

- [x] Three-way comparison table complete (Classical CNN vs DV-QNN vs CV-QNN)
- [x] AUROC/F1/CI95 reported for both datasets (VinDr-SpineXR + BUU-LSPINE)
- [x] Cross-dataset generalization gap documented for all three architectures
- [x] Runtime and parameter count table generated
- [x] Comparison bar plot SVG generated (`figures/comparison_barplot.svg`)
- [x] Cross-dataset generalization SVG generated (`figures/cross_dataset_generalization.svg`)
- [x] Winner identified with quantitative justification
- [x] Results files written: `results/auroc_summary.csv`, `results/comparison_table.csv`
- [x] Analysis script written: `scripts/run_q59_statistical_analysis.py`

---

## 10. Output Files

| File | Description |
|------|-------------|
| `results/auroc_summary.csv` | Per-seed AUROC, F1, CI95, runtime, params for all 6 architecture×dataset combinations |
| `results/comparison_table.csv` | Row-per-metric three-way comparison table with winner column |
| `figures/comparison_barplot.svg` | Grouped bar chart: AUROC by dataset and architecture with CI95 whiskers |
| `figures/cross_dataset_generalization.svg` | Side-by-side VinDr vs BUU bars with Δ generalization gap annotations |
| `scripts/run_q59_statistical_analysis.py` | Reproducible analysis script (stdlib-only, no runtime training) |
