# Q66 — Pure Quantum Visualization Package

**Slice ID**: Q66-PURE-QUANTUM-VISUALIZATION-PACKAGE  
**Campaign**: pure_quantum_readout_smoke  
**Status**: READY  
**Depends on**: none  
**Estimated runtime**: LOW (< 10 min)  
**Date planned**: 2026-06-04  

---

## 1. Objective

Generate publication-ready visualization figures for the pure quantum readout study.
Document the dataset pipeline, binary mapping rationale, quantum architecture data flow,
and all proposed readout mechanisms.

No training. No benchmarks. Reads Q49 embeddings and qcore source code only.

---

## 2. Inputs

| Artifact | Path | Notes |
|---|---|---|
| Train embeddings | `workspace/experiments/Q49/embeddings/train_embeddings.npy` | (6712, 128) |
| Train labels | `workspace/experiments/Q49/embeddings/train_labels.npy` | (6712,) |
| Val embeddings | `workspace/experiments/Q49/embeddings/val_embeddings.npy` | (1677, 128) |
| Val labels | `workspace/experiments/Q49/embeddings/val_labels.npy` | (1677,) |
| DV-QNN results | `workspace/experiments/Q57/results/vindr_metrics.csv` | reference |
| CV-QNN results | `workspace/experiments/Q58/results/vindr_metrics.csv` | reference |
| Existing sample grids | `workspace/experiments/Q55/figures/` | may reuse |

---

## 3. Figures to Generate

### 3.1 Dataset Figures

| Figure | Description | Output |
|--------|-------------|--------|
| raw_sample_grid | 4×4 grid of raw VinDr-SpineXR ROI crops (2 rows class=0, 2 rows class=1) | figures/raw_sample_grid.svg |
| processed_roi_grid | Same crops after CLAHE (clip=3.0, tile=4×4) preprocessing | figures/processed_roi_grid.svg |
| class_distribution | Bar chart of class 0 vs class 1 sample counts in train/val/test splits | figures/class_distribution.svg |
| multiclass_to_binary_mapping | Diagram: original VinDr-SpineXR pathology labels → binary Any Pathology/No Finding | figures/multiclass_to_binary_mapping.svg |

### 3.2 Data Flow Figures

| Figure | Description | Output |
|--------|-------------|--------|
| data_flow_pipeline | Full pipeline: raw image → CLAHE → MobileNetV3-L → 960-dim → Linear(960→128) → 128-dim embedding → quantum encoding → measurement → readout | figures/data_flow_pipeline.svg |

### 3.3 Quantum Architecture Figures

| Figure | Description | Output |
|--------|-------------|--------|
| dv_qnn_circuit_diagram | 4-qubit medical_ansatz: H-init → data reuploading (RY/RZ via atan) → variational (RX/RY/RZ) → ring+cross CNOT → Born-rule measurement | figures/dv_qnn_circuit_diagram.svg |
| cv_qnn_circuit_diagram | 2-mode Gaussian circuit: encoder → vacuum → D+S+R+BS ring → homodyne X-quadrature measurement | figures/cv_qnn_circuit_diagram.svg |

### 3.4 Readout Diagrams

| Figure | Description | Output |
|--------|-------------|--------|
| probability_readout_diagram | DV-QNN: 16-dim Born-rule probability vector → parity grouping / top-k assignment / expectation value → P(class=1) score | figures/probability_readout_diagram.svg |
| homodyne_readout_diagram | CV-QNN: (X_mode0, X_mode1) homodyne outputs → threshold on X_mode0 (single) → binary class | figures/homodyne_readout_diagram.svg |
| dual_homodyne_readout_diagram | CV-QNN: (X_mode0, X_mode1) 2D scatter → centroid-based binary class assignment | figures/dual_homodyne_readout_diagram.svg |

---

## 4. Implementation Notes

- Use `matplotlib` with SVG backend for all figures.
- Color scheme: class 0 = `#2196F3` (blue), class 1 = `#FF5722` (orange).
- Save to `workspace/experiments/Q66/figures/` and copy key figures to `reports/`.
- Circuit diagrams may be drawn using matplotlib patches/lines (no external circuit library).
- For raw/processed sample grids: sample 8 images per class from Q49 embedding indices; reconstruct from the raw dataset if available, or use placeholder grayscale tiles if raw images are not accessible from the embedding-only environment.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q66/figures/*.svg` | All 10 figures |
| `workspace/experiments/Q66/reports/q66_visualization_report.md` | Figure captions and generation notes |
| `reports/q66_pure_quantum_visualization_package.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] All 10 SVG figures generated
- [ ] No training executed
- [ ] No external quantum frameworks used
- [ ] Binary mapping rationale documented in report
- [ ] Data flow pipeline figure covers full end-to-end path including quantum readout
- [ ] Both parity and homodyne readout variants diagrammed
---

## Mode

documentation

## Validation Commands

- sliceforge campaign validate --project qstrata
