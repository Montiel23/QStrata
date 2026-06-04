# Q55 — Dataset Sample Visualization Pipeline

**Slice ID**: Q55-DATASET-SAMPLE-VISUALIZATION-PIPELINE  
**Campaign**: binary_classical_publication_package  
**Status**: READY  
**Depends on**: (none — first slice in campaign)  
**Estimated runtime**: LOW (5–15 min)  
**Date planned**: 2026-06-02  

---

## 1. Objective

Produce publication-ready dataset visualizations for the binary classification paper:

1. Class distribution bar charts for train/val/test splits
2. Sample image grid with CLAHE preprocessing applied (Any Pathology vs No Finding)
3. UMAP and t-SNE projections of the 128-dim MobileNetV3-Large embeddings from Q49

---

## 2. Inputs

| Artifact | Path | Notes |
|---|---|---|
| Train embeddings | `workspace/experiments/Q49/embeddings/train_embeddings.npy` | (6712, 128) float32 |
| Train labels | `workspace/experiments/Q49/embeddings/train_labels.npy` | (6712,) int64 |
| Val embeddings | `workspace/experiments/Q49/embeddings/val_embeddings.npy` | (1677, 128) float32 |
| Val labels | `workspace/experiments/Q49/embeddings/val_labels.npy` | (1677,) int64 |
| Test embeddings | `workspace/experiments/Q49/embeddings/test_embeddings.npy` | (2077, 128) float32 |
| Test labels | `workspace/experiments/Q49/embeddings/test_labels.npy` | (2077,) int64 |
| Raw image dataset | `data/processed/vindr_binary_roi_224/` | 224x224 grayscale ROI crops |

---

## 3. Tasks

### 3.1 Class Distribution Figures

- Bar chart: class counts per split (train/val/test)
- Colors: class 0 (No Finding) = blue, class 1 (Any Pathology) = red
- Annotation: positive rate (%) on each bar
- Figure size: 8×4 inches, 300 DPI

Expected values:

| Split | N | Class 0 | Class 1 | pos% |
|---|---|---|---|---|
| train | 6712 | 3408 | 3304 | 49.2% |
| val | 1677 | 852 | 825 | 49.2% |
| test | 2077 | 1070 | 1007 | 48.5% |

### 3.2 Sample Image Grid

- 4×4 grid: 8 No Finding + 8 Any Pathology samples, randomly sampled (seed=42)
- All images with CLAHE preprocessing applied (clip=3.0, tile=4×4)
- Label overlay: white text top-left corner
- Figure size: 12×12 inches, 300 DPI

### 3.3 Embedding Space Projections

Compute UMAP and t-SNE on the train embeddings, colored by class label:

**UMAP (train):**
- `umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)`
- Color: class 0 = blue (alpha=0.3), class 1 = red (alpha=0.3)
- Figure size: 8×8 inches, 300 DPI

**t-SNE (train):**
- `sklearn.manifold.TSNE(n_components=2, perplexity=50, random_state=42, n_iter=1000)`
- Same color scheme

**UMAP (test — held-out projection):**
- Fit on train, transform test embeddings
- Same color scheme as train

---

## 4. Outputs

| File | Description |
|---|---|
| `workspace/experiments/Q55/figures/class_distribution.png` | Class counts per split |
| `workspace/experiments/Q55/figures/sample_grid_clahe.png` | 4×4 sample image grid with CLAHE |
| `workspace/experiments/Q55/figures/embedding_umap_train.png` | UMAP train embeddings |
| `workspace/experiments/Q55/figures/embedding_umap_test.png` | UMAP test embeddings (held-out) |
| `workspace/experiments/Q55/figures/embedding_tsne_train.png` | t-SNE train embeddings |
| `workspace/experiments/Q55/reports/q55_dataset_visualization_report.md` | Summary report |

---

## 5. Environment

- Container: `docker-qstrata-gpu-1` (or CPU container — no GPU needed for visualization)
- Key packages: `numpy`, `matplotlib`, `umap-learn`, `scikit-learn`, `Pillow`, `opencv-python`
- No GPU required

---

## 6. Pass Criteria

- [ ] All 5 figures produced at 300 DPI
- [ ] Class distribution values match Q49 embedding export report (train 49.2%, test 48.5%)
- [ ] UMAP and t-SNE show distinguishable cluster structure (qualitative visual check)
- [ ] Report written with figure paths and brief interpretation
- [ ] No source code modified
- [ ] No git commit made
