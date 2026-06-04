# Q68DS — CV-QNN Dual Homodyne Readout Smoke

**Slice ID**: Q68DS-CV-QNN-DUAL-HOMODYNE-READOUT-SMOKE  
**Campaign**: pure_quantum_readout_smoke  
**Status**: BLOCKED  
**Depends on**: Q66 (READY)  
**Estimated runtime**: LOW (< 2 min for 1 epoch / 64 samples)  
**Date planned**: After Q66 completes  

---

## 1. Objective

Smoke-test CV-QNN with dual homodyne readout using both X-quadrature outputs
`(X_mode0, X_mode1) = mu[::2]`. Replace `Linear(2→2)` with 2D centroid-based
class assignment. Log 2D scatter statistics to assess dual-quadrature separation.

Constraints:
- max 64 train / 32 val / 32 test samples
- 1 seed: 42, 1 epoch, VinDr-SpineXR only
- No classical `nn.Linear(2→2)` readout
- Custom qcore framework only

---

## 2. Architecture

### Shared components (same as Q68S)

| Component | Type | Params | Notes |
|-----------|------|--------|-------|
| Encoder Linear(128→4) | Classical, trainable | 516 | Maps embedding to phase-space inputs |
| GaussianVariationalAnsatz (n_modes=2, depth=1) | Quantum, trainable | 10 | D+S+R+BS ring |
| GaussianBackend | Quantum | 0 | Symplectic evolution |
| Dual homodyne measurement | Quantum | 0 | (X_mode0, X_mode1) = mu[::2] ∈ R^2 |

### Readout Variant A — Centroid Distance

```python
# features = (X_mode0, X_mode1) = mu_out[::2]  ∈ R^2
# centroid_c0, centroid_c1 = trainable nn.Parameter vectors, shape (2,), init=0
# Initialize centroids from class-conditioned means after first forward pass (1 pass, no grad)
# class = argmin_c ||features - centroid_c||_2
# score = ||features - centroid_c0||_2 - ||features - centroid_c1||_2
# (positive score → class 1 more likely)
```

Trainable params: 516 + 10 + 4 (2 centroids × 2 dims) = 530.
Report centroids as calibration. Ablate: frozen (init-only) vs trained.

### Readout Variant B — Linear Combination Score (minimal)

```python
# score = w[0]*X_mode0 + w[1]*X_mode1 + b
# w = fixed [1.0, -1.0], b = 0.0  (no training)
# class = 1 if score > 0 else 0
```

Zero trainable params after measurement: 516 + 10 = 526 total.

### Readout Variant C — 2D Gaussian Density Ratio

```python
# Fit class-conditioned 2D Gaussian to training set (mu_c, Sigma_c per class)
# score = log N(features; mu_c1, Sigma_c1) - log N(features; mu_c0, Sigma_c0)
# Fit only once before training loop; parameters are not gradient-trained
```

Zero gradient-trained params after measurement: 516 + 10 = 526 total.
Class-conditioned Gaussian params are estimated statistics, not nn.Parameters.

---

## 3. Protocol

- Load first 64 train + 32 val samples from Q49 embeddings.
- Train for 1 epoch with Adam (lr=1e-3, wd=1e-4), batch_size=8.
- Evaluate on 32 val samples.
- Repeat for all 3 variants.

---

## 4. Quantum State Logging (required)

For each val sample, log:
- `x_mode0`: `mu_out[0]`
- `x_mode1`: `mu_out[2]`
- `class_label`

Compute and log:
- `mean_X_mode0_class0`, `mean_X_mode0_class1`
- `mean_X_mode1_class0`, `mean_X_mode1_class1`
- `dual_homodyne_separation`: `||mean_features_class1 - mean_features_class0||_2`
- 2D scatter data: `(x_mode0, x_mode1, class)` for plotting

Save as `workspace/experiments/Q68DS/results/q68ds_dual_homodyne_stats.csv`.
Generate scatter figure: `workspace/experiments/Q68DS/figures/q68ds_dual_homodyne_scatter.svg`.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q68DS/results/q68ds_smoke_results.json` | Full results dict |
| `workspace/experiments/Q68DS/results/q68ds_dual_homodyne_stats.csv` | Dual homodyne statistics |
| `workspace/experiments/Q68DS/figures/q68ds_dual_homodyne_scatter.svg` | 2D scatter by class |
| `workspace/experiments/Q68DS/reports/q68ds_cv_qnn_dual_homodyne_smoke_report.md` | Analysis |
| `reports/q68ds_cv_qnn_dual_homodyne_readout_smoke.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] All 3 variants run without error
- [ ] AUROC, F1, accuracy logged per variant
- [ ] `n_quantum_params = 10` confirmed
- [ ] No `nn.Linear(2,2)` readout
- [ ] `dual_homodyne_separation` computed and logged
- [ ] 2D scatter SVG generated
- [ ] Delta AUROC vs Q58 hybrid (0.9534) and Q68S single homodyne documented
- [ ] Runtime < 30 min
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
