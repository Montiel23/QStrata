# Q68S — CV-QNN Single Homodyne Readout Smoke

**Slice ID**: Q68S-CV-QNN-HOMODYNE-READOUT-SMOKE  
**Campaign**: pure_quantum_readout_smoke  
**Status**: BLOCKED  
**Depends on**: Q66 (READY)  
**Estimated runtime**: LOW (< 2 min for 1 epoch / 64 samples; CV sim is fast)  
**Date planned**: After Q66 completes  

---

## 1. Objective

Smoke-test CV-QNN with single homodyne X-quadrature readout, replacing the classical
`Linear(2→2)` with a threshold-based class assignment on `X_mode0 = mu[0]`.
Log homodyne X-quadrature statistics to assess class separation.

Constraints:
- max 64 train / 32 val / 32 test samples
- 1 seed: 42, 1 epoch, VinDr-SpineXR only
- No classical `nn.Linear(2→2)` readout
- Custom qcore framework only

---

## 2. Architecture

### Shared components

| Component | Type | Params | Notes |
|-----------|------|--------|-------|
| Encoder Linear(128→4) | Classical, trainable | 516 | Maps embedding to 2*n_modes phase-space inputs |
| GaussianVariationalAnsatz (n_modes=2, depth=1) | Quantum, trainable | 10 | D+S+R+BS ring |
| GaussianBackend | Quantum | 0 | Symplectic evolution of (mu, cov) |
| Homodyne X measurement | Quantum measurement | 0 | mu[::2] = (X_mode0, X_mode1) |

### Readout Variant A — Single Homodyne Threshold

```python
# X_mode0 = mu_out[0]  (X-quadrature of mode 0 after ansatz)
# score = X_mode0
# threshold = trainable scalar nn.Parameter, init=0.0
# class = 1 if (score - threshold) > 0 else 0
# For AUROC: use sigmoid(score - threshold) or raw score
```

Trainable params: 516 + 10 + 1 (threshold) = 527.  
Report threshold as calibration parameter. Ablate: frozen threshold (=0) vs trained.

### Readout Variant B — Homodyne Difference Threshold

```python
# score = X_mode0 - X_mode1 = mu_out[0] - mu_out[2]
# threshold = trainable scalar, init=0.0
# class = 1 if (score - threshold) > 0 else 0
```

Trainable params: 516 + 10 + 1 = 527.

### Readout Variant C — Positive X-Quadrature Score (no threshold param)

```python
# score = X_mode0  (direct; no threshold)
# class = 1 if score > 0 else 0  (fixed threshold at 0)
# AUROC from raw score values
```

Trainable params: 516 + 10 = 526. Zero classical params after measurement.

---

## 3. Protocol

- Load first 64 train + 32 val samples from Q49 embeddings.
- Train for 1 epoch with Adam (lr=1e-3, wd=1e-4), batch_size=8.
- Use BCEWithLogitsLoss (or CrossEntropyLoss with 2-class; document which).
- Evaluate on 32 val samples.
- Repeat for all 3 variants.

---

## 4. Quantum State Logging (required)

For each sample in val set, log:
- `x_quadrature_mode0`: `mu_out[0]` per sample
- `x_quadrature_mode1`: `mu_out[2]` per sample
- `class_label`: ground truth

Compute and log:
- `mean_X_mode0_class0`, `mean_X_mode0_class1`
- `homodyne_X_separation`: `|mean_X_mode0_class1 - mean_X_mode0_class0|`
- `homodyne_X_snr`: `(mean_X_class1 - mean_X_class0)^2 / (var_X_class1 + var_X_class0)`
- `squeezing_values`: `tanh(squeezing_raw) * squeezing_cap` for each mode

Save as `workspace/experiments/Q68S/results/q68s_homodyne_stats.csv`.

---

## 5. Outputs

| File | Description |
|------|-------------|
| `workspace/experiments/Q68S/results/q68s_smoke_results.json` | Full results dict |
| `workspace/experiments/Q68S/results/q68s_homodyne_stats.csv` | Homodyne statistics per variant |
| `workspace/experiments/Q68S/reports/q68s_cv_qnn_homodyne_readout_smoke_report.md` | Analysis |
| `reports/q68s_cv_qnn_homodyne_readout_smoke.md` | Publication copy |

---

## 6. Pass Criteria

- [ ] All 3 variants run without error
- [ ] AUROC, F1, accuracy logged per variant
- [ ] `n_quantum_params = 10` confirmed
- [ ] No `nn.Linear(2, 2)` readout — only threshold scalar (≤1 param) or no params
- [ ] `homodyne_X_separation` computed and logged
- [ ] `homodyne_X_snr` computed and logged
- [ ] Delta AUROC vs Q58 hybrid (0.9534) documented
- [ ] Runtime < 30 min
---

## Mode

analysis

## Validation Commands

- sliceforge campaign validate --project qstrata
