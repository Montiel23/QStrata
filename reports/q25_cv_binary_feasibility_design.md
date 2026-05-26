# Q25: Continuous-Variable Binary Feasibility Design

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

The QStrata research program benchmarks both discrete-variable (DV) and continuous-variable (CV) quantum models for medical image classification. Slices Q17–Q23 completed the VinDr-SpineXR DV binary benchmarking phase, establishing Q21 (DV hybrid pretrained, AUROC 0.6800) as the primary DV benchmark and Q22 (approximate trainable-parameter-matched classical control, AUROC 0.6625) as its control. Full VinDr binary quantum closure requires the CV binary phase (Q25–Q28) to also be executed, compared against the same classical controls, and formally documented before multiclass benchmarking begins.

This document (Q25) defines the minimal CV hybrid binary architecture that enables Q26 smoke testing. It is a design document only. No implementation is produced here. All design decisions are grounded in the existing QStrata CV infrastructure (`qcore/backends/cvBackend.py`, `qcore/ansatz/cv_spine_ansatz.py`, `qcore/physics/symplectic.py`, `qcore/physics/cv_measurement.py`) to ensure the architecture is feasible within the project's framework constraint (QStrata only — no external quantum libraries).

---

## 2. Prior DV Baseline Summary

The following results from the completed DV phase serve as reference points for the CV design:

| Slice | Model | AUROC | F1 | Trainable Params | Notes |
|---|---|---|---|---|---|
| Q21 | DV Hybrid Pretrained | 0.6800 | 0.6159 | 574 | Frozen C006-D040 backbone + 4-qubit variational circuit |
| Q22 | Tiny Classical Control | 0.6625 | 0.5961 | 526 | Frozen C006-D040 backbone + Linear(128→4)→ReLU→Linear(4→2) |

The CV design must:
- Reuse the same frozen C006-D040 backbone and feature extraction path as Q21
- Target a comparable trainable parameter regime (~500–800 params) for scientific comparability
- Use the same binary task (Any Pathology vs No Finding on VinDr-SpineXR)
- Produce binary logits compatible with CrossEntropyLoss

Exact parameter matching to Q21/Q22 is not required for the CV phase. Approximate matching is the goal. Exact count must be printed at Q26 startup.

---

## 3. Design Objective

Design a minimal, stable CV hybrid binary model for VinDr-SpineXR that:

- Reuses the frozen C006-D040 backbone (`checkpoints/c006_d040_classical_anchor.pt`)
- Reuses the exact feature extraction path from Q21: `with torch.no_grad(): features = self.backbone(x)` → (B, 128)
- Replaces the DV quantum head with a compact CV quantum head using the existing QStrata CV infrastructure
- Produces binary logits via a classical linear readout from Gaussian state moments
- Is feasible to smoke-test in Q26 (one batch: forward pass, backward pass, one optimizer step)
- Uses only the QStrata framework — no PennyLane, no Strawberry Fields, no external CV libraries
- Processes one sample at a time through the CV circuit (per-sample loop, as in Q21's DV design)

This is a design document. No implementation code is written here.

---

## 4. QStrata CV Infrastructure Inventory

The following existing QStrata modules are available and will be used directly in Q26/Q27:

| Module | Path | Role in CV design |
|---|---|---|
| `GaussianBackend` | `qcore/backends/cvBackend.py` | Manages Gaussian state (mu, cov), applies symplectic transforms |
| `GaussianVariationalAnsatz` | `qcore/ansatz/cv_spine_ansatz.py` | Spine-specific CV ansatz with trainable D/S/R/BS params and squeezing cap |
| `get_rotation_matrix` | `qcore/physics/symplectic.py` | Symplectic rotation gate R(φ) |
| `get_beamsplitter_matrix` | `qcore/physics/symplectic.py` | Symplectic beamsplitter gate BS(θ) |
| `get_squeezing_matrix` | `qcore/physics/symplectic.py` | Symplectic squeezing gate S(r) |
| `get_displacement_vector` | `qcore/physics/symplectic.py` | Phase-space displacement D(α) |
| `realistic_homodyne_readout` | `qcore/physics/cv_measurement.py` | Homodyne readout with noise — **NOT used** in Q26/Q27 (see Section 9) |

The `GaussianVariationalAnsatz` from `cv_spine_ansatz.py` is the preferred ansatz for this task. It includes bounded squeezing (`squeezing_cap=1.5` via `tanh`), correct data re-uploading via `encoded_input`, and circular entanglement via beamsplitter operations.

---

## 5. Proposed CV Hybrid Architecture

```
Frozen C006-D040 backbone (nn.Sequential[:4])
  → [exact Q21 feature path: with torch.no_grad()]
  → feature vector: (B, 128)

  ── per-sample loop (as in Q21) ──────────────────────────────────

  → feature_compression: nn.Linear(128, 2 × n_modes)   [trainable]
    → compressed: (2 × n_modes,)
    → split into re-upload displacements:
        encoded_input = get_displacement_vector(mode_i, alpha_i)
        where alpha_i = complex(compressed[2i], compressed[2i+1])

  → Gaussian state initialised from vacuum: (mu_0, cov_0)
    → GaussianVariationalAnsatz.apply(mu_0, cov_0, backend, encoded_input)
        per-depth layer: data re-upload → D → S → R (single mode)
                                        → R → BS (two-mode circular)
    → evolved state: (mu_final, cov_final)

  → deterministic first-moment readout:
      readout_vec = mu_final   [2 × n_modes values: ⟨x̂_1⟩, ⟨p̂_1⟩, …]

  → readout_layer: nn.Linear(2 × n_modes, 2)   [trainable]
  → logits: (2,)

  ── end per-sample loop ──────────────────────────────────────────

  → stack logits: (B, 2)
  → CrossEntropyLoss / softmax → binary classification
```

All components after the frozen backbone are trainable. The backbone is frozen via `param.requires_grad = False` and `model.backbone.eval()`, using the same freeze logic as Q21 (including backbone-in-eval override in `train()`). Backbone gradient is asserted zero at every batch.

---

## 6. Feature Compression Layer

The frozen backbone produces a 128-dimensional feature vector. This dimension is confirmed from Q21's feature extraction path — print `features.shape` on the first batch in Q26 as a sanity check before training.

**Proposed compression:**

```python
feature_compression = nn.Linear(128, 2 * n_modes)
# For n_modes=2: nn.Linear(128, 4)
# Output: (4,) per sample
# Interpretation:
#   compressed[0], compressed[1]  →  alpha_real, alpha_imag for mode 0
#   compressed[2], compressed[3]  →  alpha_real, alpha_imag for mode 1
```

The compression output is used to construct the `encoded_input` displacement vector for data re-uploading in the `GaussianVariationalAnsatz`. It is the primary interface between classical image features and the CV head.

**Activation:** None on the compression output before CV encoding. The `GaussianVariationalAnsatz` receives the raw linear outputs as displacement amplitudes. The `tanh` squeezing cap inside the ansatz handles the bounded-parameter constraint independently.

The compression layer is fully classical and fully differentiable. Gradients flow from `readout_layer` → moments → `GaussianVariationalAnsatz` → `encoded_input` → `feature_compression`.

---

## 7. CV Encoding Plan

The compression layer output is used as per-sample data re-uploading displacement. At each circuit depth layer, the `GaussianVariationalAnsatz` adds `encoded_input` to `mu` before applying single-mode gates.

**For n_modes = 2:**

```
nn.Linear(128 → 4)
Output: [c_0, c_1, c_2, c_3]

Encoding:
  Mode 0: alpha_0 = complex(c_0, c_1)   →  encoded_input_0
  Mode 1: alpha_1 = complex(c_2, c_3)   →  encoded_input_1

Ansatz per layer (depth d):
  mu += encoded_input    (data re-upload)
  For mode i:
    D(alpha_i):   mu += displacement_vector(mode_i, alpha_complex_i)
    S(r_i):       mu, cov = apply_symplectic(mu, cov, S(squeezing_raw[d,i]))
    R(phi_i):     mu, cov = apply_symplectic(mu, cov, R(rot_phi[d,i]))
  For pair (m1, m2) in circular topology:
    R(phi_m1):    mu, cov = apply_symplectic(mu, cov, R(rot_phi[d, m1]))
    BS(theta):    mu, cov = apply_symplectic(mu, cov, BS(bs_theta[d, m1]))
```

**Squeezing constraint:** The `cv_spine_ansatz.py` ansatz applies `bounded_squeezing = tanh(squeezing_raw[d,i]) * squeezing_cap` (default cap = 1.5 nats). This prevents squeezing parameter explosion. Do not bypass this bound.

**Displacement and rotation:** Used directly from compression output and ansatz trainable params respectively. No additional clamping required at initialization.

**Key design principle:** The compression layer provides sample-conditioned, per-sample displacements. The ansatz's own trainable parameters (squeezing, beamsplitter, rotation) are global — they do not vary per sample. This is the same pattern as Q21's projection + theta design.

---

## 8. Gaussian Ansatz Plan

Use `GaussianVariationalAnsatz` from `qcore/ansatz/cv_spine_ansatz.py` directly.

| Parameter | Value | Rationale |
|---|---|---|
| n_modes | 2 | Minimal configuration for entanglement; matches smoke test stability requirements |
| depth | 1 | Single layer sufficient to validate full pipeline; revisit in Q27 |
| squeezing_cap | 1.5 | Default from cv_spine_ansatz.py; prevents squeezing explosion |
| Entanglement | Circular beamsplitter | Built into cv_spine_ansatz.py: (m1, m2) = (i, (i+1) % n_modes) |
| Data re-uploading | YES | encoded_input added to mu at each depth layer |

**Gate sequence per depth layer (n_modes=2):**
```
Single-mode gates (for each mode i = 0, 1):
  D(alpha_i)     — displacement from compression output
  S(r_i)         — squeezing with tanh cap
  R(phi_i)       — phase rotation

Two-mode gates (circular: (0,1), (1,0)):
  R(phi_m1)      — pre-rotation for interferometer
  BS(theta_m1)   — beamsplitter mixing
```

**Trainable ansatz parameters (depth=1, n_modes=2):**

| Parameter | Shape | Init | Count |
|---|---|---|---|
| `disp_real` | (1, 2) | N(0, 0.02) | 2 |
| `disp_imag` | (1, 2) | N(0, 0.02) | 2 |
| `squeezing_raw` | (1, 2) | N(0, 0.01) | 2 |
| `bs_theta` | (1, 2) | U(0, 2π) | 2 |
| `rot_phi` | (1, 2) | U(0, 2π) | 2 |
| **Total ansatz** | | | **10** |

This ansatz should be revisited in Q27 based on Q26 smoke test gradient health and latency results. Increasing depth to 2 or n_modes to 4 are the primary tuning levers.

---

## 9. Readout Plan

After the Gaussian ansatz, extract the first moments of the evolved state as the classical signal. Do **not** use `realistic_homodyne_readout` from `cv_measurement.py` — that function introduces stochastic noise and is intended for hardware simulation. For training, deterministic moment readout is required.

**First-moment readout (deterministic):**

```python
# After GaussianVariationalAnsatz.apply(mu_0, cov_0, backend, encoded_input):
mu_final, cov_final = ansatz.apply(mu_0, cov_0, backend, encoded_input)

# First moments — the mean quadrature values:
readout_vec = mu_final    # shape: (2 * n_modes,) = (4,) for n_modes=2
# Elements: [⟨x̂_0⟩, ⟨p̂_0⟩, ⟨x̂_1⟩, ⟨p̂_1⟩]
```

**Optional second-moment augmentation (deferred to Q27):**

```python
# Diagonal of covariance matrix (variances per quadrature):
diag_cov = torch.diag(cov_final)    # shape: (2 * n_modes,)
readout_vec = torch.cat([mu_final, diag_cov], dim=0)
# shape: (4 * n_modes,) for n_modes=2 → (8,) for readout_layer input
```

For Q26 and Q27 initial run: first moments only (readout_dim = 2 * n_modes = 4).

**Readout layer:**

```python
readout_layer = nn.Linear(readout_dim, 2)
# For n_modes=2, first moments only: nn.Linear(4, 2)
```

The readout vector must be checked for finiteness at every forward pass in Q26. NaN or inf in `mu_final` indicates moment explosion and triggers a hard stop.

---

## 10. Trainable Parameter Estimate

All trainable parameters are in the compression layer, the CV ansatz, and the readout layer. The backbone contributes zero trainable parameters.

**For n_modes = 2, depth = 1, first-moment readout (readout_dim = 4):**

```
Compression:  nn.Linear(128, 2 × 2) = nn.Linear(128, 4)
              weights: 128 × 4 = 512
              bias:    4
              subtotal: 516

CV Ansatz:    GaussianVariationalAnsatz(n_modes=2, depth=1)
              disp_real:      1 × 2 = 2
              disp_imag:      1 × 2 = 2
              squeezing_raw:  1 × 2 = 2
              bs_theta:       1 × 2 = 2
              rot_phi:        1 × 2 = 2
              subtotal: 10

Readout:      nn.Linear(4, 2)
              weights: 4 × 2 = 8
              bias:    2
              subtotal: 10

Total trainable: 516 + 10 + 10 = 536
```

The compression layer output directly defines per-sample data re-uploading displacements. The ansatz's own scalar parameters (squeezing, beamsplitter, rotation) are separate trainable scalars required by the QStrata CV backend infrastructure — they are counted above (10 params). Do not count them twice.

| Component | Params | Type |
|---|---|---|
| Compression | 516 | Classical, feature-conditioned per sample |
| CV Ansatz | 10 | Quantum, global scalars (squeezing, BS, rotation) |
| Readout | 10 | Classical, linear map from moments |
| **Total** | **536** | |

**Comparison to DV phase:**

| Model | Trainable Params |
|---|---|
| Q21 DV Hybrid | 574 |
| Q22 Tiny Classical | 526 |
| Q27 CV Hybrid (estimated) | **~536** |

536 is within ±5% of both Q21 and Q22. This supports meaningful cross-model comparison.

**Exact parameter count must be printed at Q26 startup. If it differs from 536, report the actual count and explain the source of deviation.**

---

## 11. Stability and Health Checks

The following checks are required in Q26. All must PASS before Q27 full training begins.

### Forward pass checks

| Check | Description | Hard stop? |
|---|---|---|
| Output shape | `(batch_size, 2)` binary logits | YES — wrong shape → exit |
| Output finiteness | No NaN or inf in logits | YES — NaN/inf → exit |
| Compression finiteness | No NaN or inf in compression output | YES |
| Encoding finiteness | No NaN or inf in encoded_input | YES |
| mu_final finiteness | No NaN or inf in first moments | YES — moment explosion |
| cov_final finiteness | No NaN or inf in covariance diagonal | YES |
| Feature shape | `features.shape[1] == 128` | YES — backbone dimension mismatch |

### Gradient flow checks

| Check | Description | Hard stop? |
|---|---|---|
| Compression grad | Non-zero and finite after backward() | YES — zero grad = no signal |
| Ansatz grad | Non-zero and finite for all 5 param groups | YES |
| Readout grad | Non-zero and finite | YES |
| Backbone grad | Must be None or zero | YES — BACKBONE GRADIENT DETECTED |
| Grad finiteness | No NaN or inf in any grad tensor | YES |

### Optimizer check

| Check | Description | Hard stop? |
|---|---|---|
| Step executes | optimizer.step() completes without error | YES |
| Params change | At least one trainable param changes after step | YES — dead optimizer |

### CV-specific health checks

| Check | Description | Hard stop? |
|---|---|---|
| First moment magnitude | \|mu_final\| < 1e4 for all elements | YES — moment explosion |
| Squeezing bounds | \|squeezing_raw\| < squeezing_cap before tanh | WARN |
| Cov diagonal positive | All diagonal entries of cov_final > 0 | YES — invalid Gaussian state |
| Cov symmetry | \|cov_final − cov_final.T\| < 1e-6 | WARN |
| Vacuum init | mu_0 is all-zeros, cov_0 = (ħ/2) * I | YES — wrong vacuum state |

### Initialization checks

| Check | Description |
|---|---|
| No NaN params | All trainable params initialized without NaN or inf |
| Squeezing near zero | squeezing_raw initialized with σ=0.01 |
| BS theta bounded | bs_theta initialized with U(0, 2π) |
| Compression near zero | nn.Linear default init (Kaiming uniform) |

---

## 12. Metrics

### ML metrics (identical to Q21/Q22)

| Metric | Description |
|---|---|
| train_loss | Mean CrossEntropyLoss per epoch |
| train_acc | Training classification accuracy |
| val_loss | Mean CrossEntropyLoss on validation set |
| val_acc | Validation accuracy |
| val_precision | sklearn precision_score |
| val_recall | sklearn recall_score |
| val_f1 | sklearn f1_score |
| val_auroc | sklearn roc_auc_score |
| val_auprc | sklearn average_precision_score |
| test_loss | Final test loss (analysis only) |
| test_acc, precision, recall, f1, auroc, auprc | Final test metrics |
| confusion_matrix | 2×2 TN/FP/FN/TP |
| trainable_params | Exact count printed at startup |
| latency | ms/sample on CUDA |

### Gradient norm metrics (per epoch, last batch)

| Metric | Description |
|---|---|
| compression_grad_norm | L2 norm of compression layer gradients |
| ansatz_grad_norm | L2 norm of all CV ansatz parameter gradients |
| readout_grad_norm | L2 norm of readout layer gradients |
| total_grad_norm | L2 norm of all trainable parameter gradients |

### CV-specific health metrics (per epoch)

| Metric | Description |
|---|---|
| mu_magnitude_mean | Mean absolute value of all first moment elements |
| mu_magnitude_max | Max absolute value across first moments |
| cov_diag_mean | Mean diagonal value of cov_final (reflects variance magnitude) |
| squeezing_norm | L2 norm of squeezing_raw parameters |
| ansatz_param_norm | L2 norm of all ansatz parameters |
| quadrature_finite | PASS/FAIL: all mu_final and cov_final values finite |

---

## 13. Risks and Caveats

**1. Numerical instability in Gaussian moments.** Squeezing operations amplify one quadrature exponentially (`exp(r)`). If `squeezing_raw` parameters grow, `mu` and `cov` elements can diverge rapidly. The `tanh`-bounded squeezing in `cv_spine_ansatz.py` mitigates this for well-initialized runs, but extreme gradients from the readout can still push squeezing parameters toward saturation. Hard stops on NaN/inf are required; moment magnitude monitoring is required per epoch.

**2. Per-sample loop latency.** The `GaussianVariationalAnsatz.apply()` operates on a single Gaussian state `(mu, cov)` per call. As in Q21's DV per-sample quantum loop, Q27 will process samples sequentially in a for-loop inside the batch forward pass. Latency will be substantially higher than the Q22 GPU-parallel classical baseline. This is a known constraint of the current QStrata CV backend; batch-parallel Gaussian simulation is deferred.

**3. Feature-to-CV mapping is not analytically grounded.** The compression layer maps 128-dimensional CNN features to 4 displacement amplitudes. This is a pragmatic interface choice. There is no theoretical grounding for why displacement parameterization is an appropriate encoding for CNN features. The design is practical rather than principled.

**4. First-moment-only readout may lose information.** The covariance matrix encodes variance and correlation structure of the Gaussian state. Using only `mu_final` as readout discards this information. Second-moment augmentation (appending `diag(cov_final)` to readout) is deferred to Q27 but should be considered if Q26 shows shallow gradients in the readout.

**5. Entanglement in smoke test.** The `cv_spine_ansatz.py` includes circular beamsplitter entanglement even at depth=1. This is more complex than a strictly unentangled smoke test, but it exercises the full intended circuit topology from the start. Gradient flow through beamsplitter operations must be explicitly verified.

**6. No quantum advantage claim.** No result from the CV binary phase will establish quantum advantage. CV benchmarking is a comparative scientific exercise to determine whether continuous-variable circuit geometry provides a different inductive bias relative to DV circuits and classical controls.

**7. Framework constraint.** Only the QStrata framework may be used. The existing `qcore/` CV modules (`cvBackend.py`, `cv_spine_ansatz.py`, `symplectic.py`) are the sole implementation substrate. Do not import PennyLane, Strawberry Fields, or any external CV quantum library.

---

## 14. Q26 Implementation Boundary

### Q26 MUST implement

- Frozen C006-D040 backbone, identical freeze logic to Q21 (`param.requires_grad = False`, `backbone.eval()`, backbone-in-eval override in `train()`)
- Exact Q21 feature extraction path: `with torch.no_grad(): features = self.backbone(x)` → (B, 128)
- `feature_compression`: `nn.Linear(128, 2 * n_modes)` — classical, trainable
- CV encoding: map compression output to `encoded_input` displacement vector per mode
- `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` from `qcore/ansatz/cv_spine_ansatz.py`
- `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` from `qcore/backends/cvBackend.py` — CV circuit is CPU-only
- Deterministic first-moment readout: `readout_vec = mu_final` (no stochastic homodyne noise)
- `readout_layer`: `nn.Linear(2 * n_modes, 2)` — classical, trainable
- All forward-pass, gradient, optimizer, and CV health checks from Section 11
- Trainable parameter count printed at startup
- Feature shape printed on first batch
- One batch only: one forward pass, one backward pass (CrossEntropyLoss), one optimizer step (AdamW, lr=1e-3)
- CUDA availability check (mandatory); backbone runs on CUDA if available; CV circuit runs on CPU (same asymmetry as Q21)

### Q26 MUST NOT implement

- Full training loop (smoke test only — one batch)
- Validation or test evaluation
- Per-epoch metrics or reporting
- Second-moment readout augmentation (defer to Q27)
- Entangling operations beyond the beamsplitter already in `cv_spine_ansatz.py`
- Any hyperparameter search
- `realistic_homodyne_readout` from `cv_measurement.py` (stochastic — not for training)
- Any architecture changes beyond the design specified here
- Any external quantum framework

---

## 15. Next Slice

**Q26 — Continuous-Variable Binary Smoke Test**

Purpose: implement the minimal CV pipeline defined in this document and validate:
- Forward pass executes without error (one batch, batch_size=4)
- All health checks from Section 11 PASS
- Gradient flow confirmed through all three trainable components (compression, ansatz, readout)
- Backbone receives zero gradient throughout
- One optimizer step executes and parameters update
- No NaN or inf at any point

Q26 produces a smoke test report only. No training, no validation, no test evaluation. One batch, all checks, PASS/FAIL verdict.

---

```
CV binary feasibility design status: COMPLETE
Next: Q26 — CV binary smoke test
```
