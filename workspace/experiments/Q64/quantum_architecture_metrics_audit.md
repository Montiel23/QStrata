# Q64 — Quantum Architecture and Metrics Scientific Audit

**Slice ID:** Q64-QUANTUM-ARCHITECTURE-AND-METRICS-AUDIT  
**Date:** 2026-06-03  
**Mode:** Analysis (no training, no benchmark execution, no push)  
**Framework:** Custom in-house `qcore` only — no PennyLane, Qiskit, Strawberry Fields, TorchQuantum  

---

## 1. Executive Summary

The QSTRATA campaign evaluated three model families: a classical CNN baseline (Q56), a discrete-variable quantum neural network (DV-QNN, Q57), and a continuous-variable quantum neural network (CV-QNN, Q58). All models share the same frozen MobileNetV3-Large + frozen linear projection backbone (5,606,040 frozen params) and differ only in their trainable head.

**Critical finding:** Neither the DV-QNN nor the CV-QNN is a pure quantum classifier. Both are **hybrid quantum-classical classifiers** with a classical linear encoder before the quantum block and a classical linear readout after it. Only 24/574 parameters (DV-QNN) and 10/532 parameters (CV-QNN) are purely quantum gate angles. The quantum blocks act as **quantum feature maps**; classification decisions are made by a classical `Linear` layer.

**Performance summary (VinDr-SpineXR, 3-seed mean AUROC, validation set):**

| Model | AUROC | F1 | Accuracy | Trainable Params | Runtime/seed |
|-------|-------|-----|----------|-----------------|--------------|
| Classical CNN (Q56) | 0.9731 | 0.9152 | 0.9175 | 2,250 | 0.57s |
| DV-QNN (Q57) | 0.8842 | 0.7888 | 0.8007 | 574 | 263s |
| CV-QNN (Q58) | **0.9534** | 0.8789 | 0.8823 | 532 | 56s |

CV-QNN is substantially stronger than DV-QNN (+6.9pp AUROC) and only 2pp below the classical baseline, with fewer trainable parameters. Both quantum heads are more parameter-efficient than the classical MLP head (2,250 params) but only the CV-QNN approaches classical performance.

---

## 2. DV-QNN Architecture

### Pipeline

```
Input embeddings (B, 128)
  [pre-computed Q49 MobileNetV3-Large + frozen Linear(960→128)]
        |
        | CLASSICAL: Linear(128 → 4)  [trainable, 516 params]
        |   proj layer; no activation; scales 128-dim embedding to n_qubits=4
        |
        v
  x_i ∈ R^4  (one sample at a time)
        |
        | QUANTUM: medical_ansatz(x_i, theta, n_qubits=4, depth=1, alpha=0.1)
        |   1. H gate on each qubit (superposition initialization)
        |   2. Data reuploading (depth=1):
        |      for q in [0,1,2,3]:
        |        RY(torch.atan(x_i[q]) * alpha, q)
        |        RZ(x_i[q] * alpha, q)
        |   3. Variational layer:
        |      for q in [0,1,2,3]:
        |        RX(theta[0,0,q,0], q)
        |        RY(theta[0,0,q,1], q)
        |        RZ(theta[0,0,q,2], q)
        |   4. Entanglement (ring + cross):
        |      CNOT(q, (q+1)%4) for q in [0,1,2,3]  (ring)
        |      CNOT(q, (q+2)%4) for q in [0,1,2,3]  (cross, n_qubits>2)
        |   Total gates per sample: 4H + 8 encoding + 12 variational + 8 CNOT = 32
        |
        | Backend: circuit → unitary U (16×16 via kronecker products)
        |          |ψ⟩ = U |0000⟩
        |          probs = |ψ|² ∈ R^16  (Born rule measurement)
        |
        v
  probs_i ∈ R^16  (probability distribution over 16 basis states)
        |
        | CLASSICAL: Linear(16 → 2)  [trainable, 34 params]
        |   readout layer; maps probability vector to class logits
        |
        v
  logits (B, 2)  → CrossEntropyLoss
```

### DV-QNN Parameter Accounting

| Component | Type | Params | Trainable |
|-----------|------|--------|-----------|
| MobileNetV3-L backbone | Classical (frozen) | 5,483,032 | No |
| Linear(960→128) projection | Classical (frozen) | 123,008 | No |
| Linear(128→4) proj | **Classical** | 516 | **Yes** |
| theta angles (1×2×4×3) | **Quantum** | 24 | **Yes** |
| quantum circuit + backend | Quantum (no params) | 0 | — |
| Linear(16→2) readout | **Classical** | 34 | **Yes** |
| **DV-QNN head total** | **Hybrid** | **574** | **Yes** |

### DV-QNN Structural Properties

- **Input dimensionality to quantum block:** 4 (after proj)
- **Hilbert space dimension:** 2^4 = 16
- **Quantum gate parameter count:** 24 (theta only)
- **Circuit gates per sample:** 32
- **Measurement:** Born rule — `|ψ|²` produces 16 probabilities
- **Gradient path:** `torch.atan(x_i[q])` preserves autograd; full end-to-end backprop through proj → circuit → readout (Q5 fix)
- **Scalability limit:** kronecker-product embedding is O(4^n) in memory; practical limit ~6 qubits
- **Device:** CPU-only

---

## 3. CV-QNN Architecture

### Pipeline

```
Input embeddings (B, 128)
  [pre-computed Q49 MobileNetV3-Large + frozen Linear(960→128)]
        |
        | CLASSICAL: Linear(128 → 4)  [trainable, 516 params]
        |   encoder layer; maps to 2*n_modes=4 phase-space coordinates
        |   enc_i = encoder(x_i)  ∈ R^4
        |
        v
  Gaussian state: vacuum (mu=0, cov=(hbar/2)*I)  [2n_modes=4 dim]
        |
        | QUANTUM: GaussianVariationalAnsatz.apply(mu, cov, backend, encoded_input=enc_i)
        |   For depth=1:
        |     mu = mu + enc_i  [data reuploading: add encoded input to mean vector]
        |     For each mode i in [0, 1]:
        |       alpha = disp_real[0,i] + j*disp_imag[0,i]
        |       D(alpha): displacement — mu[2i] += Re(alpha)*sqrt(2*hbar)
        |       S(r): squeezing — r = tanh(squeezing_raw[0,i]) * 1.5  [bounded]
        |         apply_symplectic(mu, cov, S_squeeze)
        |       R(phi): rotation — phi = rot_phi[0,i]
        |         apply_symplectic(mu, cov, S_rotate)
        |     For each mode pair (i, (i+1)%n_modes):
        |       R(rot_phi[0,i]): phase realignment rotation
        |       BS(theta): beamsplitter — theta = bs_theta[0,i]
        |         apply_symplectic(mu, cov, S_bs)
        |
        | Measurement: Homodyne X-quadrature
        |   feats = mu[::2]  (even indices = X quadratures) ∈ R^2
        |
        v
  feats_i ∈ R^2  (n_modes X-quadrature means)
        |
        | CLASSICAL: Linear(2 → 2)  [trainable, 6 params]
        |   readout layer; maps homodyne output to class logits
        |
        v
  logits (B, 2)  → CrossEntropyLoss
```

### CV-QNN Parameter Accounting

| Component | Type | Params | Trainable |
|-----------|------|--------|-----------|
| MobileNetV3-L backbone | Classical (frozen) | 5,483,032 | No |
| Linear(960→128) projection | Classical (frozen) | 123,008 | No |
| Linear(128→4) encoder | **Classical** | 516 | **Yes** |
| disp_real (1×2) | **Quantum** | 2 | **Yes** |
| disp_imag (1×2) | **Quantum** | 2 | **Yes** |
| squeezing_raw (1×2) | **Quantum** | 2 | **Yes** |
| bs_theta (1×2) | **Quantum** | 2 | **Yes** |
| rot_phi (1×2) | **Quantum** | 2 | **Yes** |
| GaussianBackend (no params) | Quantum | 0 | — |
| Linear(2→2) readout | **Classical** | 6 | **Yes** |
| **CV-QNN head total** | **Hybrid** | **532** | **Yes** |

### CV-QNN Structural Properties

- **Input dimensionality to quantum block:** 4 (2*n_modes, after encoder)
- **Quantum modes:** 2
- **State representation:** Gaussian (μ ∈ R^4, Σ ∈ R^4×4)
- **Quantum gate parameter count:** 10 (ansatz only)
- **Squeezing cap:** 1.5 (tanh-bounded; prevents unphysical states)
- **Initial squeezing:** max=0.037, mean=0.037 (Q58S); deep in sub-cap regime
- **Measurement:** Homodyne X-quadrature — `mu[::2]` = 2 real values
- **Gradient path:** symplectic matrix ops are PyTorch tensors; full autograd
- **Runtime advantage:** symplectic ops faster than DV circuit compilation (56s vs 263s per seed)
- **Device:** CPU-only

---

## 4. Are These True Quantum Classifiers?

### Classification

| Model | Classification | Reason |
|-------|---------------|--------|
| DV-QNN | **Hybrid quantum-classical / Quantum feature map + classical readout** | Classical proj → quantum circuit → classical readout; only 24/574 params are quantum; final class decision made by Linear(16→2) |
| CV-QNN | **Hybrid quantum-classical / Quantum feature map + classical readout** | Classical encoder → Gaussian circuit → classical readout; only 10/532 params are quantum; final class decision made by Linear(2→2) |

### Detailed Classification Rationale

**DV-QNN** is closest to: **CNN feature extractor + quantum head + classical readout**

- The large frozen CNN (MobileNetV3-L) dominates representational capacity
- The quantum block applies a parameterized unitary transformation to a 4-dim classical embedding
- The 16 output probabilities are fed to a classical linear classifier
- The quantum block functions as a **non-linear feature map** in a 16-dimensional Hilbert space
- This is analogous to a quantum kernel method with trainable encoding, not a standalone quantum classifier

**CV-QNN** is closest to: **CNN feature extractor + quantum feature map + classical readout**

- Same frozen CNN backbone; same frozen projection
- The Gaussian circuit manipulates 2-mode phase-space state via symplectic transformations
- The homodyne measurement extracts 2 real quadrature values
- A classical Linear(2→2) makes the final classification decision
- The Gaussian circuit is an even simpler quantum feature map than the DV circuit (10 quantum params vs 24)

**Neither architecture is a pure quantum classifier.** A pure quantum classifier would:
1. Accept raw or minimally-preprocessed inputs
2. Encode data into a quantum state with no preceding classical MLP
3. Perform measurement whose statistics directly yield class probabilities without a subsequent classical layer

---

## 5. MLP / Classical Component Analysis

### Explicit Component Audit

| Component | DV-QNN | CV-QNN |
|-----------|--------|--------|
| Classical layer **before** quantum circuit | **YES** — `Linear(128→4)` | **YES** — `Linear(128→4)` |
| Classical layer **inside** quantum circuit | No | No |
| Classical layer **after** quantum circuit | **YES** — `Linear(16→2)` | **YES** — `Linear(2→2)` |
| Final logits from quantum measurement directly | **No** | **No** |
| Final logits from classical layer | **YES** | **YES** |
| Fraction of head params that are quantum | 24/574 = **4.2%** | 10/532 = **1.9%** |

### Component Role Analysis

**Before quantum block:**
- DV-QNN `proj = Linear(128, 4)`: reduces 128-dim CNN embedding to 4 scalar inputs for the circuit. This is a mandatory classical dimensionality reduction — the circuit can only accept n_qubits=4 inputs.
- CV-QNN `encoder = Linear(128, 4)`: same purpose; outputs 2*n_modes=4 values used as phase-space displacement inputs.
- Both use standard `nn.Linear` with no non-linear activation before the quantum block.

**After quantum block:**
- DV-QNN `readout = Linear(16, 2)`: maps the 16-dim Born-rule probability distribution to 2 class logits. This is equivalent to a linear probe on quantum features — identical to linear probing on classical neural representations.
- CV-QNN `readout = Linear(2, 2)`: maps 2 homodyne X-quadrature values to 2 logits. Extremely small (6 params), almost a threshold — but it is still a classical linear decision boundary.

**Conclusion:** Both models are classical-quantum-classical sandwiches. The quantum component is the middle layer; it does not independently classify. This design is consistent with the variational quantum circuit (VQC) paradigm, but it does not satisfy the requirements for a pure quantum advantage claim.

---

## 6. Quantum Metric Availability Matrix

See: `quantum_metrics_availability_matrix.csv` (this directory)

### Summary by Availability Status

| Status | DV-QNN Metrics | CV-QNN Metrics |
|--------|----------------|----------------|
| **Already computed** | hilbert_space_dim, n_qubits, circuit_depth, n_quantum_layers, n_trainable_quantum_params, compression_ratio, gate_count_per_sample, unitary_matrix_dim | n_modes, circuit_depth, n_gaussian_layers, n_trainable_quantum_params, compression_ratio, squeezing_values, squeezing_cap_utilization |
| **Computable from artifacts** | — | shot_noise_quadrature_variance (from Sigma if logged) |
| **Computable by re-running inference** | state_purity, fidelity_class_conditioned, trace_distance, measurement_entropy, gradient_norm, expressibility_proxy | fidelity_class_conditioned_gaussian, state_separation, covariance_matrix_diagnostics, mean_displacement_separation, homodyne_statistics, gradient_norm |
| **Not currently supported** | signal_to_noise_ratio | signal_to_noise_ratio, electronic_noise_proxy |
| **Not applicable** | — | state_purity (always pure for unitary Gaussian; Tr(rho^2)=1 for vacuum-initialized circuits) |

**Key observation:** The most scientifically valuable quantum metrics — state separation, class-conditioned fidelity, measurement entropy, gradient norms — are all **computable by re-running inference with additional logging**. None requires re-training.

---

## 7. Current Quantum Metrics Extracted

### Performance Metrics (from Q57, Q58, Q56 result CSVs)

| Dataset | Model | Mean AUROC | Std | CI95 | Mean F1 | Mean Accuracy |
|---------|-------|-----------|-----|------|---------|---------------|
| VinDr-SpineXR | CNN (Q56) | 0.9731 | 0.0018 | [0.9686, 0.9775] | 0.9152 | 0.9175 |
| VinDr-SpineXR | DV-QNN (Q57) | 0.8842 | 0.0008 | [0.8822, 0.8863] | 0.7888 | 0.8007 |
| VinDr-SpineXR | CV-QNN (Q58) | 0.9534 | 0.0010 | [0.9510, 0.9558] | 0.8789 | 0.8823 |
| BUU-LSPINE | CNN (Q56) | 1.0000 | 0.0000 | — | — | — |
| BUU-LSPINE | DV-QNN (Q57) | 1.0000 | 0.0000 | — | — | — |
| BUU-LSPINE | CV-QNN (Q58) | 1.0000 | 0.0000 | — | — | — |

### Circuit / Architecture Metrics (already computed)

| Metric | DV-QNN | CV-QNN |
|--------|--------|--------|
| n_qubits / n_modes | 4 qubits | 2 modes |
| Hilbert space dim / state space dim | 2^4 = 16 | R^4 (mu) + R^4×4 (Sigma) |
| Circuit depth | 1 | 1 |
| Quantum gate params | 24 | 10 |
| Classical head params | 550 | 522 |
| Total head trainable params | 574 | 532 |
| Compression ratio (CNN→quantum) | 128→4 = 32:1 | 128→4 = 32:1 |
| Runtime per seed (VinDr) | ~263s | ~56s |
| Runtime ratio vs CNN baseline | 263/0.57 ≈ 461× slower | 56/0.57 ≈ 98× slower |

### Squeezing Statistics (from Q58S circuit manifest)

| Metric | Value |
|--------|-------|
| squeezing_cap | 1.5 |
| squeezing_max_observed (init) | 0.03744 |
| squeezing_mean_observed (init) | 0.03719 |
| squeezing_cap_utilization (init) | 2.5% |
| squeezing_bound_check | PASS |

**Note:** Post-training squeezing values are available in Q62A checkpoints (`cv_qnn_seed42.pt`, `cv_qnn_seed7.pt`, `cv_qnn_seed123.pt`) and can be extracted without re-training.

### Parameter Efficiency

| Model | Trainable Head Params | AUROC (VinDr) | Params per AUROC point |
|-------|----------------------|----------------|----------------------|
| CNN (Q56) | 2,250 | 0.9731 | 2,313 |
| DV-QNN (Q57) | 574 | 0.8842 | 649 |
| CV-QNN (Q58) | 532 | 0.9534 | 558 |

CV-QNN achieves 4.2× higher parameter efficiency vs CNN (AUROC per param); DV-QNN achieves 3.6× higher efficiency but at much lower absolute AUROC.

---

## 8. Quantum Advantage Interpretation

### Is There Evidence of Quantum Advantage?

**Short answer: No clear quantum advantage is demonstrated, but there is evidence of quantum-competitive performance with parameter efficiency for CV-QNN.**

#### Evidence Against Quantum Advantage

1. **Classical CNN baseline outperforms both quantum models** on VinDr-SpineXR (0.9731 vs 0.9534 CV, 0.8842 DV). A quantum model must at minimum match classical performance to claim advantage.
2. **The dominant component is classical.** The frozen MobileNetV3-L backbone (5.48M params) does the heavy lifting. The quantum heads operate on 128-dim pre-extracted features, not raw images. Any AUROC gap vs. classical can be attributed to the classical projection layer's effect, not quantum properties per se.
3. **Neither model uses genuinely quantum measurements.** DV-QNN uses Born-rule probabilities fed to a classical linear layer; CV-QNN uses homodyne X-quadrature means fed to a classical linear layer. Both are classically simulable.
4. **The quantum blocks have very few quantum parameters** (24 and 10, respectively) relative to the classical components. The models are 95–98% classical by parameter count.
5. **Runtime is dramatically worse** — CV-QNN is ~98× slower than the classical baseline per seed; DV-QNN is ~461× slower. No runtime advantage exists.

#### Evidence Suggestive of Quantum-Competitive Performance

1. **CV-QNN matches CNN within 2pp AUROC** with 4× fewer trainable parameters. This is parameter efficiency, though attributable in part to the simpler architecture.
2. **CV-QNN significantly outperforms DV-QNN** (+6.9pp AUROC) with fewer parameters and 5× faster runtime. This suggests the Gaussian/photonic paradigm is better suited to the feature structure of MobileNetV3-L embeddings.
3. **Both quantum models achieve perfect AUROC on BUU-LSPINE synthetic data**, matching the classical baseline. On sufficiently separable data, quantum feature maps are competitive.
4. **3-seed CI95 bounds are tight** (±0.002 AUROC), confirming reproducibility and statistical significance of the DV vs. CV performance gap.

#### What Would Make a Quantum Advantage Claim Credible

1. A quantum model that **outperforms the classical baseline** on VinDr-SpineXR under identical conditions.
2. A **pure quantum readout** experiment demonstrating that the quantum measurement alone (without the classical readout layer) gives comparable performance.
3. **Scaling experiments**: demonstrating that increasing circuit depth or modes improves performance in a way not achievable with equivalent classical parameters.
4. **Trainability analysis**: demonstrating that the quantum circuits do not suffer from barren plateaus at the current scale.
5. **Hardware execution** or certified quantum hardware noise benchmarks to distinguish quantum simulation from quantum hardware advantage.

---

## 9. What Is Required for 100% Quantum Experiments

### Proposed Fully-Quantum Experimental Design

To test genuinely quantum classifiers, the following modifications are required:

#### 9.1 Remove Classical Readout Layer

**Current:** quantum measurement → Linear(n→2) → logits  
**Proposed:** quantum measurement → direct class assignment via Born-rule argmax or threshold

For DV-QNN: split 16 basis states into class-0 group and class-1 group; classify by total probability mass in each group. No classical layer after measurement.

For CV-QNN: replace Linear(2→2) with a quantum-measurement-defined threshold on the homodyne output difference: `sign(mu[0] - mu[2])` or similar. Alternatively use both X-quadrature outcomes as sufficient statistics without a classical affine transform.

#### 9.2 Minimize or Remove Classical Encoder

**Current:** Linear(128→4) maps CNN embedding to quantum inputs.  
**Option A (minimal encoder):** Replace with fixed normalization + feature selection (no trainable parameters before circuit).  
**Option B (no CNN):** Apply quantum circuit directly to raw pixel features after PCA reduction (CPU-feasible for 4 qubits / 2 modes).  
**Option C (fixed encoder):** Freeze the encoder at random init (same random seed) to isolate quantum circuit training.

#### 9.3 Experimental Ablation Matrix

| Experiment | CNN backbone | Encoder type | Quantum block | Readout | Hypothesis |
|------------|-------------|--------------|---------------|---------|------------|
| E0 (current) | Frozen MobileNetV3-L | Trainable Linear | Trainable quantum | Trainable Linear | Baseline |
| E1 | Frozen MobileNetV3-L | Frozen random | Trainable quantum | Trainable Linear | Isolate quantum block contribution |
| E2 | Frozen MobileNetV3-L | Frozen random | Trainable quantum | **Fixed threshold** | Near-pure quantum readout |
| E3 | Frozen MobileNetV3-L | **None (direct)** | Trainable quantum (PCA-4 input) | Fixed threshold | No classical encoder |
| E4 | **None** | PCA(raw pixels→4) | Trainable quantum | Fixed threshold | Fully quantum on raw features |

#### 9.4 Required Infrastructure Changes

1. `qcore` DV backend: add `_run_and_return_statevector()` method for state separation metrics.
2. `qcore` CV backend: add `get_covariance()` return from `apply()` for covariance diagnostics.
3. Add gradient norm logging to training loops in `run_q57` / `run_q58` scripts.
4. Define `quantum_readout()` function: maps Born-rule probabilities or homodyne values to class logits without a trainable layer.
5. Implement `class_conditioned_state_separation()` metric: collects (μ, Σ) or |ψ⟩ per class and computes fidelity / trace distance.

#### 9.5 Fixed Constraints for Fair Comparison

- Identical train/test/val splits and seeds [42, 7, 123] as Q56/Q57/Q58.
- Same epochs (4) and optimizer (Adam, lr=1e-3, wd=1e-4).
- Same batch size (8).
- Same evaluation metrics: AUROC, F1, accuracy.
- Same datasets: VinDr-SpineXR and BUU-LSPINE.

---

## 10. Recommended Next Slices

### Q65 — QUANTUM-METRICS-EXTRACTION

**Goal:** Re-run inference on saved Q62A checkpoints with additional logging to extract:
- Per-sample and per-class quantum state statistics (DV: measurement entropy; CV: covariance matrix, mean displacement)
- Class-conditioned state separation (fidelity, trace distance proxy)
- Post-training squeezing value distribution from CV-QNN checkpoints
- Gradient norm proxy by running one training epoch with gradient logging

**Constraints:** No re-training; use saved checkpoints from Q62A; no external quantum frameworks.

---

### Q66 — PURE-QUANTUM-READOUT-ABLATION

**Goal:** Remove the classical Linear readout from both DV-QNN and CV-QNN; replace with a measurement-based threshold rule; compare AUROC vs. current hybrid models.

For DV-QNN: split 16 basis states by label-conditioned Born-rule class assignment.  
For CV-QNN: use sign of homodyne difference as classifier.

**Constraints:** Same splits/seeds/epochs as Q57/Q58; report AUROC delta vs. hybrid baseline.

---

### Q67 — HYBRID-VS-PURE-QUANTUM-COMPARISON

**Goal:** Run the full 5-experiment ablation matrix from Section 9.3 (E0–E4) on VinDr-SpineXR; compare AUROC, F1, runtime, and trainable parameter count; generate publication-quality comparison figure.

**Constraints:** No external quantum frameworks; same evaluation protocol as Q56–Q58; report which design achieves best AUROC per trainable parameter ratio.

---

## Appendix A: Codebase Reference

| File | Purpose |
|------|---------|
| `qcore/ansatz/medical_ansatz.py` | DV-QNN circuit definition |
| `qcore/ansatz/cv_spine_ansatz.py` | CV-QNN Gaussian ansatz |
| `qcore/backends/base.py` | DV unitary compiler and runner |
| `qcore/backends/cvBackend.py` | CV Gaussian symplectic backend |
| `qcore/physics/symplectic.py` | Symplectic gate matrices |
| `qcore/models/dv_hybrid_cnn_qnn.py` | DVHybridCNNQNN model class |
| `scripts/run_q57_dv_qnn_benchmark.py` | DV-QNN full benchmark |
| `scripts/run_q58_cv_qnn_benchmark.py` | CV-QNN full benchmark |
| `workspace/experiments/Q57/results/vindr_metrics.csv` | DV-QNN per-seed VinDr results |
| `workspace/experiments/Q58/results/vindr_metrics.csv` | CV-QNN per-seed VinDr results |
| `workspace/experiments/Q58S/results/q58s_smoke_results.json` | CV circuit manifest + squeezing stats |
| `workspace/experiments/Q62A/checkpoints/` | Saved model checkpoints (all 3 models × 3 seeds) |
