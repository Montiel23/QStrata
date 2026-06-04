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

**Neither architecture is a pure quantum classifier.** Both are classical-quantum-classical sandwiches. The quantum component applies a parameterized non-linear transformation between two classical linear layers. The frozen CNN backbone dominates representational capacity.

---

## 5. MLP / Classical Component Analysis

| Component | DV-QNN | CV-QNN |
|-----------|--------|--------|
| Classical layer **before** quantum circuit | **YES** — `Linear(128→4)` | **YES** — `Linear(128→4)` |
| Classical layer **inside** quantum circuit | No | No |
| Classical layer **after** quantum circuit | **YES** — `Linear(16→2)` | **YES** — `Linear(2→2)` |
| Final logits from quantum measurement directly | **No** | **No** |
| Final logits from classical layer | **YES** | **YES** |
| Fraction of head params that are quantum | 24/574 = **4.2%** | 10/532 = **1.9%** |

---

## 6. Quantum Metric Availability Matrix

Full matrix: `workspace/experiments/Q64/quantum_metrics_availability_matrix.csv`

| Status | DV-QNN | CV-QNN |
|--------|--------|--------|
| Already computed | hilbert_space_dim, n_qubits, circuit_depth, n_trainable_quantum_params, compression_ratio, gate_count | n_modes, circuit_depth, n_trainable_quantum_params, compression_ratio, squeezing_values, cap_utilization |
| Computable by re-running inference | state_purity, fidelity, trace_distance, measurement_entropy, gradient_norm, expressibility | fidelity, state_separation, covariance_diagnostics, mean_displacement_separation, homodyne_stats, gradient_norm |
| Not currently supported | SNR (no shot noise) | SNR, electronic_noise_proxy |

---

## 7. Current Quantum Results

### Benchmark Results (3-seed, validation set)

| Dataset | Model | AUROC | F1 | Accuracy | Trainable Params | Runtime/seed |
|---------|-------|-------|-----|----------|-----------------|--------------|
| VinDr-SpineXR | CNN Q56 | 0.9731 ± 0.0018 | 0.9152 | 0.9175 | 2,250 | 0.57s |
| VinDr-SpineXR | DV-QNN Q57 | 0.8842 ± 0.0008 | 0.7888 | 0.8007 | 574 | 263s |
| VinDr-SpineXR | CV-QNN Q58 | 0.9534 ± 0.0010 | 0.8789 | 0.8823 | 532 | 56s |
| BUU-LSPINE | CNN Q56 | 1.0000 | — | — | 2,250 | — |
| BUU-LSPINE | DV-QNN Q57 | 1.0000 | — | — | 574 | — |
| BUU-LSPINE | CV-QNN Q58 | 1.0000 | — | — | 532 | — |

### Circuit Specifications

| Metric | DV-QNN | CV-QNN |
|--------|--------|--------|
| Quantum parameter count | 24 | 10 |
| Hilbert/state space dim | 16 | R^4 (Gaussian) |
| Compression ratio | 32:1 | 32:1 |
| Runtime vs CNN | 461× slower | 98× slower |

---

## 8. Quantum Advantage Interpretation

**No clear quantum advantage is demonstrated.** The classical CNN baseline outperforms both quantum models on VinDr-SpineXR. The quantum blocks add non-linear feature transformation but the dominant representational capacity resides in the frozen MobileNetV3-L backbone.

**Evidence suggestive of quantum-competitive performance:**
- CV-QNN reaches 97.99% of CNN AUROC with 23.7% of CNN trainable parameters.
- CV-QNN significantly outperforms DV-QNN (+6.9pp AUROC), suggesting continuous-variable Gaussian circuits are better matched to the embedding structure.
- 3-seed CI95 bounds are tight, confirming statistical significance.

**What is needed for a credible quantum advantage claim:**
1. A quantum model that outperforms the classical baseline.
2. A pure quantum readout (no classical layer after measurement).
3. Scaling experiments showing that circuit depth/modes improve performance beyond equivalent classical parameters.
4. Gradient/trainability analysis to rule out barren plateau effects.

---

## 9. What Is Required for 100% Quantum Experiments

1. **Remove classical readout:** Replace `Linear(n→2)` with Born-rule argmax (DV) or homodyne threshold (CV). No trainable classical layer after quantum measurement.
2. **Minimize/remove classical encoder:** Use fixed normalization or PCA instead of trainable `Linear(128→4)`. Option: freeze encoder at random init.
3. **Add quantum-specific logging:** state separation metrics, covariance matrix diagnostics, gradient norms per epoch.
4. **Run ablation matrix (E0–E4):** Compare frozen vs. trainable encoder, with vs. without classical readout, with vs. without CNN backbone.
5. **Use identical experimental controls:** same seeds [42,7,123], same splits, same epochs=4, same optimizer.

---

## 10. Recommended Next Slices

- **Q65-QUANTUM-METRICS-EXTRACTION**: Re-run inference on Q62A checkpoints with logging; extract state separation, measurement entropy, post-training squeezing, gradient norms.
- **Q66-PURE-QUANTUM-READOUT-ABLATION**: Remove classical Linear readout from DV-QNN and CV-QNN; replace with measurement-based threshold; compare AUROC vs. hybrid baseline.
- **Q67-HYBRID-VS-PURE-QUANTUM-COMPARISON**: Run full 5-experiment ablation matrix (E0–E4) on VinDr-SpineXR; generate publication-quality comparison figure.

---

*Generated by Q64-QUANTUM-ARCHITECTURE-AND-METRICS-AUDIT. No training executed. No benchmarks executed. No external quantum frameworks used. No push.*
