# Q65 — Pure Quantum Readout and Quantum Metrics Research Plan

**Slice ID:** Q65-PURE-QUANTUM-READOUT-AND-METRICS-RESEARCH-PLAN  
**Date:** 2026-06-04  
**Mode:** Analysis (no training, no benchmark execution, no push)  
**Framework:** Custom in-house `qcore` only  

---

## 1. Current Architecture Diagnosis

### The Core Problem

Both the DV-QNN (Q57) and CV-QNN (Q58) are **classical-quantum-classical sandwiches**. The quantum circuit is a middle layer between two classical `nn.Linear` blocks. Classification decisions are made by the final classical layer, not by quantum measurement outcomes.

| Model | Classical params in head | Quantum params in head | % Quantum |
|-------|--------------------------|------------------------|-----------|
| DV-QNN | 550 (proj 516 + readout 34) | 24 (theta) | **4.2%** |
| CV-QNN | 522 (encoder 516 + readout 6) | 10 (ansatz) | **1.9%** |

The quantum block applies a parameterized transformation, but the resulting measurement output — 16 Born-rule probabilities (DV) or 2 homodyne X-quadrature values (CV) — is immediately fed to a classical linear classifier. The quantum circuit is functioning as a **non-linear feature map**, not as a classifier.

### Model-by-Model Classification

**DV-QNN (Q57):**  
**Classification: CNN feature extractor + quantum feature map + classical readout**

- Frozen MobileNetV3-L extracts features (5.48M params, dominant representational capacity)
- Frozen Linear(960→128) reduces to standardized embedding
- Trainable Linear(128→4) compresses 128-dim embedding to 4 qubit inputs (classical, 516 params = 89.9% of trainable head)
- `medical_ansatz` maps 4 classical values through H-init → data reuploading → variational (RX/RY/RZ) → ring+cross CNOT → unitary U
- Born rule: |ψ⟩ = U|0000⟩ → probs ∈ R^16
- Trainable Linear(16→2) maps probability vector to logits (classical, 34 params)
- CrossEntropyLoss on logits

The circuit is not a quantum classifier. It is a quantum embedding of a 4-dimensional classical vector into a 16-dimensional Hilbert space, followed by linear probing on the probability distribution.

**CV-QNN (Q58):**  
**Classification: CNN feature extractor + Gaussian quantum feature map + classical readout**

- Same frozen backbone as DV-QNN
- Trainable Linear(128→4) encoder maps to 2*n_modes=4 phase-space inputs (classical, 516 params = 97.0% of trainable head)
- GaussianVariationalAnsatz applies D+S+R+BS symplectic transformations to Gaussian state (μ, Σ)
- Homodyne X-quadrature measurement: mu[::2] → 2 real values
- Trainable Linear(2→2) maps homodyne outputs to logits (classical, 6 params)

The 2-dim homodyne output is rich enough to carry class-discriminative information, but the classical Linear(2→2) is doing the actual thresholding. At 6 params and only 2 inputs, this is essentially a learnable threshold — but it is still a classical decision boundary.

### Why This Is Scientifically Insufficient

1. **No quantum advantage claim is possible** when final decisions are classical. Any improvement in AUROC could be attributed to the classical encoder learning a good projection rather than the quantum circuit learning discriminative quantum structure.

2. **Quantum metrics are unmeasured.** State separation, fidelity, measurement entropy, and homodyne SNR — the metrics that would establish quantum advantage — are not extracted by the current pipeline.

3. **The quantum/classical parameter ratio is too low** (4.2% / 1.9%) to draw conclusions about quantum representation. A model dominated by classical parameters cannot demonstrate quantum advantage even if the quantum block is perfect.

4. **The current experiments establish a useful hybrid baseline** but cannot answer the core research question: *Can quantum measurement outcomes directly classify medical imaging data?*

---

## 2. Proposed Corrected Quantum Model Designs

### Design A — DV-QNN Pure Probability Readout

**Goal:** Replace the classical `Linear(16→2)` readout with a Born-rule measurement-based class assignment. No trainable classical layer after the quantum measurement.

#### Architecture

```
Input embedding (128-dim, pre-computed)
    |
    | CLASSICAL ENCODER (allowed as data compression only):
    | Linear(128→4) [trainable, 516 params]
    |
    v
x_i ∈ R^4
    |
    | QUANTUM CIRCUIT: medical_ansatz(x_i, theta, n_qubits=4, depth=1, alpha)
    | H-init → data reuploading (torch.atan) → variational RX/RY/RZ → ring+cross CNOT
    |
    v
|ψ⟩ = U|0000⟩   probs = |ψ|² ∈ R^16
    |
    | PURE QUANTUM READOUT (three variants — no trainable classical layer):
    |
    | Variant A — Parity readout:
    |   P(class=1) = Σ p_i  for i where popcount(i) is odd
    |   P(class=0) = Σ p_i  for i where popcount(i) is even
    |   AUROC computed from P(class=1)
    |
    | Variant B — Top-k probability mass:
    |   Assign 8 highest-probability basis states to class 1, 8 to class 0
    |   (assignment learned on training set in one pass; frozen during eval)
    |   P(class=1) = sum of p_i for states assigned to class 1
    |
    | Variant C — Expectation value:
    |   score = <Z⊗I⊗I⊗I> = Σ_i (-1)^{bit_0(i)} p_i
    |   class = 1 if score > 0 else 0 (or threshold learned as 1 scalar)
    |
    v
Binary prediction + probability score → AUROC/F1/confusion matrix
```

**Optional calibration layer:** If any affine post-processing is unavoidable (e.g., learned threshold), use `Linear(1→1, bias=True)` initialized to `(weight=1.0, bias=0.0)`. This must be explicitly reported and ablated (frozen vs trained).

**Trainable params (pure):** 516 (encoder) + 24 (theta) = 540 total; 0 classical after measurement.  
**Trainable params (with calibration):** 540 + 2 = 542 total; 2 classical after measurement.  
**Quantum fraction:** 24/540 = **4.4%** (up from 4.2% — marginal improvement; the encoder remains classical).

**Ablation plan:**
1. Parity vs top-k vs expectation value readout
2. Encoder frozen (random init) vs trained
3. Calibration layer frozen vs trained
4. alpha sweep: {0.05, 0.1, 0.2, 0.5}
5. depth sweep: {1, 2, 3} (requires `medical_ansatz` depth parameter)

---

### Design B — CV-QNN Single Homodyne Readout

**Goal:** Replace the classical `Linear(2→2)` readout with a homodyne X-quadrature threshold or distance rule. The measurement outcome itself — not a learned linear map of it — drives classification.

#### Architecture

```
Input embedding (128-dim, pre-computed)
    |
    | CLASSICAL ENCODER (allowed as data compression only):
    | Linear(128→4) [trainable, 516 params]
    |
    v
enc_i ∈ R^4
    |
    | QUANTUM CIRCUIT: GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)
    | vacuum (mu=0, cov=(hbar/2)I) → D+S+R+BS ring → (mu_out, cov_out)
    | data reuploading: mu = mu + enc_i at each depth layer
    |
    v
mu_out ∈ R^4,  cov_out ∈ R^4x4
    |
    | HOMODYNE X-QUADRATURE MEASUREMENT:
    | x_q = mu_out[::2] = (X_mode0, X_mode1) ∈ R^2
    |
    | PURE QUANTUM READOUT (three variants — minimal/no trainable params after measurement):
    |
    | Variant A — Single homodyne threshold:
    |   score = x_q[0] = X_mode0
    |   class = 1 if score > threshold else 0
    |   threshold is 1 trainable scalar, init=0  (1 param; must be ablated: frozen vs trained)
    |
    | Variant B — Dual homodyne centroid:
    |   features = (x_q[0], x_q[1]) = (X_mode0, X_mode1)
    |   class = argmin_c ||features - centroid_c||_2
    |   centroids: 2 trainable 2D vectors (4 params), init at class means from one forward pass
    |   (must be ablated: frozen vs trained)
    |
    | Variant C — Homodyne difference threshold:
    |   score = x_q[0] - x_q[1] = X_mode0 - X_mode1
    |   threshold is 1 trainable scalar, init=0  (1 param; must be ablated)
    |
    v
Binary prediction + probability score → AUROC/F1/confusion matrix
```

**Trainable params (pure, no calibration):** 516 (encoder) + 10 (ansatz) = 526.  
**Trainable params (with threshold/centroid):** 526 + 1 to 4 = 527–530.  
**Quantum fraction:** 10/526 = **1.9%** (same quantum fraction; encoder still dominates).

**Key diagnostic to log per run:**
- Per-sample `(X_mode0, X_mode1)` and class labels → homodyne class separation plot
- Class-conditioned mean: `E[X_mode0 | class=0]`, `E[X_mode0 | class=1]`
- Homodyne SNR: `(E[X | c=1] - E[X | c=0])^2 / (Var[X|c=1] + Var[X|c=0])`

---

### Design C — Hybrid Control Model (for comparison)

**Purpose:** Quantify the delta between pure quantum readout and classical readout to isolate the value of the classical final layer.

This is the **existing Q57/Q58 architecture** used as a controlled reference. For fair comparison:
- Same encoder architecture
- Same quantum circuit
- Same seeds [42, 7, 123]
- Same epochs (4)
- Same datasets
- Same optimizer (Adam, lr=1e-3, wd=1e-4)

Results from Q57 and Q58 serve as the hybrid control without re-running.

---

## 3. Visualization Package Plan

### 3.1 Dataset and Preprocessing Figures

| Figure | Content | Output |
|--------|---------|--------|
| raw_sample_grid | 4×4 grid of raw VinDr-SpineXR DICOM crops (class 0 and class 1) | Q66/figures/raw_sample_grid.svg |
| processed_roi_grid | Same samples after CLAHE preprocessing | Q66/figures/processed_roi_grid.svg |
| batch_sample_visualization | One batch (B=8) with labels | Q66/figures/batch_sample_visualization.svg |
| class_distribution | Bar chart: class 0 vs class 1 counts (train/val/test) for both datasets | Q66/figures/class_distribution.svg |
| multiclass_to_binary_mapping | Diagram: original spine pathology labels → binary fracture/no-fracture | Q66/figures/multiclass_to_binary_mapping.svg |

### 3.2 Data Flow Figures

| Figure | Content | Output |
|--------|---------|--------|
| preprocessing_pipeline | Image → CLAHE → normalize → CNN backbone → avgpool → 960-dim → Linear(960→128) → 128-dim | Q66/figures/data_flow_preprocessing.svg |
| embedding_to_quantum_encoding | 128-dim → Linear(128→4) → 4 qubit inputs (DV) or 4 phase-space coords (CV) | Q66/figures/embedding_to_quantum_encoding.svg |
| dv_circuit_to_measurement | 4-qubit circuit diagram → Born-rule probabilities → readout | Q66/figures/dv_qnn_circuit_diagram.svg |
| cv_circuit_to_measurement | 2-mode Gaussian circuit diagram → homodyne X-quadrature → readout | Q66/figures/cv_qnn_circuit_diagram.svg |

### 3.3 Quantum Readout Diagrams

| Figure | Content | Output |
|--------|---------|--------|
| probability_readout_diagram | DV-QNN: 16-dim Born-rule probs → parity/top-k/expectation grouping → binary score | Q66/figures/probability_readout_diagram.svg |
| homodyne_readout_diagram | CV-QNN: (X_mode0, X_mode1) → threshold or centroid → binary class | Q66/figures/homodyne_readout_diagram.svg |
| dual_homodyne_scatter | 2D scatter of (X_mode0, X_mode1) colored by class | Q71/figures/dual_homodyne_class_scatter.svg |
| homodyne_distribution_comparison | Histograms of X_mode0 for class 0 vs class 1 | Q70/figures/homodyne_class_separation.svg |

### 3.4 Quantum Metrics Figures

| Figure | Content | Output |
|--------|---------|--------|
| dv_measurement_entropy | Histogram of S = -Σ p_i log p_i per sample, colored by class | Q72/figures/dv_measurement_entropy_by_class.svg |
| cv_displacement_phase_space | 2D phase space (X, P) scatter of class-conditioned mu_out | Q72/figures/cv_displacement_phase_space.svg |
| auroc_vs_quantum_param_ratio | Scatter plot: x=quantum/classical param ratio, y=AUROC for all model variants | Q74/figures/auroc_vs_quantum_param_ratio.svg |
| hybrid_vs_pure_comparison | Bar chart: AUROC for all 6 model variants (CNN/DV-hybrid/CV-hybrid/DV-pure/CV-single/CV-dual) | Q74/figures/hybrid_vs_pure_bar_comparison.svg |

---

## 4. Quantum Metrics Plan

See `quantum_metrics_matrix.csv` in this directory for full matrix with formulas and computation requirements.

### Priority Metrics by Phase

**Phase 1 (Q67/Q68 smoke tests — computable immediately):**
- Born-rule probability margin (DV)
- Homodyne X-quadrature class separation (CV)
- Per-sample measurement entropy (DV)
- Class-conditioned homodyne SNR (CV)

**Phase 2 (Q72 metrics extraction — requires inference with logging):**
- Class-conditioned fidelity (DV): `|<ψ_class0|ψ_class1>|²`
- Trace distance (DV): `T(ρ_0, ρ_1) = 0.5 ||ρ_0 - ρ_1||_1`
- Mean displacement separation (CV): `||E[μ|c=0] - E[μ|c=1]||_2`
- Covariance trace by class (CV)
- Post-training squeezing values (CV, from Q62A checkpoints)
- Gradient norms (both models, from instrumented training run)

**Phase 3 (Q73 ablations — requires modified inference):**
- Homodyne SNR under electronic noise sweep
- DV-QNN AUROC vs theta noise level
- CV-QNN AUROC vs squeezing cap
- CV-QNN AUROC vs detector efficiency

---

## 5. Ablation Study Plan

### 5.1 Readout Ablations (Q67/Q68/Q69/Q70/Q71)

| Ablation | DV-QNN | CV-QNN |
|----------|--------|--------|
| Readout type | parity vs top-k vs expectation | single homodyne vs dual homodyne vs homodyne difference |
| Calibration frozen vs trained | linear(1→1) frozen vs 1e-3 lr | threshold scalar frozen vs trained |
| Calibration removed entirely | no calibration vs with calibration | no calibration vs with calibration |

### 5.2 Encoder Ablations (Q69/Q70)

| Ablation | Description |
|----------|-------------|
| Encoder trained (current) | Linear(128→4) with gradient updates |
| Encoder frozen (random) | Linear(128→4) with requires_grad=False, random seed=42 |
| Encoder frozen (PCA) | Replace Linear with offline PCA(128→4); no trainable encoder |
| No encoder | Take first 4 dims of 128-dim embedding directly (no compression layer) |

### 5.3 Circuit Ablations (Q73)

| Parameter | Sweep values |
|-----------|-------------|
| DV-QNN n_qubits | {2, 4, 6} |
| DV-QNN depth | {1, 2, 3} |
| DV-QNN alpha | {0.05, 0.1, 0.2, 0.5} |
| CV-QNN n_modes | {1, 2, 4} |
| CV-QNN depth | {1, 2, 3} |
| CV-QNN squeezing_cap | {0.5, 1.0, 1.5, 2.0, 2.5} |

### 5.4 Noise/Detector Ablations (Q73)

| Parameter | Sweep values |
|-----------|-------------|
| DV-QNN theta noise σ | {0.0, 0.01, 0.05, 0.1, 0.2} |
| DV-QNN input noise σ | {0.0, 0.01, 0.05, 0.1} |
| CV-QNN electronic noise σ_e | {0.0, 0.01, 0.05, 0.1, 0.5} |
| CV-QNN detector efficiency η | {0.5, 0.7, 0.9, 1.0} |

---

## 6. Classification Metrics Plan

For every model variant, report the following on VinDr-SpineXR test set (held from Q49):

| Metric | Computation | Notes |
|--------|-------------|-------|
| AUROC | sklearn.metrics.roc_auc_score | Primary metric |
| AUPRC | sklearn.metrics.average_precision_score | For imbalanced data |
| Accuracy | sklearn.metrics.accuracy_score | |
| F1 | sklearn.metrics.f1_score | binary, positive class = fracture |
| Precision (PPV) | sklearn.metrics.precision_score | |
| Recall (Sensitivity) | sklearn.metrics.recall_score | |
| Specificity | TN/(TN+FP) | |
| NPV | TN/(TN+FN) | |
| Confusion matrix | sklearn.metrics.confusion_matrix | |
| Runtime (s/seed) | wall clock | |
| Total params | sum(p.numel() for p in model.parameters()) | |
| Trainable params | sum(p.numel() for p in model.parameters() if p.requires_grad) | |
| Quantum params | manually counted | |
| Classical params | trainable - quantum | |
| Quantum/classical ratio | quantum/classical | Key scientific metric |

**Models to compare:**

| Model | Slice | Hybrid or Pure |
|-------|-------|---------------|
| Classical CNN baseline | Q56 | classical |
| DV-QNN hybrid | Q57 | hybrid (4.2% quantum) |
| CV-QNN hybrid | Q58 | hybrid (1.9% quantum) |
| DV-QNN parity readout | Q69 | near-pure (4.4% quantum, 0 after-measurement params) |
| DV-QNN top-k readout | Q69 | near-pure |
| DV-QNN expectation readout | Q69 | near-pure |
| CV-QNN single homodyne | Q70 | near-pure (1.9% + threshold) |
| CV-QNN dual homodyne | Q71 | near-pure (1.9% + 4 centroid params) |

---

## 7. Scientific Framing

### Why the Previous Hybrid Architecture Is Insufficient for Quantum Advantage Analysis

The Q57/Q58 hybrid models are valuable engineering baselines: they confirm that a quantum circuit can be integrated into a functional classification pipeline using the custom `qcore` framework, and they establish reproducible AUROC benchmarks across multiple seeds and datasets.

However, they cannot support quantum advantage claims because:

1. **The quantum block is sandwiched between classical layers.** The final classification is performed by a classical linear map, not by quantum measurement statistics. Any performance difference between the hybrid model and the CNN baseline could be entirely due to the classical projection layer learning a different compression of the 128-dim embedding — unrelated to quantum effects.

2. **The quantum/classical parameter ratio is too low** (4.2% / 1.9%). In any model where 95–98% of the trainable capacity is classical, attributing performance to the quantum component requires controlled ablation, not correlation.

3. **No quantum-specific metrics are extracted.** The Q57/Q58 pipelines log AUROC, F1, and runtime — all classical metrics. Quantum advantage analysis requires class-conditioned state separation, measurement entropy, homodyne SNR, and purity — none of which are computed by the current pipeline.

### Why Pure Quantum Readout Is Needed

A pure quantum readout design answers the question: *What does the quantum circuit contribute, in isolation?* By removing the classical final layer and using Born-rule probabilities or homodyne outcomes directly for classification:

- The AUROC of the pure-readout model measures the discriminative power of the quantum measurement itself.
- The delta AUROC between pure-readout and hybrid quantifies the value added by the classical final layer.
- If AUROC(pure) ≈ AUROC(hybrid), the quantum measurement is already discriminative and the classical layer adds only calibration.
- If AUROC(pure) << AUROC(hybrid), the classical layer is doing the work and the quantum circuit is not learning discriminative structure.

This comparison is **the core experiment** needed before any quantum advantage claim can be made.

### Why Homodyne/Dual-Homodyne Readout Matters for CV-QNN

The Gaussian state representation (μ, Σ) contains all physically accessible information about the quantum state via the Wigner function. Homodyne measurement is the standard CV quantum measurement: it projects the state onto the X or P quadrature, yielding a continuous real value with statistics determined by the Gaussian parameters.

For classification, the homodyne X-quadrature mean `E[X_mode_i]` = `mu[2*i]` directly encodes the class-discriminative phase-space structure learned by the circuit. If the circuit learns to displace class-0 and class-1 states to different regions of phase space, then `X_mode0` alone — without any classical linear layer — should discriminate the classes.

Dual-homodyne (measuring X on both modes simultaneously) provides a 2D phase-space signature that may contain richer class structure than either quadrature alone. The joint scatter plot `(X_mode0, X_mode1)` colored by class is the key diagnostic figure.

### Why Born-Rule Probability Readout Matters for DV-QNN

The 16-element Born-rule probability vector `probs = |ψ|²` is the complete information accessible from a single-shot measurement of the 4-qubit state. Different basis states correspond to different bit patterns, and the circuit learns to concentrate probability mass differently depending on the class.

If the circuit learns class-discriminative structure, then simple groupings of basis states — by parity, by the value of the first qubit, or by learned class assignment — should yield class probabilities that discriminate without a classical linear layer.

The probability margin (`max(P(class=1 states)) - max(P(class=0 states))`) is the measurement-space equivalent of a decision margin and is directly interpretable as a confidence score.

### Why Quantum Metrics Are Necessary Beyond AUROC/F1

AUROC and F1 measure end-to-end classification performance. They do not distinguish between:
- A quantum circuit that learns class-discriminative unitary transformations
- A quantum circuit that learns an arbitrary non-linear function equivalent to a classical sigmoid
- A quantum circuit stuck in a barren plateau, contributing nothing, while the classical encoder does all the work

The following quantum-specific metrics are needed to make this distinction:

| Metric | What It Tests |
|--------|-------------|
| Class-conditioned state fidelity | Whether DV quantum states of class 0 and class 1 are distinguishable |
| Mean displacement separation | Whether CV Gaussian states are phase-space-separated by class |
| Measurement entropy | Whether DV circuit output is concentrated or diffuse |
| Homodyne SNR | Whether CV X-quadrature reliably separates classes |
| Gradient norm of theta/ansatz | Whether quantum parameters receive useful gradients (not barren plateau) |
| Post-training squeezing | Whether CV circuit is using squeezing to compress quadrature variance |

A model can achieve 0.9534 AUROC (as CV-QNN does) while the quantum circuit learns nothing — if the classical encoder is doing the discrimination and the quantum circuit is an expensive identity. Without quantum metrics, this cannot be ruled out.

### How the New Campaign Tests Quantum Usefulness

The proposed campaign (Q66–Q75) is designed to answer:

1. **Can quantum measurement outcomes classify?** (Q67/Q68 smoke, Q69/Q70/Q71 benchmarks)
2. **How much does the classical readout layer help?** (Q74 hybrid vs pure comparison)
3. **What does the quantum circuit learn?** (Q72 metrics extraction: state separation, homodyne SNR)
4. **Is the quantum circuit trainable?** (Q72 gradient norm logging)
5. **Is the quantum readout robust?** (Q73 noise/detector ablation)
6. **Is CV better than DV for this task?** (Q74 cross-model comparison)

If Q69/Q70 show AUROC competitive with Q57/Q58 at pure quantum readout, and Q72 shows class-conditioned state separation, and Q73 shows robustness to realistic noise — then a credible quantum advantage narrative can be constructed for publication.

---

## Appendix: Artifacts

| File | Description |
|------|-------------|
| `quantum_architecture_component_audit.csv` | Per-component audit of DV-QNN, CV-QNN, CNN baseline |
| `quantum_metrics_matrix.csv` | 40+ metrics with formulas, computation requirements, priorities |
| `recommended_experiment_campaign.yaml` | Full Q66–Q75 campaign definition with architecture specs |
| `quantum_model_redesign_plan.md` | This document |
| `reports/q65_pure_quantum_readout_and_metrics_research_plan.md` | Publication copy |
