# QStrata QNN Integration Plan v1

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/qnn-integration
- **Classical anchor tag:** v1_classical_anchor

---

## 2. Purpose

This document defines the minimal technical approach for integrating a QNN classifier head on top of the frozen C006-D040 CNN feature extractor following the close of the classical optimization phase. The plan is grounded in the existing QStrata quantum framework — `qcore/ansatz/`, `qcore/circuit/`, `qcore/backends/`, `qcore/physics/`, and `experiments/models/` — and proposes no new external quantum dependencies. Nothing is implemented in this document — it establishes the architecture approach, module boundaries, interface contracts, and the scope of the first QNN execution slice so that implementation can proceed in a controlled, verifiable way. The plan is intentionally minimal: the first priority is a working end-to-end hybrid benchmark, not an optimised or theoretically motivated quantum architecture.

---

## 3. Frozen Classical Anchor

| Field | Value |
|---|---|
| Candidate ID | C006-D040 |
| block_type | depthwise_sep |
| conv_channels | [64, 128] |
| dropout | 0.40 |
| params | 9,870 |
| best_val_acc | 91.79% |
| test_acc (analysis only) | 86.86% |
| latency (ms/batch) | 0.474 |
| Git tag | v1_classical_anchor |

The CNN backbone is built by `build_model()` in `qcore/models/cnn.py`. For C006-D040, the architecture is:

```
Input (B, 1, 28, 28)
  → build_block(depthwise_sep, 1, 64)   — depthwise + pointwise conv, BN, ReLU
  → build_block(depthwise_sep, 64, 128) — depthwise + pointwise conv, BN, ReLU
  → AdaptiveAvgPool2d((1, 1))           — global average pool
  → Flatten()                            → (B, 128)
  → Dropout(0.40)
  → Linear(128, 2)                       → (B, 2) logits
```

The feature vector after `Flatten()` and before `Dropout + Linear` is `(B, 128)`. This is dimension **D = 128** — the backbone output that the projection layer will consume. This value is deterministic from `conv_channels[-1] = 128` and `AdaptiveAvgPool2d((1, 1))`.

---

## 4. Existing Quantum Framework Survey

The QStrata repository contains a complete quantum simulation framework used for prior DV and CV experiments. No external libraries (PennyLane, Qiskit, Strawberry Fields) are present or required. The framework is divided into two paths:

### 4a. Discrete-Variable (DV) Path

| Component | Location | Role |
|---|---|---|
| Operators | `qcore/operators/dv/rotations.py` | `RX`, `RY`, `RZ`, `H` rotation gates |
| Operators | `qcore/operators/dv/entanglers.py` | `CNOT` entangling gate |
| Circuit | `qcore/circuit/circuit.py` | `Circuit` class; `circuit.matrix()` compiles to unitary via PyTorch ops |
| Backend | `qcore/backends/base.py` | `Backend`; `compile(circuit) → U`; `run(U, state) → U @ state` |
| Initial state | `qcore/states/vacuum.py` | `vacuum_state(n_qubits)` — computational basis |0⟩⊗n |
| Measurement | `qcore/measurement/probability.py` | `measure_probability(state)` → probability vector \|ψ\|² of length 2^n_qubits |
| Primary ansatz | `qcore/ansatz/medical_ansatz.py` | `medical_ansatz(x, theta, n_qubits, depth, alpha) → Circuit` |
| Ansatz shape | `qcore/ansatz/medical_ansatz.py` | `get_ansatz_shape(n_qubits, depth) → (depth, 2, n_qubits, 3)` |
| Simpler ansatz | `qcore/ansatz/test_ansatz.py` | Minimal DV ansatz; same theta shape |
| DV model | `experiments/models/basic_qmodel.py` | `TwoDQClassifier(nn.Module)` — quantum-only classifier |
| DV training | `experiments/train_medmnist.py` | Training loop; sample-by-sample; Adam + ExponentialLR |

**`medical_ansatz` design:**
- Initialisation: `H` on all qubits
- Per layer: non-linear data reuploading (`arctan(x[q])*alpha` → `RY`, `x[q]*alpha` → `RZ`), variational `RX/RY/RZ` (theta-driven), ring CNOT `(q, (q+1)%n)` + cross CNOT `(q, (q+2)%n)`
- Theta shape `(depth, 2, n_qubits, 3)`: layer × (data-reupload vs variational) × qubit × (X/Y/Z rotation)
- Gradient path: `circuit.matrix()` uses PyTorch tensor ops → autograd tracks through `theta → U → U @ state → |state|² → readout_layer`; confirmed working in `train_medmnist.py`

**`TwoDQClassifier` design:**
- `theta = nn.Parameter(alpha * torch.randn(*get_ansatz_shape(n_qubits, depth)))`
- `readout_layer = nn.Linear(2**n_qubits, n_classes)` — maps full prob distribution to class logits
- `forward(x)`: builds circuit → compile to `U` → run on vacuum → measure probs (16 values for n=4) → readout → logits
- **Limitation:** processes one sample at a time; no native batch parallelism

### 4b. Continuous-Variable (CV) Path

| Component | Location | Role |
|---|---|---|
| Backend | `qcore/backends/cvBackend.py` | `GaussianBackend(n_modes, hbar)` — Gaussian state simulator |
| Physics | `qcore/physics/symplectic.py` | `get_displacement_vector`, `get_beamsplitter_matrix`, `get_rotation_matrix` |
| Measurement | `qcore/physics/cv_measurement.py` | `realistic_homodyne_readout(mu, cov, mode, angle)` — quadrature measurement |
| Visualisation | `qcore/circuit/drawer.py` | `draw_cv_ascii(ansatz)` — ASCII circuit diagram |
| Base ansatz | `qcore/ansatz/cv_ansatz.py` | `GaussianVariationalAnsatz(nn.Module)` — symplectic Gaussian evolution |
| Spine ansatz | `qcore/ansatz/cv_spine_ansatz.py` | Bounded squeezing variant (`squeezing_cap=1.5`) for spine data |
| CV model | `experiments/models/cv_2d_classifier.py` | `CV2DClassifier(nn.Module)` — quantum-only classifier with hybrid MLP stub |
| Spine model | `experiments/models/cv_spine_model.py` | `SpineCVQNN(nn.Module)` — VinDr-SpineXR path; incomplete |
| CV training | `experiments/train_cv_medmnist.py` | Training loop; batched DataLoader; Adam + ReduceLROnPlateau; gradient clipping |

**`GaussianVariationalAnsatz` design:**
- Parameters: `disp_real`, `disp_imag`, `squeezing_r`, `bs_theta`, `rot_phi` — all `nn.Parameter`
- `apply(mu, cov, backend, encoded_input=None) → (mu_out, cov_out)` — evolves Gaussian state
- Layer-by-layer data reuploading: `mu = mu + encoded_input` at each depth layer
- Ring beamsplitter topology: `(i, (i+1)%n_modes)` entanglement pattern
- All parameters are differentiable via PyTorch autograd

**`CV2DClassifier` design:**
- Encoding: displacement with angle distribution (`i * π / n_modes`), `encoding_multiplier=20.0` scaling
- Readout: `dual_homodyne` — X and P quadratures per mode → `2*n_modes` scalars
- Output path: `gain (nn.Parameter)` × readout + `bias (nn.Parameter)` → logits
- Hybrid stub: `mlp_head = nn.Sequential(Linear(2*n_modes, 16), ReLU, Linear(16, n_classes))` — present but not wired as default forward path; signals anticipated hybrid use
- **Limitation:** processes one sample at a time in `forward()` loop

### 4c. PyTorch Integration Layer

`qcore/torch/layers.py` defines `QuantumLayer(nn.Module)` — a minimal wrapper for quantum operators. It is not used by the current DV or CV models and is not needed for the hybrid integration approach proposed here.

---

## 5. Architecture Options — 2×2 Analysis

Four combinations are possible: (DV vs CV) × (quantum-only vs hybrid).

| Option | Backbone | Quantum component | Readout | Status |
|---|---|---|---|---|
| A — DV quantum-only | None | `medical_ansatz` + `TwoDQClassifier` | `nn.Linear(2^n_qubits, n_classes)` | Working (`train_medmnist.py`) |
| B — DV hybrid | C006-D040 (frozen) | `medical_ansatz` as classifier head | `nn.Linear(2^n_qubits, 1)` | Not yet built |
| C — CV quantum-only | None | `GaussianVariationalAnsatz` + `CV2DClassifier` | dual homodyne + gain/bias | Working (`train_cv_medmnist.py`) |
| D — CV hybrid | C006-D040 (frozen) | `GaussianVariationalAnsatz` as classifier head | dual homodyne + `mlp_head` | Not yet built; `mlp_head` stub present |

**Option A — DV quantum-only:**
The existing baseline. `TwoDQClassifier` encodes raw PCA-reduced features directly. No CNN backbone. Not suitable as a hybrid benchmark because it does not exercise the CNN-QNN interface at all.

**Option B — DV hybrid (recommended for MVP):**
Freeze the C006-D040 CNN backbone. Extract the `(B, 128)` feature vector (before dropout and linear). Add a projection layer `nn.Linear(128, n_qubits) + tanh` (scaled to `[-π, π]`). Pass each projected vector through `medical_ansatz` to build a DV circuit. Measure probabilities (16-dim for n=4). Feed into a `readout_layer = nn.Linear(16, 1)`. This is the primary recommended approach: `medical_ansatz` is the production-designated ansatz for medical imaging, gradient flow through the circuit to `theta` is confirmed, and the end-to-end interface is straightforward to implement.

**Option C — CV quantum-only:**
The existing CV baseline. `CV2DClassifier` encodes raw features via displacement. No CNN backbone. Not suitable as a hybrid benchmark for the same reason as Option A.

**Option D — CV hybrid:**
Replace the final Linear layers of C006-D040 with a `GaussianVariationalAnsatz` head. The `mlp_head` stub in `CV2DClassifier` signals this was anticipated. However, the CV path requires the full Gaussian state pipeline (`GaussianBackend`, symplectic evolution, homodyne readout) and has more implementation surface area than the DV path. CV hybrid is the correct longer-term research direction given that medical images are continuous-valued, but it introduces additional complexity at the integration boundary.

**Recommendation: Option B (DV hybrid) for the MVP.** The DV path is lower-risk for initial integration: the circuit-to-unitary compilation path is simpler, gradient flow is confirmed, `medical_ansatz` is the primary named production ansatz, and `TwoDQClassifier` provides a reference implementation pattern. CV hybrid (Option D) is deferred until after the DV hybrid baseline is working and committed.

---

## 6. Recommended Initial Hybrid Design

The proposed hybrid architecture follows a linear composition:

```
Input (B, 1, 28, 28)
  → CNN backbone: C006-D040 (depthwise_sep, frozen weights)
      build_block(1→64) → build_block(64→128) → AdaptiveAvgPool2d → Flatten
  → Feature vector (B, 128)
  → Projection layer: nn.Linear(128, n_qubits) + tanh × π
  → Angle values (B, n_qubits) ∈ [-π, π]
  → medical_ansatz: 4-qubit DV variational circuit (one sample at a time)
      H init → data reuploading (arctan+x*alpha → RY/RZ) → variational RX/RY/RZ → CNOT ring+cross
  → measure_probability → prob vector (16,)
  → readout_layer: nn.Linear(16, 1)
  → Binary logit (B, 1)
```

The CNN feature extractor weights are frozen during initial hybrid training. The projection layer and `theta` (the variational circuit parameters) are the only trainable components in the first baseline. The `readout_layer` is also trainable. The dropout layer from C006-D040 is excluded from the backbone slice used in the hybrid model (the feature vector is taken after `Flatten`, before `Dropout`).

**Quantum circuit specification for MVP:**

| Design choice | Value |
|---|---|
| Framework | QStrata DV framework (`qcore/ansatz/`, `qcore/circuit/`, `qcore/backends/`) |
| Ansatz | `medical_ansatz` from `qcore/ansatz/medical_ansatz.py` |
| Qubits | 4 |
| Depth | 1 (initial; can be increased) |
| Embedding | Non-linear angle reuploading (`arctan(x)*alpha` for RY, `x*alpha` for RZ) |
| Projection | `nn.Linear(128, 4)` + `tanh × π` |
| Theta shape | `(1, 2, 4, 3)` — from `get_ansatz_shape(4, 1)` |
| Readout | `measure_probability` → `nn.Linear(16, 1)` |
| Backend | `Backend` from `qcore/backends/base.py` |
| Gradient path | PyTorch autograd through `circuit.matrix()` → `theta` |

---

## 7. Proposed Module Boundaries

The following files are proposed for future implementation slices. **Do not create any of these now.**

| Proposed file | Purpose |
|---|---|
| `qcore/quantum/hybrid_head.py` | DV quantum classifier head — wraps `medical_ansatz` + `measure_probability` as `nn.Module`; accepts `(B, n_qubits)` angle values, returns `(B, 2**n_qubits)` probability vector; handles per-sample loop internally |
| `qcore/models/hybrid_cnn_qnn.py` | Hybrid model combining frozen CNN backbone + projection layer + quantum head + readout into a single `nn.Module`; `forward(x) → logits` with the same signature as the existing classical model |
| `experiments/configs/binary_hybrid_qnn_baseline.yaml` | Training config for the first hybrid run — epochs, lr, batch_size, n_qubits, depth, alpha |
| `scripts/run_hybrid_qnn_baseline.py` | Training and evaluation runner for the hybrid model; follows stable benchmark protocol v1 conventions (seed 42, best-val checkpoint, test accuracy analysis-only) |
| `reports/hybrid_qnn_baseline.md` | Results report for the first hybrid benchmark — per-epoch metrics, gate criteria, interpretation |

---

## 8. Minimum Interface Design

The hybrid model must satisfy the following interface contract at every component boundary:

| Component | Input | Output |
|---|---|---|
| CNN backbone | Image tensor `(B, 1, 28, 28)` | Feature vector `(B, 128)` (after Flatten, before Dropout) |
| Projection layer | Feature vector `(B, 128)` | Angle values `(B, 4)` ∈ [-π, π] |
| Quantum head | Angle values `(B, 4)` | Probability vector `(B, 16)` — 2^4 = 16 |
| Readout layer | Probability vector `(B, 16)` | Binary logit `(B, 1)` |

**Notes:**

- The CNN backbone output dimension is **D = 128**, confirmed from `conv_channels[-1] = 128` and `AdaptiveAvgPool2d((1, 1))` in `build_model()`. This does not need to be re-confirmed at implementation time.
- The projection layer maps `D=128 → n_qubits=4` and scales values into `[-π, π]` for angle embedding. `tanh` activation followed by multiplication by `π` is the standard approach; this is consistent with the `arctan`-based scaling already used inside `medical_ansatz` for data reuploading.
- The quantum head iterates over each sample in the batch individually, building and compiling a `Circuit` per sample. This is consistent with the existing `TwoDQClassifier.forward()` pattern and `train_medmnist.py`.
- The `readout_layer = nn.Linear(16, 1)` aggregates the 16-element probability distribution into a single binary logit, compatible with `BCEWithLogitsLoss`.
- The full hybrid model must implement `forward(x) → logits` with the same signature as the existing classical model, so it can be substituted into the existing training loop with minimal changes.
- `n_qubits = 4` and `depth = 1` are fixed for the MVP; both are hyperparameters that can be varied in later slices.
- CNN backbone weights are loaded from the C006-D040 checkpoint at tag `v1_classical_anchor` and frozen (`requires_grad = False`) before training begins.

---

## 9. Risks

| Risk | Description |
|---|---|
| Per-sample circuit compilation overhead | Neither `TwoDQClassifier` nor the proposed quantum head supports batched circuit execution; each sample requires an independent `circuit.matrix()` call. Training over 4,708 samples × epochs will be orders of magnitude slower than classical inference. This is expected and documented; the MVP is a proof of integration, not a performance claim. |
| Gradient vanishing through circuit depth | Variational quantum gradients via automatic differentiation through `circuit.matrix()` can be numerically unstable at depth > 1. The MVP uses depth 1 to minimise this risk. Gradient monitoring (`theta.grad.norm()`) must be included in the training runner. |
| Small qubit bottleneck | Compressing a 128-dimensional CNN feature vector into 4 angle values discards significant information. The expressivity of the quantum head may be the binding constraint on hybrid accuracy. This is expected at MVP scale; qubit count is a hyperparameter for post-MVP exploration. |
| Projection layer dominance | If the `nn.Linear(128, 4)` projection layer is the only significant learnable component, the hybrid model may be functionally equivalent to a small classical MLP rather than a genuine QNN. The distribution of gradient magnitudes between `projection.weight`, `theta`, and `readout_layer.weight` should be tracked during training. |
| Scope mismatch | The MVP uses a 4-qubit, depth-1 simulator. Results must be framed as a hybrid benchmark and research exploration — not a claim of quantum superiority or performance advantage over the classical anchor. |

---

## 10. Non-Goals

This plan explicitly does not cover:

- **Quantum advantage claims** — no such claims will be made at any stage of the MVP.
- **QNN architecture search** — no automated or NAS-based search for the quantum circuit structure.
- **VinDr-SpineXR dataset integration** — deferred; current scope is PneumoniaMNIST only. `SpineCVQNN` and `cv_spine_ansatz.py` are out of scope for all MVP slices.
- **CV hybrid implementation (Option D)** — deferred until after the DV hybrid baseline (Option B) is working and committed.
- **CV quantum-only baseline (Option C)** — deferred; not a hybrid benchmark.
- **Fine-tuning the CNN backbone** during hybrid training — CNN weights are frozen at `v1_classical_anchor` for the first baseline.
- **Batched quantum circuit execution** — per-sample circuit compilation is accepted for the MVP; batching optimisation is deferred.
- **External quantum frameworks** — PennyLane, Qiskit, Strawberry Fields, and any other external quantum dependency are not in scope. Only the existing QStrata framework is used.
- **`qcore/torch/layers.py` `QuantumLayer` wrapper** — not integrated into the hybrid design; the hybrid model will construct its own module boundary directly.
- **`qcore/nas/evaluator.py` modifications** — the hybrid training runner will not use the NAS evaluator interface. A new standalone runner will be created.
- **Multi-qubit entanglement strategies** beyond the existing `medical_ansatz` ring + cross CNOT pattern — deferred to post-MVP exploration.

---

## 11. Slice Q2 Proposal

**Slice Q2 — QNN Stack Smoke Test**

**Goal:** Verify that the existing QStrata DV quantum stack (`medical_ansatz`, `Backend`, `vacuum_state`, `measure_probability`) runs a complete forward and backward pass with no errors, and that gradients flow through the variational parameters `theta`. This confirms that the integration foundation is operational before any hybrid model construction begins. No external dependencies are added.

**Allowed in Slice Q2:**

- Create a minimal standalone smoke test script (`scripts/qnn_smoke_test.py`) that:
  - Imports `medical_ansatz`, `get_ansatz_shape` from `qcore.ansatz.medical_ansatz`
  - Imports `Backend` from `qcore.backends.base`
  - Imports `vacuum_state` from `qcore.states.vacuum`
  - Imports `measure_probability` from `qcore.measurement.probability`
  - Defines a 4-qubit, depth-1 variational circuit using `medical_ansatz`
  - Creates `theta = nn.Parameter(torch.zeros(*get_ansatz_shape(4, 1)))` as the trainable variational parameter
  - Runs a single forward pass with a dummy input tensor of shape `(4,)` (one sample, 4 angle features)
  - Computes a scalar loss from the probability output (e.g. `loss = probs.sum()`)
  - Runs a backward pass and verifies that `theta.grad is not None` and `theta.grad.norm() > 0`
  - Prints `QNN SMOKE TEST PASSED` or `QNN SMOKE TEST FAILED` with a one-line diagnostic
- Verify the smoke test prints `PASSED` when executed inside the GPU Docker container.

**Not allowed in Slice Q2:**

- No hybrid model construction (`hybrid_cnn_qnn.py` must not be created)
- No dataset loading or CNN model training
- No changes to any existing source file (including `qcore/nas/evaluator.py`)
- No new training configs
- No new external Python dependencies

**Deliverables for Slice Q2:**

- `scripts/qnn_smoke_test.py`
- Confirmed `QNN SMOKE TEST PASSED` output from inside the Docker container
- One logical commit: smoke test script and its verified output

---

## 12. Exit Criteria for Slice Q1

This planning slice is complete when:

- `docs/design/qnn_integration_plan_v1.md` is created and contains all 12 required sections.
- The plan is grounded in the existing QStrata quantum framework — no external quantum dependencies are referenced as required components.
- The 2×2 architecture analysis (DV vs CV) × (quantum-only vs hybrid) is present and a recommendation is stated with justification.
- No source code has been modified.
- No dependencies have been added or changed.
- No training has been executed.
- No QNN implementation of any kind exists in the repository.
- One documentation commit has been created on `feature/qnn-integration`.
