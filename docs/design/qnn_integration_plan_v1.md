# QStrata QNN Integration Plan v1

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/qnn-integration
- **Classical anchor tag:** v1_classical_anchor

---

## 2. Purpose

This document defines the minimal technical approach for integrating a QNN classifier head on top of the frozen C006-D040 CNN feature extractor following the close of the classical optimization phase. Nothing is implemented in this document — it establishes the architecture approach, module boundaries, interface contracts, and the scope of the first QNN execution slice so that implementation can proceed in a controlled, verifiable way. The plan is intentionally minimal: the first priority is a working end-to-end hybrid benchmark, not an optimised or theoretically motivated quantum architecture.

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

---

## 4. Initial Hybrid Design

The proposed hybrid architecture follows a linear composition:

```
Input (28×28 grayscale)
  → CNN feature extractor (C006-D040 architecture, frozen weights)
  → Small projection layer (maps CNN output to n_qubits scalar features)
  → QNN classifier head (gate-based variational circuit)
  → Binary logit output
```

The CNN feature extractor is used as-is from the C006-D040 architecture — its weights may be frozen during hybrid training or optionally fine-tuned at a later stage. The projection layer is a small trainable linear map that compresses and scales the CNN feature vector into the input space expected by the QNN. The QNN head replaces the classical dense classifier entirely.

---

## 5. First QNN MVP Recommendation

**Recommended approach: gate-based QNN using PennyLane**

| Design choice | Value |
|---|---|
| Framework | PennyLane with PyTorch interface |
| Qubits | 4 |
| Embedding | Angle embedding |
| Variational layers | 1–2 |
| Backend | Simulator (default.qubit) |
| Output | Binary classifier head |

**Rationale:**

- **Fastest path to a working end-to-end benchmark.** The PennyLane PyTorch interface (`qml.qnn.TorchLayer`) allows a quantum circuit to be dropped into a standard `nn.Sequential` stack. Autograd, Adam, and the existing training loop all carry over without modification.
- **Angle embedding + variational layers is the established minimal gate-based QNN pattern.** Each qubit receives one scalar feature as a rotation angle; a parameterised ansatz layer applies entangling rotations; expectation values of Pauli-Z operators provide the output vector. This is the lowest-complexity path to a classically verifiable quantum model.
- **Simulator backend avoids cloud hardware friction for the MVP.** `default.qubit` runs on CPU with no account, queue, or transpilation overhead. Hardware noise can be introduced later.
- **Lower integration risk than CV-QNN.** Continuous-variable models require a different framework (Strawberry Fields) and different embedding strategies; the gate-based approach reuses the existing discrete-input pipeline.
- **PennyLane's PyTorch interface preserves standard autograd.** Gradients flow through the quantum layer via the parameter-shift rule with no custom backward pass required.

---

## 6. Future CV-QNN Direction

Continuous-variable QNN (CV-QNN) is a valid longer-term research direction for this project, particularly given that medical images are naturally continuous-valued inputs. CV-QNN models can represent transformations in infinite-dimensional Hilbert spaces and may offer richer feature interaction than gate-based architectures at equivalent qubit counts. However, CV-QNN currently introduces additional framework complexity — specifically Strawberry Fields and the Gaussian simulator backend — and requires separate validation of the end-to-end training pipeline. CV-QNN work should begin only after the gate-based hybrid baseline is working, validated, and committed. This ensures any comparison between classical, gate-QNN, and CV-QNN results is grounded in a confirmed working pipeline rather than untested infrastructure.

---

## 7. Proposed Module Boundaries

The following files are proposed for future implementation slices. **Do not create any of these now.**

| Proposed file | Purpose |
|---|---|
| `qcore/quantum/qnn_head.py` | Gate-based QNN classifier head — PennyLane variational circuit wrapped as `nn.Module` |
| `qcore/models/hybrid_cnn_qnn.py` | Hybrid model combining CNN extractor + linear projection + QNN head into a single `nn.Module` |
| `experiments/configs/binary_hybrid_qnn_baseline.yaml` | Training config for the first hybrid run — epochs, lr, batch size, n_qubits, n_layers |
| `scripts/run_hybrid_qnn_baseline.py` | Training and evaluation runner for the hybrid model; follows the same stable benchmark protocol v1 conventions (seed, best-val checkpoint, test accuracy analysis-only) |
| `reports/hybrid_qnn_baseline.md` | Results report for the first hybrid benchmark — per-epoch metrics, gate criteria, interpretation |

---

## 8. Minimum Interface Design

The hybrid model must satisfy the following interface contract at every component boundary:

| Component | Input | Output |
|---|---|---|
| CNN backbone | Image tensor `(B, 1, 28, 28)` | Compact feature vector `(B, D)` |
| Projection layer | Feature vector `(B, D)` | Scaled angle values `(B, n_qubits)` |
| QNN head | Angle values `(B, n_qubits)` | Expectation values `(B, n_qubits)` |
| Output layer | Expectation values `(B, n_qubits)` | Binary logit `(B, 1)` |

**Notes:**

- The CNN backbone output dimension `D` depends on the C006-D040 architecture's feature map size after pooling. This must be confirmed at implementation time by inspecting the model output shape.
- The projection layer must map `D → n_qubits` and scale values into `[−π, π]` for angle embedding. A `tanh` activation followed by multiplication by `π` is a standard approach.
- `n_qubits` is fixed at 4 for the MVP; this is a hyperparameter that can be increased in later slices.
- The output layer aggregates the `n_qubits` expectation values into a single binary logit for `CrossEntropyLoss` or `BCEWithLogitsLoss` compatibility.
- The full hybrid model must implement `forward(x) -> logits` with the same signature as the existing classical model, so it can be dropped into the existing training loop with minimal changes.

---

## 9. Risks

| Risk | Description |
|---|---|
| Simulator slowness | CPU-based quantum simulation scales exponentially with qubit count; even 4 qubits will be orders of magnitude slower than classical inference per batch. Training will require significantly more wall-clock time. |
| Gradient instability | Variational QNN gradients computed via the parameter-shift rule can be noisy and slow to converge, especially at small qubit counts where the loss landscape is flat (barren plateaus). |
| Small qubit bottleneck | Compressing the CNN feature vector into 4 qubit rotation angles discards significant information; the expressivity of the QNN head may be the binding constraint on hybrid accuracy. |
| Dependency friction | PennyLane version pinning must be compatible with PyTorch 2.2.2+cu121 and `numpy<2` as constrained by the existing environment. An incompatible install would break the entire training stack. |
| Overclaiming quantum advantage | No quantum speedup or accuracy advantage is expected at MVP scale on a 4-qubit CPU simulator. Results must be framed as a hybrid benchmark and research exploration, not a claim of quantum superiority. |
| Scope mismatch | Research ambition may outpace what a 4-qubit simulator can demonstrate meaningfully on a 28×28 binary classification task. The MVP is a proof of integration, not a performance claim. |

---

## 10. Non-Goals

This plan explicitly does not cover:

- **Quantum advantage claims** — no such claims will be made at any stage of the MVP.
- **QNN architecture search** — no automated or NAS-based search for the quantum circuit structure.
- **VinDr-SpineXR dataset integration** — deferred; current scope is PneumoniaMNIST only.
- **CV-QNN implementation** — deferred until after the gate-based hybrid baseline is working and committed.
- **Cloud quantum hardware** (IBM Quantum, IonQ, Quantinuum, etc.) — not in scope for MVP.
- **NSGA-II, pymoo, Ray, or any distributed search framework** — not applicable to the QNN integration phase.
- **Multi-qubit entanglement strategies** beyond standard strongly-entangling layers — deferred to post-MVP exploration.
- **Fine-tuning the CNN backbone** during hybrid training — the CNN weights are frozen at the classical anchor for the first baseline.

---

## 11. Slice Q2 Proposal

**Slice Q2 — Add Quantum Dependency and Smoke Test**

**Goal:** Establish that PennyLane integrates cleanly with the existing environment before building any hybrid model. A smoke test is the minimum viable signal that the quantum stack is operational.

**Allowed in Slice Q2:**

- Add PennyLane to the dependency stack, version-pinned to a release compatible with PyTorch 2.2.2+cu121 and `numpy<2`.
- Create a minimal standalone smoke test script (`scripts/qnn_smoke_test.py`) that:
  - Imports PennyLane and the PyTorch interface (`pennylane.qnn.TorchLayer` or equivalent)
  - Defines a trivial 4-qubit variational circuit with angle embedding and one parameterised layer
  - Runs a single forward pass with a dummy input tensor of shape `(1, 4)` (one sample, 4 features)
  - Runs a backward pass and verifies that gradients flow through the circuit parameters
  - Prints `QNN SMOKE TEST PASSED` or `QNN SMOKE TEST FAILED` to stdout with a one-line diagnostic
- Verify the smoke test prints `PASSED` when executed inside the GPU Docker container.

**Not allowed in Slice Q2:**

- No hybrid model construction (`hybrid_cnn_qnn.py` must not be created)
- No dataset loading or model training
- No changes to `qcore/nas/evaluator.py` or any existing source file
- No new training configs

**Deliverables for Slice Q2:**

- Updated dependency file (`requirements.txt` or equivalent) with PennyLane version pinned
- `scripts/qnn_smoke_test.py`
- Confirmed `QNN SMOKE TEST PASSED` output from inside the Docker container
- Two logical commits: one for the dependency update, one for the smoke test script and its verified output

---

## 12. Exit Criteria for Slice Q1

This planning slice is complete when:

- `docs/design/qnn_integration_plan_v1.md` is created and contains all 12 required sections.
- No source code has been modified.
- No dependencies have been added or changed.
- No training has been executed.
- No QNN implementation of any kind exists in the repository.
- One documentation commit has been created on `feature/qnn-integration`.
