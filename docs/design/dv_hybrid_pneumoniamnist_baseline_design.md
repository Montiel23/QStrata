# DV Hybrid PneumoniaMNIST Baseline Design

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/qnn-integration
- **Classical anchor tag:** v1_classical_anchor

---

## 2. Context

The classical anchor C006-D040 (`depthwise_sep`, `conv_channels: [64, 128]`, `dropout: 0.40`, `params: 9,870`, `best_val_acc: 91.79%`) is frozen at git tag `v1_classical_anchor`. No further classical tuning will be performed.

Slice Q2 validated the DV quantum train-step interface end-to-end: 13/13 validations passed inside the GPU Docker container. Key confirmed values: `theta.grad.norm()` = 4.52e-02, `max|Δtheta|` = 1.0e-02, `prob_sum` = 1.000000. Gradient flow through `medical_ansatz` → `circuit.matrix()` → `theta` is confirmed.

The next implementation target is the first DV hybrid binary PneumoniaMNIST baseline. This document defines the complete design before Slice Q4 implementation begins.

---

## 3. Design Objective

The goal is a working end-to-end hybrid pipeline — not performance optimisation. Correctness and completeness of the pipeline take priority over validation accuracy in the first baseline.

**Target architecture:**

```
C006-D040 CNN feature extractor (frozen)
  → trainable projection layer
  → medical_ansatz DV quantum head
  → measure_probability
  → binary classification output (CrossEntropyLoss)
```

The first baseline establishes that the CNN-to-quantum interface works correctly, that gradients flow through both the projection layer and the quantum circuit parameters, and that a complete train/val/test evaluation cycle can run on PneumoniaMNIST. Performance relative to the classical anchor is recorded for reference only — no accuracy gate is applied to the first hybrid baseline.

---

## 4. Frozen CNN Backbone

**Source:** `qcore/models/cnn.py`, function `build_model()`

For C006-D040, `build_model()` constructs an `nn.Sequential` with the following layer structure (confirmed from source):

```
Index  Layer                                          Output shape (B=batch)
----------------------------------------------------------------------
  0    build_block("depthwise_sep", 1,   64)          (B, 64,  28, 28)
  1    build_block("depthwise_sep", 64, 128)          (B, 128, 28, 28)
  2    nn.AdaptiveAvgPool2d((1, 1))                   (B, 128,  1,  1)
  3    nn.Flatten()                                    (B, 128)
  4    nn.Dropout(p=0.40)                              (B, 128)
  5    nn.Linear(128, 2)                               (B, 2)
```

Each `build_block("depthwise_sep", in_ch, out_ch)` expands to:
```
nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)  — depthwise
nn.BatchNorm2d(in_ch)
nn.ReLU()
nn.Conv2d(in_ch, out_ch, kernel_size=1)                           — pointwise
nn.BatchNorm2d(out_ch)
nn.ReLU()
```

**Feature extraction point:** After `nn.Flatten()` (index 3), before `nn.Dropout` (index 4) and `nn.Linear` (index 5). The hybrid model takes `model[:4]` — the first four elements of the `nn.Sequential` — as the frozen backbone slice.

**CNN backbone output feature dimension: D = 128.** This is determined by `conv_channels[-1] = 128` and `AdaptiveAvgPool2d((1, 1))`, producing a `(B, 128)` vector after flattening. This value does not need to be re-confirmed at implementation time.

**CNN weights:** Frozen for the first hybrid baseline. The backbone weights are loaded from the C006-D040 checkpoint at tag `v1_classical_anchor` and `requires_grad` is set to `False` on all backbone parameters before training begins. Future slices may evaluate selective fine-tuning after the first baseline is proven; this design does not preclude that.

---

## 5. Quantum Head Design

**Source:** `qcore/ansatz/medical_ansatz.py`, `qcore/backends/base.py`, `qcore/states/vacuum.py`, `qcore/measurement/probability.py`

| Parameter | Value | Grounding |
|---|---|---|
| Ansatz | `medical_ansatz` | `qcore/ansatz/medical_ansatz.py` |
| Backend | `Backend` | `qcore/backends/base.py` |
| Initial state | `vacuum_state(n_qubits)` | `qcore/states/vacuum.py` |
| Readout | `torch.abs(out)**2` (full state probability) | `experiments/models/basic_qmodel.py` |
| `n_qubits` | 4 | MVP value from integration plan |
| `depth` | 1 | MVP value; one variational layer |
| `alpha` | 0.1 | Inherited from `TwoDQClassifier` default |
| `theta` shape | `get_ansatz_shape(4, 1)` = `(1, 2, 4, 3)` | `qcore/ansatz/medical_ansatz.py` |
| Probability output dimension | 16 (= 2^4) | State vector length for 4 qubits |

**`medical_ansatz` circuit structure** (confirmed from source, `depth=1`, `n_qubits=4`):
1. Hadamard on all 4 qubits (superposition initialisation)
2. Data reuploading layer:
   - `RY(np.arctan(x[q]) * alpha, q)` for each qubit — arctan bounds input, alpha=0.1 scales
   - `RZ(x[q] * alpha, q)` for each qubit — direct scaling
3. Variational layer:
   - `RX(theta[0, 0, q, 0], q)`, `RY(theta[0, 0, q, 1], q)`, `RZ(theta[0, 0, q, 2], q)` for each qubit
4. Entanglement: ring CNOT `(q, (q+1)%4)` + cross CNOT `(q, (q+2)%4)` for each qubit

Total circuit ops for `n_qubits=4, depth=1`: 32 (confirmed in Q2 smoke test output).

**`Backend`** (confirmed from `qcore/backends/base.py`):
- `compile(circuit)` → `circuit.matrix()` — compiles circuit to unitary `U` via PyTorch tensor operations
- `run(U, state)` → `U @ state` — applies unitary to state vector

**`vacuum_state(4)`** (confirmed from `qcore/states/vacuum.py`):
- Returns complex64 tensor of shape `(16,)` with `state[0] = 1.0 + 0.0j`, all others zero
- Represents `|0000⟩` — all qubits in ground state

**Full probability vector** (from `experiments/models/basic_qmodel.py`):
```python
probs = torch.abs(out)**2  # shape: (16,) — full basis state probability distribution
```
This is the full 16-element probability distribution over all basis states `|0000⟩` through `|1111⟩`. It is the approach used in the current production `TwoDQClassifier`, replacing the earlier per-qubit marginal approach (the commented z-expectation code).

---

## 6. Projection Layer Design

**Source:** `qcore/ansatz/medical_ansatz.py`, `experiments/models/basic_qmodel.py`

**Fixed constraints:**

| Parameter | Value |
|---|---|
| Input dimension | 128 (confirmed from Section 4) |
| Output dimension | 4 (`n_qubits`) |

**`medical_ansatz` input convention** (confirmed from source):

The ansatz applies two transformations to each input element `x[q]`:
- `np.arctan(x[q]) * alpha` → RY rotation — `arctan` bounds to `(-π/2, π/2)`, `alpha=0.1` scales to `(-0.157, 0.157)` radians
- `x[q] * alpha` → RZ rotation — direct scaling by `alpha=0.1`

The `arctan` function handles arbitrary-magnitude inputs gracefully — it is designed to compress any real-valued input into a bounded rotation angle. The existing `TwoDQClassifier` passes raw PCA feature values (which can be any real number) directly to `medical_ansatz` without any pre-activation. The ansatz's internal scaling (`arctan`, `alpha`) is the normalisation mechanism.

**Recommendation: `nn.Linear(128, 4)` with no additional activation.**

Rationale: This is directly consistent with the existing repo design pattern — `TwoDQClassifier` passes raw unbounded features to `medical_ansatz` and relies on `arctan(x[q]) * alpha` inside the ansatz to bound the RY encoding. Adding `tanh` or `sigmoid` before the ansatz would introduce redundant bounding that duplicates the `arctan`'s role. A plain linear map is the minimal correct choice.

The projection layer learns to select and combine the 128 CNN features into 4 scalar quantum input values. Gradient flow through this layer is straightforward (standard `nn.Linear` backward pass).

**Projection layer specification:**
```python
self.proj = nn.Linear(128, 4)
# No activation — medical_ansatz handles angle scaling internally
```

---

## 7. Classification Output Design

**Source:** `experiments/models/basic_qmodel.py`, `experiments/train_medmnist.py`

**Repo design review:**

`TwoDQClassifier` uses:
```python
self.readout_layer = torch.nn.Linear(2**n_qubits, n_classes)  # Linear(16, 2)
probs = torch.abs(out)**2        # (16,)
logits = self.readout_layer(probs.unsqueeze(0))  # → (1, 2)
return logits.view(-1), out      # logits: (2,)
```

`train_medmnist.py` uses:
```python
criterion = nn.CrossEntropyLoss(weight=weights)  # balanced class weights
loss = criterion(logits.unsqueeze(0), target)    # logits (1,2), target (1,)
```

The commented-out alternative — per-qubit Z-expectation values fed through a `Linear(n_qubits, n_classes)` — was explicitly abandoned in the current codebase in favour of the full probability distribution readout.

**Chosen option: Option B2 — `nn.Linear(16, 2)` + `CrossEntropyLoss`**

This is the exact approach used in `TwoDQClassifier` and `train_medmnist.py`. It is the natural choice for the hybrid baseline:
1. **Consistency with existing repo design.** The readout layer `nn.Linear(2**n_qubits, n_classes)` is the production choice in `TwoDQClassifier`. Changing the classification head for the hybrid model would introduce unnecessary divergence.
2. **More information than per-qubit marginals.** The full 16-element distribution encodes correlations between qubit states that per-qubit marginals discard.
3. **Training stability.** `CrossEntropyLoss` with balanced class weights is well-tested on PneumoniaMNIST in `train_medmnist.py`. Class imbalance handling via `weights = 1.0 / (class_counts + 1e-6)` is inherited from the existing training pattern.
4. **Binary classification via 2-class logit.** `CrossEntropyLoss` with `n_classes=2` is equivalent to binary cross-entropy over two classes and is numerically stable.

**Readout specification:**
```python
self.readout = nn.Linear(16, 2)   # 2**n_qubits → n_classes
# Loss: nn.CrossEntropyLoss(weight=balanced_weights)
```

---

## 8. Forward Interface

Complete forward path with tensor shapes at each stage:

| Stage | Operation | Output shape |
|---|---|---|
| Input | Image batch from DataLoader | `(B, 1, 28, 28)` |
| CNN backbone | `model[:4]` — depthwise-sep blocks + AdaptiveAvgPool + Flatten, frozen weights | `(B, 128)` |
| Projection | `nn.Linear(128, 4)` — no activation | `(B, 4)` |
| Quantum loop | Per-sample: `medical_ansatz(x_i, theta, 4, 1, alpha)` → `Backend.compile/run` → `vacuum_state` → `torch.abs(out)**2` | `(B, 16)` |
| Readout | `nn.Linear(16, 2)` — trainable output map | `(B, 2)` |
| Loss | `nn.CrossEntropyLoss(weight=balanced_weights)` | scalar |

**Quantum loop detail:** The 4-dimensional projected feature vector for each sample `x_i = proj_output[i]` (shape `(4,)`) is passed to `medical_ansatz(x_i, theta, 4, 1, alpha)`. The resulting circuit is compiled to unitary `U = Backend().compile(circuit)`, applied to `vacuum_state(4)` via `U @ state`, and the output probability vector is computed as `probs_i = torch.abs(out)**2`. The batch probability matrix is formed by stacking: `probs = torch.stack([probs_i for i in range(B)])` → `(B, 16)`.

**Note on per-sample loop:** The quantum circuit does not support native batching in the current framework — `medical_ansatz` builds and compiles a separate `Circuit` per sample. This limitation is accepted for the MVP. The CNN backbone and projection steps are batched efficiently. The quantum loop scales linearly with batch size. Batched quantum execution may be addressed in future slices after the baseline is proven.

---

## 9. Trainable Parameters

| Component | Frozen / Trainable | Shape |
|---|---|---|
| CNN backbone (`model[:4]`) | **Frozen** — `requires_grad=False` on all params | N/A for gradient |
| Projection layer (`nn.Linear(128, 4)`) | **Trainable** | weight: `(4, 128)`, bias: `(4,)` |
| Quantum `theta` (`nn.Parameter`) | **Trainable** | `(1, 2, 4, 3)` = 24 values |
| Readout layer (`nn.Linear(16, 2)`) | **Trainable** | weight: `(2, 16)`, bias: `(2,)` |

**Trainable parameter count (hybrid components only):**
- Projection: 4 × 128 + 4 = 516
- Theta: 1 × 2 × 4 × 3 = 24
- Readout: 2 × 16 + 2 = 34
- **Total trainable: 574 parameters**

This is compared against the classical anchor C006-D040 (9,870 total parameters, all trainable).

---

## 10. Training Protocol for Q4

| Parameter | Value | Grounding |
|---|---|---|
| Dataset | PneumoniaMNIST binary classification | Project scope |
| Input shape | `(B, 1, 28, 28)`, grayscale | PneumoniaMNIST format |
| Seed | 42 | Stable benchmark protocol v1 |
| Initial epochs | 3 (sanity check — not full training) | Q4 scope |
| Batch size | 8 | Conservative; quantum loop is per-sample |
| Optimizer | `torch.optim.Adam` | Consistent with `train_medmnist.py` |
| Learning rate | `1e-3` | Conservative start; `train_medmnist.py` uses config-driven LR |
| Checkpoint selection | Best validation accuracy (save model state dict at peak val acc) | Stable benchmark protocol v1 |
| Test accuracy | Analysis only — not fitness signal | Project-wide constraint |
| Multi-seed | No — single seed for first baseline | Q4 scope |
| Scheduler | None for first baseline | Added in follow-up slices if needed |
| Class weights | Balanced — `1.0 / (class_counts + 1e-6)`, normalised | Inherited from `train_medmnist.py` |

**Training loop pattern** (mini-batch, not per-sample update):
```
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    # CNN forward (batched, GPU-efficient)
    features = cnn_backbone(batch_x)             # (B, 128)
    proj_out = projection(features)              # (B, 4)
    # Quantum forward (per-sample loop)
    probs = quantum_forward(proj_out, theta)     # (B, 16)
    logits = readout(probs)                      # (B, 2)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
```

This departs from `train_medmnist.py`'s per-sample `optimizer.step()` in favour of proper mini-batch gradient accumulation. The mini-batch pattern is more statistically sound and aligns with the existing `run_c006_stability_validation.py` / NAS evaluator training style. It is the correct default for neural network training.

**Seeding:** Python, NumPy, PyTorch, and CUDA seeds all set to 42 at the start of the runner, matching stable benchmark protocol v1 (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic = True`).

---

## 11. Metrics Plan

### Machine Learning Metrics

| Metric | Role | Frequency |
|---|---|---|
| Train loss | Training progress | Per epoch |
| Val loss | Generalisation signal | Per epoch |
| Train accuracy | Training progress | Per epoch |
| Val accuracy | **Primary fitness signal** | Per epoch |
| Test accuracy | Analysis only — not fitness | At best-val checkpoint only |
| Precision | Model quality characterisation | Final report |
| Recall | Model quality characterisation | Final report |
| F1-score | Balanced summary metric | Final report |
| Confusion matrix | Error pattern analysis | Final report |
| AUROC | Ranking quality | Final report |
| AUPRC | Imbalance-aware ranking quality | Final report |

### Quantum Metrics

| Metric | Role | Frequency |
|---|---|---|
| `theta` gradient norm | Verify gradient flow to quantum params | Per epoch |
| Projection gradient norm | Verify gradient flow to projection layer | Per epoch |
| Readout gradient norm | Verify gradient flow to classification head | Per epoch |
| Probability distribution summary | Mean and std of `prob_sum` per batch — verify unitary preservation | Per epoch |
| State entropy | `get_entropy(out, n_qubits)` from `experiments/metrics.py` — circuit expressivity indicator | Per epoch |
| Mean probability per basis state | Per epoch if feasible — track which basis states the circuit activates | Per epoch |
| Epoch wall-clock time | Track per-sample quantum overhead | Per epoch |

**Note on entropy:** `get_entropy` is available in `experiments/metrics.py` (imported in `train_medmnist.py`). It is computed per sample in `train_medmnist.py` as `entropy = get_entropy(out, n_qubits)`. The hybrid runner should compute the mean epoch entropy similarly.

---

## 12. Report Format

**Target report path:** `reports/dv_hybrid_pneumoniamnist_baseline.md`

The report must include:

### Required sections:

1. **Title, status, date, branch, run configuration**

2. **Architecture summary table:**
   - Component, type, frozen/trainable, parameter count for each layer
   - Total trainable parameters

3. **Training configuration table:**
   - Dataset, seed, epochs, batch size, optimizer, LR, loss function

4. **Per-epoch training table** (or curves if visualisation is available):
   - Epoch, train loss, val loss, train acc, val acc, theta grad norm, proj grad norm, readout grad norm, mean entropy, epoch time

5. **Best-epoch summary:**
   - Best val acc epoch, best val acc value
   - Test accuracy at best-val checkpoint (analysis only, clearly labelled)

6. **Final ML metrics table** (at best-val checkpoint):
   - Precision, recall, F1, AUROC, AUPRC, confusion matrix

7. **Quantum metrics section:**
   - Gradient norm evolution table per epoch
   - Probability distribution validity (mean `prob_sum`, std)
   - Mean entropy per epoch

8. **Comparison against classical anchor:**

   | Model | Val acc | Test acc | Params (trainable) | Latency |
   |---|---|---|---|---|
   | C006-D040 (classical anchor) | 91.79% | 86.86% | 9,870 | 0.474 ms/batch |
   | DV Hybrid (first baseline) | [result] | [result] (analysis only) | 574 hybrid | [result] |

9. **Limitations section:**
   - Per-sample quantum loop overhead
   - Single seed, 3 epochs only
   - No CNN fine-tuning
   - No hyperparameter search

10. **Verdict and next-step recommendation**

**Format conventions** (consistent with existing `reports/` files such as `c006_dropout_expansion.md`):
- Markdown with `##` section headers
- Tables using GitHub-flavoured markdown syntax
- No embedded images (text/table format only for first baseline)
- Verdict emitted as a clearly labelled block

---

## 13. Implementation Files Proposed for Q4

The following files are proposed for Q4 implementation. **Do not create any of these now.**

| File | Purpose |
|---|---|
| `qcore/models/dv_hybrid_cnn_qnn.py` | Hybrid model class — frozen CNN backbone (`model[:4]`) + projection `nn.Linear(128, 4)` + quantum head (`medical_ansatz` loop) + readout `nn.Linear(16, 2)` as a single `nn.Module` |
| `scripts/run_dv_hybrid_pneumoniamnist_baseline.py` | Training and evaluation runner — seed setup, data loading, mini-batch loop, best-val checkpoint, metric collection, report generation; follows stable benchmark protocol v1 |
| `reports/dv_hybrid_pneumoniamnist_baseline.md` | Results report — all sections defined in Section 12 above |
| `experiments/configs/binary_dv_hybrid_pneumoniamnist.yaml` | Training config (only if required by the runner design; prefer script-level defaults for first baseline) |

**Note on `dv_hybrid_cnn_qnn.py`:** The hybrid model must load the C006-D040 `nn.Sequential` from `qcore/models/cnn.py` using the original config, take the `[:4]` slice as the frozen backbone, and freeze all its parameters. The forward method must accept `(B, 1, 28, 28)` image tensors and return `(B, 2)` logits, with the same call signature as the existing classical model so it can slot into a standard training loop.

---

## 14. Guardrails

Hard constraints for Q4 implementation:

- **No external quantum framework** — no PennyLane, Qiskit, Strawberry Fields; use only `qcore/ansatz/`, `qcore/backends/`, `qcore/states/`, `qcore/measurement/`
- **No CV work** — DV path only; `qcore/ansatz/cv_ansatz.py`, `qcore/backends/cvBackend.py` are out of scope
- **No NAS, no QNN search, no NSGA-II, no pymoo, no Ray, no cloud**
- **No CNN fine-tuning** — backbone weights must remain frozen throughout Q4 training
- **Do not modify `qcore/nas/evaluator.py`** or any existing source file
- **No multi-seed validation** in first baseline
- **No VinDr-SpineXR integration** — PneumoniaMNIST only
- **No multiclass extension** — binary classification only
- **No dashboards, no MLflow, no monitoring stack**
- **No new external Python packages** — all dependencies must be satisfiable by the existing GPU Docker image

---

## 15. Slice Q4 Proposal

**Slice Q4 — Implement DV Hybrid PneumoniaMNIST Baseline**

**Goal:** Implement the minimal hybrid model as specified in this design document, run a 3-epoch sanity training run, and generate the baseline technical report.

**Required implementation:**

1. `qcore/models/dv_hybrid_cnn_qnn.py` — hybrid `nn.Module` matching the architecture in Section 8
2. `scripts/run_dv_hybrid_pneumoniamnist_baseline.py` — training/evaluation runner

**Required execution:**

- Set seed 42 (stable benchmark protocol v1)
- Load PneumoniaMNIST training and validation splits
- Train for 3 epochs with batch size 8, Adam lr=1e-3
- Record all metrics defined in Section 11
- Save best-val checkpoint
- Evaluate test accuracy at best-val checkpoint (analysis only)
- Write `reports/dv_hybrid_pneumoniamnist_baseline.md`

**Deliverables for Q4:**

- `qcore/models/dv_hybrid_cnn_qnn.py` — committed
- `scripts/run_dv_hybrid_pneumoniamnist_baseline.py` — committed
- `reports/dv_hybrid_pneumoniamnist_baseline.md` — committed after run completes
- Config file only if required by runner design

**Q4 success criterion:** The runner completes 3 epochs without error, produces finite metrics, gradient flow is confirmed on all three trainable components (projection, theta, readout), and the report is generated.

---

## 16. Exit Criteria for Slice Q3

This design slice is complete when:

- `docs/design/dv_hybrid_pneumoniamnist_baseline_design.md` is created and contains all 16 required sections.
- CNN backbone output dimension D=128 is confirmed from `qcore/models/cnn.py` and documented.
- Feature extraction point (`model[:4]`) is specified and grounded in the actual layer structure found in `cnn.py`.
- `medical_ansatz` input convention is confirmed from source and used to justify the projection layer design.
- Classification head option is selected (Option B2) and justified by reference to `basic_qmodel.py` and `train_medmnist.py`.
- Training loop pattern (mini-batch, not per-sample update) is specified and justified.
- No source code has been modified.
- No scripts have been created.
- No configs have been created.
- No training has been executed.
- One documentation commit has been created on `feature/qnn-integration`.
