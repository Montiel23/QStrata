# QStrata Quantum Roadmap Refinement v1

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/qnn-integration
- **Classical anchor tag:** v1_classical_anchor

---

## 2. Context

The classical optimisation phase is closed. The CNN anchor candidate C006-D040 (`depthwise_sep`, `conv_channels: [64, 128]`, `dropout: 0.40`, `params: 9,870`, `best_val_acc: 91.79%`) is frozen at git tag `v1_classical_anchor`. No further classical tuning will be performed.

Slice Q1 reviewed the existing quantum framework in the repository and produced the initial QNN integration plan (`docs/design/qnn_integration_plan_v1.md`). That review confirmed:

- **DV path:** `medical_ansatz` (`qcore/ansatz/medical_ansatz.py`) + `Backend` (`qcore/backends/base.py`) + `measure_probability` (`qcore/measurement/probability.py`). Gradient flow through the variational circuit parameters `theta` is confirmed via existing `experiments/train_medmnist.py`.
- **CV path:** `GaussianVariationalAnsatz` (`qcore/ansatz/cv_ansatz.py`) with symplectic Gaussian state evolution (`mu`, `cov`) + `GaussianBackend` (`qcore/backends/cvBackend.py`) + quadrature and homodyne readout (`qcore/physics/cv_measurement.py`). A working CV training loop exists in `experiments/train_cv_medmnist.py`.

Both paths are implemented in the existing QStrata quantum framework. No external quantum libraries are present or required. The quantum roadmap now requires explicit prioritisation before execution begins.

---

## 3. Refined Strategic Decision

**The project will execute DV first, then CV.**

### DV Execution Order

1. DV binary classification — PneumoniaMNIST
2. DV binary classification — VinDr-SpineXR
3. DV multiclass benchmark
4. DV multiclass — VinDr-SpineXR (if labels support it)

### CV Execution Order (after DV path is proven)

1. CV binary classification — PneumoniaMNIST
2. CV binary classification — VinDr-SpineXR
3. CV multiclass benchmark
4. CV multiclass — VinDr-SpineXR (if labels support it)

---

## 4. Rationale

**DV executes first because:**

- **Lower risk for first end-to-end quantum model validation.** Gradient flow through `medical_ansatz` → `circuit.matrix()` → `measure_probability` → `theta` is already confirmed by `experiments/train_medmnist.py`. No unknowns exist in the autograd path before training begins.
- **Simpler integration boundary.** The DV circuit compiles to a unitary matrix via PyTorch tensor operations; the interface to the CNN backbone is a straightforward projection from the 128-dimensional CNN feature vector to 4 angle values. The CV path requires Gaussian state initialisation, symplectic matrix evolution, and a distinct readout formalism — more moving parts at the integration boundary.
- **Validates the full quantum workflow first.** Running DV first lets the team establish and verify the complete quantum experiment pipeline — data pipeline, training loop, metric logging (ML and quantum), reporting format, gate criteria — before adding CV complexity. Lessons from the DV baseline will directly inform the CV implementation.
- **`medical_ansatz` is the production-designated ansatz for medical imaging.** Its naming and design (non-linear reuploading, `arctan + ring + cross CNOT`) reflect prior project intent for this domain.

**CV is deferred in execution order only — not deprioritised as a research direction:**

- CV is better aligned with photonic quantum computing platforms, where continuous-variable operations (displacement, squeezing, beamsplitter, rotation) are native hardware primitives.
- The symplectic first and second moment formalism (`mu`, `cov`) provides an exact and compact representation of Gaussian quantum states, well-suited to continuous-valued medical imaging inputs.
- Gaussian gates and homodyne/quadrature measurement outputs are physically interpretable in a way that is distinct from DV probability distributions — this distinction carries independent research value.
- The `GaussianVariationalAnsatz` is already implemented and differentiable. CV execution will proceed as soon as the DV binary milestones are proven. Deferral is a sequencing decision, not a deprioritisation decision.

---

## 5. Immediate Next Slice

**Slice Q2 — DV Quantum Stack Smoke Validation**

**Goal:** Validate that the existing DV quantum stack runs cleanly end-to-end in the current environment before building any classifier. This confirms the integration foundation is operational. No hybrid model, no dataset loading, no training.

**Required validations:**

| Validation | What is checked |
|---|---|
| `medical_ansatz` instantiates | Import succeeds; `medical_ansatz()` returns a `Circuit` without error |
| `Backend` initialises | `Backend()` instantiates correctly |
| `vacuum_state` produces expected output | Returns a complex state vector of length `2**n_qubits` with norm = 1.0 |
| `measure_probability` returns valid distribution | Returns a real-valued probability vector summing to 1.0 |
| Forward pass completes | Full path: dummy input → `medical_ansatz` → `circuit.matrix()` → `vacuum_state` → `measure_probability` runs without error |
| Backward pass completes | `loss.backward()` runs without error |
| Nonzero gradient on `theta` | `theta.grad is not None` and `theta.grad.norm() > 0` confirmed |

**Constraints:**

- No training
- No dataset loading
- No hybrid CNN model construction
- Use existing repo entrypoints and experiment code paths wherever possible
- Create new validation code only if no reusable existing entrypoint serves this purpose

---

## 6. Updated Q2 Non-Goals

The following are explicitly out of scope for Slice Q2:

- **No CV smoke test** — CV validation is deferred until after DV binary and multiclass milestones are proven
- **No hybrid model construction** — `hybrid_cnn_qnn.py` must not be created in Q2
- **No dataset loading** — no PneumoniaMNIST or VinDr-SpineXR data access
- **No training runs** — no optimiser steps, no epoch loops
- **No reports** — no metric output documents
- **No new configs** — no YAML experiment files
- **No external framework installation** — no PennyLane, Qiskit, Strawberry Fields, or any other quantum library
- **No changes to `qcore` source files or any other existing source files**

---

## 7. Future Slices

Proposed high-level sequence. Implementation details are not specified here and will be determined at each slice boundary.

| Slice | Goal |
|---|---|
| Q2 | DV quantum stack smoke validation |
| Q3 | DV quantum-only or DV hybrid binary PneumoniaMNIST baseline (direction determined by Q2 output) |
| Q4 | DV binary PneumoniaMNIST training report |
| Q5 | DV binary VinDr-SpineXR data integration planning |
| Q6 | DV binary VinDr-SpineXR baseline |
| Q7 | DV multiclass extension planning |
| Q8 | CV roadmap activation planning |

---

## 8. Metrics Reminder

All future quantum experiment reports must include both ML and quantum metrics. This applies to DV and CV experiments alike.

### Machine Learning Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- AUROC
- AUPRC

### Quantum Metrics (where available in repo experiments)

- Gate fidelity
- State fidelity
- Entropy
- Purity
- State evolution across epochs
- Gradient norm evolution (tracked per parameter group: projection, theta, readout)
- Measurement distributions (DV: probability distributions over basis states; CV: quadrature histograms)
- Observable evolution across epochs (DV: Pauli expectation values where computable)
- Quadrature evolution across epochs (CV: X and P quadrature means from homodyne readout)

---

## 9. Explicit Stop Line

The following items are explicitly stopped. They will not be executed.

- **No further classical tuning.** The C006-D040 anchor is frozen. No dropout sweeps, no weight decay expansion (Slice 32 is cancelled), no learning rate search, no architecture changes.
- **Slice 32 weight decay expansion is not executed.** The Slice 31 planning document (`docs/design/slice31_c006_d040_manual_search_plan.md`) is superseded by the decision to proceed to QNN integration.
- **The next executable slice is Slice Q2 — DV smoke validation.** Nothing runs before Q2 clears.
- **CV execution does not begin until DV binary and multiclass milestones are proven.** No CV smoke test, no CV hybrid model, no CV training run may begin while any DV milestone remains unproven.

---

## 10. Exit Criteria

This slice is complete when:

- `docs/design/quantum_roadmap_refinement_v1.md` is created and contains all 10 required sections.
- DV-first execution order is clearly stated with an explicit ordered list for both DV and CV paths.
- CV is preserved as a core research direction and deferred in execution order only.
- Slice Q2 is defined as DV-only smoke validation with explicit required validations and non-goals.
- No source code has been modified.
- No scripts have been created.
- No configs have been created.
- No training has been executed.
- One documentation commit has been created on `feature/qnn-integration`.
