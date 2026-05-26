# Q24: Roadmap Realignment for Continuous-Variable Binary Quantum Phase

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Problem

Marking VinDr binary benchmarking as fully CLOSED after Q23 is scientifically incomplete.

The QStrata research program benchmarks two families of quantum models: discrete-variable (DV) and continuous-variable (CV). The DV binary phase covers quantum circuits that operate on discrete qubit states using gate-based operations, entangling layers, and measurement in the computational basis. The CV binary phase covers quantum circuits that operate on continuous-variable (bosonic) modes using Gaussian operations, quadrature measurements, and moment-based readout in infinite-dimensional Hilbert space. These are qualitatively distinct quantum computing paradigms, and their respective inductive biases for machine learning tasks cannot be assumed equivalent.

Q23 closed the DV binary phase only — it consolidated results from Q17 through Q22 and declared the VinDr DV binary benchmarking sequence complete. No CV benchmark has been designed, implemented, or evaluated. Marking the overall VinDr binary quantum benchmarking as CLOSED at Q23 would silently drop the CV phase from the research program, producing an incomplete and scientifically misleading record. The correct state is: DV binary CLOSED, CV binary PENDING, overall binary quantum IN PROGRESS.

This slice (Q24) corrects that wording in the roadmap and inserts the CV binary phase (Q25–Q29) as the immediate next active work block before any multiclass benchmarking begins.

---

## 2. DV Binary Status

The VinDr-SpineXR DV binary benchmarking phase is **COMPLETE**.

| Slice | Description | Status |
|---|---|---|
| Q17 | Classical VinDr binary baseline | COMPLETE |
| Q19 | DV random frozen backbone benchmark | COMPLETE (invalid benchmark) |
| Q20 | Pretrained backbone feasibility | COMPLETE |
| Q21 | DV hybrid pretrained benchmark | COMPLETE |
| Q22 | Approximate trainable-parameter-matched classical control | COMPLETE |
| Q23 | VinDr binary DV comparative report | COMPLETE |

**VinDr DV binary benchmarking: CLOSED.**

Key results from the DV phase:

| Metric | Q17 Classical | Q21 DV Hybrid | Q22 Tiny Classical |
|---|---|---|---|
| Test AUROC | 0.6224 | 0.6800 | 0.6625 |
| Test F1 | 0.5355 | 0.6159 | 0.5961 |
| Test Accuracy | 60.66% | 63.84% | 64.37% |
| Trainable Params | 23,650 | 574 | 526 |

Primary finding: Q21 outperforms Q22 by +0.0175 AUROC after controlling for backbone and parameter count. The residual gap is scientifically interesting but does not establish quantum advantage. Further investigation under stronger controls is warranted — this is the motivation for the CV binary phase.

---

## 3. CV Binary Status

The VinDr-SpineXR CV binary benchmarking phase has **not started**.

| Slice | Description | Status |
|---|---|---|
| Q25 | CV binary feasibility design | PLANNED |
| Q26 | CV binary smoke test | PLANNED |
| Q27 | CV binary full training | PLANNED |
| Q28 | DV vs CV binary comparative report | PLANNED |
| Q29 | Binary quantum release tagging | PLANNED |

**VinDr CV binary benchmarking: PENDING.**

The CV binary phase will implement and evaluate a continuous-variable quantum hybrid model for the same VinDr-SpineXR binary task. The architectural approach — Gaussian ansatz, quadrature outputs, moment-based readout, symplectic formalism — differs fundamentally from the DV approach used in Q21. The CV phase provides an additional comparative data point for understanding what properties of quantum circuit structure (discrete vs continuous, qubit vs bosonic, gate-based vs symplectic) affect generalization performance on this task.

---

## 4. Updated Roadmap — Q17 to Q29

Full table of all slices covering the VinDr binary quantum benchmarking program:

| Slice | Description | Status |
|---|---|---|
| Q17 | Classical VinDr binary baseline | COMPLETE |
| Q19 | DV random frozen backbone | COMPLETE (invalid) |
| Q20 | Pretrained backbone feasibility | COMPLETE |
| Q21 | DV hybrid pretrained | COMPLETE |
| Q22 | Approximate trainable-parameter-matched classical control | COMPLETE |
| Q23 | VinDr DV binary comparative report | COMPLETE |
| Q24 | Roadmap realignment for CV binary phase | COMPLETE |
| Q25 | CV binary feasibility design | PLANNED |
| Q26 | CV binary smoke test | PLANNED |
| Q27 | CV binary full training | PLANNED |
| Q28 | DV vs CV binary comparative report | PLANNED |
| Q29 | Binary quantum release tagging | PLANNED |
| Multiclass | Multiclass benchmarking phase | BLOCKED until Q29 |

---

## 5. Tagging Strategy

| Tag | Trigger condition |
|---|---|
| `vindr-binary-dv-v1` | After Q23 — VinDr DV binary phase closure |
| `vindr-binary-cv-v1` | After Q28 — VinDr CV binary phase closure |
| `vindr-binary-complete-v1` | After Q29 — full VinDr binary quantum closure |

Tags must not be created ahead of their trigger conditions. The `vindr-binary-dv-v1` tag is eligible to be created now (Q23 is complete); the other tags are blocked pending their respective slices.

---

## 6. Closure Logic

**DV binary closure does not equal full binary quantum closure.**

Full binary quantum closure requires all of the following:

| Condition | Status |
|---|---|
| DV binary benchmark and controls (Q17–Q23) | **COMPLETE** |
| CV binary benchmark and controls (Q25–Q28) | PENDING |
| DV vs CV binary comparative report (Q28) | PENDING |
| Release tags (Q29) | PENDING |

The research program is **IN PROGRESS** until all four conditions are met.

The DV vs CV comparative report (Q28) is the scientific capstone of the binary quantum phase. It will directly compare the DV hybrid (Q21 architecture, qubit-based, gate-based variational circuit) with the CV hybrid (Q27 architecture, bosonic modes, Gaussian circuit), both benchmarked against the same classical controls (Q17, Q22) under identical experimental conditions. This comparison is required before any interpretation of quantum model class differences can be grounded in data.

---

## 7. Multiclass Gate

Multiclass benchmarking is **BLOCKED** until Q29 is complete.

The multiclass phase (PathMNIST, VinDr-SpineXR multiclass) must not begin until:
- VinDr DV binary benchmarking (Q17–Q23): COMPLETE ✓
- VinDr CV binary benchmarking (Q25–Q28): PENDING
- DV vs CV binary comparative report (Q28): PENDING
- Binary quantum release tagging (Q29): PENDING

This gate is intentional. Multiclass quantum circuits introduce additional complexity (multi-output readout, class imbalance, encoding strategies) that should not be started while the simpler binary case is still yielding scientifically open questions. Beginning multiclass work before binary closure is resolved would fragment the research program and compromise interpretability.

---

## 8. Next Slice

**Q25 — Continuous-Variable Binary Feasibility Design**

Purpose: Design the CV binary experiment architecture for VinDr-SpineXR using:
- **Gaussian ansatz** — Gaussian boson sampling or Gaussian circuit formalism as the quantum layer
- **Quadrature outputs** — position and momentum quadrature measurements as circuit outputs
- **Moment-based readout** — first and second moments of the output quadrature distribution as features for the classical readout layer
- **Symplectic formalism** — gate operations represented as symplectic matrices acting on the phase-space covariance matrix and mean vector
- **QStrata-only integration** — no external quantum libraries; all CV circuit machinery implemented within the QStrata backend infrastructure

Q25 produces a design document only. No training, no code implementation, no checkpoint production. The feasibility design must specify: input encoding strategy, circuit topology, measurement protocol, readout layer design, and trainable parameter structure. Q25 gates Q26 (smoke test) — the smoke test cannot begin until the design is formally documented and reviewed.

---

```
Roadmap realignment status: COMPLETE
VinDr DV binary: CLOSED
VinDr CV binary: PENDING
Overall VinDr binary quantum: IN PROGRESS
```
