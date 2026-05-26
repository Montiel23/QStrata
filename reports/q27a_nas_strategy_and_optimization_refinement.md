# Q27A: NAS Strategy and Optimization Phase Refinement

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Q27A is a documentation-only slice executed immediately after Q27 (CV Binary Full Training, COMPLETE). The Q30–Q35 automation phase placeholders in `docs/roadmaps/binary_classical_quantum_closure_plan.md` were created during Q25A as generic stubs. They did not distinguish between classical and quantum NAS, lacked a gating rationale tied to Q29, and contained no optimization philosophy. Q27A replaces those stubs with a structured, sequenced, and principled optimization roadmap.

**What Q27A does:**
- Adds Q27A as a COMPLETE slice in the roadmap slice table
- Replaces Q30–Q35 placeholder entries with purpose-differentiated slice definitions
- Updates NAS/AWS/Ray gating from Q28 to Q29 (binary quantum release tagging)
- Adds a new Section 3e: Optimization Philosophy covering four principles
- Creates this report documenting the rationale and full structure

**What Q27A does NOT do:**
- No training, no model changes, no scripts
- No NAS unblocking (still BLOCKED until Q29)
- No AWS or Ray provisioning (still BLOCKED)
- No multiclass work (still BLOCKED until Q29)
- No branch changes (`feature/qnn-integration`)

---

## 2. Why Q30–Q35 Needed Refinement

The original Q30–Q35 entries created in Q25A were:

| Slice | Original Description |
|---|---|
| Q30 | Experiment Automation Design |
| Q31 | Local GPU Experiment Runner |
| Q32 | Lightweight NAS Search Space Design |
| Q33 | Local NAS Pilot |
| Q34 | AWS / Ray Distributed Design |
| Q35 | Distributed NAS Pilot |

These entries had three structural problems:

**Problem 1 — No classical/quantum distinction in NAS.** Q32 and Q33 treated NAS as a single undifferentiated phase. Classical NAS and quantum NAS require different search spaces, different evaluation criteria, and different scientific interpretations. They must be sequenced: classical ceiling first, then quantum search.

**Problem 2 — Gating tied to Q28, not Q29.** Q28 produces a comparative report; Q29 formally closes the binary quantum benchmarking phase with release tagging. Starting automation work after Q28 but before Q29 would mean building search infrastructure before the scientific baseline is formally closed and tagged. The correct gate is Q29 completion.

**Problem 3 — No optimization philosophy.** The original entries had no stated principles governing how optimization decisions would be made, what multi-objective criteria to use, or how infrastructure complexity should grow relative to validated scientific need.

---

## 3. Scientific Baseline First Principle

Before any NAS or automated search begins, the scientific baseline must be complete, validated, and formally closed.

The current scientific baseline status:

| Phase | Status |
|---|---|
| VinDr DV binary benchmarking (Q16–Q23) | CLOSED |
| VinDr CV binary benchmarking (Q25–Q28) | IN PROGRESS |
| DV vs CV comparative report (Q28) | NEXT |
| Binary quantum release tagging (Q29) | PLANNED |

NAS may not begin until the row marked Q29 reads COMPLETE. This is not a preference — it is an execution constraint. Searching for better architectures before understanding the baseline produces a reference-free result: you cannot evaluate whether a found architecture is good without a valid comparison point.

---

## 4. Classical Ceiling Principle

The classical ceiling principle governs the sequencing of Q32 and Q33:

> **Classical NAS (Q32) must always precede quantum NAS (Q33).**

A quantum architecture search that has not been preceded by a classical architecture search produces results that cannot be interpreted. If the quantum NAS finds an architecture with AUROC 0.72 but no compact classical architecture was searched, it is impossible to determine whether the result reflects quantum effects or simply a better architecture that any model family could achieve.

Q32 (NAS Search Space Design — Classical Feature Extractors) establishes:
- The strongest compact classical architecture under the target parameter budget
- The performance ceiling for classical CNNs on the VinDr-SpineXR binary task
- The evaluation criteria and stopping conditions that Q33 will inherit

Q33 (NAS Search Space Design — Quantum Heads) begins only after Q32 has produced a validated classical ceiling result. The classical ceiling is not a suggestion — it is the scientific reference for all quantum NAS conclusions.

---

## 5. Refined Q30–Q35 Slice Table

The following replaces the original Q30–Q35 entries in the roadmap:

| Slice | Description | Purpose |
|---|---|---|
| Q30 | Experiment Automation Framework Design | Design reproducible local orchestration, metric tracking, checkpoint management, sweep configuration. Documentation only — no code. |
| Q31 | Local GPU Experiment Runner | Implement YAML-driven runner with run queuing, resume of failed runs, metric collection, result ranking by configurable criteria. |
| Q32 | NAS Search Space Design — Classical Feature Extractors | Search compact classical CNN space (block types, channel widths, depth, pooling). Establishes classical ceiling before quantum NAS. |
| Q33 | NAS Search Space Design — Quantum Heads (DV/CV) | Search DV (qubits, depth, ansatz) and CV (modes, depth, squeezing, encoding) quantum head spaces. Requires classical ceiling from Q32. |
| Q34 | Local Multi-Objective NAS Pilot | Joint optimization across AUROC, F1, parameter count, latency, stability on single GPU. No AWS/Ray. |
| Q35 | AWS / Ray Distributed Scaling Design | Distributed NAS design only after Q34 local NAS is validated. Covers AWS provisioning, Ray orchestration, artifact management, cost controls. |

**Gating:** All Q30–Q35 slices are BLOCKED until Q29 (Binary Quantum Release Tagging) is complete. Q35 is additionally blocked until Q34 (local NAS pilot) is complete and validated.

---

## 6. Multi-Objective Optimization Philosophy

Single-metric optimization is insufficient for medical imaging research. The Q34 NAS pilot and all subsequent NAS work optimize jointly across the following objectives:

| Objective | Rationale |
|---|---|
| Test AUROC | Primary discriminative performance metric; threshold-independent |
| Test F1 | Class-balance-aware; critical for imbalanced datasets (VinDr binary is imbalanced) |
| Trainable parameter count | Model compactness; resource efficiency; practical deployment constraint |
| Inference latency (ms/sample) | Clinical deployment feasibility; real-time constraint proxy |
| Training stability | Variance across random seeds; reproducibility under fixed data splits |
| Generalization gap | val loss − train loss at convergence; overfitting signal |

No single trade-off point is pre-selected. The Pareto frontier is explored, and final model selection is a scientific decision guided by the research context — not an automated argmax on any single objective. This approach prevents spurious findings from optimizing one metric at the expense of all others.

---

## 7. NAS Philosophy

The NAS procedure implemented in Q32–Q34 follows these constraints:

**Constrained:** Search spaces are bounded by hardware budget (single GPU, target parameter ceiling). Unbounded search is computationally infeasible and scientifically unnecessary for the research questions addressed here.

**Reproducible:** All trials are seeded. Metric tables report exact values, not approximations. Each trial log records seed, split, hyperparameters, metrics, and wall-clock cost. Any result must be exactly reproducible from the trial log.

**Interpretable:** Each dimension of the search space has a stated scientific motivation. Search spaces that cannot be explained in terms of the research question are not included. After NAS completes, the best architectures must be analyzable — what specifically about their structure drives their performance.

**Budget-aware:** Wall-clock time and GPU memory are tracked as first-class trial metadata alongside ML metrics. A result that achieves AUROC 0.75 but requires 10× the training budget of the baseline is not unconditionally superior. Cost is part of the result.

**Non-exhaustive by design:** The search is guided, not grid-over-everything. Search strategies (random, Bayesian, evolutionary) are chosen based on the dimensionality of the search space. Exhaustive search is only used for spaces with ≤3 discrete dimensions.

---

## 8. Infrastructure Sequencing

Automation and infrastructure complexity must grow in proportion to validated scientific need. The six-stage sequence:

| Stage | Slice | What is validated |
|---|---|---|
| 1 | Q17–Q27 | Manual baselines: classical, DV hybrid, CV hybrid — scientifically sound |
| 2 | Q28–Q29 | Comparative report + binary closure — benchmarks formally closed |
| 3 | Q30 | Automation design — approach validated on paper before code |
| 4 | Q31 | Local runner — automation validated locally before NAS |
| 5 | Q32–Q34 | NAS on local GPU — search procedure validated before scaling |
| 6 | Q35 | Distributed scaling — scaled only after local NAS produces sound results |

Each stage is a hard prerequisite for the next. Skipping a stage is not a shortcut — it removes the validation that makes the next stage's results interpretable.

---

## 9. Updated Gating Logic

### NAS / AWS / Ray Gate (Q30–Q35)

BLOCKED until ALL of the following are complete:

| Gate condition | Status |
|---|---|
| Q26: CV binary smoke test PASSES | ✓ COMPLETE |
| Q27: CV binary full training COMPLETE | ✓ COMPLETE |
| Q28: DV vs CV binary comparative report COMPLETE | PENDING |
| Q29: Binary quantum release tagging COMPLETE | PLANNED |

No automation, NAS, or infrastructure work begins until the Q29 gate is cleared.

### Distributed Scaling Additional Gate (Q35 only)

Q35 (AWS / Ray Distributed Scaling Design) is additionally blocked until:

| Gate condition | Status |
|---|---|
| Q34: Local multi-objective NAS pilot COMPLETE and validated | PLANNED |

AWS and Ray infrastructure is not provisioned before local NAS demonstrates the search procedure is sound and cost-effective on a single machine.

### Multiclass Gate (unchanged)

All multiclass work (PathMNIST, VinDr-SpineXR multiclass, PneumoniaMNIST CV) remains BLOCKED until Q29 (Binary Quantum Release Tagging) is complete. This gate is unchanged by Q27A.

---

## 10. Risks of Premature Optimization

The following risks justify the staged approach and the Q29 gate:

**Risk 1 — Reference-free NAS results.** If NAS begins before the comparative report (Q28) is finalized, found architectures cannot be evaluated against validated baselines. The NAS result is uninterpretable.

**Risk 2 — Infrastructure built before scientific need is confirmed.** Building a Ray cluster before confirming that NAS is necessary and tractable wastes resources. Q34 must validate local NAS before Q35 distributes it.

**Risk 3 — Classical ceiling missing.** If quantum NAS (Q33) runs without a classical ceiling (Q32), any quantum improvement found cannot be attributed to the quantum components. The result conflates architecture search with quantum effects.

**Risk 4 — Stability unknown.** Training stability (seed variance) is a required metric for NAS evaluation. If stability is not measured in the baseline phase, NAS trial comparisons will conflate architecture effects with initialization noise.

**Risk 5 — Premature stopping conditions.** NAS requires well-defined stopping conditions (convergence criterion, budget limit, performance threshold). These stopping conditions must be derived from validated baselines, not guessed. Premature NAS without validated baselines produces arbitrary stopping.

---

## 11. Immediate Priorities Unchanged

Q27A is documentation-only and does not change the immediate execution priority. The immediate next slice remains Q28 (DV vs CV Binary Comparative Report).

| Slice | Status | Action |
|---|---|---|
| Q28 | NEXT | DV vs CV Binary Comparative Report — compare Q21, Q27, Q17, Q22 |
| Q29 | PLANNED | Binary Quantum Release Tagging — after Q28 complete |
| Q30–Q35 | BLOCKED | All gated on Q29 completion |
| Multiclass | BLOCKED | All gated on Q29 completion |

The Q28 comparative report will use the following confirmed result table:

| Model | AUROC | F1 | Params | Report |
|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 | `reports/vindr_classical_baseline_full_training.md` |
| Q21 DV Hybrid (pretrained backbone) | 0.6800 | 0.6159 | 574 | `reports/vindr_dv_hybrid_pretrained_full_training.md` |
| Q22 Tiny Classical (param-matched control) | 0.6625 | 0.5961 | 526 | `reports/vindr_classical_control_tiny_head.md` |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 | `reports/q27_cv_binary_full_training.md` |

---

## 12. Next Slice: Q28

With Q27A complete, the roadmap is:

- Q26: COMPLETE — PASS (2026-05-26)
- Q27: COMPLETE — PASS (2026-05-26): Test AUROC 0.6708
- Q27A: COMPLETE — documentation refinement (2026-05-26)
- Q28: NEXT — DV vs CV Binary Comparative Report
- Q29: PLANNED — Binary Quantum Release Tagging
- Q30–Q35: BLOCKED until Q29

Q28 produces the first formal scientific comparison of DV and CV quantum hybrid models on VinDr-SpineXR binary classification under the QStrata framework. It must not claim quantum advantage — it must report results accurately and provide honest interpretation within the limitations of the current benchmarking scope.

---

```
Q27A status: COMPLETE
Type: Documentation only — no training, no scripts, no model changes
Files changed: docs/roadmaps/binary_classical_quantum_closure_plan.md, reports/q27a_nas_strategy_and_optimization_refinement.md
Q28 status: NEXT — DV vs CV Binary Comparative Report
```
