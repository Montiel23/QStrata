# Q25A: Roadmap Prioritization and Experiment Automation Planning

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

The QStrata research project began as a series of isolated benchmark experiments. Q17 established the VinDr-SpineXR classical CNN baseline. Q18–Q21 built and validated the DV hybrid benchmark sequence. Q22 introduced a trainable-parameter-matched classical control. Q23 produced the first comparative report. Q24 corrected the roadmap to reflect DV binary closure only, inserting the CV binary phase. Q25 completed the CV feasibility design, grounding the architecture in existing QStrata infrastructure and specifying the implementation contract for Q26.

The roadmap now spans a structured multi-phase quantum benchmarking program: classical baselines, DV hybrid benchmarks, CV hybrid benchmarks, comparative reports, release tagging, and eventual experiment automation and neural architecture search. This evolution requires explicit priority separation. Without clear sequencing, there is a risk of premature infrastructure expansion — implementing automation, NAS search, or distributed AWS infrastructure before the CV scientific baseline exists. This slice (Q25A) documents that separation, gates automation work appropriately, and corrects any earlier Q26 drafts using the final Q25 design decisions.

---

## 2. Immediate Priorities

The following four slices constitute the exclusive scientific execution focus until Q28 is complete. No automation, distributed infrastructure, or NAS design work should begin before this block is done. Executing them sequentially on a single GPU is sufficient and scientifically appropriate — no scaling is required for the binary phase.

| Slice | Description | Status |
|---|---|---|
| Q26 | CV Binary Smoke Test | **NEXT** |
| Q27 | CV Binary Full Training | PLANNED |
| Q28 | DV vs CV Binary Comparative Report | PLANNED |
| Q29 | Binary Quantum Release Tagging | PLANNED |

**Why these four slices and nothing else first:**

Q26 validates that the CV pipeline designed in Q25 is numerically stable, gradient-healthy, and correctly integrated with the QStrata backend. Without this validation, Q27 full training cannot begin — a broken or unstable CV pipeline running for 15 epochs wastes compute and produces uninterpretable results.

Q27 produces the first valid VinDr CV binary benchmark. Without Q27, there is no CV result to compare against the DV benchmark (Q21) and the classical controls (Q17, Q22).

Q28 is the scientific capstone of the binary quantum phase. It directly compares DV and CV hybrid architectures on the same task under identical conditions — this is the primary scientific output of the entire Q17–Q28 sequence.

Q29 creates immutable release tags for the completed DV and CV binary phases. Tags cannot be created before their trigger conditions are met.

---

## 3. Why NAS Is Deferred

The CV baseline has not been trained or validated. Designing or running automated search over quantum architectures before a single CV baseline exists produces results with no interpretable reference point.

**The primary risk is automation of an unstable or poorly understood search space.** NAS requires:
1. A validated baseline to define meaningful search bounds — What parameter range is reasonable? What AUROC is achievable? What depth causes instability?
2. Clear evaluation criteria — What metric defines "better"? AUROC? F1? Both?
3. Reliable stopping conditions — How many epochs are needed? When does val loss plateau?

None of these are known for CV circuits on VinDr-SpineXR. Q27 will establish them. Running NAS before Q27 completes is premature and scientifically ungrounded. It would also create a risk of investing significant compute budget optimizing a search space that may itself need fundamental redesign after Q26 smoke test results are reviewed.

The correct sequencing is: baseline first, then automation. This is not a limitation — it is scientific method.

---

## 4. Future Automation Phase

After Q28 completes, the program enters an experiment automation and NAS phase. All slices below are planned but not yet active.

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q30 | Experiment Automation Design | PLANNED | Q28 complete |
| Q31 | Local GPU Experiment Runner | PLANNED | Q30 complete |
| Q32 | Lightweight NAS Search Space Design | PLANNED | Q31 complete |
| Q33 | Local NAS Pilot | PLANNED | Q32 complete |
| Q34 | AWS / Ray Distributed Design | PLANNED | Q33 complete |
| Q35 | Distributed NAS Pilot | PLANNED | Q34 complete |

**Brief slice purposes:**

- **Q30** — Design a reproducible experiment execution framework: config management, seed tracking, artifact logging, and metric collection that works uniformly across DV, CV, and classical model runs.
- **Q31** — Implement a local GPU experiment runner that can execute the Q30 framework for single-model runs without cloud infrastructure.
- **Q32** — Define a constrained NAS search space covering backbone variants, CV mode counts, DV qubit counts, learning rates, and compact head configurations. Bounded and interpretable.
- **Q33** — Run a constrained NAS pilot on a single local GPU. Validate that the search space is stable, that trials are reproducible, and that the runner correctly logs results.
- **Q34** — Design AWS GPU backend and Ray orchestration for distributed trial execution. Artifact management and trial scheduling.
- **Q35** — Validate distributed experiment execution with a small pilot. Confirm that distributed results match local single-GPU results.

---

## 5. NAS Philosophy

Future NAS work in QStrata must be:

- **Constrained** — bounded search space with explicit limits on modes, qubits, layers, and parameter budgets. An unconstrained search over all possible quantum architectures is neither feasible nor interpretable.

- **Budget-aware** — each trial must have a defined compute ceiling (maximum epochs, time limit, parameter count). Trials that exceed budget are terminated, not continued.

- **Reproducible** — fixed seeds, logged configs, deterministic trial execution. Every result must be exactly reproducible from its logged config.

- **Non-exhaustive** — random or Bayesian sampling over a well-defined space. Not grid search over all combinations. The goal is to identify promising regions of the search space, not to evaluate every point.

- **Scientifically interpretable** — results must be explainable in terms of the architectural decisions being varied. "More qubits is better" is not a finding. "Increasing n_modes from 2 to 4 while holding depth constant improves AUROC by X on VinDr binary" is a finding.

**Avoid:**

- Giant unconstrained search spaces (exponential trial counts, no interpretable structure)
- Premature distributed scaling before local validation (wasted cloud spend on broken pipelines)
- Black-box optimization with no architectural interpretation (treats the quantum circuit as an opaque hyperparameter)
- NAS as a substitute for understanding why individual architectures work (NAS finds; understanding explains)

---

## 6. Infrastructure Sequencing

The recommended infrastructure progression is sequential. Each stage validates the next. Do not skip stages.

```
Stage 1 — Single GPU (current, Q26–Q29)
  Purpose: validate CV baseline, produce binary comparative report, create release tags
  Hardware: single local GPU (existing)
  Duration: Q26 smoke test → Q27 full training → Q28 report → Q29 tags

Stage 2 — Local Automation (Q30–Q31)
  Purpose: reproducible experiment execution framework
  Hardware: single local GPU (no new hardware required)
  Duration: framework design → local runner implementation

Stage 3 — Lightweight NAS (Q32–Q33)
  Purpose: constrained search over validated search space
  Hardware: single local GPU
  Duration: search space design → local NAS pilot

Stage 4 — Distributed AWS / Ray (Q34–Q35)
  Purpose: scale what is already understood and validated
  Hardware: AWS GPU instances + Ray orchestration
  Duration: distributed design → pilot validation
```

**Principle:** do not scale infrastructure before the science is understood. Each stage produces results that inform the next. Skipping to Stage 4 without Stages 1–3 produces expensive, uninterpretable results.

---

## 7. Q26 Technical Correction

The Q25 feasibility design finalized the CV architecture. The following table supersedes any earlier Q26 drafts or assumptions from pre-Q25 planning. These are the precise technical parameters that Q26 must implement.

| Parameter | Value | Source |
|---|---|---|
| Compression layer | `nn.Linear(128 → 4)` | Q25 Section 6: n_modes=2 → 2×n_modes=4 |
| CV encoding | Complex displacement parameterization | Q25 Section 7: `alpha_i = complex(c[2i], c[2i+1])` |
| Ansatz | `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` | Q25 Section 8, from `qcore/ansatz/cv_spine_ansatz.py` |
| Backend | `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` | Q25 Section 14, from `qcore/backends/cvBackend.py` |
| Readout | Deterministic first-moment readout: `readout_vec = mu_final` | Q25 Section 9: no stochastic homodyne |
| Readout layer | `nn.Linear(4 → 2)` | Q25 Section 9: readout_dim = 2×n_modes = 4 |
| Expected trainable params | ≈ 536 (compression 516 + ansatz 10 + readout 10) | Q25 Section 10 arithmetic |
| CV backend device | CPU-only accepted | Q25 Section 14: same asymmetry as Q21 |
| Separate CV scalar gate params | None unless QStrata backend strictly requires them | Q25 Section 7: compression output is the per-sample parameterization; ansatz scalar params are global and already counted |
| Parameter accounting | Compression + ansatz + readout; no double-counting | Q25 Section 10: total = 516 + 10 + 10 = 536 |
| Feature extraction path | `with torch.no_grad(): features = self.backbone(x)` → (B, 128) | Q25 Section 5: exact Q21 semantics |
| Backbone freeze | `param.requires_grad = False`, `backbone.eval()`, train-mode override | Q25 Section 5: identical to Q21 |

These assumptions are binding for Q26. If any deviation is required (e.g., QStrata backend imposes additional parameter requirements), the deviation must be logged and the exact count printed at startup.

---

## 8. Multiclass Gate

Multiclass benchmarking remains **BLOCKED** until all of the following are complete:

| Condition | Status |
|---|---|
| VinDr DV binary benchmarking (Q17–Q23) | **CLOSED** ✓ |
| VinDr CV binary benchmarking (Q25–Q28) | PENDING |
| DV vs CV binary comparative report (Q28) | PENDING |
| Binary quantum release tagging (Q29) | PENDING |

No multiclass work, design, or planning should begin before Q29 is complete. This gate is hard. The scientific foundation of the binary phase must be complete — including the DV vs CV comparison — before introducing the additional complexity of multi-class quantum classification.

---

## 9. Roadmap State After Q25A

| Phase | Status |
|---|---|
| VinDr DV binary (Q17–Q23) | **CLOSED** |
| CV feasibility design (Q25) | **COMPLETE** |
| Roadmap prioritization (Q25A) | **COMPLETE** |
| CV smoke test (Q26) | **NEXT** |
| CV full training (Q27) | PLANNED |
| DV vs CV comparative (Q28) | PLANNED |
| Release tagging (Q29) | PLANNED |
| Experiment automation (Q30–Q31) | PLANNED — gated by Q28 |
| NAS (Q32–Q33) | PLANNED — gated by Q31 |
| Distributed AWS / Ray (Q34–Q35) | PLANNED — gated by Q33 |
| Multiclass | BLOCKED — gated by Q29 |

---

## 10. Next Slice

**Q26 — Continuous-Variable Binary Smoke Test**

Purpose: implement the minimal CV pipeline defined in Q25 (with parameters confirmed in Section 7 of this document) and validate:
- Forward pass executes without error — one batch, batch_size=4
- All health checks PASS (forward, gradient, optimizer, CV-specific) per Q25 Section 11
- Gradient flow confirmed through compression layer, `GaussianVariationalAnsatz`, and readout layer
- Backbone receives zero gradient throughout
- One optimizer step (AdamW, lr=1e-3) executes and parameters update
- No NaN or inf at any point in forward or backward pass

Q26 produces a smoke test report only. No training loop. No validation. No test evaluation. One batch, all checks, PASS/FAIL verdict.

---

```
Roadmap prioritization status: COMPLETE
Immediate priority: Q26 — CV binary smoke test
Automation phase: BLOCKED until Q28
```
