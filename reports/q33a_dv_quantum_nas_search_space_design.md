# Q33A: DV Quantum NAS Search Space Design

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q33A — DV Quantum NAS Search Space Design  
**Status:** COMPLETE — design only; no NAS execution

---

## 1. Title

Q33A: DV Quantum NAS Search Space Design

---

## 2. Context

The QStrata binary benchmarking phase (Q17–Q29) established four frozen compact benchmarks on VinDr-SpineXR binary classification:

| Model | AUROC | F1 | Params |
|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 574 |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 526 |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 |

Q21 is the critical DV finding for Q33A: a 4-qubit variational DV quantum head over a frozen pretrained backbone (C006-D040) achieved AUROC 0.6800 and F1 0.6159 with 574 trainable parameters. Q21 outperformed the parameter-matched classical control Q22 by AUROC +0.0175 and F1 +0.0198. However, Q21 is a single fixed-point DV architecture. Whether this residual DV advantage survives systematic classical competition — or whether it reflects an underexplored classical design space — is the open question that Q33A, Q34, and Q35 are designed to answer.

Q28 documented the DV vs CV frontier: Q21 led Q27 on AUROC (+0.0092) while Q27 led Q21 on F1 (+0.0124). Neither advantage is statistically validated (single seed, no confidence intervals). Q28 explicitly identified the absence of a systematic classical ceiling as the critical limitation preventing any quantum advantage interpretation.

Q32 established the classical ceiling methodology and defined the classical NAS search space. Q34 will execute the Q32 classical search and produce a Pareto frontier that serves as the formal classical ceiling. Q33A now defines the bounded DV quantum search space required before Q34 can execute joint classical and quantum NAS trials.

**Q33A defines the search space only. No NAS execution occurs in Q33A.** No trial configs are generated. No models are trained. No architecture search is run. Q34 is the first DV NAS execution phase.

The full DV search space specification is in `docs/architecture/q33a_dv_quantum_nas_search_space.md`.

---

## 3. Why DV Quantum NAS Requires Constrained Design

### The Unconstrained DV Search Failure Mode

Unconstrained DV search faces three compounding problems that make it scientifically useless for the QStrata comparison question:

**Barren plateaus.** Gradient magnitudes of quantum circuits become exponentially small as circuit depth and qubit count increase. A circuit with 16 qubits and depth 10 — well within the realm of unconstrained search — will have gradient variances suppressed by a factor of ~1/65,536 relative to a 1-qubit circuit. Training such a circuit with gradient-based optimization is impossible; results would reflect random initialization, not learned representations.

**Simulator intractability.** The Hilbert space of an n-qubit circuit has dimension 2^n. A 20-qubit circuit requires a state vector of dimension 1,048,576 complex numbers. Full-state simulation of circuits beyond 16 qubits is computationally intractable on a local GPU without matrix product state approximations that introduce truncation errors not present in the QStrata DV backend.

**Uninterpretable results.** A DV circuit that requires 100,000 trainable parameters and 20 qubits to match the classical ceiling is not a compact DV head — it is a large DV model that happens to equal a compact classical model. This comparison does not answer the scientific question of whether compact DV heads can match compact classical heads; it answers a different question about whether large quantum systems can eventually match small classical systems, which is trivially true.

### Constrained Design Produces a Meaningful Pareto Frontier

Q33A constrains the DV search to circuits that are simulator-tractable on local GPU (≤ 8 qubits, depth ≤ 4), parametrically compact (≤ 100,000 head parameters), numerically stable (no NaN/inf in forward or backward pass), and scientifically interpretable (each dimension corresponds to a specific, comprehensible architectural choice).

These constraints ensure that the DV Pareto frontier produced by Q34 is comparable to the classical Pareto frontier from Q32: both frontiers are produced under the same backbone, dataset, orchestration, and compactness regime. The comparison is then scientifically clean.

---

## 4. Optimization Philosophy

Q33A adopts the Q32 three-tier multi-objective optimization structure in full, extended for quantum-specific concerns. **Single-metric optimization is forbidden.** Quantum NAS must optimize compactness and stability simultaneously, not just maximize AUROC.

### Primary Objectives (Must Optimize)

Both objectives are tracked for every trial. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance; primary cross-model comparison metric
- **Maximize F1** — threshold-dependent precision-recall tradeoff at 0.5 threshold; clinical relevance

### Secondary Objectives (Should Optimize)

Required in every trial result record:

- **Minimize trainable parameter count** — quantum head parameters only; frozen backbone excluded; compact quantum heads are the scientific target
- **Minimize inference latency** — ms/sample on local hardware; Q21 required 54.79 ms/sample (CPU-bound); DV NAS may discover better latency-performance tradeoffs
- **Minimize numerical instability** — NaN/inf rate across training epochs; gradient norm collapse frequency; barren plateau candidate detection

### Tertiary Objectives (Consider)

Recorded but may not enter primary Pareto analysis:

- **Minimize quantum simulator runtime** — wall-clock time for quantum circuit forward pass alone
- **Minimize memory usage** — GPU and CPU peak memory
- **Minimize barren plateau behavior** — gradient variance collapse metric: ratio of final-epoch theta gradient norm to initial-epoch theta gradient norm

A DV circuit that maximizes AUROC while requiring 10,000 trainable parameters is not a compact DV ceiling. A DV circuit that achieves high AUROC in one seed but crashes with NaN in three others is not deployable. Quantum NAS must optimize compactness and stability simultaneously.

---

## 5. DV Pareto Frontier Philosophy

### Non-Dominated DV Solutions

A DV trial is Pareto-optimal (non-dominated) if no other evaluated DV trial is strictly better on all tracked objectives simultaneously. The DV Pareto frontier is the set of all non-dominated DV trials over AUROC, F1, parameter count, latency, and stability.

A trial that achieves lower AUROC than another but uses 5× fewer parameters and runs 10× faster remains on the frontier if no single trial dominates it across all four axes simultaneously.

### Compact Quantum Tradeoffs

The DV Pareto frontier exhibits interpretable tradeoffs:
- Deeper circuits may achieve higher AUROC at the cost of higher latency and greater barren plateau risk
- Larger qubit counts expand the Hilbert space but increase simulation cost exponentially
- Re-uploading improves gradient flow at the cost of multiple circuit passes and increased runtime
- Shallow, small-qubit circuits occupy the low-latency, low-parameter region directly comparable to the most compact classical head configurations from Q32

These tradeoffs must be preserved in the frontier, not collapsed to a scalar winner.

### Compare Against the Q32 Classical Frontier, Not Q17

The DV Pareto frontier produced by Q34 must be compared directly against the Q32 classical Pareto frontier — not against Q17. Comparing quantum NAS results against Q17 would be methodologically invalid: Q17's weakness is primarily attributable to random initialization and large parameter count, not to the classical architecture family. An optimized DV frontier must be compared against an optimized classical frontier.

### Preserve the Full DV Frontier for Q34

Q34 must preserve the entire DV Pareto frontier before any comparative claims are made. The DV frontier may dominate the classical frontier in some regions (e.g., very compact low-parameter architectures) while being dominated in others (e.g., high-AUROC moderate-latency configurations). Only the full frontier comparison reveals the pattern of quantum vs. classical tradeoffs.

**The goal is the frontier, not a single best model.**

---

## 6. Searchable DV Dimensions

All dimensions are discrete. The Q34 search algorithm samples from these options.

| Dimension | Options |
|---|---|
| **Qubit count** | 2, 4, 6, 8 |
| **Ansatz depth** | 1, 2, 3, 4 |
| **Rotation families** | `RX`, `RY`, `RZ`, `RX+RY`, `RX+RY+RZ` |
| **Entanglement topology** | `linear`, `circular`, `full` (≤ 4 qubits only), `nearest_neighbor` |
| **Encoding strategy** | `angle_encoding`, `amplitude_lite_encoding`, `reuploading_angle_encoding` |
| **Re-uploading frequency** | `none`, `once`, `every_layer` |
| **Measurement strategy** | `pauli_z_per_qubit`, `multi_qubit_expectation_avg`, `concatenated_expectation_vector` |
| **Compression dimension** (classical projection before quantum encoding) | 2, 4, 8, 16 |
| **Classical projection layer** (after measurement, before binary logits) | `linear`, `linear_plus_activation`, `none` |

**Search space size (rough estimate):** 4 × 4 × 5 × 4 × 3 × 3 × 3 × 4 × 3 ≈ 103,680 discrete configurations (subject to compatibility constraints between qubit count, encoding strategy, and entanglement topology). Random or Bayesian sampling over 20–50 trials is a small but interpretable pilot sample of this space.

**Q21 position in this space:** qubit count = 4, depth = 1, rotation family = `RX+RY+RZ` (implicit in the 4-qubit variational ansatz with 24 theta parameters = 4 qubits × 1 depth × 6 rotation axes), entanglement topology = `circular`, encoding strategy = `angle_encoding`, re-uploading frequency = `none`, measurement strategy = `pauli_z_per_qubit`, compression dimension = 4, projection layer = `linear`.

Full dimension specifications with scientific rationale: `docs/architecture/q33a_dv_quantum_nas_search_space.md` Section 6.

---

## 7. Forbidden DV Dimensions

The following are explicitly out of scope for Q33A, with scientific rationale:

| Forbidden | Scientific Rationale |
|---|---|
| Qubit count > 8 | Hilbert space dimension 2^n grows exponentially; local GPU simulation intractable; barren plateau severity increases exponentially |
| Unrestricted circuit depth | Depth > 4 produces exponentially suppressed gradients (barren plateau); simulation time grows super-linearly; results not interpretable against shallow classical heads |
| Adaptive circuit topology mutation at runtime | Breaks reproducibility; incompatible with frozen YAML config requirement |
| Dynamic quantum graph generation | Modifies circuit connectivity during training; breaks the frozen-config requirement; unrecoverable from a static config |
| Hardware-aware qubit routing | Assumes target physical quantum device; incompatible with local simulator-based execution |
| Quantum error correction | Logical qubit overhead far exceeds compact parameter budget; requires syndrome measurement circuits not in QStrata DV backend |
| Cloud quantum hardware assumptions | Q33A is local-only; NISQ device execution is blocked until local NAS produces validated results |
| Transformer-style quantum attention mechanisms | Not compact DV head architectures; changes the comparison baseline |
| Hybrid large classical heads (> 10K parameters in the head alone) | Violates the compactness principle; not comparable to Q21/Q22/Q27 in the ≤600 trainable parameter regime |

**Rationale:** Q33A focuses exclusively on compact simulator-based DV systems compatible with local GPU experimentation and the QStrata framework. The compactness of existing quantum benchmarks (Q21: 574 params) defines the regime of comparison.

---

## 8. Constraint System

### Hard Constraints — Violation Marks Trial `invalid`

| Constraint | Value |
|---|---|
| Total trainable parameters | ≤ 100,000 |
| Execution environment | Local single GPU only; no cloud infrastructure |
| Quantum simulator forward pass | OOM-free at batch size 4 on local GPU |
| Numerical validity | No NaN or inf at any point in forward or backward pass |
| Reproducibility | Seed set from config; git commit captured |
| Batch size compatibility | Batch size 4 required |

Hard constraint violations cause the trial to be marked `invalid` and excluded from Pareto frontier computation. Invalid trials are retained in the leaderboard for diagnostic purposes.

### Soft Constraints — Violation Flagged but Trial Retained

| Constraint | Value |
|---|---|
| Inference latency | Within 2× Q21 latency (~110 ms/sample) |
| Compact quantum heads preferred | < 5,000 trainable parameters |
| Shallow circuits preferred | Depth ≤ 2 for pilot |
| Low simulator runtime | < 5× classical head runtime |

Soft constraint violations are recorded in the result JSON and leaderboard. Trials flagged for soft violations remain in the Pareto frontier with their constraint status visible in all tables and reports.

---

## 9. Barren Plateau Mitigation Philosophy

Barren plateaus arise when gradient magnitudes of quantum circuits become exponentially small as circuit depth or qubit count increases. A training landscape that is flat everywhere provides no gradient signal; the circuit cannot be trained with gradient-based optimization.

Q33A mitigates barren plateau risk through structural constraints:

- **Depth constraint (depth ≤ 4):** Limits the parameter landscape flatness relative to deeper circuits while preserving sufficient expressivity for binary classification
- **Qubit count constraint (≤ 8 qubits):** Keeps gradient variance in a regime where training is feasible; suppression factor ~1/256 at 8 qubits vs. ~1/2^20 at 20 qubits
- **Compact parameterization:** Fewer rotation parameters per layer reduces the joint parameter space dimension; single-axis rotation options (`RX`, `RY`) minimize per-qubit parameter count
- **Limited entanglement topology:** Linear and nearest-neighbor topologies reduce quantum volume and associated barren plateau risk; full entanglement permitted only up to 4 qubits
- **Re-uploading strategies:** Feature re-uploading interleaves input encoding with variational layers, providing a persistent classical gradient signal that can alleviate flatness in encoding layers

Q21 demonstrated full gradient health across all 15 training epochs: non-zero theta, projection, and readout gradient norms throughout; no NaN/inf; no gradient collapse. Q34 DV NAS trials must implement equivalent gradient health tracking. Trials exhibiting gradient norm collapse (theta gradient norm below 1e-8 for ≥ 3 consecutive epochs) are flagged as barren plateau candidates.

**Q33A intentionally constrains search entropy to reduce barren plateau risk.** The constraint system is not conservative out of caution — it is conservative because the scientific question requires circuits that can actually be trained.

---

## 10. Experiment Budget Philosophy

### Local Single-GPU Execution

All Q34 DV NAS trials run sequentially on one local GPU using the Q31/Q31A runner infrastructure. One DV trial runs at a time. No parallelism across trials. This is a hard constraint enforced by the local-first principle.

### Bounded Trial Count

Q34 pilot uses approximately 20–50 DV trials as a planning reference. This count is sufficient to sample the DV search space with meaningful coverage, produce a DV Pareto frontier with enough non-dominated points to identify tradeoff patterns, and complete within a time-bounded local session. The exact trial count is a Q34 decision based on available hardware, per-trial wall-time, and session budget at execution time.

### Bounded Runtime Per Trial

Each DV trial has a configurable wall-time ceiling. Trials exceeding it are marked `timeout` and excluded from Pareto frontier computation but retained in the leaderboard. An initial timeout of 30–60 minutes per DV trial is a reasonable planning estimate — DV circuit simulation is CPU-bound and substantially slower than classical GPU training.

### No Exhaustive Grid Search

Q34 uses random search or lightweight Bayesian sampling. The Q33A search space contains approximately 77,760–103,680 discrete configurations (subject to compatibility constraints). Exhaustive enumeration is infeasible at any bounded trial budget. Random or Bayesian sampling produces an interpretable Pareto frontier without search algorithm artifacts.

**Q34 prioritizes interpretable signal over brute-force quantum search scale.**

---

## 11. Reproducibility Requirements

**Frozen YAML Config Per Trial:** Every DV NAS trial produces a frozen YAML config via the Q31 runner framework, written to `experiments/configs/<experiment_id>.yaml` and set read-only (`chmod 444`) before trial execution. All searchable dimension values are captured in the config.

**Seed Locking:** Each DV trial uses a fixed seed from `reproducibility.seed` in the config. The seed is set before dataset loading, model initialization, augmentation, and quantum circuit parameter initialization. The seed is recorded in the result JSON.

**Git Commit Tracking:** Every DV trial records the exact code state at execution time via `git rev-parse HEAD` or the `QSTRATA_GIT_COMMIT` env var (validated in Q31A). Trials with `git_commit: unknown` are flagged. Published DV NAS results require a clean commit.

**Leaderboard Integrity:** DV trial results are immutable once recorded. No post-hoc editing. Erroneous trials are retained with `status: retracted`; corrected re-runs produce new entries.

**Deterministic Orchestration:** Given the same config and seed, re-running any DV trial must produce results within tolerance (AUROC ±0.0001, F1 ±0.0001). This was validated by Q31A (loss_delta = 0.0). DV quantum circuit simulation must be deterministic given a fixed seed.

The Q31/Q31A infrastructure is the prerequisite for Q34 DV NAS precisely because it enforces frozen configs, seed locking, git tracking, and leaderboard immutability mechanically. **Quantum NAS without reproducibility is scientifically invalid.**

---

## 12. Relationship to Q32 Classical Ceiling

Q32 defines the search space for compact classical CNN head architectures over the same frozen pretrained backbone (C006-D040) and the same VinDr-SpineXR binary task. Q34 executes this classical search and produces a Pareto frontier over AUROC, F1, parameter count, and latency — the classical ceiling.

Q33A defines the DV quantum search space over the same backbone, task, and orchestration framework. Q34 executes DV NAS trials from this space and produces a DV Pareto frontier.

**DV NAS results must be compared against Q32 classical results, not against the unoptimized Q17 baseline.** Comparing against Q17 would be methodologically invalid: Q17's weakness is attributable to random initialization and large parameter count, not to the classical architecture family. An optimized DV frontier must be compared against an optimized classical frontier.

Q34 evaluates both classical and DV NAS under identical orchestration, datasets, seeds, and constraints. Any difference in Pareto frontier position between classical and DV architectures is attributable to the head architecture, not to differences in experimental conditions. This is the scientific requirement that Q32 and Q33A jointly satisfy: defining both search spaces before any NAS execution begins, under a shared methodology that ensures the comparison is fair.

A DV architecture that is not dominated by any Q32 classical architecture is a meaningful positive result. A DV frontier that is fully dominated by the classical frontier is also a meaningful result — it establishes that compact classical heads fully match or exceed compact DV heads under the defined constraints.

---

## 13. Limitations

**No NAS execution in this slice.** Q33A is design only. No trial configs are generated, no models are trained, no architecture search is run.

**No AWS or Ray.** All execution is local. Distributed infrastructure remains blocked until Q34 validates local NAS.

**No distributed execution.** Q34 is local, sequential, single-GPU.

**No generated architectures.** The search space definition does not produce candidate architectures. Candidate configs are generated by the Q34 search algorithm at execution time.

**No multiclass work.** Multiclass benchmarking is blocked until Phase 3 (Q32 ceiling), Phase 4 (Q33A/Q33B quantum NAS), and Phase 5 (Q34 optimized binary release) are all complete.

**No object detection work.** Object detection is outside the current QStrata roadmap scope.

**No hardware quantum execution.** Q33A assumes local quantum circuit simulation on CPU. Physical NISQ hardware execution is blocked until local NAS produces validated results.

**Search space is a planning artifact.** The exact search space dimensions (qubit counts, depth limits, compression options, trial count) are defined here as planning inputs. Q34 may refine these based on hardware constraints, preliminary trial results, and wall-time budgets at execution time. Refinement of the search space in Q34 does not require a new design slice — it requires documenting the changes in the Q34 report.

**Single-seed evaluation in Q34 pilot.** The Q34 pilot will execute with a single seed per trial to maintain bounded wall-time. Multi-seed validation of DV NAS results is deferred to Q35. All Q34 DV Pareto frontier results are therefore single-seed point estimates without confidence intervals.

---

## 14. Next Slice

**Q33B — Continuous-Variable Quantum NAS Search Space Design**

Q33B is also design only. It will define the GaussianVariationalAnsatz search dimensions: n_modes, cv_depth, squeezing_cap, displacement_cap, encoding scheme, readout strategy — using the same multi-objective framework, the same runner infrastructure, and the same constraint system as Q33A. Q33B does not execute any NAS trials.

**Q34 is the first execution phase.** Q33B defines the CV search space; Q34 executes classical (Q32), DV (Q33A), and CV (Q33B) search spaces in a joint pilot. The classical ceiling, DV quantum Pareto frontier, and CV quantum Pareto frontier are produced in the same pilot and compared in the same report.

```
Q33A status: COMPLETE — design only; no NAS execution
Q33B status: NEXT — CV Quantum NAS Search Space Design (design only)
Q34 status: PLANNED — first local NAS execution (classical + DV + CV)
Q35 status: PLANNED — Pareto analysis and NAS hardening
Q36 status: BLOCKED — requires validated Q34/Q35 local NAS
DV quantum ceiling: UNDEFINED — will be produced by Q34
Classical ceiling: UNDEFINED — will be produced by Q34 (Q32 space)
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 local NAS validated
Object detection: BLOCKED — out of current roadmap scope
```
