# Q33B: CV Quantum NAS Search Space Design

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q33B — CV Quantum NAS Search Space Design  
**Status:** COMPLETE — design only; no NAS execution

---

## 1. Title

Q33B: CV Quantum NAS Search Space Design

---

## 2. Context

The QStrata binary benchmarking phase (Q17–Q29) established four frozen compact benchmarks on VinDr-SpineXR binary classification:

| Model | AUROC | F1 | Params |
|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 574 |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 526 |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 |

Q27 is the critical CV finding for Q33B: a 2-mode, depth-1 GaussianVariationalAnsatz CV quantum head over a frozen pretrained backbone (C006-D040) achieved AUROC 0.6708 and F1 0.6283 with 536 trainable parameters. Q26 validated the full CV pipeline across 14 health checks before training; Q27 confirmed all four Gaussian-state health checks (COV_PSD, COV_SYMMETRIC, QUAD_FINITE, NO_NAN_INF) PASS across all 15 training epochs.

Q27 outperformed the parameter-matched classical control Q22 on F1 by +0.0322 and on AUROC by +0.0083. As documented in Q28, Q27 and Q21 exhibit an AUROC-F1 inversion: Q21 (DV) leads on AUROC (+0.0092 over Q27), while Q27 (CV) leads on F1 (+0.0124 over Q21). Neither difference is statistically validated (single seed). Q28 identified the absence of systematic classical and quantum NAS as the critical limitation preventing any quantum advantage interpretation.

Q26 also documented the stability profile of the Q27 baseline: squeezing norm grew from 0.0640 to 0.1674 (controlled); covariance diagonal remained near vacuum (1.009–1.063); max mu displacement reached 3.22 in one epoch; no instability events of any type across 15 epochs. This Q27 stability record is the empirical floor that the Q33B constraint system is designed to preserve across the broader CV search space.

Q32 defined the classical ceiling methodology (compact CNN head search space). Q33A defined the DV quantum search space (qubit count, ansatz depth, rotation families, entanglement topology). Q33B now defines the bounded CV quantum search space required before Q34C can execute CV NAS trials, with **numerical stability as the primary design constraint** distinguishing Q33B from Q32 and Q33A.

**Q33B defines the search space only. No NAS execution occurs in Q33B.** No trial configs are generated. No models are trained. No Gaussian simulation is run. Q34C is the first CV NAS execution phase.

The full CV search space specification is in `docs/architecture/q33b_cv_quantum_nas_search_space.md`.

---

## 3. Why CV Quantum NAS Requires Stability-First Design

### The Covariance Explosion Failure Mode

CV circuits are more susceptible to covariance explosion than DV circuits are to barren plateaus. A squeezing operation with parameter r amplifies quadrature variance by exp(2r). Applied across multiple depth layers with large squeezing values, this produces covariance matrices whose entries grow as exp(2 × squeezing_cap × depth). At squeezing_cap=2.0 and depth=3, the worst-case covariance amplification is exp(12) ≈ 162,754 — far beyond float32 reliability.

Unlike DV barren plateaus (which produce zero gradients and stall learning without corrupting predictions), covariance explosion actively corrupts the learning signal: inflated covariance entries produce logits that appear numerically valid but represent physically invalid Gaussian states. A covariance that is no longer positive semi-definite (PSD) does not correspond to any realizable physical state. Predictions derived from such a state are artifacts, not genuine circuit outputs.

This is the core methodological risk of unconstrained CV search: an unstable trial may appear to perform well due to numerical artifacts in a degenerate covariance state, producing a spuriously high AUROC that disappears on re-evaluation or seed change. The stability-first design of Q33B is designed to prevent this failure mode from corrupting the Q34C Pareto frontier.

### Stability-First Design Prevents Silent Failures

Without stability constraints, unconstrained CV search faces three compounding problems:

**Covariance explosion produces misleading results.** A non-PSD covariance can still produce finite logits in some forward pass implementations. Those logits may happen to produce high AUROC on the training set while representing physically invalid states. The CV stability taxonomy (Section 9) is designed to detect and label these cases explicitly, preventing them from entering the Pareto frontier.

**Unstable trials waste search budget.** A CV trial that fails at epoch 3 due to covariance explosion consumes the same wall-time allocation as a trial that completes 15 epochs. Stability-first design reduces the proportion of invalid trials in the search budget by constraining the configurations most likely to produce covariance explosion.

**Unstable results are not reproducible.** A trial that exhibits covariance explosion on one seed may not exhibit it on another seed — the instability is a function of both the architecture and the parameter initialization trajectory. This makes unstable results seed-specific and unreproducible, violating the Q31/Q31A reproducibility requirement.

**Constrained design produces a meaningful, reproducible Pareto frontier.** A search over configurations where valid Gaussian-state evolution is guaranteed by construction produces results that are reproducible, physically meaningful, and directly comparable to the Q32 classical and Q33A DV frontiers.

---

## 4. Optimization Philosophy

Q33B adopts the Q32/Q33A three-tier multi-objective optimization structure in full, extended for CV-specific stability concerns. **Single-metric optimization is forbidden.** CV NAS must jointly optimize performance, compactness, and Gaussian-state stability.

### Primary Objectives (Must Optimize)

Both objectives are tracked for every valid trial. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance; primary cross-model comparison metric
- **Maximize F1** — threshold-dependent precision-recall tradeoff at 0.5 threshold; Q27 led Q21 on F1, making this objective essential for full DV-CV comparison

### Secondary Objectives (Should Optimize)

Required in every trial result record:

- **Minimize trainable parameter count** — quantum head parameters only; frozen backbone excluded; compact CV heads are the scientific target
- **Minimize inference latency** — ms/sample on local hardware; Q27 achieved 2.15 ms/sample (substantially faster than Q21's 54.79 ms/sample); CV latency advantage over DV is an important optimization axis
- **Minimize numerical instability** — NaN/inf rate; covariance PSD violation rate; exploding squeezing event rate; recorded via stability taxonomy

### Tertiary Objectives (Consider)

Recorded but may not enter primary Pareto analysis:

- **Minimize Gaussian simulator runtime** — GaussianBackend forward pass time; separable from backbone inference
- **Minimize peak memory usage** — covariance matrix scales O(n²) with mode count; memory grows with n_modes
- **Minimize Gaussian-state instability events** — count of epochs with any taxonomy label other than `valid`

A CV circuit that maximizes AUROC through covariance explosion is not a valid result. A CV circuit with perfect Gaussian stability but 50,000 trainable parameters is not compact. CV NAS must jointly optimize across all three tiers.

---

## 5. Stability-Aware Pareto Frontier Philosophy

### Non-Dominated Stable CV Solutions

Only trials that satisfy all hard constraints (Section 10) — specifically all six valid Gaussian-state conditions (Section 8) — enter the Pareto frontier computation. A CV trial that produces an invalid Gaussian state at any point during training is assigned a stability taxonomy label and excluded from the frontier, regardless of its apparent AUROC or F1.

**Stability-aware Pareto filtering:** A CV solution with high AUROC but invalid Gaussian states is excluded regardless of its apparent performance. This is more stringent than the Q32 or Q33A Pareto filter, reflecting the additional failure mode unique to CV circuits: covariance explosion can produce numerically plausible but physically invalid predictions.

### Unstable Solutions Cannot Dominate Stable Solutions

The stability filter is applied before Pareto dominance is computed. An unstable trial with AUROC 0.80 does not dominate a stable trial with AUROC 0.68. The unstable trial is excluded from the frontier computation entirely.

### Preserve Compact Stable Solutions on the Frontier

A CV solution that achieves lower AUROC with guaranteed Gaussian stability is a valid frontier point. Stability is a necessary condition for any CV result to be scientifically interpretable. Compact stable solutions at modest AUROC occupy an important region of the frontier: they demonstrate that CV circuits can produce valid, reproducible, compact results — even if not state-of-the-art AUROC.

### Preserve Full Stable CV Frontier for Q34

Q34C must preserve the entire stable CV Pareto frontier before any comparative claims are made. The CV frontier may dominate the classical (Q34A) and DV (Q34B) frontiers in some regions (e.g., very compact stable circuits at F1) while being dominated in others (e.g., high-AUROC configurations). Only the full frontier comparison in Q35 reveals the pattern.

**The goal of CV NAS is the stable frontier, not a single best model.**

---

## 6. Searchable CV Dimensions

All dimensions are discrete. The Q34C search algorithm samples from these options.

| Dimension | Options |
|---|---|
| **Number of modes (n_modes)** | 1, 2, 4, 6 |
| **CV depth** | 1, 2, 3 |
| **Squeezing cap** | 0.5, 1.0, 1.5, 2.0 |
| **Displacement cap** | 0.5, 1.0, 2.0, 4.0 |
| **Encoding strategy** | `displacement_encoding`, `reuploading_displacement_encoding`, `hybrid_displacement_phase_encoding` |
| **Beam splitter topology** | `linear`, `circular`, `nearest_neighbor` |
| **Readout strategy** | `first_moments`, `second_moments`, `concatenated_moments` |
| **Compression dimension** (classical projection before CV encoding) | 2, 4, 8, 16 |
| **Covariance parameterization** | `direct_covariance`, `symplectic_parameterization` |

**Search space size (rough estimate):** 4 × 3 × 4 × 4 × 3 × 3 × 3 × 4 × 2 ≈ 41,472 discrete configurations, subject to compatibility constraints between compression_dim, n_modes, and encoding strategy. Random or Bayesian sampling over 20–40 valid trials is a small but interpretable pilot sample.

**Q27 position in this space:** n_modes=2, cv_depth=1, squeezing_cap=1.5, displacement_cap (not explicitly bounded in Q27; effective max mu observed ~3.2), encoding_strategy=`displacement_encoding`, beam_splitter_topology=`circular`, readout_strategy=`first_moments`, compression_dim=4, covariance_parameterization=`direct_covariance` (Q27 used tanh-bounded squeezing without full symplectic group parameterization).

**Compatibility constraints enforced at config generation time:**
- For `displacement_encoding`: compression_dim must equal 2 × n_modes
- Full all-to-all beam-splitter topology excluded for n_modes > 2 (scales quadratically with mode count)
- `none` readout excluded (requires classical readout layer)

Full dimension specifications with physical rationale: `docs/architecture/q33b_cv_quantum_nas_search_space.md` Section 6.

---

## 7. Forbidden CV Dimensions

The following are explicitly out of scope for Q33B, with scientific rationale:

| Forbidden | Scientific Rationale |
|---|---|
| Unrestricted squeezing (squeezing_cap > 2.0) | Covariance amplification >55× per layer; high probability of non-PSD states during training; results not comparable to stable circuits |
| Unrestricted mode counts (n_modes > 6) | Covariance matrix >12×12; batch training memory cost prohibitive on local GPU; instability probability grows with phase space dimension |
| Unrestricted covariance dimensions | Same as n_modes > 6; excluded for tractability and stability |
| Adaptive Gaussian graph mutation at runtime | Changes circuit structure during training; breaks reproducibility; incompatible with frozen YAML config and Q31 runner |
| Cloud quantum hardware assumptions | Q33B is local simulator-only; photonic quantum hardware blocked until local NAS produces validated results |
| Non-Gaussian photonic gates (photon-number-resolving detection, cat states, Fock-basis operations) | Break the GaussianBackend's analytical covariance propagation; require exponentially expensive Fock-basis simulation not present in QStrata |
| Hybrid large classical heads (> 10K parameters in the head alone) | Violates compactness principle; not comparable to Q27/Q22 in the ≤600 trainable parameter regime |
| Unrestricted CV depth (depth > 3 without validated stability justification) | Accumulated covariance growth analytically unbounded at depth > 3 under high squeezing; excluded from pilot |
| Covariance evolution strategies known to produce non-PSD states | Any strategy bypassing the symplectic structure is excluded |

**Rationale:** Q33B focuses exclusively on compact Gaussian simulator-compatible CV systems within the existing QStrata infrastructure. The stability of Q27 — 15/15 epoch PASS on all health checks — is the empirical floor that these constraints are designed to preserve.

---

## 8. Valid Gaussian-State Requirements

All six conditions must hold at every forward pass for a trial to be classified as `valid`. A single violation at any point during training triggers the corresponding stability taxonomy label and invalidates the trial.

| Condition | Description |
|---|---|
| Covariance symmetric | `‖cov − covᵀ‖ < 1e-5` within numerical tolerance |
| Covariance PSD | All eigenvalues of cov ≥ 0 (tolerance −1e-6 for float32 rounding) |
| Finite covariance entries | No NaN or inf in any element of the covariance matrix |
| Finite first moments | No NaN or inf in `mu_final` or any intermediate `mu` |
| Valid symplectic evolution | `cov + i × ℏ/2 × Ω ≥ 0` (uncertainty principle; Ω is the symplectic form) |
| No NaN/inf during evolution | All intermediate quantum states (mu and cov at each gate step) remain finite |

These conditions were validated in Q26 (14/14 PASS) and Q27 (15/15 epochs PASS for COV_PSD, COV_SYMMETRIC, QUAD_FINITE, NO_NAN_INF). Q34C must implement equivalent per-forward-pass checks for all CV NAS trials.

**Invalid Gaussian states invalidate trials.** These checks must be performed at every forward pass in Q34C trial execution — not only at epoch end.

---

## 9. CV Stability Taxonomy

The Q34C leaderboard must record the stability taxonomy label for every CV trial. The taxonomy provides diagnostic information about the class of instability that caused a trial failure.

| Status | Trigger Condition | Trial Invalidated? |
|---|---|---|
| `valid` | All six Gaussian-state validity conditions pass for all epochs | No |
| `unstable_covariance` | Covariance diagonal entries exceed 1000× vacuum value (1000.0 at hbar=2.0) | Yes |
| `covariance_not_psd` | Any eigenvalue of cov < −1e-6 at any forward pass | Yes |
| `exploding_squeezing` | Squeezing parameter raw value exceeds squeezing_cap × 2.0 before tanh saturation | Yes |
| `nan_state` | NaN detected in any element of mu or cov at any point during training | Yes |
| `invalid_symplectic` | Symplectic condition `cov + i × ℏ/2 × Ω ≥ 0` violated at any epoch end | Yes |
| `timeout` | Trial wall-time ceiling exceeded | Yes |
| `unstable_training` | Gradient norm exceeds 1e4 or training loss is non-finite for ≥ 1 batch | Yes |

All invalidated trials are recorded in the leaderboard with their taxonomy label. They are excluded from the Pareto frontier but not silently discarded. Taxonomy patterns across trials provide diagnostic signal: repeated `covariance_not_psd` at high squeezing_cap indicates the squeezing constraint needs tightening; repeated `timeout` at high n_modes and depth indicates the compute budget needs adjustment.

---

## 10. Constraint System

### Hard Constraints — Violation Marks Trial with Taxonomy Label

| Constraint | Value |
|---|---|
| Total trainable parameters | ≤ 100,000 |
| Local GPU execution only | No cloud infrastructure; CUDA backbone + CPU GaussianBackend only |
| Gaussian simulation batch compatibility | OOM-free at batch size 4 on local GPU |
| Covariance PSD required throughout training | No eigenvalue < −1e-6 at any forward pass; taxonomy: `covariance_not_psd` |
| Finite Gaussian state required at every forward pass | No NaN or inf in mu or cov at any point; taxonomy: `nan_state` |
| Reproducibility | Seed set from config; git commit captured |
| Batch size 4 compatible | Required for comparison with Q21, Q22, Q27 benchmarks |

Hard constraint violations cause the trial to be recorded with the appropriate taxonomy label and excluded from the Pareto frontier.

### Soft Constraints — Violation Flagged but Trial Retained

| Constraint | Value |
|---|---|
| Compact mode counts preferred | n_modes ≤ 4 for pilot |
| Shallow CV depth preferred | depth ≤ 2 for pilot |
| Stable covariance evolution | No instability events of any taxonomy label |
| Low simulator runtime | < 5× classical head runtime (~7.4 ms/sample) |

Soft constraint violations are recorded in the result JSON and leaderboard with their constraint status visible.

---

## 11. Covariance Explosion Mitigation Philosophy

CV NAS faces covariance explosion as its primary stability risk, analogous to barren plateaus in DV NAS. Q33B mitigates this through five structural constraints:

**Bounded squeezing (squeezing_cap ≤ 2.0):** Squeezing is parameterized via `tanh(squeezing_raw) × squeezing_cap`, directly limiting covariance amplification to exp(4.0) ≈ 55× per layer. The cap is enforced by construction — the squeezing parameter can never exceed the cap regardless of raw parameter values. Q27 validated squeezing_cap=1.5 with stable covariance (diagonal ending at 1.0628 after 15 epochs).

**Shallow CV depth (depth ≤ 3):** Fewer transformation layers limit the accumulated covariance perturbation. Q27 used depth=1 and produced stable covariance throughout training. At depth=3 with squeezing_cap=2.0, the worst-case accumulated amplification is exp(12) ≈ 162,754 — this combination is flagged by the soft constraint system for extra stability monitoring.

**Constrained mode counts (n_modes ≤ 6):** More modes increase the covariance matrix dimension (2n × 2n) and the probability of instability from mode-to-mode coupling via beam splitters. Limiting n_modes ≤ 6 keeps the covariance matrix at most 12×12, tractable for eigenvalue PSD checking at batch size 4.

**Symplectic parameterization:** Parameterizing all circuit transformations through the symplectic group Sp(2n) guarantees valid symplectic evolution by construction — any symplectic transformation maps a valid Gaussian state to another valid Gaussian state. Symplectic trials cannot trigger `covariance_not_psd` or `invalid_symplectic` taxonomy labels by construction, eliminating the most common CV instability failure mode.

**Bounded displacement (displacement_cap ≤ 4.0):** Displacement affects mean vectors but not covariance matrices. Bounded displacement caps prevent first-moment divergence while allowing sufficient feature encoding range. Q27 observed max mu values of 3.22, fitting comfortably within displacement_cap=4.0.

**Q33B intentionally constrains CV search entropy to preserve numerical stability as the primary research integrity requirement.** A small stable search produces more scientifically interpretable results than a large unstable one.

---

## 12. Experiment Budget Philosophy

### Local Single-GPU Execution

All Q34C CV NAS trials execute sequentially on a single local GPU using the Q31/Q31A runner infrastructure. One CV trial runs at a time. No parallelism. This is a hard constraint enforced by the local-first principle.

### Bounded Trial Count

Q34C pilot uses approximately 20–40 trials as a planning reference — slightly fewer than Q34A (classical) and Q34B (DV) budgets, accounting for the higher expected invalid trial rate due to CV stability failures. A 20–30% invalid trial rate is expected at aggressive squeezing and mode count configurations. Even with this failure rate, 20–40 trial attempts should produce 15–32 valid trials — sufficient for a meaningful Pareto frontier.

The exact trial count is a Q34C decision based on session wall-time, observed per-trial runtime, and the stability profile of early trials.

### Bounded Runtime Per Trial

Each CV trial has a configurable wall-time ceiling. Q27's 15-epoch training required approximately 22 minutes (82-88 s/epoch). Q34C trials should use a 30–60 minute timeout — generous enough for 15 full epochs at Q27 configuration, accounting for overhead and potential slowdown at higher mode counts or depth.

### No Exhaustive Grid Search

Q34C uses random search or lightweight Bayesian sampling. The Q33B search space contains approximately 41,472 discrete configurations (subject to compatibility constraints). Random sampling provides an unbiased initial pilot sample. Bayesian methods may be introduced in Q35 after the pilot results inform the instability profile of the search space.

**Q34C prioritizes interpretable stable signal over brute-force CV search scale.**

---

## 13. Reproducibility Requirements

**Frozen YAML Config Per Trial:** Every CV NAS trial produces a frozen YAML config via the Q31 runner framework, written to `experiments/configs/<experiment_id>.yaml` and set read-only (`chmod 444`) before execution. All nine searchable dimension values are captured. The stability taxonomy label is recorded in the result JSON and leaderboard, not in the frozen config.

**Seed Locking:** Each CV trial uses a fixed seed from `reproducibility.seed`. The seed determines initial parameter values for disp_real, disp_imag, squeezing_raw, bs_theta, rot_phi and all other trainable parameters. The seed is recorded in the result JSON.

**Git Commit Tracking:** Every CV trial records the exact code state via `git rev-parse HEAD` or `QSTRATA_GIT_COMMIT` env var (validated in Q31A). Trials with `git_commit: unknown` are flagged. Published CV NAS results require a clean commit.

**Leaderboard Integrity with Stability Taxonomy:** CV trial results are immutable once recorded — including the stability taxonomy label. The taxonomy label is part of the scientific result and cannot be retroactively modified. If a trial is misclassified due to a runner bug, a corrected re-run produces a new entry; the original is retained with `status: retracted`.

**Deterministic Orchestration:** Given the same config and seed, re-running any CV trial must produce results within tolerance (AUROC ±0.0001, F1 ±0.0001). The Q31A reproducibility test (loss_delta = 0.0 across two sequential runs) validated the runner infrastructure. Gaussian circuit simulation is deterministic given fixed seed and parameter initialization.

The Q31/Q31A infrastructure is the prerequisite for Q34C because it enforces frozen configs, seed locking, git tracking, and leaderboard immutability mechanically. **CV NAS without reproducibility is scientifically invalid.**

---

## 14. Relationship to Q32 and Q33A

Q32 defines the compact classical Pareto frontier over CNN head architectures. Q33A defines the compact DV Pareto frontier over quantum circuit head architectures. Q33B defines the compact CV Pareto frontier over Gaussian circuit head architectures. Q34 evaluates all three under identical orchestration, datasets, seeds, and constraints, executing them incrementally: Q34A (classical) → Q34B (DV) → Q34C (CV). Q35 performs unified three-way Pareto analysis.

**CV NAS results must be compared against Q32 classical and Q33A DV results, not only against the Q27 baseline.** Q27 is a single fixed-point CV architecture, not a systematic ceiling. The scientific question is not whether Q33B/Q34C produces a circuit better than Q27 — it is whether the stable CV Pareto frontier dominates, overlaps with, or falls below the classical (Q32/Q34A) and DV (Q33A/Q34B) frontiers.

Comparing only against Q27 would be methodologically incomplete for the same reason that comparing Q22 against Q17 without Q32 NAS was insufficient: a single fixed-point comparison cannot distinguish architectural effects from search space effects. The full Pareto frontier comparison in Q35 answers the architectural question definitively.

The Q34 comparison must be conducted on identical axes: same backbone (frozen C006-D040), same dataset (VinDr-SpineXR binary, canonical split), same evaluation metrics (AUROC, F1, parameter count, latency), same orchestration (Q31/Q31A runner). Any difference in Pareto frontier position between CV and classical/DV architectures is attributable to the head architecture, not to experimental conditions.

---

## 15. Limitations

**No NAS execution in this slice.** Q33B is design only. No trial configs are generated, no models are trained, no Gaussian simulation is run.

**No AWS or Ray.** All execution is local. Distributed infrastructure remains blocked until Q34 validates local NAS.

**No distributed execution.** Q34C is local, sequential, single-GPU.

**No generated architectures.** The search space definition does not produce candidate architectures. Candidate configs are generated by the Q34C search algorithm at execution time.

**No multiclass work.** Multiclass benchmarking is blocked until Phase 3 (Q32 ceiling), Phase 4 (Q33A/Q33B quantum NAS), and Phase 5 (Q34A–Q34C optimized binary release) are all complete.

**No object detection work.** Object detection is outside the current QStrata roadmap scope.

**No hardware quantum execution.** Q33B assumes local Gaussian circuit simulation on CPU (GaussianBackend). Physical photonic quantum hardware execution is blocked until local NAS produces validated results.

**Single-seed evaluation in Q34C pilot.** The Q34C pilot executes with a single seed per trial. Multi-seed validation of CV NAS results is deferred to Q35. All Q34C CV Pareto frontier results are therefore single-seed point estimates without confidence intervals.

**Stability taxonomy coverage may be incomplete.** The eight taxonomy labels in Section 9 cover the known failure modes from Q26/Q27 health check analysis. Novel instability modes may emerge at higher mode counts or depths not previously tested. Q34C may need to add taxonomy labels based on observed failure patterns.

**Search space is a planning artifact.** Exact search space dimensions may be refined by Q34C based on hardware constraints, preliminary trial stability profiles, and wall-time budgets. Refinement in Q34C does not require a new design slice — it requires documenting the changes in the Q34C report.

---

## 16. Next Slice

**Q33C — NAS Execution Protocol Design**

Q33C is also design only. It will finalize the Q34A/Q34B/Q34C execution plan: trial sampling strategy, timeout values, leaderboard format, stability monitoring protocol for CV trials, and the incremental execution ordering (Q34A classical first, Q34B DV second, Q34C CV third). Q33C does not execute any NAS trials.

**Q34A is the first local NAS execution phase.** Q34A executes classical NAS over the Q32 search space. Q34B executes DV NAS over the Q33A search space. Q34C executes CV NAS over the Q33B search space. All three are executed sequentially — do not attempt all three simultaneously on the first execution session.

```
Q33B status: COMPLETE — design only; no NAS execution
Q33C status: NEXT — NAS Execution Protocol Design (design only)
Q34A status: PLANNED — classical NAS pilot (first execution)
Q34B status: PLANNED — DV NAS pilot (second execution)
Q34C status: PLANNED — CV NAS pilot (third execution)
Q35 status: PLANNED — unified Pareto analysis (after Q34A–Q34C)
Q36 status: BLOCKED — requires validated Q35 three-frontier comparison
CV quantum ceiling: UNDEFINED — will be produced by Q34C
DV quantum ceiling: UNDEFINED — will be produced by Q34B
Classical ceiling: UNDEFINED — will be produced by Q34A
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 local NAS validated
Object detection: BLOCKED — out of current roadmap scope
```
