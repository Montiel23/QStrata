# Q33B: Continuous-Variable Quantum NAS Search Space

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q33B — CV Quantum NAS Search Space Design  
**Status:** DESIGN ONLY — no NAS execution in Q33B

---

## 1. Title

Q33B: Continuous-Variable Quantum NAS Search Space

---

## 2. Purpose

Q27 established the baseline CV hybrid benchmark achieving AUROC 0.6708 and F1 0.6283 with 536 trainable parameters on VinDr-SpineXR binary classification. This result used a single fixed architecture: a frozen pretrained C006-D040 backbone feeding a 2-mode, depth-1 GaussianVariationalAnsatz with a classical compression projection and a deterministic first-moment readout. Q26 validated the same architecture across 14 health checks before full training. Q27 is one point in a much larger space of CV quantum head configurations that have not been explored.

Q32 established the classical ceiling methodology and defined the optimized classical search space over compact CNN head architectures. Q33A established the DV quantum NAS methodology and defined the DV search space over qubit count, ansatz depth, rotation families, entanglement topology, and encoding strategies. Q33B now defines the bounded CV quantum search space required before meaningful joint NAS execution in Q34.

By defining the CV search space in Q33B before Q34 executes, Q34 can run joint classical (Q32), DV (Q33A), and CV (Q33B) NAS trials under identical orchestration, datasets, seeds, and constraints — ensuring that the resulting Pareto frontiers are directly comparable. The CV search space must be defined before any CV NAS trial is run, for the same reason the classical search space was defined in Q32 before any classical NAS trial runs: the search space definition is a scientific commitment, not a runtime convenience.

**Q33B defines the search space only. No NAS execution occurs in Q33B.**

No architectures are generated. No trials are configured. No training runs are started. No Gaussian simulation occurs in Q33B. Q34C is the first CV NAS execution phase.

---

## 3. Scientific Motivation

### Gaussian Quantum Representations

CV circuits encode information in continuous quadrature variables — the real and imaginary components of the complex amplitude of each optical mode — rather than in discrete qubit states. This provides a fundamentally different inductive bias for feature compression. Where DV circuits operate on probability amplitudes over a discrete Hilbert space, CV circuits propagate Gaussian probability distributions through symplectic transformations in continuous phase space.

For the QStrata binary classification task, this means: the frozen backbone's 128-dimensional feature vector is compressed and embedded as a displacement in a 2n-dimensional phase space (where n is the number of modes), then evolved through a sequence of Gaussian operations (displacement, squeezing, rotation, beam splitting), and finally read out as the first moment of the evolved Gaussian state. The Gaussian inductive bias — the constraint that the state remains a multivariate Gaussian throughout — is both a stability advantage and an expressivity limit.

### Compact CV Decision Layers

Q27 achieved AUROC 0.6708 and F1 0.6283 with 536 trainable parameters, validating that compact Gaussian heads are feasible and competitive. The Q27 ansatz — 2 modes, depth 1, squeezing cap 1.5 — contributed only 10 trainable parameters (disp_real, disp_imag, squeezing_raw, bs_theta, rot_phi, each shape (1, 2)). The majority of trainable parameters are in the compression projection (516 params: Linear(128, 4)) and the readout layer (10 params: Linear(4, 2)).

This parameter distribution reveals that the Gaussian ansatz itself is extremely compact — far more compact than the DV variational circuit in Q21. CV NAS can explore whether the expressivity of the Gaussian circuit (its capacity to encode useful phase-space transformations) can be improved by varying mode count, depth, squeezing, and readout strategy without sacrificing numerical stability or compactness.

### Structured Covariance Evolution

The covariance matrix of a Gaussian state encodes all pairwise correlations between quadrature variables. Its evolution under displacement, squeezing, beam-splitter, and rotation operations is analytically tractable: each operation corresponds to a symplectic transformation of the covariance matrix. This analytical tractability means that CV circuit behavior is, in principle, more interpretable than DV circuit behavior — the effect of each gate on the covariance structure can be computed explicitly.

This interpretability advantage motivates the Q33B search space design: each searchable dimension corresponds to a meaningful change in the phase-space geometry of the circuit. Increasing mode count expands the phase space dimension. Increasing squeezing cap allows more correlated Gaussian states. Changing beam-splitter topology alters how modes interact. Each change has a well-understood effect on the covariance evolution, making the Q33B search space physically motivated rather than empirically arbitrary.

### Continuous-Variable Feature Encoding

The Q27 encoding strategy uses complex displacement encoding: the compressed backbone features are mapped to complex displacement amplitudes, embedding them directly as first moments of the initial Gaussian state. This is a physically natural encoding: displacement shifts the Gaussian's mean in phase space by the encoded value, preserving the state's Gaussian character and providing a direct linear influence on all subsequent operations.

Q33B explores alternative encoding strategies — re-uploading displacement encoding and hybrid displacement-phase encoding — that may improve the circuit's ability to encode non-linear feature relationships through multiple encoding passes or phase-space rotation of the initial state. Each alternative encoding is physically meaningful and analytically tractable, not empirically guessed.

### Q27 Residual Gap vs Classical Control

Q27 AUROC (0.6708) exceeded Q22 AUROC (0.6625) by +0.0083 with approximately matched parameter counts (536 vs 526). Q27 also led Q22 on F1 by +0.0322. These residuals are small, arise from a single seed, and have no statistical validation. As documented in Q28, they could reflect a genuine CV inductive bias effect, an underexplored classical design space (partially addressed by Q32), or seed-specific variance.

Q33B motivates constrained CV head search because Q27 is a single fixed-point CV architecture. Whether the Q27 AUROC-F1 tradeoff represents the best achievable CV performance — or a suboptimal configuration within a larger unexplored space — is unknown. Q34C will search the Q33B space to produce a CV Pareto frontier that answers this question.

### Importance of Constrained CV Optimization

Unconstrained CV search risks covariance explosion, non-PSD states, and numerically invalid results. A Gaussian circuit with large squeezing parameters and many depth layers can produce covariance matrices whose entries grow without bound, violating the positive semi-definiteness (PSD) requirement and the uncertainty principle bound. The resulting "states" are not valid Gaussian states; the "predictions" they produce are numerical artifacts, not genuine CV classification outputs.

**CV NAS without stability constraints does not produce a meaningful CV ceiling — it produces a space of experiments that fail silently.** Q33B constrains the CV search to configurations where valid Gaussian-state evolution is guaranteed by construction, using bounded squeezing, bounded depth, bounded mode counts, and symplectic parameterization.

**Q33B does not assume quantum advantage.** CV optimization must prioritize numerical stability and valid Gaussian-state evolution before expressivity scaling. The outcome of the Q33B/Q34C comparison against the Q32 classical ceiling is not predetermined.

---

## 4. Optimization Philosophy

Q33B adopts the Q32/Q33A three-tier multi-objective optimization structure in full, extended for CV-specific stability concerns. **Single-metric optimization is forbidden.** CV NAS must jointly optimize performance, compactness, and Gaussian-state stability. A CV circuit that maximizes AUROC through covariance explosion is not a valid result. A CV circuit that maintains perfect Gaussian stability but uses 50,000 trainable parameters is not a compact CV ceiling.

### Primary Objectives (Must Optimize)

Both objectives are tracked for every trial. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance. The primary cross-model comparison metric. The main axis for CV vs. classical ceiling comparison.
- **Maximize F1** — threshold-dependent precision-recall tradeoff at 0.5 threshold. Q27 led Q21 on F1 (+0.0124), making this objective essential for full DV-CV comparison.

### Secondary Objectives (Should Optimize)

Required in every trial result record:

- **Minimize trainable parameter count** — quantum head parameters only; frozen backbone excluded. Q27 used 536 trainable parameters. Compact CV heads are the scientific target.
- **Minimize inference latency** — ms/sample on local hardware. Q27 required 2.15 ms/sample (CUDA backbone + CPU CV circuit), substantially faster than Q21's 54.79 ms/sample DV circuit. This latency advantage of CV over DV is an important optimization axis.
- **Minimize numerical instability** — NaN/inf rate in Gaussian state quantities; covariance PSD violation rate; exploding squeezing event rate. These are CV-specific instability signals that have no direct analog in classical or DV NAS.

### Tertiary Objectives (Consider)

Recorded in every trial result but may not enter primary Pareto analysis:

- **Minimize Gaussian simulator runtime** — wall-clock time for GaussianBackend forward pass alone; separable from backbone inference. Distinguishes covariance computation cost from other latency sources.
- **Minimize peak memory usage** — GPU and CPU peak memory. Gaussian covariance matrices scale as O(n²) in mode count; larger mode counts require more memory.
- **Minimize Gaussian-state instability events** — count of epochs with covariance non-PSD, exploding squeezing, or non-finite state quantities. A trial with 1 instability event in 15 epochs is different from one with 15 instability events in 15 epochs.

### Single-Metric Optimization Is Forbidden

A CV circuit that maximizes AUROC while producing non-PSD covariance states is not a valid CV ceiling — its "predictions" are numerical artifacts. A CV circuit that achieves 0.68 AUROC in one seed but diverges in three others is not deployable. A CV circuit that requires 20 modes and depth 5 to achieve any improvement over Q27 is not compact.

CV NAS must jointly optimize performance, compactness, and Gaussian-state stability. The Pareto framework is the mechanism for handling this three-way tradeoff tractably.

---

## 5. Stability-Aware Pareto Frontier Philosophy

### Non-Dominated Stable CV Solutions

A CV trial is Pareto-optimal (non-dominated) if:
1. It satisfies all hard constraints (valid Gaussian state, no NaN/inf, PSD covariance throughout)
2. No other valid CV trial is strictly better on all tracked objectives simultaneously

Only trials that pass all hard constraints (Gaussian-state validity conditions, see Section 8) enter the Pareto frontier computation. Trials that violate any hard constraint are excluded from the frontier regardless of their AUROC or F1 values.

**Stability-aware Pareto filtering:** A CV solution with high AUROC but invalid or unstable Gaussian states is excluded from the frontier regardless of its apparent performance. This is more stringent than classical or DV NAS, where the primary hard constraint is parameter count and OOM. In CV NAS, a degenerate covariance state can produce arbitrary logit values that appear meaningful but reflect numerical accidents, not genuine quantum circuit behavior. Excluding such trials from the frontier is a scientific integrity requirement.

### Unstable CV Solutions Cannot Dominate Stable Solutions

The stability filter is applied before Pareto dominance is computed. An unstable trial with AUROC 0.80 and F1 0.75 does not dominate a stable trial with AUROC 0.68 and F1 0.63. The unstable trial is excluded from the frontier computation entirely; its apparent performance is treated as scientifically invalid.

### Preserving Compact Stable Solutions

A CV solution that achieves lower AUROC with guaranteed Gaussian stability is a valid frontier point. Stability is a necessary condition for any CV result to be scientifically interpretable. A compact stable CV circuit at AUROC 0.66 is a more meaningful result than an unstable circuit at AUROC 0.70 — the former reflects a genuine Gaussian circuit's performance, the latter may reflect numerical pathology.

### Avoiding Unstable High-Scoring Artifacts

Q34C must not select models that appear to perform well due to numerical artifacts from degenerate covariance states. This failure mode is specific to CV NAS: a non-PSD covariance can still produce finite logits in some forward pass implementations, and those logits may happen to produce high AUROC on the training set while representing physically invalid states. The stability taxonomy (Section 9) is designed to detect and label these cases before they enter the Pareto frontier.

### Preserve Full Stable CV Frontier for Q34

Q34C must preserve the entire stable CV Pareto frontier before any comparative claims are made. A single "best CV model" cannot be selected and the rest discarded before the classical (Q32) and DV (Q33A) frontier comparisons are complete. The CV frontier may dominate the classical and DV frontiers in some regions while being dominated in others.

**The goal of CV NAS is the stable frontier, not a single best model.**

---

## 6. Searchable CV Dimensions

All dimensions are discrete. The Q34C search algorithm samples from these options. Every dimension corresponds to a physically meaningful architectural decision in the CV quantum head design, grounded in the Gaussian quantum circuit formalism.

### Number of Modes (n_modes)

| Option | Phase Space Dim | Notes |
|---|---|---|
| 1 | 2 | Minimal; single quadrature pair; no inter-mode entanglement; simplest covariance evolution |
| 2 | 4 | Q27 validated baseline; two modes with beam-splitter interaction; 4×4 covariance matrix |
| 4 | 8 | Moderate expressivity; 8×8 covariance matrix; supports richer beam-splitter topologies |
| 6 | 12 | Upper bound for Q33B; 12×12 covariance matrix; highest expressivity; highest instability risk |

Small mode counts preserve Gaussian simulator tractability and covariance stability. Q27 used 2 modes as the validated baseline with 15/15 epochs of PASS health checks. Mode counts above 6 are excluded because the 12n×12n covariance computation (for n=6, a 72×72 matrix) becomes computationally prohibitive on local GPU for batch training, and the larger covariance matrices increase the probability of numerical instability during symplectic evolution.

The covariance matrix size scales as (2×n_modes)²: for n_modes=1 this is 2×2; for n_modes=6 this is 12×12. This scaling is polynomial (not exponential like DV qubit count) and remains tractable for all Q33B mode counts.

### CV Depth

| Option | Variational Layers | Notes |
|---|---|---|
| 1 | 1 | Q27 validated baseline; single application of the gate sequence |
| 2 | 2 | Two gate sequence applications; moderate expressivity; acceptable covariance accumulation |
| 3 | 3 | Three gate sequence applications; highest expressivity; must be monitored for covariance drift |

Deep CV circuits are intentionally constrained. Depth > 3 risks covariance explosion during training: each variational layer applies squeezing, rotation, and beam-splitter transformations that can coherently amplify phase-space variance if the squeezing parameters are large. At depth 3 with squeezing_cap 1.5, the covariance diagonal entries can grow to approximately exp(2 × 1.5 × 3) ≈ 8,000 from the vacuum value of 1.0, which is large but still finite. Depth > 3 with bounded squeezing can still produce valid states; however, the search space at depth ≥ 4 is excluded from the Q33B pilot because the accumulated covariance growth at larger depths interacts non-linearly with mode count and beam-splitter topology in ways that are difficult to bound analytically without prior trial data.

### Squeezing Cap

| Option | Max Squeezing Magnitude | Notes |
|---|---|---|
| 0.5 | 0.5 rad | Conservative; minimal covariance amplification; guaranteed stability |
| 1.0 | 1.0 rad | Moderate; covariance diagonal amplification up to exp(2.0) ≈ 7.4× per layer |
| 1.5 | 1.5 rad | Q27 validated baseline; covariance amplification up to exp(3.0) ≈ 20× per layer |
| 2.0 | 2.0 rad | Aggressive; covariance amplification up to exp(4.0) ≈ 55× per layer; highest instability risk |

Large squeezing values destabilize covariance evolution. Q27 validated squeezing_cap=1.5 with stable covariance (diagonal mean ended at 1.0628, close to the vacuum value of 1.0) and squeezing norm growing from 0.0640 to 0.1674 over 15 epochs — well within the 1.5 cap. Values above 2.0 are excluded from Q33B because exp(4.0) ≈ 55× amplification per squeezing operation can produce covariance diagonal entries in the hundreds even at depth 1, creating numerical precision issues in the PSD check and the uncertainty principle bound verification.

The squeezing is parameterized via `tanh(squeezing_raw) × squeezing_cap`, ensuring that the physical squeezing parameter never exceeds the cap regardless of the raw parameter value. This parameterization is inherited from the Q27 GaussianVariationalAnsatz and is required for all Q33B trials.

### Displacement Cap

| Option | Max Displacement Magnitude | Notes |
|---|---|---|
| 0.5 | 0.5 (complex amplitude) | Conservative |
| 1.0 | 1.0 | Moderate |
| 2.0 | 2.0 | Moderate-aggressive; Q27 observed max mu values up to 3.2 in some epochs |
| 4.0 | 4.0 | Aggressive; displacement is more stable than squeezing and can use higher caps |

Displacement is typically more stable than squeezing. Displacement operations shift the Gaussian mean but do not alter the covariance matrix — they have no direct effect on phase-space variance or PSD status. Displacement instability manifests as diverging first moments (mu_final → ∞) rather than covariance explosion, and is detectable via the FINITE_FIRST_MOMENTS check.

Q27 observed max mu values of 3.2152 (epoch 6) while maintaining finite mu throughout training. This suggests that displacement caps up to 4.0 are tractable. Higher displacement caps are permitted relative to squeezing caps because the covariance stability implications are entirely separate.

### Encoding Strategy

| Option | Description |
|---|---|
| `displacement_encoding` | Q27 baseline: backbone features → compression → complex displacement via `encoded = compressed × sqrt(2 × hbar)`; encodes features as displacement amplitudes in phase space |
| `reuploading_displacement_encoding` | Displacement encoding repeated at each depth layer; features re-encoded before each variational block; increases circuit expressivity at cost of additional displacement operations per layer |
| `hybrid_displacement_phase_encoding` | Features split into two groups: one group encoded as displacement amplitudes, one as phase rotation angles; provides two distinct encoding pathways into the Gaussian state |

Q27 validated complex displacement encoding as the stable baseline. Re-uploading displacement encoding extends the Q33A re-uploading concept to CV circuits: features are re-injected into the circuit at each depth layer via fresh displacement operations, providing a persistent classical gradient signal throughout the circuit. Hybrid displacement-phase encoding encodes features into both quadrature displacement (covariance-independent) and phase rotation (covariance-correlated), potentially capturing different feature geometry information through both encoding channels.

### Beam Splitter Topology

| Option | Interaction Pattern | Notes |
|---|---|---|
| `linear` | Mode i connects to mode i+1; no wrap-around; n-1 beam splitters per layer | Most tractable; scales linearly with mode count; no all-to-all interaction |
| `circular` | Linear plus last-to-first mode connection; n beam splitters per layer; Q27 baseline | Q27 validated 2-mode circular topology; for n=2, circular = full connectivity |
| `nearest_neighbor` | Synonym for linear in 1D mode register; specified separately for extension to 2D register layouts | Equivalent to linear in all Q33B mode count configurations |

Topology governs how modes interact via beam-splitter operations. For n_modes=2, circular and linear topologies are identical (both produce one beam splitter connecting the two modes). For n_modes=4, circular adds one additional beam splitter (connecting mode 4 back to mode 1) relative to linear. For n_modes=6, circular adds one beam splitter to the linear topology's five.

Full all-to-all connectivity (n×(n-1)/2 beam splitters) is excluded from Q33B for mode counts > 2 to prevent the number of beam-splitter parameters from scaling quadratically with mode count, which would violate the compact parameterization requirement.

### Readout Strategy

| Option | Output Dimension | Notes |
|---|---|---|
| `first_moments` | 2 × n_modes | Q27 validated baseline: `mu_final` directly used as readout vector; deterministic, no sampling noise |
| `second_moments` | n_modes × (2 × n_modes) | Covariance diagonal or full covariance elements used as readout; adds expressivity by encoding phase-space variance information; higher computational cost |
| `concatenated_moments` | 2 × n_modes + n_modes | Concatenation of first moments and covariance diagonal; combines both readout strategies; most expressive; highest computational cost |

Q27 validated first-moment readout stability: mu_final values remained bounded throughout training (mean ~0.5, max ~3.2), providing a stable 4-dimensional readout vector for the 2-mode case. Second-moment readout uses elements of the covariance matrix as features — encoding information about phase-space uncertainty rather than just the mean displacement. Concatenated moments combines both, providing the richest readout at the cost of a larger output vector dimension.

The output dimension of the readout strategy determines the input dimension of the classical readout layer. For `first_moments` with n_modes=2: output dimension = 4, readout layer = `Linear(4, 2)` (Q27 baseline). For `second_moments` with n_modes=4: covariance diagonal has 8 elements, readout layer = `Linear(8, 2)`. For `concatenated_moments` with n_modes=2: dimension = 4+2 = 6, readout layer = `Linear(6, 2)`.

### Compression Dimension (classical projection before CV encoding)

| Option | Notes |
|---|---|
| 2 | Extremely compact; compresses 128-dim backbone output to 2 features; requires n_modes ≥ 1 |
| 4 | Q27 validated baseline; 128 → 4 compression; used with n_modes=2 and angle encoding |
| 8 | Moderate compression; 128 → 8 features; more information retained; suitable for n_modes=4 |
| 16 | Light compression; 128 → 16 features; highest information retention; suitable for n_modes=6 |

The compression dimension controls the size of the classical linear layer `Linear(128, compression_dim)` that projects the backbone's 128-dimensional output before CV encoding. For displacement encoding, the compression dimension must satisfy `compression_dim = 2 × n_modes` (each mode requires one complex displacement amplitude = two real values). This constraint is enforced at config generation time: for example, n_modes=4 with displacement encoding requires compression_dim=8.

For hybrid displacement-phase encoding, the compression dimension splits between the two encoding pathways, and the constraint becomes `compression_dim ≥ 2 × n_modes`. Compatibility between compression_dim, n_modes, and encoding strategy is enforced by the Q34C trial generator before any trial is run.

### Covariance Parameterization

| Option | Notes |
|---|---|
| `direct_covariance` | Covariance updates applied directly; valid evolution depends on parameter initialization and training trajectory; faster but less stable |
| `symplectic_parameterization` | All transformations parameterized as elements of the symplectic group Sp(2n); guarantees valid symplectic evolution by construction; preferred for numerical stability |

Symplectic parameterization is preferred for numerical stability because it guarantees that every covariance update corresponds to a valid symplectic transformation. A valid symplectic transformation maps any valid Gaussian state (positive semi-definite covariance satisfying the uncertainty principle) to another valid Gaussian state. This means that if the circuit begins with a valid Gaussian state (vacuum: identity covariance), it remains valid after every gate operation — there is no path by which the covariance becomes non-PSD during forward evaluation under symplectic parameterization.

Direct covariance parameterization is faster and more flexible but requires the training trajectory to remain in the valid Gaussian state manifold. If gradient updates push the covariance parameters outside the PSD manifold, a `covariance_not_psd` instability event occurs. For the Q33B pilot, symplectic parameterization is recommended and should be prioritized in random search sampling.

---

## 7. Explicitly Forbidden CV Dimensions

The following are explicitly out of scope for Q33B. Including them would violate the stability, tractability, or interpretability requirements of the CV quantum ceiling experiment.

| Forbidden | Scientific Rationale |
|---|---|
| Unrestricted squeezing (squeezing_cap > 2.0) | Covariance diagonal amplification >55× per layer; high probability of non-PSD states during training; results not comparable to stable circuits |
| Unrestricted mode counts (n_modes > 6) | Covariance matrix dimension >12×12; batch training memory cost prohibitive on local GPU; instability probability grows with phase space dimension |
| Unrestricted covariance dimensions | Same as n_modes > 6; excluded for tractability |
| Adaptive Gaussian graph mutation at runtime | Changes circuit structure during training; breaks reproducibility; cannot be captured in a frozen YAML config; incompatible with Q31 runner |
| Cloud quantum hardware assumptions | Q33B is local simulator-only; photonic quantum hardware (NISQ photonic devices) is blocked until local NAS produces validated results |
| Non-Gaussian photonic gates (photon-number-resolving detection, cat states, Fock-basis operations) | Non-Gaussian gates break the GaussianBackend's analytical covariance propagation; require Fock-basis simulation which is exponentially more expensive; not present in the QStrata CV backend |
| Hybrid large classical heads (> 10K parameters in the head alone) | Violates the compactness principle; not comparable to Q27/Q22 in the ≤600 trainable parameter regime |
| Unrestricted CV depth (depth > 3 without validated stability justification) | Accumulated covariance growth at depth > 3 is analytically unbounded under high squeezing; excluded from pilot without empirical stability data |
| Covariance evolution strategies known to produce non-PSD states | Direct covariance parameter updates without symplectic constraint can produce non-PSD states; any strategy that bypasses the symplectic structure is excluded |

**Q33B focuses exclusively on compact Gaussian simulator-compatible CV systems within the existing QStrata infrastructure.** The stability of the Q27 baseline — 15/15 PASS on all four health checks — is the floor that Q33B's constraint system is designed to preserve across the broader search space.

---

## 8. Valid Gaussian-State Requirements

All six conditions must hold at every forward pass for a trial to be classified as `valid`. A single violation at any point during training triggers the corresponding stability taxonomy label and potentially invalidates the trial.

| Condition | Description | Check Frequency |
|---|---|---|
| Covariance symmetric | `‖cov − covᵀ‖ < 1e-5` within numerical tolerance | Every forward pass |
| Covariance PSD | All eigenvalues of cov ≥ 0 (with tolerance −1e-6 for float32 rounding) | Every forward pass |
| Finite covariance entries | No NaN or inf in any element of the covariance matrix | Every forward pass |
| Finite first moments | No NaN or inf in `mu_final` or any intermediate `mu` | Every forward pass |
| Valid symplectic evolution | `cov + i × hbar/2 × Ω ≥ 0` (uncertainty principle; Ω is the symplectic form) | Every epoch end |
| No NaN/inf during evolution | All intermediate quantum states (mu and cov at each gate step) remain finite | Every forward pass |

These conditions were validated in Q26 (14/14 health checks PASS) and Q27 (15/15 epochs PASS for COV_PSD, COV_SYMMETRIC, QUAD_FINITE, NO_NAN_INF). Q34C must implement equivalent per-forward-pass checks for all CV NAS trials. The Q27 GaussianBackend already performs COV_PSD and COV_SYMMETRIC checks; Q34C must ensure these checks are executed and recorded for every trial.

**Invalid Gaussian states invalidate trials.** These checks must be performed at every forward pass in Q34C trial execution — not only at epoch end or during smoke tests.

---

## 9. CV Stability Taxonomy

The Q34C leaderboard must record the stability taxonomy label for every CV trial. The taxonomy provides diagnostic information about the class of instability that caused a trial failure, enabling systematic analysis of which search space configurations are prone to which failure modes.

| Status | Trigger Condition | Trial Invalidated? |
|---|---|---|
| `valid` | All six Gaussian-state validity conditions pass for all epochs | No |
| `unstable_covariance` | Covariance diagonal entries exceed 1000× the vacuum value (hbar/2 = 1.0 at hbar=2.0) | Yes |
| `covariance_not_psd` | Any eigenvalue of cov < −1e-6 detected at any forward pass | Yes |
| `exploding_squeezing` | Squeezing parameter raw value exceeds squeezing_cap × 2.0 before tanh saturation | Yes |
| `nan_state` | NaN detected in any element of mu or cov at any point during training | Yes |
| `invalid_symplectic` | Symplectic condition `cov + i × hbar/2 × Ω ≥ 0` violated at any epoch end | Yes |
| `timeout` | Trial wall-time ceiling exceeded | Yes |
| `unstable_training` | Gradient norm exceeds 1e4 or training loss is non-finite for ≥ 1 batch | Yes |

All invalidated trials are recorded in the leaderboard with their taxonomy label. They are excluded from the Pareto frontier computation but are not silently discarded. The taxonomy labels provide diagnostic value: a pattern of `covariance_not_psd` failures at high squeezing_cap values would indicate that the squeezing constraint needs to be tightened in subsequent search iterations. A pattern of `timeout` failures at high mode count and depth would indicate that the compute budget per trial needs to be adjusted.

The `unstable_covariance` threshold (1000× vacuum value) is a generous warning threshold designed to catch pathological growth before NaN occurs. A covariance diagonal of 1000 corresponds to approximately exp(2 × squeezing_cap × depth) ≈ 1000, which implies effective squeezing magnitude × depth > 3.45 — achievable at squeezing_cap=1.5 and depth=2 with borderline parameter values.

---

## 10. CV Constraint System

Constraints are divided into hard (violation invalidates the trial) and soft (violation is flagged but the trial is retained) categories, consistent with Q32 and Q33A.

### Hard Constraints — Violation Marks Trial with Taxonomy Label

| Constraint | Value | Enforcement |
|---|---|---|
| Total trainable parameters | ≤ 100,000 | Checked post-model construction; trial aborted if exceeded |
| Local GPU execution only | No cloud infrastructure | No cloud execution permitted; CUDA backbone + CPU GaussianBackend only |
| Gaussian simulation batch compatibility | OOM-free at batch size 4 on local GPU | Runner marks trial `invalid` on OOM exit |
| Covariance PSD required throughout training | No eigenvalue < −1e-6 at any forward pass | Taxonomy: `covariance_not_psd` |
| Finite Gaussian state required at every forward pass | No NaN or inf in mu or cov at any point | Taxonomy: `nan_state` |
| Reproducibility | Seed set from config; git commit captured | Experiments without fixed seed or with `git_commit: unknown` are flagged |
| Batch size 4 compatible | Forward pass must complete at batch size 4 | Required for comparison with Q21, Q22, Q27 benchmarks |

### Soft Constraints — Violation Flagged but Trial Retained

| Constraint | Value | Notes |
|---|---|---|
| Compact mode counts preferred | n_modes ≤ 4 for pilot | n_modes=6 trials are permitted but flagged; pilot prioritizes tractable configurations |
| Shallow CV depth preferred | depth ≤ 2 for pilot | depth=3 trials are permitted but flagged for extra stability monitoring |
| Stable covariance evolution | No instability events of any taxonomy label | Trials with soft-but-not-hard instabilities flagged for investigation |
| Low simulator runtime | < 5× classical head runtime at Q22 latency (~7.4 ms/sample) | Trials exceeding this flagged for latency analysis |

Soft constraint violations are recorded in the result JSON and leaderboard. Trials flagged for soft violations remain in the Pareto frontier analysis with their constraint status visible in all tables.

---

## 11. Covariance Explosion Mitigation Philosophy

### Why Covariance Explosion Is the Primary CV Stability Risk

Barren plateaus are the primary stability risk in DV NAS; covariance explosion is the analogous risk in CV NAS. A squeezing operation with parameter r amplifies the variance in one quadrature direction by exp(2r) while squeezing it by exp(−2r) in the conjugate direction. Applied repeatedly across depth layers, squeezing can produce covariance matrices with entries that grow as exp(2 × squeezing_cap × depth). At squeezing_cap=2.0 and depth=3, this is exp(12.0) ≈ 162,754 — far beyond the range where float32 arithmetic is reliable.

Unlike DV barren plateaus (which produce zero gradients and prevent learning), covariance explosion actively corrupts the learning signal: the inflated covariance entries produce logits that may appear numerically valid but reflect an unphysical Gaussian state. The stability taxonomy is designed to catch this failure mode before it corrupts the Pareto frontier.

### Bounded Squeezing

The squeezing_cap directly limits the maximum squeezing magnitude achievable during training by bounding the output of the `tanh(squeezing_raw)` function. This is the primary defense against covariance explosion. By limiting squeezing to 2.0 at most (Q33B's upper bound), the maximum per-layer covariance amplification is exp(4.0) ≈ 55×. At depth 3, the worst-case covariance amplification is exp(4.0 × 3) = exp(12) ≈ 162,754. This is still potentially problematic, which is why depth=3 with squeezing_cap=2.0 is flagged by the soft constraint system for extra stability monitoring.

### Shallow CV Depth

Fewer transformation layers reduce the accumulated covariance perturbation. Q27 used depth=1 and validated stable covariance across 15 epochs (diagonal mean ending at 1.0628, close to the vacuum value of 1.0). The Q33B depth constraint (≤ 3) limits the worst-case accumulated squeezing to at most exp(2 × squeezing_cap × 3) = exp(12) at squeezing_cap=2.0 — aggressive but still within the PSD condition if the squeezing parameters remain well below their caps during training.

### Constrained Mode Counts

More modes increase the covariance matrix dimension and the probability of instability from mode-to-mode covariance coupling via beam-splitter operations. A beam splitter transforms the covariance of two coupled modes jointly; at larger mode counts with circular or near-all-to-all topology, the covariance matrix entries are coupled across many modes, increasing the probability of a covariance update that violates PSD. Limiting mode counts to ≤ 6 keeps the covariance matrix at most 12×12, which remains tractable for eigenvalue computation and PSD checking at batch size 4.

### Symplectic Parameterization

Parameterizing all circuit transformations through the symplectic group Sp(2n) guarantees valid symplectic evolution by construction. Any symplectic transformation maps a valid Gaussian state to another valid Gaussian state — it cannot produce a non-PSD covariance from a PSD starting point. This is a fundamental property of symplectic linear maps and holds regardless of the squeezing magnitude, depth, or mode count.

In practice, symplectic parameterization is more computationally expensive than direct covariance updates because it requires computing matrix exponentials or Cayley transforms of antisymmetric matrices to generate symplectic matrices from Lie algebra parameters. However, the stability guarantee is unconditional — symplectic trials cannot trigger `covariance_not_psd` or `invalid_symplectic` taxonomy labels by construction.

### Bounded Displacement

Displacement operations affect mean vectors but not covariance matrices. They shift the Gaussian's center in phase space without changing its shape. Bounded displacement caps (≤ 4.0 in Q33B) prevent first-moment divergence (`nan_state` with infinite mu) while allowing sufficient feature encoding range. Q27 observed max mu values of 3.2152, which comfortably fits within the displacement_cap=4.0 bound.

**Q33B intentionally constrains CV search entropy to preserve numerical stability as the primary research integrity requirement.** A small, stable search over a well-defined CV space produces more scientifically interpretable results than a large search over an unstable one.

---

## 12. Experiment Budget Philosophy

### Local Single-GPU Execution

All Q34C CV NAS trials execute sequentially on a single local GPU using the Q31/Q31A runner infrastructure. One CV trial runs at a time. No parallelism across trials. This is a hard constraint enforced by the local-first principle. Sequential single-GPU execution is maximally reproducible and provides the cleanest possible comparison between CV and classical (Q34A) and DV (Q34B) trials.

### Bounded Trial Count

Q34C pilot uses approximately 20–40 trials as a planning reference — slightly fewer than the Q34A classical and Q34B DV budgets, accounting for the higher expected failure rate relative to classical or DV trials. The higher failure rate is inherent to CV NAS: covariance explosion, non-PSD states, and timeout failures are more common than their DV analogs (gradient collapse, OOM). A budget of 20–40 valid trials is achievable within a bounded session even accounting for a 20–30% invalid trial rate.

The exact trial count is a Q34C decision based on the session wall-time budget, the observed per-trial runtime, and the stability profile of early trials in the search.

### Bounded Runtime Per Trial

Each CV trial has a configurable wall-time ceiling. The Q27 benchmark required approximately 82–88 seconds per epoch at batch size 4, or approximately 22 minutes for 15 epochs. Q34C trials should use a per-trial timeout of 30–60 minutes to allow for minor overhead. Trials exceeding the timeout are marked `timeout` and recorded in the leaderboard.

Q27's 15-epoch training at 82-88 s/epoch was substantially faster than Q21's 543-548 s/epoch, because the Gaussian backend is mathematically simpler than the DV circuit simulator for the 2-mode, depth-1 configuration. At higher mode counts or depth, CV trial runtime will increase.

### No Exhaustive Grid Search

Q34C uses random search or lightweight Bayesian sampling over the defined discrete space. No exhaustive grid search. The Q33B search space contains approximately:

4 × 3 × 4 × 4 × 3 × 3 × 3 × 4 × 2 ≈ 41,472 discrete configurations

(n_modes × cv_depth × squeezing_cap × displacement_cap × encoding strategy × beam_splitter_topology × readout_strategy × compression_dim × covariance_parameterization)

Subject to compatibility constraints between compression_dim, n_modes, and encoding strategy, and the n_modes ≤ 4 constraint for full/all-to-all topology. Random sampling provides an unbiased initial sample; Bayesian methods may be introduced in Q35 after the pilot results are available.

**Q34C prioritizes interpretable stable signal over brute-force CV search scale.** A small search over a well-defined stable space produces more useful results than a large search over an unstable one.

---

## 13. Reproducibility Requirements

### Frozen YAML Config Per Trial

Every CV NAS trial produces a frozen YAML config via the Q31 runner framework. The config is written to `experiments/configs/<experiment_id>.yaml` and set read-only (`chmod 444`) before the trial starts. All searchable dimension values are captured in the config: n_modes, cv_depth, squeezing_cap, displacement_cap, encoding_strategy, beam_splitter_topology, readout_strategy, compression_dim, covariance_parameterization.

The stability taxonomy label is recorded in the result JSON and leaderboard, not in the frozen config. The config describes the intended architecture; the taxonomy label describes the outcome.

### Seed Locking

Each CV trial uses a fixed seed from `reproducibility.seed` in the config. The seed is set before dataset loading, model initialization, and augmentation. It is recorded in the result JSON. Gaussian circuit parameter initialization (disp_real, disp_imag, squeezing_raw, bs_theta, rot_phi) is deterministic given the seed. Trials run without a fixed seed are invalid and excluded from all analysis.

### Git Commit Tracking

Every CV trial records the exact code state at execution time via `git rev-parse HEAD` or the `QSTRATA_GIT_COMMIT` env var (validated in Q31A for Docker container execution). Trials with `git_commit: unknown` are flagged in the leaderboard. Published CV NAS results require a clean git commit and must be reproducible from that commit and the frozen config.

### Leaderboard Integrity with Stability Taxonomy

CV trial results are immutable once recorded — including the stability taxonomy label. The runner writes the result JSON and leaderboard entry immediately after trial completion. The taxonomy label cannot be retroactively modified after the trial is recorded; if a trial is misclassified due to a runner bug, a corrected re-run produces a new entry with a new `experiment_id`, and the original entry is retained with `status: retracted`.

This immutability requirement is especially important for CV NAS: the taxonomy label is part of the scientific result, not just a diagnostic annotation. A trial that is retroactively reclassified from `covariance_not_psd` to `valid` without a new execution is scientifically invalid.

### Deterministic Orchestration

Given the same config and seed, re-running any CV trial must produce results within numerical tolerance (AUROC ±0.0001, F1 ±0.0001). Gaussian circuit simulation is deterministic given a fixed seed and fixed parameter initialization. The Q31A reproducibility test (loss_delta = 0.0 across two sequential runs) validated this property for the runner infrastructure; Q34C inherits this validation.

**CV NAS without reproducibility is scientifically invalid.** The Q31/Q31A infrastructure is the prerequisite for Q34C precisely because it enforces frozen configs, seed locking, git tracking, and leaderboard immutability mechanically.

---

## 14. Search Space Interpretability

### Physically Meaningful Architecture Space

Every searchable dimension in Q33B corresponds to a physically meaningful, well-understood decision in Gaussian quantum circuit design:

| Dimension | Physical Meaning |
|---|---|
| n_modes | How many modes of the electromagnetic field are used; how large the phase space is |
| cv_depth | How many times the full gate sequence (displacement → squeeze → rotate → beamsplit) is applied |
| squeezing_cap | How much the circuit can amplify or de-amplify quadrature variance |
| displacement_cap | How far the circuit can shift the Gaussian mean in phase space |
| encoding_strategy | How backbone features are embedded into the initial Gaussian state |
| beam_splitter_topology | How modes interact via beam-splitter coupling |
| readout_strategy | What information from the final Gaussian state is extracted as a classical readout vector |
| compression_dim | How many features from the backbone the circuit receives |
| covariance_parameterization | Whether transformations are guaranteed to preserve Gaussian-state validity by construction |

### Interpretable Circuit Evolution

Moving from one point in the CV search space to another corresponds to a comprehensible Gaussian circuit change:
- Increasing n_modes from 2 to 4 doubles the phase space dimension and allows four-mode beam-splitter interactions
- Increasing cv_depth from 1 to 2 applies the gate sequence twice, accumulating more phase-space transformation
- Reducing squeezing_cap from 1.5 to 0.5 restricts covariance amplification to exp(1.0) ≈ 2.7× per layer — dramatically more conservative than the Q27 baseline
- Switching from first_moments to concatenated_moments readout adds covariance diagonal information to the readout vector, encoding phase-space uncertainty as a classification feature

### Explainable Stable Pareto Tradeoffs

The stable CV Pareto frontier produced by Q34C should be describable in plain language. Expected interpretable frontier statements include:
- "Increasing mode count from 2 to 4 improves F1 at the cost of higher covariance instability events and 3× latency"
- "Re-uploading displacement encoding matches first-moment-only encoding on AUROC with 20% more parameters due to additional encoding layers"
- "Symplectic parameterization produces a cleaner Pareto frontier than direct covariance at equivalent performance levels, with zero covariance_not_psd failures"

### Avoid CV Architecture Chaos

Architectures that mix incompatible covariance parameterizations, encoding strategies with mismatched compression dimensions, or mode counts incompatible with the selected topology are excluded. Specifically:
- Displacement encoding with compression_dim ≠ 2×n_modes is excluded (config generation time constraint)
- Full all-to-all beam-splitter topology is excluded for n_modes > 2 (scales quadratically with mode count)
- Direct covariance parameterization with squeezing_cap > 1.5 is flagged for extra stability monitoring (highest risk combination)

**Scientific interpretability is prioritized over maximal search entropy.** A stable CV search over physically motivated dimensions produces results that can be explained, compared against the classical and DV frontiers, and used to guide the Q35 Pareto analysis.

---

## 15. Relationship to Q32 and Q33A

### Q32 Defines the Compact Classical Pareto Frontier

Q32 defines the search space for compact classical CNN head architectures over the same frozen pretrained backbone (C006-D040) and the same VinDr-SpineXR binary task. Q34A executes this classical search and produces a Pareto frontier over AUROC, F1, parameter count, and latency — the classical ceiling that replaces Q17 and Q22 as the classical reference.

### Q33A Defines the Compact DV Pareto Frontier

Q33A defines the DV quantum search space over the same backbone, task, and orchestration framework. Q34B executes DV NAS trials and produces a DV Pareto frontier. The DV frontier represents the best achievable compact DV circuit performance under the Q33A constraints.

### Q33B Defines the Compact CV Pareto Frontier

Q33B defines the CV Gaussian search space over the same backbone, task, and orchestration framework. Q34C executes CV NAS trials and produces a stable CV Pareto frontier — the CV quantum ceiling. The CV frontier represents the best achievable compact, stable Gaussian circuit performance under the Q33B constraints.

### Q34 and Q35 Compare All Three Frontiers

Q34 evaluates all three search spaces (classical, DV, CV) under identical orchestration, datasets, seeds, and constraints, executing them incrementally (Q34A → Q34B → Q34C). Q35 performs unified Pareto analysis across all three frontiers. The three-way comparison is scientifically meaningful because:
- The backbone, dataset, and evaluation protocol are identical across all three
- The reproducibility infrastructure (Q31/Q31A runner) is identical
- The multi-objective framework (three-tier optimization, Pareto frontier philosophy) is identical
- The only experimental variable is the head architecture family

**CV NAS results must be compared against Q32 classical and Q33A DV results, not only against the Q27 baseline.** Comparing only against Q27 would be methodologically incomplete: Q27 is a single fixed-point CV architecture, not a systematic ceiling. The comparison question is: does the stable CV Pareto frontier produced by Q33B/Q34C dominate, overlap with, or fall below the classical (Q32/Q34A) and DV (Q33A/Q34B) frontiers?

**Q33B does not predetermine the comparison outcome.** The goal is a fair, optimized, three-way comparison under a shared scientific framework.

---

## 16. Local-First CV NAS Philosophy

All Q34C CV NAS trials execute on local GPU only. This is a hard constraint, not a soft preference.

**AWS and Ray remain blocked until local NAS is validated.** The sequencing is:

1. Q33B — design CV quantum search space (this slice; design only)
2. Q33C — NAS execution protocol design (design only; finalizes Q34A/B/C execution plan)
3. Q34A — classical NAS pilot (first execution; Q32 search space)
4. Q34B — DV NAS pilot (second execution; Q33A search space)
5. Q34C — CV NAS pilot (third execution; Q33B search space)
6. Q35 — unified Pareto analysis and NAS hardening (after Q34A–Q34C)
7. Q36 — distributed scaling design (only after Q35 validated)

**Distributed CV orchestration remains blocked.** Gaussian circuit simulation on distributed infrastructure introduces additional failure modes (network-partitioned covariance state synchronization, distributed seed management) that are not present in local sequential execution. These must not be introduced before local CV NAS is validated.

**Local reproducibility and stability validation precedes scaling.** The Q27 benchmark demonstrated that the GaussianBackend produces stable results in sequential local execution. Q34C must validate that the Q33B search space search also produces stable, reproducible results before any distributed extension is considered.

**Execute Q34A (classical) before Q34B (DV) before Q34C (CV).** Do not attempt all three simultaneously on the first execution session. Running all three simultaneously introduces debugging complexity that makes it difficult to attribute any failure to its correct source (classical runner issue? DV barren plateau? CV stability failure?). Sequential execution allows each failure mode to be diagnosed and corrected before the next search phase begins.

---

## 17. Future Phases

The planned execution sequence after Q33B:

```
Q33C  → NAS execution protocol design (design only)
        finalizes the Q34A/Q34B/Q34C execution plan
        defines trial sampling strategy, timeout values, leaderboard format
        defines stability monitoring protocol for CV trials
        no execution; no training; no architectures generated

Q34A  → classical NAS pilot (first execution)
        executes Q32 classical search space on local GPU
        produces classical Pareto frontier
        validates runner infrastructure under full training workload

Q34B  → DV NAS pilot (second execution)
        executes Q33A DV search space on local GPU
        produces DV Pareto frontier
        compares against Q34A classical frontier

Q34C  → CV NAS pilot (third execution)
        executes Q33B CV search space on local GPU
        stability taxonomy recorded for every trial
        produces stable CV Pareto frontier
        compares against Q34A classical and Q34B DV frontiers

Q35   → unified Pareto analysis and NAS hardening
        full three-way comparison: classical (Q34A) vs DV (Q34B) vs CV (Q34C)
        stability taxonomy analysis for CV trials
        identifies which Q33A/Q33B dimensions drive Pareto-optimal quantum performance
        produces NAS hardening recommendations for Q36

Q36   → distributed scaling design (design only; after Q35 validated)
        designs distributed extension of Q34 infrastructure
        no cloud provisioning before this design is approved
        blocked until Q35 produces stable, validated three-frontier comparison
```

**No NAS execution occurs in Q33B or Q33C.** The first CV NAS trial runs in Q34C. Q33B and Q33C are complete when their design documents are committed and the roadmap is updated — not when any trials are run.

**Execute Q34A before Q34B before Q34C.** Do not run all three search spaces simultaneously on the first execution day.

---

## 18. Required Scientific Guardrail

> The QStrata CV quantum NAS program prioritizes scientifically interpretable, reproducible, compact, numerically stable, and valid Gaussian-state optimization before scaling circuit complexity or infrastructure. CV quantum NAS exists to explore controlled tradeoffs, not uncontrolled covariance entropy.

---

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
```
