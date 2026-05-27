# Q33A: Discrete-Variable Quantum NAS Search Space

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q33A — DV Quantum NAS Search Space Design  
**Status:** DESIGN ONLY — no NAS execution in Q33A

---

## 1. Title

Q33A: Discrete-Variable Quantum NAS Search Space

---

## 2. Purpose

Q21 established the baseline DV hybrid benchmark achieving AUROC 0.6800 and F1 0.6159 with 574 trainable parameters on VinDr-SpineXR binary classification. This result used a single fixed architecture: a frozen pretrained C006-D040 backbone feeding a 4-qubit variational ansatz with a compression projection and a linear readout. Q21 is one point in a much larger space of DV quantum head configurations that have not been explored.

Q32 established the classical ceiling methodology and defined the optimized classical search space over compact CNN head architectures. The Q32 search space covers backbone family, channel width, depth, compression dimension, activation, normalization, dropout, and pooling — producing a multi-objective Pareto frontier that replaces the unoptimized Q17 and Q22 single-point classical references. Q34 will execute classical NAS over the Q32 space and produce a Pareto frontier that serves as the formal classical ceiling.

Q33A now defines the bounded DV quantum search space required for meaningful comparison against optimized classical baselines in Q34. By defining the DV search space in Q33A and the CV search space in Q33B before Q34 executes, Q34 can run joint classical and quantum NAS trials under identical orchestration, datasets, seeds, and constraints — ensuring that the resulting Pareto frontiers are directly comparable.

**Q33A defines the search space only. No NAS execution occurs in Q33A.**

No architectures are generated. No trials are configured. No training runs are started. Q34 is the first DV NAS execution phase.

---

## 3. Scientific Motivation

### Compact Quantum Representations

Q21 achieved competitive results with 574 trainable parameters on a binary medical imaging classification task. This demonstrates that quantum heads can encode useful decision boundaries in very small parameter budgets. The 4-qubit variational ansatz with depth-1 entanglement learned a structured transformation of the 128-dimensional frozen backbone output, producing a non-trivial classification boundary without the representational capacity of classical MLPs.

Compact quantum representations matter for QStrata's core research question: under the same frozen backbone and compact parameter budget, can quantum heads systematically outperform classical heads? Q21 showed one data point suggesting yes (AUROC +0.0175 over Q22). Q33A defines the space to explore whether this advantage is architecture-specific, generalizes across DV design choices, or disappears under systematic classical comparison.

### Frozen Backbone + Compact Quantum Decision Layer

The QStrata DV architecture uses a frozen pretrained backbone (C006-D040) as a fixed feature extractor, feeding its 128-dimensional output through a classical compression projection into the quantum circuit. The quantum head learns a structured transformation of compressed backbone features; the backbone itself receives no gradient updates. This design decouples the representation learning problem (solved by pretraining) from the decision boundary problem (explored by NAS).

Under this factorization, the quantum head's role is to learn a compact, structured mapping from compressed features to binary class probabilities. Q33A searches over the dimensions of this mapping: how many qubits, how deep the circuit, how the input is encoded, how entanglement is structured, and how the quantum state is read out.

### Entanglement as Structured Feature Interaction

Unlike classical linear projections, entangled quantum circuits encode structured correlations between input features through the action of two-qubit gates on shared quantum states. A CNOT or controlled-rotation gate between qubits i and j couples the information content of those qubits in a way that classical tensor products cannot directly replicate without explicit interaction terms.

This structured coupling may be beneficial for feature geometries that exhibit correlated structure in the compressed backbone representation space. Whether the specific feature geometry of the VinDr-SpineXR binary task benefits from quantum entanglement — or whether classical linear projections are sufficient to capture the relevant correlations — is precisely what the Q33A/Q34 comparison against the Q32 classical ceiling is designed to test.

### Residual Unexplained Gap

Q21 outperformed the parameter-matched classical control Q22 by AUROC +0.0175 and F1 +0.0198 in a single-seed experiment. The source of this residual gap is unknown. It could reflect:

1. A quantum inductive bias effect — the DV circuit's entanglement structure captures feature interactions that the Q22 MLP cannot represent with equal parameter efficiency
2. The Q22 architecture being a weak classical baseline — a suboptimal single-configuration classical head in a large unexplored space
3. Seed-specific variance — a particular random seed that happens to favor the DV head's gradient landscape

Q33A motivates constrained quantum head search precisely because interpretation 1 cannot be separated from interpretations 2 and 3 with only the Q21/Q22 single-point comparison. Q34 will produce both a classical Pareto frontier (from Q32) and a DV Pareto frontier (from Q33A), enabling a multi-point comparison that can distinguish architectural effects from single-configuration variance.

### Importance of Constrained Quantum Optimization

Unconstrained DV search risks barren plateaus, simulator intractability, and uninterpretable results. A quantum circuit with 20 qubits and depth 10 cannot be simulated efficiently on a local GPU; its gradient landscape will exhibit exponentially vanishing gradients (barren plateau effect); its results will not generalize to any physically realizable device. Unconstrained search does not produce a valid DV ceiling — it produces an infeasible architecture space.

Q33A constrains the DV search to circuits that are simulator-tractable on local GPU, parameterically compact, numerically stable, and scientifically interpretable. These constraints are not limitations on scientific ambition — they are requirements for scientifically valid comparison against the Q32 classical ceiling.

**Q33A does not assume quantum advantage.** Q33A defines the controlled DV search space required for fair comparison against the Q32 classical frontier. The outcome of that comparison is unknown and is not prejudged by the design of the search space.

---

## 4. Optimization Philosophy

Q33A adopts the Q32 three-tier multi-objective optimization structure in full, extended for quantum-specific concerns. No single-metric optimization is permitted. Single-metric optimization — for example, maximizing AUROC while ignoring parameter count, latency, or numerical stability — would produce a quantum ceiling that is not comparable to the Q32 classical ceiling, which is optimized across all three tiers simultaneously.

### Primary Objectives (Must Optimize)

Both objectives are tracked for every trial. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance. The primary cross-model comparison metric. Stable under the class imbalance present in VinDr-SpineXR. The main axis for DV vs. classical ceiling comparison.
- **Maximize F1** — threshold-dependent precision-recall tradeoff computed at the 0.5 threshold. Captures clinical relevance of the operating point. Q21 exhibited AUROC-F1 dissociation relative to Q27 (DV led on AUROC, CV led on F1), making both objectives essential for the full comparison.

### Secondary Objectives (Should Optimize)

Included in extended Pareto analysis. Required in every trial result record:

- **Minimize trainable parameter count** — quantum head parameters only; frozen backbone excluded. Compact quantum heads are the scientific target. A DV circuit that requires 50,000 trainable parameters to achieve Q21 AUROC is not a compact quantum head.
- **Minimize inference latency** — ms/sample on local hardware. Q21 required 54.79 ms/sample due to CPU-bound quantum circuit simulation. DV NAS may discover circuit configurations with better latency-performance tradeoffs. Latency is tracked to prevent the Pareto frontier from admitting circuits with high AUROC but unacceptable deployment cost.
- **Minimize numerical instability** — NaN/inf rate across training epochs; gradient health metrics. Q21 achieved zero NaN/inf across 15 epochs. DV NAS must maintain this standard. Gradient norm collapse (barren plateau) and divergence (gradient explosion) are both tracked as instability signals.

### Tertiary Objectives (Consider)

Recorded in every trial result but may not enter primary Pareto analysis:

- **Minimize quantum simulator runtime** — wall-clock time for one forward pass of the quantum circuit alone; separable from the backbone forward pass. Distinguishes circuit-complexity cost from other latency sources.
- **Minimize memory usage** — GPU and CPU peak memory during forward and backward pass. Prevents OOM-adjacent configurations from entering the Pareto frontier without flagging.
- **Minimize barren plateau behavior** — gradient variance collapse metric: ratio of final-epoch theta gradient norm to initial-epoch theta gradient norm. Circuits that begin with healthy gradients but collapse over training are candidates for barren plateau pathology.

### Single-Metric Optimization Is Forbidden

A DV circuit that maximizes AUROC while using 10,000 trainable parameters is not a compact DV ceiling; it is a different class of experiment. A DV circuit that achieves 0.70 AUROC in one of five seeds but crashes with NaN in the other four is not deployable. A DV circuit that requires 500 ms/sample due to circuit simulation overhead is not comparable to a classical GPU head at 1.5 ms/sample.

Quantum NAS must optimize compactness and stability simultaneously. The Pareto framework is the mechanism for making this tractable: identify non-dominated solutions across all tracked objectives, present the frontier, and let scientific judgment operate on the frontier — not on a scalar aggregate of weighted objectives.

---

## 5. DV Pareto Frontier Philosophy

### Non-Dominated DV Solutions

A DV trial is Pareto-optimal (non-dominated) if no other evaluated DV trial is strictly better on all tracked objectives simultaneously. The DV Pareto frontier is the set of all non-dominated DV trials. In operational terms: a compact DV architecture is Pareto-optimal if no other evaluated DV architecture simultaneously achieves higher AUROC, higher F1, fewer parameters, and lower latency.

A trial that achieves lower AUROC than another but uses 5× fewer parameters and runs 10× faster may remain on the Pareto frontier if no single trial dominates it on all four axes simultaneously.

### Compact Quantum Tradeoffs

The DV Pareto frontier is expected to exhibit interpretable tradeoffs:
- Deeper circuits (more entangling layers) may achieve higher AUROC at the cost of higher latency and greater barren plateau risk
- Larger qubit counts expand the Hilbert space but increase simulation cost exponentially
- Re-uploading strategies can improve gradient flow and AUROC at the cost of multiple circuit passes and increased runtime
- Shallow circuits with low qubit counts occupy the low-latency, low-parameter region of the frontier — directly comparable to the most compact classical head configurations from Q32

These tradeoffs must be preserved in the frontier, not collapsed to a single scalar winner. Different positions on the DV frontier correspond to different engineering decisions about where to invest the quantum computational budget.

### The Quantum Pareto Frontier Must Be Compared Against the Classical Ceiling

The DV Pareto frontier produced by Q34 must be compared directly against the classical Pareto frontier from Q32 — not against Q17 (23,650 params, random init, full training). Comparing quantum NAS results against Q17 would be methodologically invalid: Q17's weakness is primarily attributable to random initialization and large parameter count, not to the classical architecture family. An unoptimized classical baseline is not the correct reference for an optimized quantum baseline.

The comparison must be conducted on the same Pareto frontier axes: AUROC, F1, parameter count, latency. A DV configuration that is not dominated by any classical configuration on these four axes is a meaningful result. A DV configuration that is dominated by multiple classical configurations is also a meaningful result.

### Preserve the Entire DV Pareto Frontier for Q34

Q34 must preserve the entire DV Pareto frontier before any comparative claims are made. A single "best DV model" cannot be selected and the rest discarded before the classical frontier comparison is complete. The DV frontier may dominate the classical frontier in some regions (e.g., very compact low-parameter architectures) while being dominated in others (e.g., high-AUROC moderate-latency configurations). Only the full frontier comparison reveals the pattern of quantum vs. classical tradeoffs.

**The goal of DV NAS is the frontier, not a single best model.**

---

## 6. Searchable DV Dimensions

All dimensions are discrete. The Q34 search algorithm samples from these discrete options. Every dimension corresponds to a meaningful architectural decision in the DV quantum head design.

### Qubit Count

| Option | Notes |
|---|---|
| 2 | Minimal Hilbert space; 4-dimensional; entanglement limited; fewest parameters; fastest simulation |
| 4 | Q21 baseline configuration; 16-dimensional Hilbert space; supports linear and circular entanglement |
| 6 | 64-dimensional Hilbert space; moderate expressivity; tractable on local GPU |
| 8 | 256-dimensional Hilbert space; upper bound for local simulator tractability |

Small qubit counts preserve simulator tractability and compactness. Circuits beyond 8 qubits are explicitly excluded from Q33A scope. The exponential scaling of Hilbert space with qubit count makes circuits beyond 8 qubits intractable for local GPU simulation without specialized matrix product state approximations, which are not part of the QStrata quantum backend.

### Ansatz Depth

| Option | Notes |
|---|---|
| 1 | Q21 baseline depth; one variational layer; minimal barren plateau risk; fastest training |
| 2 | Two variational layers; moderate expressivity; manageable gradient landscape |
| 3 | Three variational layers; higher expressivity; increasing barren plateau risk |
| 4 | Maximum permitted depth; highest expressivity; highest barren plateau risk; slowest simulation |

Deep circuits are intentionally constrained. Depth > 4 risks barren plateau escalation and simulator explosion on local GPU. The exponential growth in simulation cost and the polynomial growth in barren plateau severity with circuit depth jointly bound the tractable region at depth ≤ 4 for the qubit counts considered in Q33A.

### Rotation Families

| Option | Notes |
|---|---|
| `RX` | Single-axis rotations around X; minimal parameters per qubit (1 per qubit per layer) |
| `RY` | Single-axis rotations around Y; alternative single-axis parameterization |
| `RZ` | Single-axis rotations around Z; phase rotations; combines with X/Y for full expressivity |
| `RX+RY` | Two-axis rotations; 2 parameters per qubit per layer; partial SU(2) coverage |
| `RX+RY+RZ` | Full single-qubit rotation; 3 parameters per qubit per layer; full SU(2) coverage |

The rotation family determines expressive power and parameter count per variational layer. For `n_qubits` qubits and `depth` layers: total rotation parameters = `n_qubits × depth × n_rotation_axes`. With `RX+RY+RZ` and 8 qubits at depth 4, this yields 96 rotation parameters, plus entangling gate parameters (if parameterized). The parameter count constraint (≤100,000 head parameters) is satisfied by all combinations in Q33A.

### Entanglement Topology

| Option | Notes |
|---|---|
| `linear` | Each qubit connected to its nearest neighbor in a chain; n−1 entangling gates per layer |
| `circular` | Linear plus connection from last qubit to first; n entangling gates per layer; Q21 configuration |
| `full` | All-to-all connections; n(n−1)/2 entangling gates per layer; permitted only for ≤ 4 qubits |
| `nearest_neighbor` | Equivalent to linear for 1D qubit register; distinguished from linear in 2D register mappings |

Topology must remain interpretable. All-to-all (`full`) entanglement is permitted only for qubit counts ≤ 4. For 6 and 8 qubits, `full` entanglement generates 15 or 28 entangling gates per layer — increasing circuit depth, simulation cost, and barren plateau risk beyond what is tractable for the Q33A constraints. Linear, circular, and nearest-neighbor topologies scale linearly with qubit count and remain tractable at all qubit counts in Q33A.

### Encoding Strategy

| Option | Notes |
|---|---|
| `angle_encoding` | Each feature encoded as a rotation angle on one qubit; requires `n_qubits` features; standard in literature |
| `amplitude_lite_encoding` | Features encoded in probability amplitudes; compact encoding for low qubit counts |
| `reuploading_angle_encoding` | Angle encoding with feature re-uploading at specified frequency; extends expressivity beyond the linear encoding limit |

The encoding strategy determines how the compressed classical features from the backbone projection are embedded into the quantum state. Angle encoding is the simplest and most common; it encodes each compression dimension as a rotation angle on one qubit, requiring that `compression_dim = n_qubits`. Amplitude-lite encoding relaxes this constraint by distributing features across the amplitude vector. Re-uploading encoding repeats the encoding block multiple times, interleaved with variational layers, to extend the circuit's effective input dimension.

### Re-Uploading Frequency

| Option | Notes |
|---|---|
| `none` | No re-uploading; input encoded once at circuit start; standard DV configuration |
| `once` | Input re-encoded at the circuit input only; equivalent to `none` in the one-layer case |
| `every_layer` | Input re-encoded before each variational layer; maximum expressivity; highest runtime cost |

Re-uploading strategies improve gradient flow and extend the circuit's ability to approximate complex functions, at the cost of multiple forward passes of the encoding block and increased wall-clock runtime. For shallow circuits (depth 1–2), re-uploading provides the most benefit per added runtime cost. For deep circuits, re-uploading every layer multiplies the encoding circuit's cost by `depth`.

### Measurement Strategy

| Option | Notes |
|---|---|
| `pauli_z_per_qubit` | Measure the expectation value ⟨Z⟩ on each qubit; produces `n_qubits` output scalars |
| `multi_qubit_expectation_avg` | Average the ⟨Z⟩ expectations across all qubits; produces 1 scalar |
| `concatenated_expectation_vector` | Concatenate ⟨Z⟩, ⟨X⟩, ⟨Y⟩ expectations per qubit; produces `3 × n_qubits` output scalars |

The measurement strategy determines the dimensionality of the quantum circuit's output vector, which is then passed to the classical readout layer. `pauli_z_per_qubit` is the Q21 baseline strategy (4 qubits → 4-dimensional output, fed to `Linear(16, 2)` readout). `concatenated_expectation_vector` increases output dimensionality and downstream readout capacity at the cost of additional measurement gates.

### Compression Dimension (classical projection before quantum encoding)

| Option | Notes |
|---|---|
| 2 | Extremely compact; compresses 128-dim backbone output to 2 features |
| 4 | Q21 baseline configuration; 128 → 4 compression |
| 8 | Moderate compression; 128 → 8 features; more information retained |
| 16 | Light compression; 128 → 16 features; highest capacity among Q33A options |

The compression dimension controls the size of the classical linear layer `Linear(128, compression_dim)` that projects the backbone's 128-dimensional output before quantum encoding. For angle encoding, `compression_dim` must equal `n_qubits` (each compressed feature is encoded as one qubit rotation). For amplitude-lite encoding, this constraint is relaxed. The compression layer is always trainable (not frozen) and contributes `128 × compression_dim` parameters to the head parameter count.

### Classical Projection Layer (after measurement, before binary logits)

| Option | Notes |
|---|---|
| `linear` | Single linear layer `Linear(measurement_dim, 2)`; Q21 baseline; minimal parameters |
| `linear_plus_activation` | Linear layer followed by activation; `Linear(measurement_dim, hidden) → Activation → Linear(hidden, 2)` |
| `none` | No projection; direct mapping from measurement output to logits (only valid when `measurement_dim = 2`) |

The classical projection layer after measurement determines how the quantum circuit's output vector is mapped to binary class logits. In Q21, a `Linear(16, 2)` readout layer was used (`n_qubits × measurement_multiplier = 4 × 4 = 16`). Q33A allows this to be either a direct linear projection, a projection with an intermediate activation, or (in special cases) no projection at all.

---

## 7. Explicitly Forbidden DV Dimensions

The following architecture choices are explicitly out of scope for Q33A. Including them would violate the compactness, tractability, or interpretability requirements of the DV quantum ceiling experiment.

| Forbidden | Scientific Rationale |
|---|---|
| Qubit count > 8 | Hilbert space dimension 2^n grows exponentially; local GPU simulation intractable beyond 8 qubits; barren plateau severity increases exponentially with qubit count |
| Unrestricted circuit depth | Depth > 4 produces circuits where gradient magnitudes are exponentially suppressed (barren plateau); simulation time grows super-linearly with depth; results are not interpretable for comparison against shallow classical heads |
| Adaptive circuit topology mutation at runtime | Changes the circuit structure during training; breaks reproducibility; cannot be represented in a frozen YAML config; incompatible with the Q31 runner framework |
| Dynamic quantum graph generation | Architectures that modify their own circuit connectivity during training; breaks the frozen-config requirement; produces results that cannot be reproduced from a static config |
| Hardware-aware qubit routing | Assumes target physical quantum device; incompatible with local simulator-based execution; introduces device-specific noise models not present in QStrata |
| Quantum error correction | Logical qubit overhead is orders of magnitude beyond the compact parameter budget; requires syndrome measurement circuits not present in the QStrata DV backend |
| Cloud quantum hardware assumptions | Q33A is local-only; NISQ device execution is blocked until local NAS produces validated results; quantum cloud compute introduces non-reproducible hardware noise |
| Transformer-style quantum attention mechanisms | Not compact DV head architectures; incompatible with the frozen-backbone + compact-head factorization; changes the comparison baseline |
| Hybrid large classical heads (> 10K parameters in the head alone) | Violates the compactness principle; not comparable to Q21/Q22/Q27 in the ≤600 trainable parameter regime |

**Q33A focuses exclusively on compact simulator-based DV systems compatible with local GPU experimentation and the QStrata framework.** The compactness of existing quantum benchmarks (Q21: 574 params, Q27: 536 params) defines the regime of comparison. A DV NAS search that produces a 50,000-parameter quantum circuit as its "ceiling" does not answer the scientific question of whether compact DV heads can match compact classical heads — it answers a different question, and trivially.

---

## 8. DV Constraint System

Constraints are divided into hard (violation invalidates the trial) and soft (violation is flagged but the trial is retained) categories, consistent with the Q32 framework.

### Hard Constraints — Violation Marks Trial `invalid`

| Constraint | Value | Enforcement |
|---|---|---|
| Total trainable parameters | ≤ 100,000 | Checked post-model construction; trial aborted if exceeded; recorded as `invalid` in leaderboard |
| Local GPU execution only | No cloud infrastructure | No cloud execution permitted; all trials on local single GPU |
| Quantum simulator forward pass | OOM-free at batch size 4 on local GPU | Runner marks trial `invalid` on OOM exit |
| Numerical validity | No NaN or inf at any point in forward or backward pass | Any NaN/inf in forward pass, loss, or gradient is a hard violation; trial marked `invalid` |
| Reproducibility | Seed set from config; git commit captured | Experiments without a fixed seed or with `git_commit: unknown` are flagged; published results require clean commit |
| Batch size compatibility | Batch size 4 required | Circuit must support batch size 4 without OOM; single-sample evaluation is not sufficient |

Hard constraint violations are enforced by the runner. Trials that violate hard constraints are recorded in the leaderboard with `status: invalid` and excluded from Pareto frontier computation. Invalid trials are retained in the leaderboard for diagnostic purposes — they document which circuit configurations are infeasible.

### Soft Constraints — Violation Flagged but Trial Retained

| Constraint | Value | Notes |
|---|---|---|
| Inference latency | Within 2× Q21 latency (~110 ms/sample) | Q21 required 54.79 ms/sample (CPU-bound quantum circuit simulation); 2× provides headroom for more complex circuits |
| Compact quantum heads preferred | < 5,000 trainable parameters | Strongly preferred; circuits in the 500–5,000 parameter range are most directly comparable to Q21/Q22/Q27 |
| Shallow circuits preferred | Depth ≤ 2 for pilot | Depth 3–4 circuits are permitted but flagged; pilot phase prioritizes tractable shallow circuits |
| Low simulator runtime | < 5× classical head runtime | Classical GPU heads run at < 2 ms/sample; DV circuits at > 10 ms/sample are flagged for latency overhead analysis |

Soft constraint violations are recorded in the result JSON and leaderboard. Trials flagged for soft violations remain in the Pareto frontier analysis but their constraint status is visible in all tables and reports.

---

## 9. Barren Plateau Mitigation Philosophy

### The Barren Plateau Problem

Barren plateaus arise when gradient magnitudes of quantum circuits become exponentially small as circuit depth or qubit count increases. A training landscape that is flat everywhere provides no gradient signal for parameter updates; the circuit cannot be trained with gradient-based optimization. Barren plateaus are a fundamental obstacle to scaling quantum machine learning circuits.

The primary manifestation is exponential decay of the variance of the gradient with respect to circuit parameters: `Var[∂L/∂θ_k] ~ O(2^{-n})` for global cost functions on n qubits. For 8 qubits, the gradient variance is suppressed by a factor of ~1/256 relative to 1-qubit circuits. This suppression makes training unreliable at deeper and wider circuits.

### Q33A Mitigation Strategies

Q33A mitigates barren plateau risk through structural constraints:

**Depth constraint (depth ≤ 4):** The barren plateau severity increases with circuit depth. Limiting depth to ≤ 4 reduces the parameter landscape flatness relative to deeper circuits while preserving sufficient expressivity for the binary classification task.

**Qubit count constraint (≤ 8 qubits):** The gradient variance decay scales exponentially with qubit count for global cost functions. Limiting qubit count to ≤ 8 keeps the gradient variance in a regime where training is feasible. For local cost functions (e.g., measuring only one or two qubits), the scaling is less severe, but the Q33A measurement strategies include global options (averaged expectations) that require the qubit count bound.

**Compact parameterization:** Fewer rotation parameters per layer reduces the probability of gradient vanishing in the joint parameter space. The `RX` and `RY` single-axis options have 1 parameter per qubit per layer, limiting parameter space dimension without sacrificing per-qubit expressivity.

**Limited entanglement topology:** Linear and nearest-neighbor topologies reduce the quantum volume — a measure of circuit complexity — and the associated barren plateau risk. Full entanglement is permitted only up to 4 qubits, where the gradient suppression is still manageable.

**Re-uploading strategies:** Feature re-uploading (encoding input features multiple times, interleaved with variational layers) has been shown to improve gradient flow in shallow circuits by providing a persistent classical signal throughout the circuit. Re-uploading does not eliminate barren plateaus but can alleviate gradient flatness in the input encoding layers.

### Gradient Health Checks in Q34

Q21 demonstrated full gradient health across all 15 training epochs: non-zero theta, projection, and readout gradient norms throughout; no NaN/inf; no gradient collapse. Q34 DV NAS trials must implement equivalent gradient health tracking. Specifically:

- Per-epoch theta gradient norm (variational circuit parameters)
- Per-epoch compression projection gradient norm
- Per-epoch readout layer gradient norm
- NaN/inf detection on all gradient tensors after each backward pass

Trials exhibiting gradient norm collapse (theta gradient norm drops below 1e-8 and remains there for ≥ 3 consecutive epochs) are flagged as barren plateau candidates in the result JSON.

**Q33A intentionally constrains search entropy to reduce barren plateau risk.** The constraint system is not conservative out of caution — it is conservative because the scientific question is whether compact DV circuits can match compact classical heads, and that question cannot be answered with circuits that cannot be trained.

---

## 10. Experiment Budget Philosophy

### Local Single-GPU Execution

All Q34 DV NAS trials execute sequentially on a single local GPU using the Q31/Q31A runner infrastructure. One DV trial runs at a time. No parallelism across trials. This is a hard constraint enforced by the local-first principle: distributed execution is blocked until local NAS produces validated, stable results.

Sequential single-GPU execution is maximally reproducible. Given the same config and seed, re-running any DV trial must produce results within tolerance (AUROC ±0.0001, F1 ±0.0001). This was validated by the Q31A reproducibility test (loss_delta = 0.0 across two sequential runs). DV NAS inherits this requirement.

### Bounded Trial Count

Q34 pilot uses a small fixed budget. Approximately 20–50 trials for the DV pilot is a reasonable starting range. This count is sufficient to:

- Sample the DV search space with meaningful coverage across qubit count, depth, and encoding strategy
- Produce a DV Pareto frontier with enough non-dominated points to identify tradeoff patterns
- Complete within a time-bounded local session
- Generate enough variance to distinguish architectural effects from noise

The exact trial count is a Q34 decision based on available hardware, per-trial wall-time, and session budget at execution time. DV circuits are more expensive to simulate than classical heads; fewer trials may be feasible per session.

### Bounded Runtime Per Trial

Each DV trial has a configurable wall-time ceiling. Trials exceeding it are marked `timeout` by the runner and excluded from Pareto frontier computation but retained in the leaderboard. An initial timeout of 30–60 minutes per DV trial is a reasonable planning estimate — DV circuit simulation is CPU-bound and substantially slower than classical GPU training.

### No Exhaustive Grid Search

Q34 uses random search or lightweight Bayesian sampling over the defined discrete space. No exhaustive grid search. The Q33A search space contains:

3 × 4 × 5 × 4 × 3 × 3 × 3 × 4 × 3 ≈ 77,760 discrete configurations

(qubit count × depth × rotation family × entanglement topology × encoding strategy × re-uploading frequency × measurement strategy × compression dimension × projection layer)

Exhaustive enumeration is infeasible at any bounded trial budget. Random or Bayesian sampling produces an interpretable Pareto frontier without search algorithm artifacts.

**Q34 prioritizes interpretable signal over brute-force quantum search scale.**

---

## 11. Reproducibility Requirements

### Frozen YAML Config Per Trial

Every DV NAS trial produces a frozen YAML config via the Q31 runner framework. The config is written to `experiments/configs/<experiment_id>.yaml` and set read-only (`chmod 444`) before the trial starts. No trial result is recorded without an associated frozen config. The config captures all searchable dimension values for the trial: qubit count, depth, rotation family, entanglement topology, encoding strategy, re-uploading frequency, measurement strategy, compression dimension, and projection layer type.

### Seed Locking

Each DV trial uses a fixed seed from `reproducibility.seed` in the config. The seed is set before dataset loading, model initialization, and augmentation. It is recorded in the result JSON. Quantum circuit parameter initialization is deterministic given the seed. Trials run without a fixed seed are invalid and excluded from all analysis.

### Git Commit Tracking

Every DV trial records the exact code state at execution time via `git rev-parse HEAD` (or the `QSTRATA_GIT_COMMIT` env var, as validated in Q31A). Trials with `git_commit: unknown` are flagged in the leaderboard. Published DV NAS results must reference a clean git commit and be reproducible from that commit and the frozen config.

### Leaderboard Integrity

DV trial results are immutable once recorded. The runner writes the result JSON and leaderboard entry immediately after trial completion. No post-hoc editing of results is permitted. If a trial is found to have an error, a corrected re-run produces a new entry with a new `experiment_id`; the original entry is retained with `status: retracted`.

### Deterministic Orchestration

Given the same config and seed, re-running any DV trial must produce results within numerical tolerance (AUROC ±0.0001, F1 ±0.0001). This property was validated for the runner infrastructure by Q31A (loss_delta = 0.0 across two sequential runs). DV quantum circuit simulation must be deterministic given a fixed seed; non-deterministic backends are not compatible with Q33A reproducibility requirements.

**Quantum NAS without reproducibility is scientifically invalid.** The Q31/Q31A infrastructure is the prerequisite for Q34 DV NAS precisely because it enforces frozen configs, seed locking, git tracking, and leaderboard immutability mechanically.

---

## 12. Search Space Interpretability

### Human-Readable Architecture Space

Every searchable dimension in Q33A corresponds to a meaningful, well-understood architectural choice in the DV quantum circuit design space:

| Dimension | Architectural Decision |
|---|---|
| Qubit count | How large is the Hilbert space? How many features can be directly encoded? |
| Ansatz depth | How many variational layers are applied? How expressive is the circuit? |
| Rotation family | How many axes of rotation are applied per qubit per layer? |
| Entanglement topology | How are qubits coupled? What feature interactions are structurally encoded? |
| Encoding strategy | How are classical features embedded into the quantum state? |
| Re-uploading frequency | How many times is the input signal injected into the circuit? |
| Measurement strategy | What quantum observables are extracted as the circuit's output? |
| Compression dimension | How many features from the backbone does the circuit receive? |
| Classical projection layer | How is the quantum output mapped to binary logits? |

### Interpretable Architecture Evolution

Moving from one point in the DV search space to another corresponds to a comprehensible architectural difference:
- Increasing qubit count from 4 to 6 doubles the Hilbert space dimension (16 → 64) and adds 2 qubits to all entanglement patterns
- Switching from `linear` to `circular` entanglement adds one entangling gate per layer (connecting last qubit to first)
- Changing rotation family from `RY` to `RX+RY+RZ` triples the rotation parameter count per layer
- Increasing compression dimension from 4 to 8 doubles the backbone projection size and decouples the compression dimension from the qubit count (for amplitude-lite encoding)

### Explainable Pareto Tradeoffs

The DV Pareto frontier produced by Q34 should be describable in plain language. Expected interpretable frontier statements include:
- "Deeper circuits (depth 3–4) with circular entanglement achieve higher AUROC at 3× the latency of depth-1 circuits"
- "4-qubit circuits with re-uploading match 6-qubit circuits without re-uploading on AUROC while using 30% fewer parameters"
- "RX+RY+RZ rotation family improves F1 relative to RY-only at the cost of 2× parameter count"

If the Pareto frontier cannot be described in plain language, the search space has too many uninterpretable interactions. This is a signal to reduce dimensionality in the next search iteration.

### Avoid Quantum Architecture Chaos

Architectures from incompatible circuit families or incompatible encoding and qubit count combinations are excluded. Specifically:
- Angle encoding requires `compression_dim = n_qubits` (each compressed feature maps to one qubit); search combinations that violate this are excluded at config generation time
- Full entanglement is excluded for qubit counts > 4; configs specifying `full` entanglement with `n_qubits > 4` are rejected by the trial generator
- The `none` projection layer is excluded for measurement strategies that produce output dimensions ≠ 2

**Scientific interpretability is prioritized over maximal search entropy.** A DV search that mixes incompatible circuit elements produces results that are harder to interpret, harder to compare against the classical frontier, and harder to generalize from.

---

## 13. Relationship to Q32 Classical Ceiling

### Q32 Defines the Compact Classical Pareto Frontier

Q32 defines the search space for compact classical CNN head architectures over the same frozen pretrained backbone (C006-D040) and the same VinDr-SpineXR binary task. Q34 executes this classical search and produces a Pareto frontier over AUROC, F1, parameter count, and latency — the classical ceiling.

The classical ceiling replaces Q17 (23,650 params, random init) and Q22 (one fixed point, 526 params) as the classical reference. It is a multi-point frontier representing the best achievable compact classical performance under systematic search over the Q32 search space.

### Q33A Defines the Compact DV Pareto Frontier

Q33A defines the DV quantum search space over the same backbone, task, and orchestration framework. Q34 executes DV NAS trials from this space and produces a DV Pareto frontier — the DV quantum ceiling.

The DV quantum frontier is produced under the same orchestration (Q31/Q31A runner), the same dataset (VinDr-SpineXR binary, canonical split), the same backbone (frozen C006-D040), and the same multi-objective framework as the classical frontier. The only experimental variable between classical and DV NAS is the head architecture family.

### Q34 Compares Both Frontiers Under Identical Conditions

Q34 evaluates both classical and DV NAS under identical orchestration, datasets, seeds, and constraints. The resulting comparison is scientifically clean: any difference in Pareto frontier position between classical and DV architectures is attributable to the head architecture, not to differences in experimental conditions.

**DV NAS results must be compared against Q32 classical results, not against the unoptimized Q17 baseline.** Comparing against Q17 would be methodologically invalid: Q17's weakness is attributable to random initialization and large parameter count, not to the classical architecture family. An optimized DV frontier must be compared against an optimized classical frontier to produce a meaningful scientific result.

### What Comparison Outcomes Mean

A DV architecture that is not dominated by any Q32 classical architecture on AUROC, F1, parameter count, and latency is a scientifically meaningful positive result: it demonstrates that DV quantum heads occupy a distinct and non-dominated region of the compact performance space.

A DV architecture that is dominated by multiple Q32 classical architectures on the primary objectives (AUROC, F1) while using fewer parameters may still represent a meaningful tradeoff: a quantum advantage in compactness, if not in raw AUROC.

A DV Pareto frontier that is fully dominated by the classical frontier across all objectives is also a scientifically meaningful result: it establishes that, under the constrained search spaces of Q32 and Q33A, compact classical heads fully match or exceed compact DV heads, eliminating the residual quantum advantage hypothesis under the compact bottleneck regime.

**Q33A does not predetermine the comparison outcome.** The goal is a fair, optimized comparison — not confirmation of any prior expectation.

---

## 14. Local-First Quantum NAS Philosophy

All Q34 DV NAS trials execute on local GPU only. This is a hard constraint, not a soft preference.

**AWS and Ray remain blocked until local NAS is validated.** The sequencing is:

1. Q33A — design DV quantum search space (this slice; design only)
2. Q33B — design CV quantum search space (design only)
3. Q34 — execute local pilot NAS using Q32, Q33A, and Q33B spaces jointly
4. Q35 — Pareto analysis and NAS hardening (after Q34 produces stable results)
5. Q36 — design distributed scaling (only after Q34 is validated)

**Distributed quantum orchestration remains blocked.** Any attempt to parallelize DV NAS across multiple GPUs or machines before local NAS produces reproducible, interpretable results would amplify whatever bugs exist in the local execution path — and would introduce additional reproducibility challenges associated with distributed state management.

**Local reproducibility precedes scaling.** The Q31A reproducibility validation (loss_delta = 0.0 across two sequential runs) established that the runner is deterministic. This property must be validated for DV NAS before any distributed extension is considered.

Local-first is a scientific integrity constraint, not a resource constraint. A small reproducible pilot on one GPU produces more interpretable results than a large irreproducible distributed search.

---

## 15. Future Phases

The planned sequence after Q33A:

```
Q33B → CV quantum NAS search-space design (design only)
        defines GaussianVariationalAnsatz search dimensions: n_modes, cv_depth,
        squeezing_cap, displacement_cap, encoding scheme, readout strategy
        same multi-objective framework, constraint system, and runner infrastructure as Q33A
        no NAS execution in Q33B

Q34   → first local NAS execution (classical + DV + CV trials)
        executes Q32 classical space on local GPU
        executes Q33A DV space on local GPU
        executes Q33B CV space on local GPU
        produces Pareto frontiers for classical, DV, and CV under identical conditions
        compares DV and CV quantum frontiers against Q32 classical ceiling
        no AWS; no Ray; no distributed execution

Q35   → Pareto analysis and NAS hardening
        full multi-frontier comparison across classical, DV, and CV Pareto frontiers
        statistical analysis of frontier differences (if multi-seed data available)
        identifies which Q33A/Q33B dimensions drive Pareto-optimal DV/CV performance

Q36   → distributed scaling design (design only; after Q35 validated)
        designs distributed extension of Q34 infrastructure
        no cloud provisioning before this design is approved
        blocked until Q34 and Q35 produce stable, validated local results
```

**No NAS execution occurs in Q33A or Q33B.** The first DV NAS trial runs in Q34. Q33A and Q33B are complete when their design documents are committed and the roadmap is updated — not when any trials are run.

---

## 16. Required Scientific Guardrail

> The QStrata DV quantum NAS program prioritizes scientifically interpretable, reproducible, compact, and numerically stable optimization before scaling circuit complexity or infrastructure. Quantum NAS exists to explore controlled tradeoffs, not unconstrained quantum architecture entropy.

---

```
Q33A status: COMPLETE — design only; no NAS execution
Q33B status: NEXT — CV quantum NAS search space design (design only)
Q34 status: PLANNED — first local NAS execution (classical + DV + CV)
Q35 status: PLANNED — Pareto analysis and NAS hardening (after Q34)
Q36 status: BLOCKED — requires validated Q34/Q35 local NAS
DV quantum ceiling: UNDEFINED — will be produced by Q34
Classical ceiling: UNDEFINED — will be produced by Q34 (Q32 space)
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 local NAS validated
```
