# Q32: Classical Feature Extractor NAS Search Space

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q32 — Classical NAS Search Space Design  
**Status:** DESIGN ONLY — no NAS execution in Q32

---

## 1. Title

Q32: Classical Feature Extractor NAS Search Space

---

## 2. Purpose

The Q17 classical CNN baseline achieved AUROC 0.6224 using 23,650 trainable parameters, trained end-to-end from random initialization. This underperformed every compact hybrid benchmark. Q21 (DV hybrid, 574 params) achieved AUROC 0.6800. Q22 (tiny classical control, 526 params) achieved AUROC 0.6625 — recovering approximately 70–75% of Q21's improvement over Q17 using only a classical MLP head attached to the same frozen pretrained backbone. Q27 (CV hybrid, 536 params) achieved AUROC 0.6708.

This finding — that a compact classical bottleneck over a frozen pretrained backbone dramatically outperforms a larger random-initialized classical CNN — reveals that the dominant driver of performance gain is the frozen backbone representation, not the quantum head. The compactness of the trainable head (whether classical or quantum) is a secondary factor, but one that has not been systematically explored.

**The classical ceiling under optimized compact architectures has not been established.** The Q22 result is a single data point — one parameter-matched classical head at one architecture configuration. Whether Q22 can be substantially improved by systematic architecture search over compact heads is unknown. Whether the residual DV and CV advantages over Q22 survive a strong classical ceiling is unknown.

Q32 designs the search space that will answer these questions. No NAS execution occurs in Q32. The search space defined here is the input to Q33 (quantum NAS search space design) and Q34 (local NAS pilot — first execution).

---

## 3. Scientific Motivation

### The Compactness Effect

Q22's result establishes what is termed the **compactness effect**: when a frozen pretrained backbone provides fixed-quality representations, a compact head trained on those representations can match or approach the performance of larger, fully-trained architectures. Q17's 23,650-parameter CNN had to learn feature extraction and classification simultaneously from scratch on a small medical imaging dataset. Q22's 526-parameter head learned only a decision boundary over already-extracted features.

This asymmetry is decisive. The backbone (C006-D040, PneumoniaMNIST-pretrained) already extracted task-relevant features from grayscale thoracic images. The compact head needs only to map those features to binary labels. The complexity required for that mapping is far lower than the complexity required for joint feature extraction and classification.

### The Pretrained Backbone Effect

All four compact models (Q21, Q22, Q27, and by design Q33 quantum NAS targets) use the same frozen C006-D040 pretrained backbone. Cross-domain transfer from PneumoniaMNIST to VinDr-SpineXR provides representations that generalize across lung and spine pathology recognition. The backbone's utility establishes a strong prior on what feature space the head operates in. Any compact head (classical or quantum) benefits from this prior equally.

This means that comparing Q21 DV hybrid against Q22 classical control is a comparison of quantum vs. classical heads under an identical representation context. The backbone advantage is held constant. The head design is the independent variable.

### Parameter Efficiency

Q21, Q22, and Q27 achieve competitive classification performance with 526–574 trainable parameters. Q17 uses 23,650 parameters and performs worse. This is not an anomaly — it reflects that end-to-end training on a small dataset with a large model is a harder optimization problem than fine-tuning a compact head over a rich frozen representation.

The NAS program uses this finding to define its compactness constraint: compact architectures (≤100,000 trainable head parameters, with the frozen backbone excluded) are the scientifically relevant regime. Larger architectures are not comparable to the existing benchmarks and do not contribute to the compact ceiling question.

### Latency-Performance Tradeoffs

Q27 (CV hybrid) achieves higher F1 than Q21 (DV hybrid) but slower inference (2.15 ms vs 54.8 ms, though the comparison is device-inconsistent). Q22 (classical, GPU) achieves 1.48 ms. Compact classical heads on GPU are the latency baseline. A classical NAS ceiling must also consider inference latency as an objective — a classical architecture that beats Q21 in AUROC but at 100× the latency is not a compact classical ceiling, it is simply a different point on the cost curve.

### Classical Ceiling Importance

Without a systematic search for the best compact classical architecture, any residual quantum hybrid advantage observed in Q21 and Q27 versus Q22 is uninterpretable. It could reflect:

1. A genuine quantum head advantage that classical compact heads cannot match
2. The particular Q22 architecture being a weak classical baseline (one fixed point in a large space)
3. Random noise in single-seed experiments

Only a systematic search over compact classical architectures can distinguish between these interpretations. Q32 defines that search. Q34 executes it. Without Q32 and Q34, any claim of quantum benefit rests on comparing quantum heads against an arbitrarily chosen classical configuration, which is methodologically insufficient.

**A strong classical ceiling is scientifically required before evaluating quantum optimization benefit.**

---

## 4. Optimization Philosophy

The QStrata NAS program uses a structured multi-objective optimization framework across three tiers. No single-metric optimization is permitted at any tier.

### Primary Objectives (Must Optimize)

These objectives are the primary scientific evaluation criteria. Every trial must track both. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance; the primary cross-model comparison metric; stable under class imbalance
- **Maximize F1** — threshold-dependent; captures precision-recall tradeoff relevant to clinical decision support; computed at 0.5 threshold

A model that maximizes AUROC while sacrificing F1 to a degenerate level (e.g. by predicting all positives) is not a valid Pareto point. Both objectives must be tracked and both contribute to dominance assessment.

### Secondary Objectives (Should Optimize)

These objectives constrain practical deployability. Pareto frontier analysis may include these as additional dimensions when the primary frontier is established:

- **Minimize parameter count** — trainable parameters only (frozen backbone excluded); proxy for deployment feasibility and hardware cost; compact models are the scientific target
- **Minimize inference latency** — ms/sample measured on the local GPU over 100 single-sample passes; clinical real-time constraint; prevents pathological architectures with acceptable AUROC but unacceptable latency
- **Minimize runtime instability** — NaN/inf count across training epochs; gradient norm outlier frequency; a model that passes AUROC but fails numerical stability is not deployable regardless of performance

### Tertiary Objectives (Consider)

These objectives inform resource planning and are recorded but may not be included in primary Pareto analysis:

- **Minimize peak memory usage** — GPU and CPU memory in MB; determines feasibility on local hardware without OOM; required for hardware budget planning
- **Minimize training wall time** — seconds from run start to experiment end; informs NAS search budget; prevents runaway trials from monopolizing the runner

### Prohibition on Single-Metric Optimization

Single-metric optimization is forbidden by design. A model that maximizes AUROC while using 10× the parameter budget of Q21 is not a valid compact classical ceiling — it is a different research question. A model that achieves 0.70 AUROC but crashes with NaN gradients on 3 of 5 seeds is not stable enough to serve as a ceiling. A model that requires 200 ms inference per sample is not a compact medical imaging classifier.

The Pareto framework is the mechanism that makes multi-objective optimization tractable: instead of weighting objectives (which introduces arbitrary weighting choices), we identify the set of non-dominated solutions and present them as the frontier. Human scientific judgment operates on the frontier, not on a scalar aggregate.

---

## 5. Pareto Frontier Philosophy

### Non-Dominated Solutions

A solution X is non-dominated (Pareto optimal) if there is no other solution Y that is strictly better than X on all tracked objectives simultaneously. The Pareto frontier is the set of all non-dominated solutions in the evaluated trial space.

In the QStrata context: a compact classical architecture is Pareto optimal if no other evaluated architecture achieves higher AUROC, higher F1, fewer parameters, and lower latency simultaneously. It is acceptable for a Pareto-optimal architecture to be worse on one or two objectives if it is better on the others.

### Tradeoff Surface

The Pareto frontier defines the tradeoff surface — the boundary of what is achievable within the compact constraint budget under the given search space. Points on the frontier represent meaningful architectural choices: different positions on the frontier correspond to different engineering decisions about where to spend the parameter and latency budget.

For example, on the AUROC vs. parameter count frontier:
- Low-parameter, moderate-AUROC architectures: suitable for the most resource-constrained deployments
- Moderate-parameter, high-AUROC architectures: suitable for standard edge deployment
- High-parameter, highest-AUROC architectures: suitable for server-side inference

Each region of the frontier is scientifically and operationally interpretable.

### Compact High-Performance as the Reference

The correct reference for quantum head comparison is not the Q17 baseline (23,650 parameters, random initialization, end-to-end training). The correct reference is the best compact classical architecture from the Q34 Pareto frontier. Quantum NAS (Q33/Q34) must beat this frontier to demonstrate quantum benefit — not just beat Q17 or Q22.

This is the classical ceiling principle in operational form: **Q32 defines the search space; Q34 produces the frontier; Q33 quantum NAS must be compared to that frontier.**

### Preserve Full Pareto Frontier for Q33 Comparison

The full Pareto frontier produced by Q34 must be preserved for Q33 comparison. Do not select a single "best" classical architecture and discard the rest before Q33 quantum NAS is complete. Quantum NAS may dominate some regions of the frontier (e.g., very compact architectures) while being dominated in others (e.g., high-AUROC large architectures). Only the full frontier comparison reveals which regions exhibit quantum residuals.

---

## 6. Search Space Boundaries

The classical NAS search space for Q32 is defined over the following dimensions. All dimensions are discrete. The search algorithm (defined in Q34) samples points from this space.

### Backbone Family

The trainable head architecture family:

| Option | Description |
|---|---|
| `standard_cnn` | Standard convolutional blocks: Conv → BN → Activation, stacked sequentially |
| `depthwise_sep` | Depthwise separable convolutional blocks: DW-Conv → PW-Conv → BN → Activation; fewer parameters per block |
| `residual_compact` | Compact residual blocks with skip connections; aggressive channel compression; maintains gradient flow in deeper heads |

The frozen C006-D040 backbone outputs 128-dimensional feature vectors (after `build_model(CNN_CONFIG)[:4]`). The head takes this 128-dim feature as input and produces 2-class logits.

### Channel Width

Discrete channel width progression options for the convolutional head blocks:

| Option | Channels |
|---|---|
| `narrow` | `[16, 32]` |
| `compact` | `[32, 64]` |
| `standard` | `[64, 128]` |
| `wide` | `[96, 192]` |

For depthwise separable and residual compact heads, these widths apply to the pointwise convolution output. For the final compression before the classifier, the compression dimension is separately controlled.

### Depth

Number of convolutional blocks in the head (each block = one element of the channel width list above, or one residual unit):

| Option | Depth | Notes |
|---|---|---|
| `shallow` | 2 conv blocks | Minimal head; rapid convergence; low parameter count |
| `medium` | 3–4 conv blocks | Moderate capacity; standard tradeoff |
| `compact_deep` | 5–6 conv blocks with aggressive channel compression | More layers but channels compressed; aims for representational depth without parameter explosion |

### Compression Dimension

The linear projection that maps from the CNN feature output to the final classifier input. This is the bottleneck dimension — analogous to the compression layer in Q21/Q22/Q27:

| Dimension |
|---|
| 2 |
| 4 |
| 8 |
| 16 |
| 32 |

The classifier is always `Linear(compression_dim, 2)`.

### Activation Function

Applied after each convolutional block and within the head:

| Option | Notes |
|---|---|
| `relu` | Standard; well-understood; computationally cheap |
| `gelu` | Smoother; commonly used in modern compact architectures |
| `silu` | Swish activation; smooth; slightly more expensive than ReLU |

### Normalization

Applied within each convolutional block:

| Option | Notes |
|---|---|
| `batch_norm` | BatchNorm; standard for CNN training; statistics over batch |
| `group_norm` | GroupNorm; stable on small batches; independent of batch size |
| `layer_norm` | LayerNorm; per-sample; used in transformer-adjacent contexts |

### Dropout

Applied after the pooling layer and optionally within blocks:

| Option |
|---|
| 0.0 |
| 0.1 |
| 0.2 |
| 0.3 |
| 0.5 |

### Pooling

Spatial pooling applied at the end of the convolutional backbone before the linear head:

| Option | Notes |
|---|---|
| `max_pool` | Takes maximum activation; emphasizes dominant features |
| `avg_pool` | Takes average activation; smoother, less prone to single-pixel dominance |
| `adaptive_avg_pool` | Adaptive average pooling to a fixed output size; flexible; standard in modern compact CNNs |

### Learning Rate

Searched over a log-uniform range. The exact range and search protocol are defined in Q34 (local NAS pilot design). The range will be anchored to the existing successful training runs (Q17: lr=1e-3; Q21/Q22/Q27: lr=1e-3).

**Likely range for Q34:** log-uniform over [1e-4, 5e-3].

### Weight Decay

Searched over a log-uniform range. The exact range and search protocol are defined in Q34.

**Likely range for Q34:** log-uniform over [0.0, 1e-2] (including zero for no regularization).

---

The search space must remain compact and scientifically interpretable. Every dimension added to the search space multiplies the number of configurations exponentially. The dimensions above are chosen for three properties: they correspond to architectural decisions with well-understood effects, they are the primary variables that differentiate existing compact CNN families, and they cover the range that produced successful results in Q17/Q22 while extending into design decisions not yet explored.

---

## 7. Explicitly Forbidden Search Dimensions

The following architecture families and design decisions are explicitly out of scope for Q32 classical NAS. Including them would invalidate the scientific purpose of the classical ceiling experiment.

### Forbidden Architectures

| Forbidden | Reason |
|---|---|
| Transformer architectures | Not compact medical imaging CNNs; incomparable to Q21/Q22/Q27; requires different training regime |
| Attention mechanisms | Adds search dimensions beyond compact CNN scope; attention in compact models changes comparison baseline |
| ResNet50+, EfficientNet-B4+, ViT variants | Giant backbone families; parameter counts orders of magnitude above compact target; not comparable to 500–5000 param range |
| Diffusion models | Generative models; not classifiers; entirely out of scope |
| Multimodal architectures | Requires multimodal data not present in VinDr-SpineXR binary setup |
| Foundation models | Require compute and data budgets incompatible with local-first constraint |
| Any architecture exceeding 5M total parameters | Incompatible with compact constraint; not comparable to Q21/Q22/Q27 |
| Cloud-scale training assumptions | Q34 executes on local GPU; cloud-scale architectures are incompatible with local-first execution |

### Scientific Rationale

Q32 focuses exclusively on compact medical-imaging CNN systems. These constraints are not arbitrary — they reflect the edge-oriented research philosophy of QStrata and ensure that classical ceiling results are comparable to the compact quantum benchmarks from Q21 and Q27.

The compact quantum heads (Q21: 574 params, Q27: 536 params) operate in a parameter regime that is fundamentally different from large neural architectures. A classical NAS experiment that produces a 50M-parameter ResNet as the "ceiling" does not answer the scientific question of whether compact classical heads can match compact quantum heads. It answers a different question (whether large classical models can outperform compact quantum models) which trivially answers yes and provides no scientific insight.

The 5M parameter hard ceiling is generous relative to Q21/Q22/Q27 and ensures that even large-ish compact CNNs are included in the search without admitting architectures that are categorically incomparable.

---

## 8. Constraint System

Constraints are divided into hard and soft categories based on whether their violation invalidates the trial or merely flags it for review.

### Hard Constraints

Violation of any hard constraint causes the trial to be marked `invalid` in the leaderboard and excluded from Pareto frontier computation. Hard constraints are enforced by the runner before or immediately after trial execution.

| Constraint | Value | Enforcement |
|---|---|---|
| Maximum trainable parameters (head only) | ≤ 100,000 | Checked post-model construction; trial aborted if exceeded |
| GPU memory ceiling | OOM-free at batch size 4 on local GPU | Runner marks trial `invalid` on OOM exit |
| Execution environment | Local single GPU only; no cloud infrastructure | No cloud execution permitted |
| Reproducibility | Seed set from config; git commit recorded | Runner enforces; experiments with `git_commit: unknown` flagged |

**Rationale for 100,000 parameter ceiling:** The existing compact benchmarks (Q21: 574, Q22: 526, Q27: 536) use under 600 trainable parameters. The 100,000 ceiling allows architectural exploration that includes moderately larger compact heads (which may be justified by increased performance) while excluding architectures that are not meaningfully compact. This range encompasses the full spectrum from the existing benchmarks to reasonably sized compact heads.

### Soft Constraints

Violation of a soft constraint is recorded in the result JSON and leaderboard but does not invalidate the trial. Soft-constrained trials appear in the Pareto frontier analysis with their constraint violation noted.

| Constraint | Value | Notes |
|---|---|---|
| Inference latency | ≤ 2× Q21 latency (~110 ms/sample) | Q21's latency is CPU-bound (quantum circuit); classical GPU heads should be substantially faster; this ceiling is intentionally generous |
| Training stability | No NaN or inf in any epoch across training | Violations recorded; unstable trials excluded from Pareto frontier recommendation but retained in leaderboard |
| Parameter count comparable to existing benchmarks | 500–5,000 trainable parameters preferred | Trials exceeding this range are noted; preference for compact, not required |

**Latency note:** The Q21 latency reference (54.78 ms/sample) is CPU-bound due to the quantum circuit simulation. Classical GPU heads should achieve sub-10 ms latency at batch size 1 on the local GPU. The 2× ceiling (~110 ms) is therefore generous and would only be violated by pathologically inefficient classical implementations.

---

## 9. Experiment Budget Philosophy

### Local Single-GPU Assumption

All Q34 NAS trials execute on a single local GPU sequentially. The Q31/Q31A runner infrastructure executes one experiment at a time. No parallelism across trials is introduced in Q34. This constraint exists because:

1. Local-first validation must precede any distributed scaling
2. Sequential single-GPU execution is maximally reproducible
3. The trial count is bounded, making sequential execution feasible

### Bounded Trial Count

Q34 will define the exact trial count. As a planning reference, approximately 20–50 trials is a reasonable starting range for a pilot NAS study. This is sufficient to:

- Cover the primary search space dimensions with adequate sampling density
- Produce a Pareto frontier with enough points to be interpretable
- Complete in a time-bounded fashion on local hardware
- Generate enough variance to distinguish meaningful architectural effects from noise

Exact trial count is a Q34 decision based on the wall-time budget available at Q34 execution time.

### Bounded Runtime Per Trial

Each trial has a configurable wall-time ceiling. Trials exceeding it are marked `timeout` by the runner and excluded from Pareto frontier computation but retained in the leaderboard with their partial metrics. The exact per-trial timeout is defined in Q34; a reasonable initial value is 15–30 minutes per trial for binary classification under compact constraints.

### Search Strategy

Q34 will use random search or lightweight Bayesian sampling (e.g., Tree-structured Parzen Estimator) over the defined discrete space. No exhaustive grid search is performed — the search space has too many combinations for exhaustive evaluation with a bounded trial budget. The exact strategy is a Q34 decision.

**Rationale for random search as baseline:** Random search over a well-designed space is a surprisingly strong baseline for NAS at pilot scale. It produces an unbiased sample of the Pareto frontier, does not require a search algorithm that is itself a source of hyperparameter decisions, and is maximally transparent. Bayesian methods provide marginal gains at small trial counts.

### Search Over Science

Q32 prioritizes scientific signal over brute-force scale. A small, well-designed search over a meaningful space produces more interpretable results than an unconstrained large search. Interpretability is a first-class constraint: the goal is to understand which architectural decisions produce compact, high-performing classical heads — not to maximize raw search throughput.

---

## 10. Reproducibility Requirements

All Q34 NAS trials executed over this search space must meet the following reproducibility requirements. These are not aspirational — they are enforced by the Q31/Q31A runner infrastructure.

### Frozen YAML Config Per Trial

Every NAS trial produces a frozen YAML config via the Q31 runner framework. The config is written to `experiments/configs/<experiment_id>.yaml` and set read-only (`chmod 444`) before the trial starts. No trial result can be recorded without an associated frozen config.

### Seed Locking

Each trial uses a fixed seed from the config (`reproducibility.seed`). The seed is set before dataset loading, model initialization, and augmentation. The seed is recorded in the result JSON. Trials run without a fixed seed are invalid and excluded from all analysis.

### Git Commit Tracking

Every trial records the exact code state at execution time via `git rev-parse HEAD` (or `QSTRATA_GIT_COMMIT` env var). Trials with `git_commit: unknown` are flagged in the leaderboard. Published NAS results must reference a clean git commit.

### Leaderboard Integrity

Trial results are immutable once recorded. The runner writes the result JSON and leaderboard entry immediately after trial completion. No post-hoc editing of results is permitted. If a trial is found to have an error, a corrected re-run produces a new entry with a new `experiment_id`; the original entry is retained with a `status: retracted` annotation.

### Deterministic Orchestration

Given the same config and seed, re-running any NAS trial must produce results within numerical tolerance (AUROC within ±0.0001, F1 within ±0.0001). This was validated by the Q31A reproducibility test (loss_delta = 0.0 across two sequential runs). NAS without deterministic execution cannot be scientifically compared.

**NAS without reproducibility is scientifically invalid.** The Q31/Q31A infrastructure is the prerequisite for Q34 precisely because it enforces these requirements mechanically.

---

## 11. Search Space Interpretability

### Human-Readable Architecture Space

The Q32 search space must be small enough that a human expert can understand what architectural decision each dimension represents without simulation or enumeration. Each dimension corresponds to a specific architectural choice:

| Dimension | Architectural Decision |
|---|---|
| Backbone family | How spatial features are extracted and mixed |
| Channel width | How much representational capacity each layer has |
| Depth | How many feature transformation stages the head applies |
| Compression dimension | How aggressively features are compressed before classification |
| Activation | How nonlinearity is introduced |
| Normalization | How feature scale is controlled |
| Dropout | How overfitting is regularized |
| Pooling | How spatial information is aggregated to a vector |

### Interpretable Architecture Evolution

Moving from one point in the search space to another corresponds to a meaningful architectural change. For example:
- Increasing channel width from `[32, 64]` to `[64, 128]` doubles representational capacity at the cost of parameter count
- Switching from `standard_cnn` to `depthwise_sep` reduces parameters while maintaining depth
- Increasing compression dimension from 4 to 16 increases the capacity of the final decision boundary

These transitions are interpretable because they correspond to specific engineering decisions with known effects in the CNN literature.

### Explainable Pareto Tradeoffs

The Pareto frontier produced by Q34 should be describable in plain language. Expected interpretable frontier statements include:
- "Increasing channel width improves AUROC at the cost of parameter count and latency"
- "Depthwise separable blocks achieve comparable AUROC to standard CNNs with 30–50% fewer parameters"
- "Shallow heads with large compression dimensions converge faster but plateau earlier"

If the Pareto frontier cannot be described in plain language, the search space has too many uninterpretable interactions, which is a signal to reduce its dimensionality in the next search iteration.

### Avoid Architecture Chaos

Architectures from incompatible families (e.g., mixing attention heads with depthwise separable CNNs in a single trial) are not searched. Each architecture family is coherent: all blocks within a trial share the same family, activation, and normalization choice. This constraint preserves interpretability: a trial's architecture can be described as a member of a single design family, not as an incoherent mixture.

**Scientific interpretability is prioritized over raw search entropy.** A larger search space with mixed families produces results that are harder to interpret, harder to generalize from, and harder to compare against quantum heads.

---

## 12. Relationship to Future Quantum NAS

### Q32 Establishes the Classical Ceiling

Q34 (first NAS execution, using the Q32 search space) will produce a Pareto frontier of compact classical architectures. This frontier is the **classical ceiling** — the best achievable compact classical performance under the defined constraints.

The classical ceiling is the correct reference for quantum NAS comparison. It replaces Q17 (23,650 params, random init) and Q22 (one fixed point) as the classical reference. The ceiling is not a single model — it is a Pareto frontier.

### Q33 Defines Quantum NAS Search Spaces

Q33 will define DV and CV quantum head search spaces using the same multi-objective framework and the same runner infrastructure. Q33 quantum NAS search spaces will vary:
- DV: n_qubits, circuit_depth, ansatz_type, compression_dim
- CV: n_modes, cv_depth, squeezing_cap, encoding_scheme, compression_dim

Q33 quantum heads will be compared against the Q32 classical ceiling, not against Q17 or Q22.

### Quantum vs. Classical Ceiling Comparison Protocol

The comparison between quantum NAS (Q33/Q34) and classical ceiling (Q32/Q34) must be performed on the same Pareto frontier space:
- Same task: VinDr-SpineXR binary classification
- Same backbone: frozen C006-D040
- Same dataset split: canonical_42
- Same evaluation metrics: AUROC, F1, parameter count, latency

**Q32 classical ceiling comparison before Q33 quantum NAS is a methodological requirement, not a preference.** Claiming quantum benefit without a strong classical ceiling is a methodological error that would invalidate any residual advantage claim.

---

## 13. Local-First NAS Philosophy

All Q34 NAS trials execute on local GPU only. This is a hard constraint, not a soft preference.

**AWS and Ray remain blocked until local NAS is validated.** The sequencing is:

1. Q32 — design classical search space (this slice; design only)
2. Q33 — design quantum search space (design only)
3. Q34 — execute local pilot NAS using Q32 and Q33 spaces
4. Q35 — design distributed scaling (only after Q34 produces stable, interpretable results)

Distributed scaling does not begin before local search produces stable, interpretable results. The local-first principle protects against:
- **Premature infrastructure cost:** Cloud compute charges accumulate before reproducibility is validated
- **Reproducibility loss at scale:** A bug in trial execution that produces undetected corrupted results on 1 GPU corrupts 20× more results when distributed
- **Interpretability loss:** Debugging a distributed NAS run is substantially harder than debugging a local one

Local-first is not a resource constraint — it is a scientific integrity constraint.

---

## 14. Future Execution Phases

Q32 and Q33 are design-only slices. No NAS execution occurs in either. The planned execution sequence is:

```
Q33 → quantum NAS search-space design (design only)
      defines DV and CV quantum head search spaces
      same framework as Q32; same runner infrastructure
      no execution; no training; no architectures generated

Q34 → local NAS pilot (first execution; classical search)
      executes Q32 classical search space on local GPU
      executes Q33 quantum head search in same pilot
      produces Pareto frontiers for both classical and quantum
      validates local NAS before any distributed infrastructure
      no AWS; no Ray; no distributed execution

Q35 → distributed scaling design (design only; after Q34)
      designs distributed extension of Q34 infrastructure
      no cloud provisioning before this design is approved
      blocked until Q34 produces stable, validated local results
```

No execution occurs in Q32 or Q33. The first NAS trial runs in Q34.

---

## 15. Required Scientific Guardrail

> The QStrata NAS program prioritizes scientifically interpretable, reproducible, and compact optimization before scaling search complexity or distributed infrastructure. NAS exists to explore controlled tradeoffs, not to maximize uncontrolled architecture entropy.

---

```
Q32 status: COMPLETE — design only; no NAS execution
Q33 status: NEXT — quantum NAS search space design (design only)
Q34 status: PLANNED — first local NAS execution phase
Q35 status: BLOCKED — requires validated Q34 local NAS
Classical ceiling: UNDEFINED — will be produced by Q34
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 validated
```
