# Q32: Classical NAS Search Space Design

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q32 — Classical NAS Search Space Design  
**Status:** COMPLETE — design only; no NAS execution

---

## 1. Title

Q32: Classical NAS Search Space Design

---

## 2. Context

The QStrata binary benchmarking phase (Q17–Q29) established four frozen compact benchmarks on VinDr-SpineXR binary classification:

| Model | AUROC | F1 | Params |
|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 574 |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 526 |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 |

Q22 is the critical finding for Q32: a 526-parameter classical MLP head over the same frozen pretrained backbone as Q21 recovered approximately 70–75% of Q21's AUROC improvement over Q17. This result suggests that the frozen backbone representation is the dominant performance driver, and that compact classical heads can be highly competitive with compact quantum heads.

**The classical ceiling under systematic architecture search has not been established.** Q22 is a single fixed-point classical architecture. Whether compact classical heads can be substantially improved through architecture search — and whether the remaining DV and CV hybrid residuals survive a strong classical ceiling — is unknown.

Q31A validated the experiment runner for reproducible sequential execution (loss_delta = 0.0 across two runs, tolerance 0.0001). The infrastructure is ready for NAS execution. Q32 now defines the classical search space that Q34 will execute.

**Q32 is a design-only slice. No NAS execution, search trial generation, or model training occurs in Q32.**

---

## 3. Why Classical Ceiling Exploration Is Required

### The Compact Bottleneck Effect

Q22's result demonstrates that the compact bottleneck effect is real: a frozen pretrained backbone with a tiny trainable head dramatically outperforms a larger randomly-initialized classical CNN (Q17: AUROC 0.6224, 23,650 params vs Q22: AUROC 0.6625, 526 params). The backbone representation matters far more than the head complexity.

This finding creates a methodological requirement: before concluding that quantum heads provide performance advantages over classical heads, the space of compact classical head designs must be explored. Q22 occupies one fixed point in that space. Q32 defines the full space; Q34 searches it.

### Uninterpretable Residuals Without a Ceiling

The residual quantum advantages observed in binary benchmarking are:
- Q21 vs Q22 on AUROC: +0.0175 (DV over classical control)
- Q27 vs Q22 on F1: +0.0322 (CV over classical control)

These residuals are observed in single-seed experiments without confidence intervals. They could reflect:
1. Genuine quantum head advantage under the compact constraint
2. Q22 being a suboptimally configured classical head (one point in a large space)
3. Seed-specific variance

Without a systematic classical ceiling, interpretation 2 cannot be ruled out. If Q34 produces a classical architecture that matches or exceeds Q21 and Q27 under the same compact constraint, the residual quantum advantage disappears — and was never real. If Q34 cannot close the gap, the residual gains strengthened as potential evidence of quantum benefit.

**Without Q32 and Q34, any claim of quantum benefit rests on an arbitrarily chosen classical comparison point.** This is methodologically insufficient and scientifically unjustifiable.

---

## 4. Optimization Philosophy

Q32 defines a three-tier multi-objective optimization structure. All NAS phases use this structure. **Single-metric optimization is forbidden.**

### Primary Objectives (Must Optimize)

Both objectives are tracked for every trial. Pareto dominance is computed over at minimum these two:

- **Maximize AUROC** — rank-based, threshold-independent discriminative performance; primary cross-model comparison metric
- **Maximize F1** — threshold-dependent precision-recall tradeoff; clinical relevance; computed at 0.5 threshold

### Secondary Objectives (Should Optimize)

Included in extended Pareto analysis and required for all trial result records:

- **Minimize parameter count** — trainable head parameters only; compact models are the scientific target
- **Minimize inference latency** — ms/sample on local GPU; prevents pathological architectures with good AUROC but unacceptable deployment cost
- **Minimize runtime instability** — NaN/inf count across training; gradient norm anomaly frequency

### Tertiary Objectives (Consider)

Recorded but may not enter primary Pareto analysis:

- **Minimize peak memory usage** — GPU and CPU memory; hardware feasibility
- **Minimize training wall time** — resource cost tracking; prevents runaway trials

A model that maximizes AUROC while using 10× the parameter budget of Q21 is not a valid compact classical ceiling. A model that passes AUROC but crashes with NaN gradients on multiple seeds is not deployable. All three tiers of objectives must be tracked to produce a scientifically complete Pareto analysis.

---

## 5. Pareto Frontier Philosophy

### Non-Dominated Solutions

A trial is Pareto-optimal (non-dominated) if no other evaluated trial is strictly better on all tracked objectives simultaneously. The Pareto frontier is the set of all non-dominated trials.

In QStrata: a compact classical architecture is Pareto-optimal if no other evaluated architecture simultaneously achieves higher AUROC, higher F1, fewer parameters, and lower latency. A trial worse on one objective but better on another remains on the frontier.

### Tradeoff Surface

The Pareto frontier defines the tradeoff surface — the achievable boundary under the compact constraint and search space. Different positions on the frontier correspond to different engineering decisions: low-parameter/moderate-AUROC vs. moderate-parameter/high-AUROC vs. high-stability/moderate-performance. Each region is interpretable.

### Compact High-Performance as the Correct Reference

The correct reference for quantum NAS comparison (Q33/Q34) is the Q32 classical Pareto frontier — not Q17 (23,650 params, random init) and not just Q22 (one fixed point). Quantum NAS must be compared against the best compact classical architecture family, not against an arbitrarily chosen or systematically weak classical reference.

### Preserve Full Pareto Frontier

The full Pareto frontier produced by Q34 must be preserved intact for Q33 quantum head comparison. A single "best classical model" cannot be selected and the rest discarded before Q33 is complete. Quantum NAS may dominate some frontier regions while being dominated in others; only the full comparison reveals the pattern.

---

## 6. Searchable Dimensions

All dimensions are discrete. The Q34 search algorithm samples from these discrete options.

| Dimension | Options |
|---|---|
| **Backbone family** | `standard_cnn`, `depthwise_sep`, `residual_compact` |
| **Channel width** | `[16,32]`, `[32,64]`, `[64,128]`, `[96,192]` |
| **Depth** | `shallow` (2 blocks), `medium` (3–4 blocks), `compact_deep` (5–6 blocks with compression) |
| **Compression dimension** | 2, 4, 8, 16, 32 |
| **Activation** | `relu`, `gelu`, `silu` |
| **Normalization** | `batch_norm`, `group_norm`, `layer_norm` |
| **Dropout** | 0.0, 0.1, 0.2, 0.3, 0.5 |
| **Pooling** | `max_pool`, `avg_pool`, `adaptive_avg_pool` |
| **Learning rate** | Log-uniform [1e-4, 5e-3] — exact range defined in Q34 |
| **Weight decay** | Log-uniform [0.0, 1e-2] — exact range defined in Q34 |

**Search space size (rough estimate):** 3 × 4 × 3 × 5 × 3 × 3 × 5 × 3 ≈ 24,300 discrete configurations (continuous lr/wd are not enumerated). A 20–50 trial pilot is a small sample of this space, not exhaustive enumeration. Random or Bayesian sampling is the appropriate strategy.

**Interpretability:** Each dimension corresponds to a specific, well-understood architectural decision. The search space is compact enough that a human expert can understand what each trial represents without simulation.

Full dimension specifications with notes: `docs/architecture/q32_classical_nas_search_space.md` Section 6.

---

## 7. Forbidden Dimensions

The following architecture families and design decisions are **explicitly out of scope** for Q32 classical NAS.

| Forbidden | Scientific Rationale |
|---|---|
| Transformer architectures | Not compact CNNs; incomparable to Q21/Q22/Q27; different training regime |
| Attention mechanisms | Adds search dimensions beyond compact CNN scope; changes comparison baseline |
| ResNet50+, EfficientNet-B4+, ViT variants | Orders of magnitude above compact parameter target; not comparable |
| Diffusion models | Generative; not classifiers; entirely out of scope |
| Multimodal architectures | Requires data modalities absent from VinDr-SpineXR binary setup |
| Foundation models | Incompatible with local-first, compact constraints |
| >5M total parameters | Not meaningfully compact; incomparable to Q21/Q22/Q27 |
| Cloud-scale training | Q34 is local-only; cloud assumptions are incompatible with local-first |

**Rationale:** Q32 focuses on compact medical-imaging CNN systems. The compactness of quantum benchmarks (Q21: 574 params, Q27: 536 params) defines the regime of comparison. A classical NAS that produces a 50M-parameter architecture as the "ceiling" does not answer the question of whether compact classical heads can match compact quantum heads — it answers a different question entirely, and trivially.

---

## 8. Constraint System

### Hard Constraints (violation → trial marked `invalid`)

| Constraint | Value |
|---|---|
| Maximum trainable parameters (head only, excluding frozen backbone) | ≤ 100,000 |
| GPU memory ceiling | OOM-free at batch size 4 on local GPU |
| Execution environment | Local single GPU; no cloud |
| Reproducibility | Seed set from config; git commit recorded |

Hard constraint violations are enforced by the runner. Trials that violate hard constraints are recorded in the leaderboard with `status: invalid` and excluded from Pareto frontier computation.

### Soft Constraints (violation → flagged but trial retained)

| Constraint | Value |
|---|---|
| Inference latency | ≤ 2× Q21 latency (~110 ms/sample) |
| Training stability | No NaN/inf across any training epoch |
| Parameter count preference | 500–5,000 trainable parameters (preference; not required) |

Soft constraint violations are recorded in the result JSON and leaderboard. Unstable trials (NaN/inf) are excluded from Pareto frontier recommendation but retained in the leaderboard for diagnostic purposes.

---

## 9. Experiment Budget Philosophy

### Local Single-GPU Execution

All Q34 NAS trials execute on a single local GPU sequentially using the Q31/Q31A runner. One experiment at a time; no parallelism across trials. This is a hard constraint enforced by the local-first principle.

### Bounded Trial Count

Approximately 20–50 trials is the planning reference for the Q34 pilot. This range is sufficient to:
- Sample the search space with meaningful coverage
- Produce an interpretable Pareto frontier
- Complete within a time-bounded local session
- Generate enough variance to distinguish architectural effects from noise

The exact trial count is a Q34 decision based on available hardware and wall-time budget at execution time.

### Bounded Runtime Per Trial

Each trial has a configurable timeout. Trials exceeding it are marked `timeout` and retained in the leaderboard with partial metrics. An initial timeout of 15–30 minutes per trial is a reasonable planning estimate for binary classification under compact constraints.

### Search Strategy

Q34 will use random search or lightweight Bayesian sampling (e.g., Tree-structured Parzen Estimator). No exhaustive grid search. The rationale:
- Random search is an unbiased, transparent baseline that is well-understood
- At 20–50 trial budgets, Bayesian methods provide marginal gains over random
- Interpretability of the resulting Pareto frontier is higher with random search (no search algorithm artifacts)

**Q32 prioritizes scientific signal over brute-force scale.** A small well-designed pilot produces more interpretable results than an unconstrained large search.

---

## 10. Reproducibility Requirements

### Frozen YAML Config Per Trial

Every Q34 NAS trial produces a frozen YAML config via the Q31 runner: written to `experiments/configs/<experiment_id>.yaml`, `chmod 444`, before trial execution. No trial result exists without an associated frozen config.

### Seed Locking

Each trial uses a fixed seed from `reproducibility.seed` in the config. Set before dataset loading, model initialization, and augmentation. Recorded in result JSON.

### Git Commit Tracking

Every trial records the code state via `QSTRATA_GIT_COMMIT` env var or `git rev-parse HEAD`. Trials with `git_commit: unknown` are flagged. Published NAS results require a clean commit.

### Leaderboard Integrity

Trial results are immutable once recorded. No post-hoc editing. Erroneous trials are retained with `status: retracted`; corrected re-runs produce new entries.

### Deterministic Execution

Given the same config and seed, re-running any trial must produce results within tolerance (AUROC ±0.0001, F1 ±0.0001). This was validated by Q31A (loss_delta = 0.0 across two runs). NAS without deterministic execution cannot produce scientifically comparable trial results.

**NAS without reproducibility is scientifically invalid.** The Q31/Q31A infrastructure is the prerequisite for Q34 because it enforces these requirements mechanically. Running NAS trials through ad hoc scripts — without frozen configs, seed enforcement, git capture, and leaderboard immutability — would produce results that cannot be verified, compared, or published.

---

## 11. Relationship to Future Quantum NAS

### Q32 Classical Ceiling Is the Quantum NAS Reference

The Pareto frontier produced by Q34 over the Q32 search space is the classical ceiling. It is the correct reference for Q33 quantum NAS comparison — replacing Q17 (arbitrary unoptimized baseline) and Q22 (one fixed configuration) with a systematic, multi-point frontier.

### Q33 Quantum NAS Must Be Compared Against This Frontier

Q33 will define DV and CV quantum head search spaces. Q34 will execute trials over both classical and quantum spaces. The quantum Pareto frontier will be compared to the classical Pareto frontier on the same axes: AUROC, F1, parameter count, latency.

**Quantum benefit cannot be evaluated against an unoptimized classical reference.** If Q34 produces a classical architecture that matches Q21 and Q27, the binary-phase quantum residuals were not due to quantum heads but due to an underexplored classical design space. If Q34 cannot close the gap, the residuals are strengthened as candidate evidence of quantum benefit — though still not statistically validated without multi-seed confidence intervals.

### Sequencing

Q32 (this slice) → Q33 (quantum space design) → Q34 (both classical and quantum execution) → Q35 (distributed design, after Q34 validated) → multiclass (after Phases 3–5).

---

## 12. Limitations

**No NAS execution in this slice.** Q32 is design only. No trial configs are generated, no models are trained, no architecture search is run.

**No AWS or Ray.** All execution is local. Distributed infrastructure remains blocked until Q34 validates local NAS.

**No distributed execution.** Q34 is local, sequential, single-GPU.

**No generated architectures.** The search space definition does not produce candidate architectures. Candidate configs are generated by the Q34 search algorithm at execution time.

**No multiclass work.** Multiclass benchmarking is blocked until Phase 3 (Q32 ceiling), Phase 4 (Q33 quantum NAS), and Phase 5 (Q34 optimized binary release) are all complete.

**No object detection work.** Object detection is outside the current QStrata roadmap scope.

**Search space is a planning artifact.** The exact search space dimensions (channel widths, depths, learning rate ranges, trial count) are defined here as planning inputs. Q34 may refine these based on hardware constraints, preliminary trial results, and wall-time budgets at execution time. Refinement of the search space in Q34 does not require a new design slice — it requires documenting the changes in the Q34 report.

---

## 13. Next Slice

**Q33 — Quantum NAS Search Space Design**

Q33 is also design only. It will define the DV and CV quantum head search spaces, using the same multi-objective framework, the same runner infrastructure, and the same constraint system as Q32. Q33 does not execute any NAS trials. Q34 is the first local NAS execution phase for both classical and quantum search spaces.

**Q34 is the first execution phase.** Q33 defines the quantum search space; Q34 executes both the Q32 classical and Q33 quantum spaces in a joint pilot. The classical ceiling (from Q32/Q34) and the quantum Pareto frontier (from Q33/Q34) are produced in the same pilot and compared in the same report.

```
Q32 status: COMPLETE — design only; no NAS execution
Q33 status: NEXT — Quantum NAS Search Space Design (design only)
Q34 status: PLANNED — first local NAS execution (classical + quantum)
Q35 status: BLOCKED — requires validated Q34 local NAS
Classical ceiling: UNDEFINED — will be produced by Q34
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 validated
Object detection: BLOCKED — out of current roadmap scope
```
