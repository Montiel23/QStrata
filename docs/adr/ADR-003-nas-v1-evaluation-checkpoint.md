# ADR-003: NAS v1 Evaluation Checkpoint

- **Status:** Accepted
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:**
  - docs/design/nas_evaluation_plan_v1.md
  - docs/design/nas_interfaces_v1.md
  - reports/nas_v1_eval.md
  - Slice 27 evaluation run

---

## Context

NAS v1 was intentionally not NSGA-II — with only three approved candidates (C001, C004, C006), exhaustive sequential enumeration was the correct and sufficient approach; evolutionary search machinery would add complexity and cost with no benefit over a three-point grid. The evaluation ran inside the GPU Docker stack using `scripts/run_nas_v1_eval.py` with `experiments/nas/nas_v1_eval.yaml`, executing each candidate sequentially on the local RTX 2060 SUPER. All evaluations followed stable benchmark protocol v1: seed 42 applied to Python, NumPy, PyTorch, and CUDA before any data loading or model instantiation, validation accuracy tracked every epoch, best-validation checkpoint selected in memory via `copy.deepcopy`, and test accuracy evaluated at the best-validation checkpoint and excluded from all fitness and ranking decisions. The Pareto ranking was computed after all three candidates completed, across three objectives: maximize best validation accuracy, minimize parameter count, and minimize mean inference latency.

---

## Results

| Candidate | block_type | conv_channels | Params | Best val acc | Best epoch | Final train acc | Test acc* | Mean epoch time | Latency (ms/batch) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C001 | standard | [32,64] | 19,138 | 92.56% | 7 | 90.46% | 86.54% | 0.48s | 0.590 |
| C004 | depthwise_sep | [32,64] | 2,894 | 91.41% | 7 | 89.95% | 87.66% | 0.44s | 0.698 |
| C006 | depthwise_sep | [64,128] | 9,870 | 91.98% | 9 | 91.53% | 86.22% | 0.57s | 0.589 |

> *Test accuracy is analysis only and must not be used as a fitness signal.

---

## Pareto Front

All three candidates — C001, C004, and C006 — are on the Pareto front. No candidate is dominated by any other across the three objectives simultaneously.

- **C001** — highest best validation accuracy (92.56%); no other candidate matches it on the accuracy objective. C001 has higher parameter count and comparable latency to C006, but its accuracy advantage means no other candidate can dominate it.

- **C004** — lowest parameter count (2,894); no other candidate matches it on the parameter efficiency objective. C004 achieves within 1.15 pp of C001's validation accuracy at 6.6× fewer parameters, and no other candidate is simultaneously better on all three objectives.

- **C006** — closest to C001 on validation accuracy (91.98%, gap of 0.58 pp) while using roughly half the parameters of C001 (9,870 vs 19,138); no other candidate strictly dominates it across all three objectives simultaneously. C006 is better than C004 on accuracy and better than C001 on parameter count, placing it firmly on the trade-off surface between them.

---

## Decision

### Accepted

- **C001 (`standard [32,64]`)** — accuracy anchor and reference baseline; highest best validation accuracy across the v1 candidate set. All other candidates are evaluated relative to C001 on the accuracy objective.

- **C004 (`depthwise_sep [32,64]`)** — ultra-light parameter-efficiency anchor; 6.6× fewer parameters than C001 with only a 1.15 pp validation gap. Represents the extreme efficiency corner of the Pareto surface.

- **C006 (`depthwise_sep [64,128]`)** — recommended practical trade-off candidate; best balance of validation accuracy, parameter count, and latency across the v1 candidate set.

### Recommended candidate for follow-up

C006 is the recommended practical candidate for follow-up validation. Not because it dominates everything — it does not — but because it offers the most balanced trade-off across all three objectives: validation accuracy within 0.58 pp of C001, parameter count approximately half of C001 (9,870 vs 19,138), and latency matching C001 within measurement variance (0.589 vs 0.590 ms/batch). It is the most actionable candidate for real-world deployment consideration, where neither maximum accuracy nor minimum parameter count alone is the decisive criterion.

---

## Guardrails

The following constraints must not be violated before explicit human approval:

- Test accuracy must not be used as a NAS fitness signal — it remains analysis-only and must never influence candidate ranking or selection
- NSGA-II must not be introduced until the search space grows beyond what exhaustive enumeration can handle efficiently
- pymoo must not be installed or used in v1
- Ray, AWS, SkyPilot, and SageMaker remain out of scope
- Dashboards, MLflow, W&B, TensorBoard, and monitoring stacks must not be added
- All benchmark outputs remain terminal-first: stdout summaries, markdown reports, and optional CSV — no databases or tracking servers

---

## Recommended Next Step

The next human-approved slice should be one of exactly two options:

**(a) Targeted follow-up validation of C006** — run additional evaluations of C006 to confirm its stability and generalisability, for example by varying the random seed or evaluating on a held-out data partition. This would increase confidence before committing to C006 for deployment.

**(b) A human-justified expansion of the search space** — add one new manually validated search variable (e.g. a new block type, a new channel configuration, or a new training hyperparameter) and define a revised candidate grid, grounded in explicit evidence that the current v1 space does not adequately cover the trade-off surface.

Neither option should be implemented in this slice. The choice between options (a) and (b) requires explicit human approval before any work begins — no code, no config, and no experiment should be created until that decision has been made and communicated.
