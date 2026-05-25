# NAS Search Space v1 — Reduced Candidate Set

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Supersedes:** docs/design/nas_search_space_v0.md
- **Informed by:** reports/nas_v0_candidate_benchmark_stable.md, Slice 21

---

## Purpose

This document records the evidence-based reduction from NAS search space v0 — six candidates across two block types and three channel configurations — to v1, which retains only three Pareto-anchor survivors. The reduction keeps only candidates that define distinct, non-dominated points on the accuracy-efficiency trade-off surface, before NSGA-II implementation begins. Expanding NAS around weak or redundant candidates would waste search budget on trajectories already shown to be dominated: if a candidate offers neither better accuracy nor better efficiency than an existing survivor, it contributes no new information to the search. All decisions are grounded in the Slice 21 stable benchmark results, which used a fixed seed of 42 and best-validation checkpoint selection per protocol v1.

---

## Evidence Summary

| Candidate | block_type | conv_channels | Params | Best val acc | Best epoch | Final train acc | Test acc* | Mean epoch (s) | Latency (ms/batch) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C001 | standard | [32,64] | 19,138 | 92.56% | 7 | 90.46% | 86.54% | 0.47 | 0.65 |
| C002 | standard | [48,96] | 42,530 | 91.41% | 5 | 90.99% | 87.82% | 0.55 | 0.56 |
| C003 | standard | [64,128] | 75,138 | 92.37% | 7 | 91.19% | 83.17% | 0.62 | 0.55 |
| C004 | depthwise_sep | [32,64] | 2,894 | 91.41% | 7 | 89.95% | 87.66% | 0.49 | 0.70 |
| C005 | depthwise_sep | [48,96] | 5,870 | 91.41% | 5 | 90.74% | 86.22% | 0.52 | 0.76 |
| C006 | depthwise_sep | [64,128] | 9,870 | 91.98% | 9 | 91.53% | 86.22% | 0.58 | 0.68 |

> *Test accuracy is reported for analysis only. It must not be used as a fitness signal during NAS search. Best validation accuracy is the sole NAS fitness signal.

---

## Selection Rule

- **Fitness signal for NAS search:** Best validation accuracy — highest value recorded across all epochs using seed 42 and the stable protocol v1
- **Efficiency objectives:** Parameter count (minimize) and mean inference latency (minimize)
- **Excluded from selection decisions:** Test accuracy — reported for analysis only; must not be used to rank or select candidates

---

## Pareto Survivors — Accepted into v1

The following three candidates are accepted into the reduced v1 search space.

**C001 — standard [32,64]**
- Params: 19,138 | Best val acc: 92.56% | Latency: 0.65 ms/batch
- Rationale: Accuracy anchor and reference baseline. Highest best validation accuracy across the full grid. All other candidates are evaluated relative to C001.

**C004 — depthwise_sep [32,64]**
- Params: 2,894 | Best val acc: 91.41% | Latency: 0.70 ms/batch
- Rationale: Ultra-parameter-efficient anchor. Achieves within 1.15 pp of C001 validation accuracy at 6.6× fewer parameters. Represents the extreme efficiency corner of the Pareto surface.

**C006 — depthwise_sep [64,128]**
- Params: 9,870 | Best val acc: 91.98% | Latency: 0.68 ms/batch
- Rationale: Balanced accuracy-efficiency anchor. Closes the validation gap to 0.58 pp of C001 at roughly half the parameter count. Defines the middle of the Pareto trade-off surface between C001 and C004.

---

## Excluded Candidates — Removed from v1

The following three candidates are excluded from the v1 search space.

**C002 — standard [48,96]**
- Params: 42,530 | Best val acc: 91.41%
- Rationale: No meaningful validation accuracy gain over C001. Parameter count is 2.2× higher than C001 with lower best validation accuracy. Adds no distinct point to the trade-off surface.

**C003 — standard [64,128]**
- Params: 75,138 | Best val acc: 92.37%
- Rationale: Near-identical validation accuracy to C001 (0.19 pp gap) but 3.9× the parameter count. Excessive model size for the current small-model objective. Dominated by C001 on the efficiency axis.

**C005 — depthwise_sep [48,96]**
- Params: 5,870 | Best val acc: 91.41%
- Rationale: Equal validation accuracy to C004 at double the parameter count, and lower best validation accuracy than C006 with higher latency. Not better than either C004 or C006 on the trade-off surface. Provides no additional information to the NAS search.

---

## Reduced v1 Search Variables

**block_type** — searchable:
- `standard`
- `depthwise_sep`

**conv_channels** — searchable:
- `[32, 64]`
- `[64, 128]`

**Explicitly excluded from v1:**
- `[48, 96]` — the midpoint channel configuration did not justify itself across either block type in the Slice 21 benchmark

**Fixed (unchanged from baseline):**
- `dropout`, `batch_norm`, `optimizer`, `learning rate`, `epochs`, `class weights`, `dataset`, `input shape`

---

## Next Step Recommendation

The next human-approved slice may prepare a minimal NSGA-II implementation plan only — defining the search loop structure, candidate evaluation protocol, and Pareto front tracking approach, without writing any NAS code. NAS implementation must not begin until that implementation plan has been reviewed and explicitly approved by the human architect. No NAS code should be written until the human architect has reviewed and approved both this reduced search space and the implementation plan.
