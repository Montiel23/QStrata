# Minimal NAS / Pareto Evaluation Implementation Plan v1

- **Status:** Draft
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:** docs/design/nas_search_space_v1_reduced.md, Slice 22

---

## Purpose

This document defines the v1 NAS evaluation strategy before any code is written, establishing the approach, tooling constraints, and implementation scope that a future implementation slice must follow. With only three fixed candidates, exhaustive sequential enumeration is the correct approach — simpler, cheaper, and more transparent than evolutionary search, and requiring no additional dependencies. pymoo and NSGA-II are explicitly deferred until the search space grows beyond a small fixed candidate set where enumeration is no longer practical and the overhead of evolutionary machinery is justified. All evaluations must follow the approved stable benchmark protocol v1 as defined in `docs/design/nas_benchmark_protocol_v1.md`.

---

## Approved v1 Strategy

- **Evaluation approach:** Exhaustive enumeration — all three approved candidates evaluated sequentially, one per run
- **Training backend:** PyTorch — using existing `build_model()` and dataset interfaces
- **Execution mode:** Sequential single-GPU — one candidate evaluated at a time on the local RTX 2060 SUPER
- **Benchmark protocol:** `nas_benchmark_protocol_v1` — seed 42, best-validation checkpoint, test accuracy excluded from fitness
- **Result tracking:** Simple local file — CSV or SQLite; no experiment tracking framework in v1
- **Output:** Pareto ranking across validation accuracy, parameter count, and latency — produced after all candidates complete
- **Parallelism:** None in v1 — sequential evaluation only
- **pymoo / NSGA-II:** Explicitly deferred — not used in v1; to be reconsidered when the search space grows beyond exhaustive enumeration

---

## Candidate Set

The v1 evaluation is scoped to exactly the three approved candidates from `nas_search_space_v1_reduced.md`:

| Candidate | block_type | conv_channels |
|---|---|---|
| C001 | standard | [32, 64] |
| C004 | depthwise_sep | [32, 64] |
| C006 | depthwise_sep | [64, 128] |

The candidate set may be expanded in a future version, at which point evolutionary search should be reconsidered.

---

## Evaluation Settings

- **Candidate set:** C001, C004, C006 — evaluated exhaustively
- **Evaluations per candidate:** One stable evaluation per candidate — no repeated trials
- **Random seed:** 42 — applied consistently with stable benchmark protocol v1
- **Evaluation order:** Sequential — C001 first, then C004, then C006
- **Distributed execution:** None — single GPU, single process
- **Parallelism:** None

---

## Objectives and Pareto Ranking

The following objectives are used to rank candidates after all evaluations complete:

1. **Maximize best validation accuracy** — primary fitness signal; best validation accuracy across all epochs using stable protocol v1
2. **Minimize parameter count** — model size efficiency objective
3. **Minimize mean inference latency** — runtime efficiency objective; measured as ms/batch on the GPU stack

After all three candidates are evaluated, a Pareto ranking is produced across these three objectives. A candidate is Pareto-dominant if no other candidate is better on all three objectives simultaneously.

The following are explicitly excluded from v1 objectives:
- Estimated cloud cost — no cloud execution in v1
- Training time — explicitly deferred as a tracked objective in v1 to keep the objective space small
- FLOPs — not measured in current benchmark infrastructure
- Energy usage — not instrumented

Test accuracy is excluded from all fitness and ranking decisions. It is reported for final analysis only after evaluation completes.

---

## When to Introduce pymoo / NSGA-II

Evolutionary search should be reconsidered when any of the following conditions apply:

- The search space grows to a size where exhaustive enumeration is no longer practical — when exhaustive evaluation becomes operationally inefficient for the available hardware and runtime budget
- Continuous or mixed-type search variables are introduced (e.g. tunable dropout, kernel size)
- Multi-objective trade-off analysis requires population-level diversity that enumeration cannot provide

None of these conditions apply to the current three-candidate v1 search space.

---

## Proposed Code Modules

The following files are proposed for implementation in a future slice. They are listed here as planned file paths only and are not created in this slice:

- `qcore/nas/evaluator.py` — wraps the stable benchmark protocol for sequential candidate evaluation
- `qcore/nas/pareto.py` — filters and ranks candidates by Pareto dominance across the three objectives
- `scripts/run_nas_v1_eval.py` — entry point for running the sequential evaluation sweep and producing the Pareto ranking
- `experiments/nas/nas_v1_eval.yaml` — configuration file for the v1 evaluation run (candidate configs, seed, output path)

The exact module structure is subject to human review before implementation begins.

---

## Out of Scope for v1

- pymoo installation or any dependency changes
- NSGA-II or any evolutionary search
- Ray or any distributed execution framework
- AWS, SkyPilot, or SageMaker
- Docker changes
- Training runs or experiments in this slice
- Notebook modifications
- Cloud cost objective
- FLOPs objective
- Energy objective
- Parallel or multi-GPU evaluation
- Any NAS code written in this slice

---

## Next Step Recommendation

The next human-approved slice may prepare a minimal NAS evaluation implementation design — specifying the module interfaces, data flow, and output format for the four proposed code modules — but must not write code or create files. A full evaluation sweep must not begin until a future implementation slice is explicitly reviewed and approved by the human architect. No implementation work of any kind should begin until this plan is explicitly approved.
