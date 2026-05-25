# QStrata v1 Classical Anchor

- **Status:** Frozen
- **Date:** 2026-05-24
- **Branch:** develop
- **Tag:** v1_classical_anchor

---

## 2. Purpose

This document freezes the selected classical CNN candidate as a reproducible research checkpoint before moving into QNN / hybrid quantum-classical integration. The checkpoint closes the classical optimization phase and establishes a stable, documented reference point — a known architecture, hyperparameter configuration, and performance baseline — against which all future hybrid quantum-classical results will be compared. No further classical micro-optimization is planned after this freeze.

---

## 3. Frozen Candidate

| Field | Value |
|---|---|
| Candidate ID | C006-D040 |
| block_type | depthwise_sep |
| conv_channels | [64, 128] |
| dropout | 0.40 |
| params | 9,870 |

---

## 4. Metrics

| Metric | Value | Notes |
|---|---|---|
| best_val_acc | 91.79% | Primary fitness signal |
| test_acc | 86.86% | Analysis only — not fitness |
| latency | 0.474 ms/batch | Slice 30 rerun timing values |
| dataset | PneumoniaMNIST | Binary classification, 28×28 grayscale |
| seed | 42 | Used for Slice 30 candidate selection |

---

## 5. Why Freeze Now

The classical pipeline is working end-to-end: data loading, model building, training with stable benchmark protocol v1 (seed 42, best-validation checkpoint), Pareto ranking, and systematic single-variable hyperparameter sweeps have all been validated across multiple slices. C006-D040 achieves 91.79% best validation accuracy at 9,870 parameters with 0.474 ms/batch latency — a compact, efficient profile that sits within 0.77 pp of the v1 Pareto accuracy anchor (C001 at 92.56%) while using roughly half the parameters. Further classical tuning — weight decay expansion, learning rate sweep, or multi-seed robustness validation — would yield incremental improvements in a regime where the accuracy ceiling is already close and the search space is well-characterized. The research value of continued classical micro-optimization is low relative to its execution cost. The next meaningful step in the QStrata research roadmap is hybrid QNN integration, not continued single-variable sweeps.

---

## 6. What Is Intentionally Stopped

The following work is stopped at this checkpoint and will not be executed:

- **Dropout tuning** — complete; C006-D040 (dropout=0.40) selected over C006-D030 and C006-D020 in Slice 30.
- **Weight decay expansion** — Slice 32 (C006-D040-WD0000/WD0001/WD0005) is deferred and will not be executed.
- **Classical micro-optimization** — no further single-variable sweeps (learning rate, batch size, conv width, etc.) are planned for the classical anchor.
- **Multi-seed robustness validation of the classical anchor** — Slice 29 confirmed training stability across four seeds at the val acc level; formal multi-seed certification is deferred.

---

## 7. Next Phase

**Next phase:** QNN / hybrid quantum-classical integration

**Initial recommended direction:**

```
CNN feature extractor (C006-D040 architecture)
  → QNN classifier head
  → Binary classification on PneumoniaMNIST
```

The C006-D040 depthwise-separable feature extractor (9,870 params) serves as the frozen classical backbone. A quantum neural network layer or variational quantum circuit is introduced as a classifier head, replacing the final linear projection. No implementation is defined in this document. This section states intent and architectural direction only; no quantum code exists at this tag.

---

## 8. Git Checkpoint

**Tag:** `v1_classical_anchor`

**Tag message:** `QStrata v1 classical anchor frozen before QNN integration`

This annotated tag marks the exact commit at which the classical optimization phase is formally closed. All future QNN and hybrid work begins from this point. The tag provides a reproducible recovery point: checking out `v1_classical_anchor` gives the complete, verified classical pipeline with C006-D040 as the practical candidate.

---

## 9. Non-Goals

This checkpoint is explicitly **not**:

- A leaderboard claim — results are from a single-GPU, single-institution R&D experiment and have not been benchmarked against published baselines.
- A final production model — C006-D040 is a research candidate, not a deployed clinical tool.
- A statistical robustness proof — candidate selection was based on single-seed (seed 42) evaluation; multi-seed certification has not been performed.
- A VinDr-SpineXR integration milestone — the current scope is PneumoniaMNIST only; spine imaging integration is out of scope.
- A QNN implementation — no quantum code, circuit, or simulator exists at this tag; the QNN phase begins after this freeze.
