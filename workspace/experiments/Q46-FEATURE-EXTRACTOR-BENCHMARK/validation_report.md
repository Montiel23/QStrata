# Q46 Feature Extractor Benchmark — Validation Report

**Slice ID**: Q46-FEATURE-EXTRACTOR-BENCHMARK  
**Date**: 2026-06-02  
**Branch**: feature/q46c_feature_extractor_benchmark_scaffold  
**Script**: `scripts/run_q46b_feature_extractor_benchmark.py`  
**Fix commit**: d2f9694  
**Full run commit**: (this report)

---

## Verdict: WINNER — Multiple Candidates Clear Q45A Baseline

All 4 torchvision backbones decisively beat the Q45A decision threshold (mean_auroc > 0.7196,
≥ 2/3 seeds above threshold). The project objectives `reach_auroc_80_binary` and
`reach_auroc_90_binary_if_feasible` are both achieved.

---

## 1. Execution Summary

| Phase | Seeds | Candidates | Status |
|---|---|---|---|
| Phase 1 — Smoke | [42] | 5 | PASS (exit 0) |
| Phase 2 — Full | [42, 7, 123] | 5 | PASS (exit 0, leaderboard written) |

**Dry-run classification**: READY  
**Total Phase 2 wall time**: ~9 min (well within 120-min cap)

---

## 2. Phase 2 — 3-Seed Summary Leaderboard

| Rank | Candidate | Mean AUROC | Std | CI95 | Mean F1 | Seeds>Q45A | Decision |
|---|---|---|---|---|---|---|---|
| 1 | mobilenetv3_large | **0.9873** | 0.0004 | [0.9862, 0.9885] | 0.9331 | 3/3 | WINNER |
| 2 | convnext_tiny | **0.9865** | 0.0008 | [0.9845, 0.9884] | 0.9343 | 3/3 | WINNER |
| 3 | efficientnet_b0 | **0.9832** | 0.0035 | [0.9746, 0.9918] | 0.9342 | 3/3 | WINNER |
| 4 | mobilenetv3_small | **0.9822** | 0.0026 | [0.9758, 0.9886] | 0.9078 | 3/3 | WINNER |
| 5 | baseline | 0.7265 | 0.0215 | [0.6730, 0.7799] | 0.6066 | 2/3 | WINNER |

**Q45A decision threshold**: 0.7196 (3-seed mean AUROC from Q45A augmentation phase)  
**Q38C best single-run**: 0.7239 (CLAHE clip=3.0 tile=4×4)

---

## 3. Per-Seed Results

### baseline
| Seed | AUROC | F1 | Acc | Δvs_Q45A |
|---|---|---|---|---|
| 42 | 0.743932 | 0.670831 | 0.685748 | +0.0243 |
| 7 | 0.733006 | 0.589260 | 0.662493 | +0.0134 |
| 123 | 0.702416 | 0.559647 | 0.643411 | −0.0172 |

### efficientnet_b0
| Seed | AUROC | F1 | Acc | Δvs_Q45A |
|---|---|---|---|---|
| 42 | 0.979869 | 0.928486 | 0.928444 | +0.2603 |
| 7 | 0.986763 | 0.942047 | 0.943948 | +0.2672 |
| 123 | 0.982929 | 0.932153 | 0.931425 | +0.2633 |

### mobilenetv3_small
| Seed | AUROC | F1 | Acc | Δvs_Q45A |
|---|---|---|---|---|
| 42 | 0.984298 | 0.922299 | 0.926655 | +0.2647 |
| 7 | 0.982949 | 0.893617 | 0.904592 | +0.2633 |
| 123 | 0.979311 | 0.907574 | 0.914132 | +0.2597 |

### mobilenetv3_large
| Seed | AUROC | F1 | Acc | Δvs_Q45A |
|---|---|---|---|---|
| 42 | 0.987149 | 0.944238 | 0.946333 | +0.2675 |
| 7 | 0.987027 | 0.952092 | 0.952892 | +0.2674 |
| 123 | 0.987855 | 0.902970 | 0.912343 | +0.2683 |

### convnext_tiny
| Seed | AUROC | F1 | Acc | Δvs_Q45A |
|---|---|---|---|---|
| 42 | 0.985628 | 0.933810 | 0.933810 | +0.2660 |
| 7 | 0.987157 | 0.932015 | 0.934407 | +0.2676 |
| 123 | 0.986682 | 0.937046 | 0.937984 | +0.2671 |

---

## 4. Project Objective Assessment

| Objective | Target | Best Achieved | Status |
|---|---|---|---|
| reach_auroc_80_binary | AUROC ≥ 0.80 | 0.9873 (mobilenetv3_large) | **EXCEEDED** |
| reach_auroc_90_binary_if_feasible | AUROC ≥ 0.90 | 0.9873 | **ACHIEVED** |

All four torchvision backbones cleared 0.97 AUROC across all seeds.

---

## 5. Runtime Blockers Fixed (SF-FIX-021)

Eight bugs were resolved in `scripts/run_q46b_feature_extractor_benchmark.py` (commit d2f9694):

1. **T.ToTensor() removed** — dataset returns float32 tensors; `T.ToTensor()` crashed on tensor input
2. **Per-candidate transforms** — torchvision backbones need 3-channel + ImageNet norm; baseline stays 1-channel grayscale
3. **CrossEntropyLoss** — head outputs `(B, 2)` logits; BCEWithLogitsLoss caused shape mismatch
4. **labels.long()** — CrossEntropyLoss requires long targets, not float
5. **.squeeze(1) removed** — no effect on `(B, 2)` tensor, was incorrectly included
6. **eval_split criterion arg** — missing required 4th argument caused TypeError
7. **Projection to BACKBONE_OUT_DIM=128** — build_nas_head expects 128-dim input; projection was targeting wrong dimension
8. **Baseline feature_dim corrected** — C006-D040 outputs 128-dim, not 512; build_backbone_extractor now returns BACKBONE_OUT_DIM

---

## 6. Decision

**WINNER**: `mobilenetv3_large` (rank 1, mean_auroc=0.9873, std=0.0004, 3/3 seeds above threshold)

Recommended next slice: Q46G — fine-tuning or head-architecture sweep on MobileNetV3-Large backbone
to further refine the model for production deployment.

---

## 7. Pass/Fail Checklist

- [x] Execution completes without runtime errors (Phase 1 exit 0, Phase 2 exit 0)
- [x] Expected output files present:
  - `scripts/run_q46b_feature_extractor_benchmark.py` (fixed, committed d2f9694)
  - `workspace/projects/qstrata/leaderboards/q46-feature-extractor-benchmark.csv`
  - `workspace/experiments/Q46-FEATURE-EXTRACTOR-BENCHMARK/validation_report.md`
  - `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv`
  - `experiments/leaderboards/q46b_extractor_full_leaderboard.csv`
- [x] Feature Extractor phase advancement confirmed: all torchvision backbones clear AUROC 0.97+
- [x] Goal met: reach_auroc_80_binary ✓ and reach_auroc_90_binary_if_feasible ✓
- [x] Local commits only (no push)
- [x] Only intended files staged

---

```
Slice: Q46-FEATURE-EXTRACTOR-BENCHMARK
Protocol: Q46A (3-seed full evaluation)
Script: scripts/run_q46b_feature_extractor_benchmark.py
Fix commit: d2f9694
Report commit: (pending)
```
