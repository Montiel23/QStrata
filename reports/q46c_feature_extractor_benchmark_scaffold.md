# Q46C — Feature Extractor Benchmark Scaffold

**Date:** 2026-05-31
**Slice:** Q46C-FEATURE-EXTRACTOR-BENCHMARK-SCAFFOLD
**Branch:** feature/q46c_feature_extractor_benchmark_scaffold
**Type:** Scaffold only — no benchmark execution, no training

---

## Summary

This report documents the creation and validation of the Q46 Feature Extractor Benchmark execution scaffold. The scaffold provides:

- `scripts/run_q46b_feature_extractor_benchmark.py` — full benchmark runner with `--smoke`, `--full`, `--seed`, `--candidate`, `--output-dir`, `--dry-run` flags
- `configs/q46_feature_extractor_benchmark.yaml` — Q46A protocol config
- Dry-run validation confirms READY_WITH_WARNINGS

---

## 1. Dry-Run Result

**Classification: READY_WITH_WARNINGS**
**Dry-run PASS**

```
DRY-RUN RESULT: PASS  (classification: READY_WITH_WARNINGS)
Recommended next slice: Q46-FEATURE-EXTRACTOR-BENCHMARK
```

---

## 2. Dependency Inventory

| Dependency | Status | Path |
|-----------|--------|------|
| Baseline checkpoint | ✓ OK | `checkpoints/c006_d040_classical_anchor.pt` |
| Dataset | ✓ OK | `data/processed/vindr_binary_roi_224/` (train/val/test) |
| Head config (q34a_trial_004) | ✓ OK | `configs/experiments/q34a_classical_nas/q34a_trial_004.yaml` |
| qcore package | ✓ OK | `qcore/` |
| Output dirs | ✓ OK | `experiments/leaderboards/`, `experiments/results/`, `reports/` |
| torch / torchvision | ⚠ WARN | Not importable outside docker; available in `docker-qstrata-gpu-1` |

---

## 3. Candidate Backbone Imports

Validated inside `docker-qstrata-gpu-1` (torchvision 0.17.2+cu121):

| Candidate | Status | Backbone | Params | Feature Dim |
|-----------|--------|----------|--------|-------------|
| baseline | ✓ OK | C006-D040 frozen ResNet | ~11,000,000 | 512 |
| efficientnet_b0 | ✓ OK | EfficientNet-B0 (IMAGENET1K_V1) | 5,288,548 | 1280 + projection |
| mobilenetv3_small | ✓ OK | MobileNetV3-Small (IMAGENET1K_V1) | 2,542,856 | 576 + projection |
| mobilenetv3_large | ✓ OK | MobileNetV3-Large (IMAGENET1K_V1) | 5,483,032 | 960 + projection |
| convnext_tiny | ✓ OK | ConvNeXt-Tiny (IMAGENET1K_V1) | 28,589,128 | 768 + projection |

All candidates use frozen ImageNet-pretrained weights. A linear projection layer maps each backbone's feature dimension to the q34a_trial_004 head input.

---

## 4. Leaderboard Schema

### Per-seed result fields (14)
```
rank, candidate_id, seed, auroc, f1, accuracy,
params_backbone, params_head, params_total,
latency_ms_per_batch, wall_time_s,
delta_auroc_vs_q45a, delta_auroc_vs_q38c, delta_f1_vs_q38c
```

### Summary aggregation fields (18)
```
rank, candidate_id, row_type,
mean_auroc, std_auroc, ci95_lo_auroc, ci95_hi_auroc,
mean_f1, std_f1, ci95_lo_f1, ci95_hi_f1, mean_accuracy,
delta_mean_auroc_vs_q45a, delta_mean_auroc_vs_q38c, delta_mean_f1_vs_q38c,
seeds_auroc_beat_q45a, seeds_auroc_beat_q38c, decision
```

---

## 5. Runtime Contract

| Phase | Seeds | Candidates | Epochs | Cap | Output |
|-------|-------|-----------|--------|-----|--------|
| Smoke (Phase 1) | `[42]` | All 5 | 4 | 60 min | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` |
| Full (Phase 2) | `[42, 7, 123]` | All 5 | 4 | 120 min | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` |
| Extended (Phase 3) | `[42, 7, 123, 999, 2025]` | Winner only | 4 | 60 min | `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv` |

**Decision rule:** Winner must achieve `mean_auroc > 0.7196` (Q45A baseline). Phase 3 triggered if Phase 2 winner is within 0.005 AUROC of ceiling (0.7239).

---

## 6. Execution Command Reference

```bash
# Dry-run (no training — validates scaffold):
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run

# Phase 1 — smoke (1 seed, ~60 min cap):
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke

# Phase 2 — full (3 seeds, ~120 min cap):
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --full

# Single candidate, single seed:
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke \
        --candidate efficientnet_b0 --seed 42
```

---

## 7. Blocker Table

| Blocker ID | Description | Severity | Required Fix |
|-----------|-------------|----------|-------------|
| BLK-001 | torch/torchvision not available outside docker | LOW | Run benchmark inside `docker-qstrata-gpu-1` |

No critical blockers. All dataset, checkpoint, and config dependencies are satisfied.

---

## 8. Readiness Classification

**READY_WITH_WARNINGS**

All structural dependencies are satisfied. The scaffold is complete and the dry-run validates all components. The only warning is that torch/torchvision must be accessed inside the GPU docker container for actual execution — this is the standard execution path for all QSTRATA benchmarks.

---

## 9. Recommended Next Slice

**Q46-FEATURE-EXTRACTOR-BENCHMARK**

The scaffold is complete. The benchmark is ready to execute inside `docker-qstrata-gpu-1` using the validated scaffold script. No dependency fixes are required.

Execute with:
```bash
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke
```
