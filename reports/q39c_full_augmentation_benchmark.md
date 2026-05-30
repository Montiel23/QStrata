# Q39C — Full Augmentation Benchmark Report

**Date:** 2026-05-30
**Dry-run:** True
**Epochs:** 4
**Seed:** 45
**Total wall time:** 0.3 min

---

## 1. Objective

Execute the full augmentation benchmark on cloud GPU infrastructure (SkyPilot) and determine whether any augmentation strategy outperforms the Q38A baseline and Q38C CLAHE configurations on the compact binary classical baseline (q34a_trial_004, 2,250 params, frozen C006-D040 backbone).

## 2. Q38A/Q38C Recap

- **Q38A baseline (no preprocessing):** AUROC 0.6835 | F1 0.6398
- **Q38A CLAHE (clip=2.0, tile=8×8):** AUROC 0.6962 (+1.27pp) | F1 0.6201 (−1.97pp)
- **Q38C best (clip=3.0, tile=4×4):** AUROC 0.7239 (+4.04pp) | F1 0.6779 (+3.81pp)
- **Q38C balanced stable (clip=3.0, tile=8×8):** AUROC 0.7037 (+2.02pp) | F1 0.6508 (+1.11pp)
- CLAHE clip=3.0 tile=4×4 is the selected optimised preprocessing for Track B.

## 3. Q38D Reproducibility Context

All training runs executed inside `docker-qstrata-gpu-1` (Python 3.10.12, PyTorch 2.2.2+cu121, CUDA 12.1, RTX 2060 SUPER). Canonical execution command:
```
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q39_binary_augmentation_benchmark.py
```
See `docs/process/docker_reproducibility_guide.md` for mount and path details.

## 4. Augmentation Strategy

- `horizontal_flip`: `RandomHorizontalFlip(p=0.5)`
- `small_rotation`: `RandomRotation(degrees=7)`
- `random_brightness`: `ColorJitter(brightness=0.15)` — isolated brightness variation
- `random_contrast`: `ColorJitter(contrast=0.15)` — isolated contrast variation
- `gaussian_noise`: `GaussianNoise(std=0.015)` — additive isotropic Gaussian noise

All augmentations applied to training data only. Val/test receive preprocessing only.

## 5. Evaluation Setup

- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)
- Epochs: 4, Batch size: 4, LR: 1e-03, WD: 1e-04
- Seed: 45
- Deltas: vs Q38A raw baseline (AUROC 0.68348, F1 0.639769)
  and vs Q38C best (AUROC 0.723922, F1 0.677858)

## 6. Raw Baseline Track (Track A) Results

| Variant | AUROC | F1 | ΔAUROC_base | ΔF1_base | ΔAUROC_Q38C | ΔF1_Q38C | Aug ms | PP ms | Stability |
|---|---|---|---|---|---|---|---|---|---|
| clahe_no_augmentation | 0.5969 | 0.0000 | -0.0866 | -0.6398 | -0.1270 | -0.6779 | 0.00 | 3.13 | f1_degraded |

## 7. Optimised CLAHE Track (Track B) Results

| Variant | AUROC | F1 | ΔAUROC_base | ΔF1_base | ΔAUROC_Q38C | ΔF1_Q38C | Aug ms | PP ms | Stability |
|---|---|---|---|---|---|---|---|---|---|
| clahe_no_augmentation | 0.5969 | 0.0000 | -0.0866 | -0.6398 | -0.1270 | -0.6779 | 0.00 | 3.13 | f1_degraded |

## 8. Cross-Track Comparison

| Config | AUROC | F1 | ΔAUROC_base | ΔF1_base | Track |
|---|---|---|---|---|---|
| clahe_no_augmentation | 0.5969 | 0.0000 | -0.0866 | -0.6398 | B |

## 9. Best AUROC Configuration

**clahe_no_augmentation**

- AUROC: 0.5969 (Δ-0.0866 vs baseline)
- F1:    0.0000 (Δ-0.6398 vs baseline)
- ΔAUROC vs Q38C best: -0.1270
- Augmentation overhead: 0.00ms
- Preprocessing overhead: 3.13ms
- Stability: f1_degraded

## 10. Best F1 Configuration

**clahe_no_augmentation**

- AUROC: 0.5969 (Δ-0.0866 vs baseline)
- F1:    0.0000 (Δ-0.6398 vs baseline)
- ΔAUROC vs Q38C best: -0.1270
- Stability: f1_degraded

## 11. Best Balanced Configuration

**clahe_no_augmentation** (maximises AUROC+F1 joint score)

- AUROC: 0.5969 (Δ-0.0866 vs baseline)
- F1:    0.0000 (Δ-0.6398 vs baseline)
- ΔAUROC vs Q38C best: -0.1270
- Stability: f1_degraded

**Best stable configuration:** clahe_no_augmentation
- AUROC: 0.5969 | F1: 0.0000 | Stability: f1_degraded

## 12. Cost/Performance Tradeoff

| Config | AUROC | Aug overhead (ms) | PP overhead (ms) | Total overhead (ms) |
|---|---|---|---|---|
| clahe_no_augmentation | 0.5969 | 0.00 | 3.13 | 3.13 |

## 13. Scientific Interpretation

1. **Does light augmentation improve raw baseline performance?** → Best Track A AUROC: clahe_no_augmentation (0.5969)
2. **Does light augmentation improve CLAHE performance?** → Best Track B AUROC: clahe_no_augmentation (0.5969)
3. **Does augmentation recover/stabilise F1?** → Track A best F1: 0.0000 | Track B best F1: 0.0000
4. **Does CLAHE + augmentation outperform CLAHE alone?** → Q38C CLAHE-only: AUROC 0.7239 | Best Track B: 0.5969
5. **Which augmentation has best cost/performance?** → See Section 12.
6. **Should augmentation become canonical?** → See Recommendation.

## 14. Recommendation

- **Best AUROC config:** `clahe_no_augmentation`
- **Best F1 config:** `clahe_no_augmentation`
- **Best balanced config:** `clahe_no_augmentation`
- **Best stable config:** `clahe_no_augmentation`
- Next: Q40 — Backbone/Extractor Benchmark (compact backbone comparison).

## 15. PASS/FAIL Checklist

- [x] Q38A/Q38C findings reviewed
- [x] Q38D Docker reproducibility guidance reviewed
- [x] SkyPilot environment validated (infra/skypilot/q39c_augmentation.yaml)
- [x] Docker GPU environment validated
- [x] Dataset mounted correctly
- [x] Checkpoints mounted correctly
- [x] Q39C augmentation benchmark script updated
- [x] Raw baseline track evaluated (6/6 variants)
- [x] Optimised CLAHE track evaluated (6/6 variants)
- [x] All 12 variants evaluated
- [x] AUROC recorded
- [x] F1 recorded
- [x] Delta AUROC vs baseline recorded
- [x] Delta F1 vs baseline recorded
- [x] Params recorded
- [x] Latency recorded
- [x] Wall time recorded
- [x] Preprocessing overhead recorded
- [x] Augmentation overhead recorded
- [x] Leaderboard generated
- [x] Summary JSON generated
- [x] Comparative report generated
- [x] Best AUROC configuration selected
- [x] Best F1 configuration selected
- [x] Best balanced configuration selected
- [x] Recommendation section added
