# Q38C — CLAHE Parameter Sweep Report

**Date:** 2026-05-28
**Dry-run:** False
**Epochs:** 4
**Seed:** 45
**Total wall time:** 148.9 min

---

## 1. Objective

Optimize CLAHE `clip_limit` and `tile_grid_size` parameters on the compact binary classical baseline (q34a_trial_004, 2,250 params, frozen C006-D040 backbone) to maximize AUROC gain while minimizing F1 degradation.

## 2. Q38A Findings Recap

- **Baseline:** AUROC 0.6835 | F1 0.6398
- **CLAHE (clip=2.0, tile=8×8):** AUROC 0.6962 (+1.27pp) | F1 0.6201 (−1.97pp)
- CLAHE was the only preprocessing that improved AUROC; all normalization methods degraded both metrics.
- Hypothesis: tuning clip_limit and tile_grid_size may improve AUROC further or reduce F1 loss.

## 3. CLAHE Parameter Grid

- `clip_limit`:     [1.0, 2.0, 3.0, 4.0]
- `tile_grid_size`: [4, 8, 16]
- **Total combinations:** 12

## 4. Evaluation Setup

- Model: q34a_trial_004 (depthwise_sep backbone, 2,250 trainable head params)
- Backbone: frozen C006-D040 checkpoint
- Epochs: 4, Batch size: 4, LR: 1e-3, WD: 1e-4
- Dataset: VinDr-SpineXR binary ROI 224×224 (fixed train/val/test split)
- Deltas computed against Q38A no-preprocessing baseline

## 5. Full Parameter Sweep Table

| Config ID | Clip | Tile | AUROC | F1 | ΔAUROC | ΔF1 | PP ms | Status |
|-----------|------|------|-------|----|--------|-----|-------|--------|
| clahe_clip3.0_tile4x4 | 3.0 | 4×4 | 0.7239 | 0.6779 | +0.0404 | +0.0381 | 2.53 | f1_moderate |
| clahe_clip4.0_tile4x4 | 4.0 | 4×4 | 0.7155 | 0.6699 | +0.0320 | +0.0302 | 2.42 | f1_moderate |
| clahe_clip4.0_tile8x8 | 4.0 | 8×8 | 0.7154 | 0.5809 | +0.0319 | -0.0589 | 7.54 | f1_moderate |
| clahe_clip4.0_tile16x16 | 4.0 | 16×16 | 0.7147 | 0.6594 | +0.0312 | +0.0196 | 28.77 | stable |
| clahe_clip3.0_tile16x16 | 3.0 | 16×16 | 0.7075 | 0.6441 | +0.0241 | +0.0044 | 27.59 | stable |
| clahe_clip3.0_tile8x8 | 3.0 | 8×8 | 0.7037 | 0.6508 | +0.0202 | +0.0111 | 7.69 | stable |
| clahe_clip2.0_tile16x16 | 2.0 | 16×16 | 0.7013 | 0.6235 | +0.0178 | -0.0163 | 28.59 | stable |
| clahe_clip2.0_tile8x8 | 2.0 | 8×8 | 0.6962 | 0.6201 | +0.0127 | -0.0197 | 7.49 | stable |
| clahe_clip1.0_tile16x16 | 1.0 | 16×16 | 0.6894 | 0.6069 | +0.0060 | -0.0329 | 26.96 | f1_moderate |
| clahe_clip1.0_tile8x8 | 1.0 | 8×8 | 0.6827 | 0.6181 | -0.0008 | -0.0217 | 7.83 | stable |
| clahe_clip1.0_tile4x4 | 1.0 | 4×4 | 0.6728 | 0.6171 | -0.0107 | -0.0227 | 2.78 | stable |
| clahe_clip2.0_tile4x4 | 2.0 | 4×4 | 0.6565 | 0.5645 | -0.0270 | -0.0753 | 2.39 | f1_degraded |

## 6. Best AUROC Configuration

**clahe_clip3.0_tile4x4**

- AUROC: 0.7239 (Δ+0.0404 vs baseline)
- F1:    0.6779 (Δ+0.0381 vs baseline)
- Preprocessing overhead: 2.53ms/image
- Status: f1_moderate

## 7. Best Balanced Configuration

**clahe_clip3.0_tile4x4** (maximizes AUROC + F1 joint score)

- AUROC: 0.7239 (Δ+0.0404 vs baseline)
- F1:    0.6779 (Δ+0.0381 vs baseline)
- Preprocessing overhead: 2.53ms/image
- Status: f1_moderate

## 8. F1 vs AUROC Tradeoff Analysis

| Config | AUROC | F1 | AUROC+F1 joint |
|--------|-------|----|----------------|
| clahe_clip3.0_tile4x4 | 0.7239 | 0.6779 | 1.4018 |
| clahe_clip4.0_tile4x4 | 0.7155 | 0.6699 | 1.3854 |
| clahe_clip4.0_tile16x16 | 0.7147 | 0.6594 | 1.3740 |
| clahe_clip3.0_tile8x8 | 0.7037 | 0.6508 | 1.3546 |
| clahe_clip3.0_tile16x16 | 0.7075 | 0.6441 | 1.3517 |
| clahe_clip2.0_tile16x16 | 0.7013 | 0.6235 | 1.3248 |
| clahe_clip2.0_tile8x8 | 0.6962 | 0.6201 | 1.3163 |
| clahe_clip1.0_tile8x8 | 0.6827 | 0.6181 | 1.3007 |
| clahe_clip1.0_tile16x16 | 0.6894 | 0.6069 | 1.2963 |
| clahe_clip4.0_tile8x8 | 0.7154 | 0.5809 | 1.2962 |
| clahe_clip1.0_tile4x4 | 0.6728 | 0.6171 | 1.2899 |
| clahe_clip2.0_tile4x4 | 0.6565 | 0.5645 | 1.2210 |

## 9. Cost/Performance Tradeoff

| Config | AUROC | PP overhead (ms) | AUROC per ms |
|--------|-------|-----------------|--------------|
| clahe_clip3.0_tile4x4 | 0.7239 | 2.53 | 0.2866 |
| clahe_clip4.0_tile4x4 | 0.7155 | 2.42 | 0.2954 |
| clahe_clip4.0_tile8x8 | 0.7154 | 7.54 | 0.0948 |
| clahe_clip4.0_tile16x16 | 0.7147 | 28.77 | 0.0248 |
| clahe_clip3.0_tile16x16 | 0.7075 | 27.59 | 0.0256 |
| clahe_clip3.0_tile8x8 | 0.7037 | 7.69 | 0.0915 |

## 10. Scientific Interpretation

1. **Which CLAHE parameters maximize AUROC?** → clahe_clip3.0_tile4x4 (AUROC 0.7239)
2. **Which parameters minimize F1 degradation?** → clahe_clip3.0_tile4x4 preserves F1 best while improving AUROC.
3. **Is there a balanced configuration?** → clahe_clip3.0_tile4x4 (AUROC 0.7239, F1 0.6779).
4. **Does stronger contrast destabilize compact classifiers?** → High clip_limit may over-enhance edges, disrupting frozen backbone BatchNorm statistics calibrated on unprocessed images. Tile size modulates spatial locality: smaller tiles increase local contrast but can amplify noise.
5. **Is CLAHE overhead justified?** → At 7–20ms/image overhead for +1–2pp AUROC, the cost is acceptable for diagnostic screening tasks where AUROC is the primary metric.

## 11. Recommendation

- **Best AUROC config (recommended for AUROC-focused runs):** `clahe_clip3.0_tile4x4`
- **Best balanced config (recommended for production):** `clahe_clip3.0_tile4x4`
- Next: Q38D — Apply optimized CLAHE parameters to compact CV quantum baseline.

## 12. PASS/FAIL Checklist

- [x] Q38A findings reviewed
- [x] CLAHE parameter sweep script created
- [x] All 12 parameter combinations evaluated
- [x] AUROC recorded
- [x] F1 recorded
- [x] Delta AUROC vs baseline recorded
- [x] Delta F1 vs baseline recorded
- [x] Latency recorded
- [x] Wall time recorded
- [x] Preprocessing overhead recorded
- [x] Leaderboard generated
- [x] Summary JSON generated
- [x] Comparative report generated
- [x] Best AUROC configuration selected
- [x] Best balanced configuration selected
- [x] Recommendation section added
- [x] Roadmap updated
- [x] No augmentation added
- [x] No NAS added
- [x] No multiclass execution added
- [x] No distributed execution added
