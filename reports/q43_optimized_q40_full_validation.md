# Q43 — Optimized Q40 Full Validation Report

**Date:** 2026-05-30
**Slice:** Q43
**Epochs:** 4
**Seeds:** [42, 7, 123, 999, 2025]
**DataLoader profile:** bs=8 nw=4 pm=True pw=True pf=2 (Q41 optimised, Q42 applied)
**Total wall time:** 12.4 min

---

## 1. Objective

Full 5-seed × 3-candidate Q40 validation run using the Q41-optimized DataLoader profile (applied by Q42). This is the first complete (non-dry-run) execution of the Q40 protocol at full depth (4 epochs per run).

## 2. Reference Baselines

- **Q38A raw baseline:** AUROC 0.6835 | F1 0.6398
- **Q38C best (CLAHE clip=3.0 tile=4×4):** AUROC 0.7239 | F1 0.6779

## 3. Candidates

| Candidate | Description |
|---|---|
| clahe_no_augmentation | CLAHE clip=3.0 tile=4×4, no augmentation |
| clahe_small_rotation | CLAHE + RandomRotation(degrees=7) |
| clahe_random_contrast | CLAHE + ColorJitter(contrast=0.15) |

## 4. Evaluation Protocol

- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)
- Epochs: 4 | Batch: 8 | LR: 1e-03 | WD: 1e-04
- Seeds: [42, 7, 123, 999, 2025]
- DataLoader: bs=8 nw=4 pm=True pw=True pf=2
- 95% CI: t-distribution, df=4 (t*=2.776)

## 5. Multi-Seed Summary

| Candidate | AUROC (mean ± std) | 95% CI AUROC | F1 (mean ± std) | 95% CI F1 | ΔAUROC vs Q38C | Seeds>Q38C | Stability |
|---|---|---|---|---|---|---|---|
| clahe_no_augmentation | 0.7168 ± 0.0087 | [0.7060, 0.7277] | 0.6255 ± 0.0368 | [0.5798, 0.6713] | -0.0071 | 1/5 | stable |
| clahe_small_rotation | 0.7134 ± 0.0147 | [0.6952, 0.7317] | 0.6185 ± 0.0533 | [0.5523, 0.6847] | -0.0105 | 1/5 | stable |
| clahe_random_contrast | 0.7084 ± 0.0077 | [0.6989, 0.7179] | 0.5940 ± 0.0854 | [0.4880, 0.7001] | -0.0155 | 0/5 | stable |

## 6. Per-Seed Results

| Candidate | Seed | AUROC | F1 | Accuracy | ΔAUROC vs Q38C |
|---|---|---|---|---|---|
| clahe_no_augmentation | 7 | 0.7094 | 0.5704 | 0.6505 | -0.0145 |
| clahe_no_augmentation | 42 | 0.7282 | 0.6535 | 0.6615 | +0.0043 |
| clahe_no_augmentation | 123 | 0.7213 | 0.6581 | 0.6403 | -0.0027 |
| clahe_no_augmentation | 999 | 0.7069 | 0.6391 | 0.6423 | -0.0170 |
| clahe_no_augmentation | 2025 | 0.7184 | 0.6065 | 0.6639 | -0.0055 |
| clahe_random_contrast | 7 | 0.6980 | 0.4718 | 0.6346 | -0.0259 |
| clahe_random_contrast | 42 | 0.7129 | 0.6420 | 0.6505 | -0.0110 |
| clahe_random_contrast | 123 | 0.7183 | 0.6661 | 0.6312 | -0.0056 |
| clahe_random_contrast | 999 | 0.7064 | 0.5370 | 0.6447 | -0.0175 |
| clahe_random_contrast | 2025 | 0.7063 | 0.6533 | 0.6346 | -0.0177 |
| clahe_small_rotation | 7 | 0.7034 | 0.5555 | 0.6432 | -0.0205 |
| clahe_small_rotation | 42 | 0.7350 | 0.6676 | 0.6538 | +0.0111 |
| clahe_small_rotation | 123 | 0.7159 | 0.6803 | 0.6018 | -0.0081 |
| clahe_small_rotation | 999 | 0.7162 | 0.5974 | 0.6586 | -0.0077 |
| clahe_small_rotation | 2025 | 0.6967 | 0.5917 | 0.6365 | -0.0272 |

## 7. Scientific Questions

**Q1: Does clahe_small_rotation outperform clahe_no_augmentation?**
→ False (Δ mean AUROC = -0.0034)

**Q2: Is improvement consistent across seeds?**
→ False (rotation beats Q38C in 1/5 seeds)

**Q3: Which candidate is most stable (lowest variance)?**
→ clahe_random_contrast (std_auroc = 0.0077)

**Q4: Which candidate becomes production default?**
→ **clahe_no_augmentation**
  Mean AUROC: 0.7168 | Mean F1: 0.6255
  Rationale: Highest mean AUROC among all candidates. Stability: stable.

## 8. Recommendation

**Production default:** `clahe_no_augmentation`

- Mean AUROC: 0.7168 (Δ -0.0071 vs Q38C)
- 95% CI AUROC: [0.7060, 0.7277]
- Mean F1: 0.6255 (Δ -0.0523 vs Q38C)
- 95% CI F1: [0.5798, 0.6713]
- Stability: stable

## 9. PASS/FAIL Checklist

- [x] Three candidates evaluated
- [x] Five seeds executed: [42, 7, 123, 999, 2025]
- [x] Full 4-epoch training (no dry-run truncation)
- [x] Q41 optimized DataLoader profile active
- [x] Mean AUROC computed
- [x] Standard deviation computed
- [x] 95% confidence intervals computed
- [x] Leaderboard CSV generated
- [x] Statistical comparison generated
- [x] Recommendation generated
