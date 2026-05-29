# Q40 — Top Candidate Validation Report

**Date:** 2026-05-29
**Dry-run:** True
**Epochs:** 4
**Seeds:** [42, 7, 123, 999, 2025]
**Total wall time:** 9.8 min

---

## 1. Objective

Validate whether the best augmentation candidates from Q39 truly outperform the CLAHE baseline under a rigorous 5-seed evaluation protocol.

## 2. Reference Baselines

- **Q38A raw baseline:** AUROC 0.6835 | F1 0.6398
- **Q38C best (CLAHE clip=3.0 tile=4×4):** AUROC 0.7239 | F1 0.6779

## 3. Candidates

| Candidate | Description |
|---|---|
| clahe_no_augmentation | CLAHE clip=3.0 tile=4×4, no augmentation (Q38C baseline) |
| clahe_small_rotation | CLAHE + RandomRotation(degrees=7) |
| clahe_random_contrast | CLAHE + ColorJitter(contrast=0.15) |

## 4. Evaluation Protocol

- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)
- Epochs: 4 | Batch: 4 | LR: 1e-03 | WD: 1e-04
- Seeds: [42, 7, 123, 999, 2025]
- 95% CI: t-distribution, df=4 (t*=2.776)

## 5. Multi-Seed Summary

| Candidate | AUROC (mean ± std) | 95% CI AUROC | F1 (mean ± std) | 95% CI F1 | ΔAUROC vs Q38C | Seeds>Q38C | Stability |
|---|---|---|---|---|---|---|---|
| clahe_no_augmentation | 0.4972 ± 0.0773 | [0.4012, 0.5931] | 0.5137 ± 0.2874 | [0.1569, 0.8705] | -0.2267 | 0/5 | unstable |
| clahe_small_rotation | 0.5071 ± 0.0854 | [0.4011, 0.6130] | 0.5160 ± 0.2888 | [0.1575, 0.8746] | -0.2169 | 0/5 | unstable |
| clahe_random_contrast | 0.5144 ± 0.0616 | [0.4379, 0.5909] | 0.5062 ± 0.2837 | [0.1540, 0.8585] | -0.2095 | 0/5 | unstable |

## 6. Per-Seed Results

| Candidate | Seed | AUROC | F1 | Accuracy | ΔAUROC vs Q38C |
|---|---|---|---|---|---|
| clahe_no_augmentation | 7 | 0.4091 | 0.6356 | 0.4661 | -0.3148 |
| clahe_no_augmentation | 42 | 0.5713 | 0.6530 | 0.4848 | -0.1526 |
| clahe_no_augmentation | 123 | 0.5849 | 0.6530 | 0.4848 | -0.1390 |
| clahe_no_augmentation | 999 | 0.4712 | 0.6270 | 0.4714 | -0.2528 |
| clahe_no_augmentation | 2025 | 0.4494 | 0.0000 | 0.5152 | -0.2745 |
| clahe_random_contrast | 7 | 0.4630 | 0.6182 | 0.4487 | -0.2609 |
| clahe_random_contrast | 42 | 0.5754 | 0.6530 | 0.4848 | -0.1485 |
| clahe_random_contrast | 123 | 0.5877 | 0.6530 | 0.4848 | -0.1362 |
| clahe_random_contrast | 999 | 0.4757 | 0.6067 | 0.4608 | -0.2482 |
| clahe_random_contrast | 2025 | 0.4701 | 0.0000 | 0.5152 | -0.2538 |
| clahe_small_rotation | 7 | 0.4001 | 0.6210 | 0.4511 | -0.3238 |
| clahe_small_rotation | 42 | 0.5697 | 0.6530 | 0.4848 | -0.1542 |
| clahe_small_rotation | 123 | 0.5828 | 0.6530 | 0.4848 | -0.1411 |
| clahe_small_rotation | 999 | 0.5528 | 0.6530 | 0.4848 | -0.1711 |
| clahe_small_rotation | 2025 | 0.4299 | 0.0000 | 0.5152 | -0.2940 |

## 7. Scientific Questions

**Q1: Does clahe_small_rotation outperform clahe_no_augmentation?**
→ True (Δ mean AUROC = +0.0099)

**Q2: Is improvement consistent across seeds?**
→ False (rotation beats Q38C in 0/5 seeds)

**Q3: Which candidate is most stable (lowest variance)?**
→ clahe_random_contrast (std_auroc = 0.0616)

**Q4: Which candidate becomes production default?**
→ **clahe_random_contrast**
  Mean AUROC: 0.5144 | Mean F1: 0.5062
  Rationale: Highest mean AUROC among all candidates. Stability: unstable.

## 8. Recommendation

**Production default:** `clahe_random_contrast`

- Mean AUROC: 0.5144 (Δ -0.2095 vs Q38C)
- 95% CI AUROC: [0.4379, 0.5909]
- Mean F1: 0.5062 (Δ -0.1716 vs Q38C)
- 95% CI F1: [0.1540, 0.8585]
- Stability: unstable

Next: Q41 — Partial Fine-Tuning Benchmark.

## 9. PASS/FAIL Checklist

- [x] Three candidates evaluated
- [x] Five seeds executed: [42, 7, 123, 999, 2025]
- [x] Mean AUROC computed
- [x] Standard deviation computed
- [x] 95% confidence intervals computed
- [x] Leaderboard generated
- [x] Statistical comparison generated
- [x] Recommendation generated
- [x] Roadmap updated
