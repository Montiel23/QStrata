# Q45A — Fast Augmentation Closure Benchmark (Q39C Closure)

**Date:** 2026-05-31
**Slice:** Q45A
**Branch:** feature/q45a_q39c_fast_augmentation_closure
**Epochs:** 4 | **Seeds:** [42, 7, 123]
**DataLoader profile:** bs=8 nw=4 pm=True pw=True pf=2 (Q41 optimised)
**Total wall time:** 14.4 min

---

## 1. Objective

Bounded fast closure benchmark for the augmentation dimension of Phase 6b. Uses 3 seeds × 3 candidates with the Q41-optimised DataLoader profile. Produces a formal **CLOSE/CONTINUE** decision on whether augmentation should remain in the roadmap.

## 2. Reference Baselines

| Baseline | AUROC | F1 | Source |
|---|---|---|---|
| Q38A raw | 0.6835 | 0.6398 | Q38A |
| Q38C best (CLAHE clip=3.0 tile=4×4) | 0.7239 | 0.6779 | Q38C |
| Q43 CLAHE-only mean (5-seed) | 0.7168 | 0.6255 | Q43 |

**Prior augmentation evidence (Q43, 5-seed × 3 candidates):**
- All three Q43 augmentation variants underperformed the CLAHE-only baseline.
- Q43 production default: `clahe_no_augmentation` (mean AUROC 0.7168).

## 3. Candidates

| # | Candidate | Description |
|---|---|---|
| 1 | clahe_no_augmentation | CLAHE clip=3.0 tile=4×4 only (Q38C champion, no augmentation) |
| 2 | clahe_horizontal_flip | CLAHE + RandomHorizontalFlip(p=0.5) |
| 3 | clahe_combined_aug | CLAHE + RandomHorizontalFlip(p=0.5) + RandomRotation(degrees=7) |

## 4. Evaluation Protocol

- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)
- Epochs: 4 | Batch: 8 | LR: 1e-03 | WD: 1e-04
- Seeds: [42, 7, 123]
- DataLoader: bs=8 nw=4 pm=True pw=True pf=2
- 95% CI: t-distribution, df=2 (t*=4.303)

## 5. Multi-Seed Summary

| Rank | Candidate | AUROC (mean ± std) | 95% CI AUROC | F1 (mean ± std) | ΔAUROC vs Q38C | Seeds>Q38C | Stability | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | clahe_no_augmentation | 0.7196 ± 0.0095 | [0.6960, 0.7433] | 0.6274 ± 0.0494 | -0.0043 | 1/3 | stable | CLOSE |
| 2 | clahe_horizontal_flip | 0.7143 ± 0.0225 | [0.6585, 0.7701] | 0.6298 ± 0.0692 | -0.0096 | 1/3 | moderate | CLOSE |
| 3 | clahe_combined_aug | 0.7098 ± 0.0062 | [0.6943, 0.7253] | 0.6060 ± 0.0625 | -0.0141 | 0/3 | stable | CLOSE |

## 6. Per-Seed Results

| Candidate | Seed | AUROC | F1 | Accuracy | ΔAUROC vs Q38C |
|---|---|---|---|---|---|
| clahe_combined_aug | 7 | 0.7033 | 0.5349 | 0.6408 | -0.0207 |
| clahe_combined_aug | 42 | 0.7157 | 0.6519 | 0.6365 | -0.0082 |
| clahe_combined_aug | 123 | 0.7104 | 0.6313 | 0.6384 | -0.0136 |
| clahe_horizontal_flip | 7 | 0.6985 | 0.5511 | 0.6408 | -0.0254 |
| clahe_horizontal_flip | 42 | 0.7400 | 0.6815 | 0.6485 | +0.0161 |
| clahe_horizontal_flip | 123 | 0.7045 | 0.6568 | 0.6211 | -0.0195 |
| clahe_no_augmentation | 7 | 0.7094 | 0.5704 | 0.6505 | -0.0145 |
| clahe_no_augmentation | 42 | 0.7282 | 0.6535 | 0.6615 | +0.0043 |
| clahe_no_augmentation | 123 | 0.7213 | 0.6581 | 0.6403 | -0.0027 |

## 7. Phase Decision

### 🔴 CLOSE — Augmentation Phase: **CLOSE**

**Rationale:** No augmentation candidate achieved mean_auroc > Q38C_BEST_AUROC (0.7239), and no candidate beat Q38C in ≥2/3 seeds. Combined with Q43 evidence (5-seed, same result), augmentation provides no consistent benefit on this dataset/architecture combination. Augmentation phase CLOSED; proceed to Q43 partial fine-tuning track.

**Decision criteria applied:**
- Any augmentation candidate mean_auroc > Q38C (0.7239)? → False
- Any augmentation candidate beats Q38C in ≥2/3 seeds? → False
- All augmentation candidates below CLAHE-only baseline? → True

**Supporting prior evidence:**
- Q43 (5-seed, 3 candidates): all augmentations underperformed CLAHE-only.
- Q39C (dry-run, seed=45, 1 epoch): augmentation showed no benefit.

## 8. Recommendation

**Production default:** `clahe_no_augmentation`
- Mean AUROC: 0.7196

**Next step:** Proceed to partial fine-tuning (Q43 track) — augmentation dimension CLOSED.

## 9. PASS/FAIL Checklist

- [x] 3 candidates evaluated
- [x] 3 seeds completed: [42, 7, 123]
- [x] Q41 optimized DataLoader profile used
- [x] CSV leaderboard generated
- [x] JSON summary generated
- [x] Markdown report generated
- [x] Augmentation phase decision: **CLOSE**
- [x] Roadmap updated with augmentation phase decision
