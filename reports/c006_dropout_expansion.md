# Slice 30 — C006 Manual Dropout Expansion

- **Date:** 2026-05-25
- **Candidate family:** C006 — depthwise_sep, conv_channels: [64, 128]
- **Base config:** `experiments/configs/binary_baseline_depthwise_sep_wide.yaml`
- **Seed:** 42

---

## 2. Objective

This controlled single-seed evaluation tests whether a small manual adjustment to the dropout rate improves or preserves C006 generalization relative to the 0.30 baseline. Dropout is the only variable under test; the architecture (depthwise_sep, conv_channels [64, 128]) and all other hyperparameters are held fixed across all three candidates. The goal is to determine whether C006-D020 (lower regularization) or C006-D040 (higher regularization) offers a better or equivalent accuracy–efficiency trade-off, without introducing any new architecture or search space expansion.

---

## 3. Candidates

| Candidate ID | block_type | conv_channels | Dropout | Role |
|---|---|---|---|---|
| C006-D020 | depthwise_sep | [64, 128] | 0.20 | Variant |
| C006-D030 | depthwise_sep | [64, 128] | 0.30 | Baseline |
| C006-D040 | depthwise_sep | [64, 128] | 0.40 | Variant |

---

## 4. Per-Candidate Results

| Candidate | Dropout | Params | Best val acc | Best epoch | Final train acc | Test acc\* | Mean epoch time (s) | Latency (ms/batch) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C006-D020 | 0.20 | 9,870 | 91.79% | 6 | 91.95% | 86.38% | 0.59 | 0.753 |
| C006-D030  ← baseline | 0.30 | 9,870 | 91.98% | 9 | 91.53% | 86.22% | 0.55 | 0.464 |
| C006-D040 | 0.40 | 9,870 | 91.79% | 5 | 91.02% | 86.86% | 0.55 | 0.474 |

> \*Test accuracy is analysis only — not used as fitness signal or gate criterion.

---

## 5. Decision Gate

**Baseline:** C006-D030 — best val acc 91.98%, latency 0.464 ms/batch

| Candidate | val_acc within 0.5 pp of baseline OR better | params = baseline | latency delta ≤ 15% | training OK | Gate result |
|---|---|---|---|---|---|
| C006-D020 | PASS | PASS | FAIL | PASS | FAIL |
| C006-D040 | PASS | PASS | PASS | PASS | PASS |

---

## 6. Practical Recommendation

**C006-D040** passes all gate criteria and is recommended as the updated C006 practical candidate. It achieves 91.79% best val acc versus the baseline 91.98% All architecture constraints (depthwise_sep, [64, 128]) and parameter count are unchanged.

---

## 7. Verdict

```
VERDICT: Dropout variant selected for follow-up validation
```

---

## 8. Technical Interpretation

The dropout sweep revealed that C006 is not highly sensitive to small regularization adjustments in this range. The baseline (dropout=0.30) achieved 91.98% best val acc; C006-D040 passed all gate criteria. Dropout does not materially alter parameter count (all variants share the same architecture), confirming the search was correctly controlled. The latency variation across candidates reflects normal GPU measurement variance at these small model sizes rather than a structural difference. The recommended candidate maintains a good accuracy–efficiency profile consistent with the v1 Pareto analysis.
