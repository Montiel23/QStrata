# C006 Stability Validation Report

- **Slice:** 29
- **Date:** 2026-05-25
- **Candidate:** C006 — depthwise_sep, conv_channels: [64, 128], 9,870 params
- **Config:** `experiments/configs/binary_baseline_depthwise_sep_wide.yaml`
- **Seeds:** 42, 7, 123, 999
- **Protocol:** nas_benchmark_protocol_v1 — best-val checkpoint, seed applied before all ops

---

## Per-Seed Results

| Seed | Params | Best val acc | Best epoch | Final train acc | Test acc\* | Mean epoch time (s) | Latency (ms/batch) |
|------|-------:|-------------:|-----------:|----------------:|----------:|--------------------:|-------------------:|
| 42 | 9,870 | 91.98% | 9 | 91.53% | 86.22% | 0.61 | 0.683 |
| 7 | 9,870 | 90.65% | 6 | 89.55% | 83.81% | 0.56 | 0.439 |
| 123 | 9,870 | 91.03% | 10 | 88.38% | 86.54% | 0.57 | 0.448 |
| 999 | 9,870 | 91.60% | 8 | 89.63% | 85.42% | 0.58 | 0.550 |

> \*Test accuracy is analysis only — not used as fitness signal or gate criterion.

---

## Aggregate Statistics

| Metric | Mean | Std | Min | Max |
|--------|-----:|----:|----:|----:|
| Best val acc (%) | 91.32% | 0.59% | 90.65% | 91.98% |
| Test acc (%) [analysis only] | 85.50% | 1.22% | 83.81% | 86.54% |
| Latency (ms/batch) | 0.530 | 0.113 | 0.439 | 0.683 |
| Mean epoch time (s) | 0.58 | 0.02 | 0.56 | 0.61 |

---

## Decision Gate

| Gate Criterion | Threshold | Actual | Result |
|----------------|-----------|--------|--------|
| std(best_val_acc) | ≤ 1.0% | 0.59% | PASS |
| No seed > 2.5% below mean val acc | ≤ 2.5% gap | 0.67% max gap | PASS |
| Latency std % of mean | ≤ 15% | 21.4% | FAIL |
| No training failures | 0 failures | 0 failure(s) | PASS |

---

## Stability Interpretation

3 of 4 gate criteria passed. The following criteria failed: latency std ≤ 15% of mean. C006 achieved a mean best validation accuracy of 91.32% (std 0.59%) across the four seeds. The failed gate(s) indicate instability that should be investigated before committing to C006 for follow-up work.

---

## Verdict

```
VERDICT: Not stable enough; stop and investigate
```
