# NAS v0 Candidate Benchmark — Stable Results

- **Status:** Complete
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Protocol:** nas_benchmark_protocol_v1 — seed 42, best-val checkpoint selection
- **Candidates:** C001–C006

---

## Full Benchmark Results

| Candidate | block_type | conv_channels | Params | Best val acc | Best epoch | Final train acc | Test acc* | Mean epoch (s) | Latency (ms/batch) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C001 | standard | [32, 64] | 19,138 | 92.56% | 7 | 90.46% | 86.54% | 0.47 | 0.65 |
| C002 | standard | [48, 96] | 42,530 | 91.41% | 5 | 90.99% | 87.82% | 0.55 | 0.56 |
| C003 | standard | [64, 128] | 75,138 | 92.37% | 7 | 91.19% | 83.17% | 0.62 | 0.55 |
| C004 | depthwise_sep | [32, 64] | 2,894 | 91.41% | 7 | 89.95% | 87.66% | 0.49 | 0.70 |
| C005 | depthwise_sep | [48, 96] | 5,870 | 91.41% | 5 | 90.74% | 86.22% | 0.52 | 0.76 |
| C006 | depthwise_sep | [64, 128] | 9,870 | 91.98% | 9 | 91.53% | 86.22% | 0.58 | 0.68 |

> *Test accuracy is reported for analysis only. It must not be used as a fitness signal during NAS search. Best validation accuracy is the sole NAS fitness signal.

---

## Rankings

### 1. By best validation accuracy — highest to lowest

- C001 — 92.56%
- C003 — 92.37%
- C006 — 91.98%
- C002 — 91.41%
- C004 — 91.41%
- C005 — 91.41%

### 2. By parameter count — lowest to highest

- C004 — 2,894 params
- C005 — 5,870 params
- C006 — 9,870 params
- C001 — 19,138 params
- C002 — 42,530 params
- C003 — 75,138 params

### 3. By mean inference latency — lowest to highest

- C003 — 0.55 ms/batch
- C002 — 0.56 ms/batch
- C001 — 0.65 ms/batch
- C006 — 0.68 ms/batch
- C004 — 0.70 ms/batch
- C005 — 0.76 ms/batch

---

## Interpretation

C001 (`standard [32, 64]`) achieved the highest best validation accuracy at 92.56%, with C003 (`standard [64, 128]`) close behind at 92.37% — a difference of only 0.19 pp at nearly 4× the parameter cost. C004 (`depthwise_sep [32, 64]`) is the most parameter-efficient candidate with 2,894 parameters, achieving a best validation accuracy of 91.41% — only 1.15 pp below C001 while using 6.6× fewer parameters, which is a compelling accuracy-efficiency trade-off. The stable results are consistent with prior benchmark evidence from Slices 12 and 13: depthwise separable blocks reach competitive validation accuracy when evaluated at their best checkpoint rather than their final epoch, confirming that the Slice 18 anomalies were a measurement artifact rather than a structural model deficiency. Notably, all six candidates fall within a narrow validation accuracy band of 91.41%–92.56%, while spanning nearly a 26× range in parameter count (2,894–75,138), indicating that the search space has meaningful diversity on the efficiency axis with limited variation on the accuracy axis — a profile well suited to multi-objective search. These stable results provide a reliable basis for proceeding to NAS search.

---

## Next Step

These stable results are ready for human review. The next step — NAS search space confirmation or NSGA-II implementation — requires human approval before proceeding.
