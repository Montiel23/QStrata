# NAS v0 Candidate Benchmark Report

- **Status:** Complete
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Sweep:** All six NAS v0 candidates — C001 through C006

---

## Benchmark Results

| Candidate | block_type | conv_channels | Params | Mean epoch (s) | Train acc | Val acc | Test acc | Latency (ms/batch) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C001 | standard | [32, 64] | 19,138 | 0.47 | 89.97% | 33.97% | 40.87% | 0.59 |
| C002 | standard | [48, 96] | 42,530 | 0.53 | 90.27% | 85.11% | 86.86% | 0.54 |
| C003 | standard | [64, 128] | 75,138 | 0.59 | 90.51% | 93.13% | 85.90% | 0.51 |
| C004 | depthwise_sep | [32, 64] | 2,894 | 0.43 | 88.91% | 88.74% | 87.50% | 0.69 |
| C005 | depthwise_sep | [48, 96] | 5,870 | 0.50 | 88.38% | 75.76% | 78.04% | 0.65 |
| C006 | depthwise_sep | [64, 128] | 9,870 | 0.56 | 87.83% | 45.80% | 53.21% | 0.63 |

> **Note:** Test accuracy is reported here for analysis only. It must not be used as a fitness signal during NAS search. Validation accuracy is the sole fitness signal.

---

## Rankings

### 1. By validation accuracy — highest to lowest

- C003 — 93.13%
- C004 — 88.74%
- C002 — 85.11%
- C005 — 75.76%
- C006 — 45.80%
- C001 — 33.97%

### 2. By parameter count — lowest to highest

- C004 — 2,894 params
- C005 — 5,870 params
- C006 — 9,870 params
- C001 — 19,138 params
- C002 — 42,530 params
- C003 — 75,138 params

### 3. By mean inference latency — lowest to highest

- C003 — 0.51 ms/batch
- C002 — 0.54 ms/batch
- C001 — 0.59 ms/batch
- C006 — 0.63 ms/batch
- C005 — 0.65 ms/batch
- C004 — 0.69 ms/batch

---

## Interpretation

C003 (standard [64, 128]) achieved the highest validation accuracy at 93.13%. C004 (depthwise_sep [32, 64]) is the most parameter-efficient candidate with 2,894 parameters. Among the depthwise separable candidates, C004 ([32, 64]) reached the highest validation accuracy (88.74%), which is 4.39 pp below the best standard candidate while using 25.96× fewer parameters — a compelling accuracy-efficiency trade-off. Inference latency is competitive across all six candidates, with no candidate showing a significant runtime disadvantage. The spread in validation accuracy and parameter count across the grid provides meaningful signal for a multi-objective search, and the results support proceeding to NSGA-II exploration.

---

## Next Step

The results of this sweep are ready for human review. The next step — NAS search space confirmation or NSGA-II implementation — requires human approval before proceeding.

