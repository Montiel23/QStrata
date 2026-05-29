# Q41 — DataLoader Throughput Optimization Report

**Date:** 2026-05-29
**Device:** cuda (NVIDIA GeForce RTX 2060 SUPER)
**Dry-run:** False
**Max train batches:** 100
**Total wall time:** 5.1 min

---

## 1. Objective

Optimize QSTRATA training throughput by finding a DataLoader configuration that achieves 70–80% GPU utilization on the RTX 2060 SUPER without overloading the local workstation. The benchmark measures samples/second and GPU utilization across a grid of `batch_size`, `num_workers`, `pin_memory`, and `persistent_workers` settings.

## 2. Q40 Runtime Bottleneck Recap

Q40 uses the default DataLoader from Q38A: `batch_size=4`, `num_workers=0`, `pin_memory=false`, `persistent_workers=false`. CLAHE preprocessing (clip=3.0, tile=4×4) costs ~2.5 ms/image and runs synchronously in the main thread, leaving the GPU starved between batches. The full Q40 5-seed validation is bottlenecked by this low DataLoader throughput.

## 3. Benchmark Design

- **Candidates:** `clahe_no_augmentation`, `clahe_small_rotation`
- **Max batches/run:** 100
- **No validation, no test evaluation**
- **GPU utilization:** nvidia-smi polled every 0.5 s during training
- **CPU utilization:** psutil if available
- **OOM handling:** RuntimeError caught; config flagged and skipped

## 4. Candidate Grid

| Dimension | Values |
|---|---|
| `batch_size` | 4, 8, 16 |
| `num_workers` | 0, 2, 4 |
| `pin_memory` | false, true |
| `persistent_workers` | false, true (only when num_workers > 0) |
| `prefetch_factor` | 2 when num_workers > 0; omitted when 0 |

Total configs: 60 × 2 candidates = 60 runs (num_workers=8 excluded to protect workstation responsiveness)

## 5. Results Leaderboard

Stable configs ranked by samples/second (descending):

| Rank | Candidate | BS | NW | PM | PW | PF | Samp/s | Batch ms | GPU avg% | GPU max% | GPU MB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | clahe_no_augmentation | 8 | 4 | true | true | 2 | 371.1 | 5.6 | 45.8 | 52.0 | 1836 | ok |
| 2 | clahe_no_augmentation | 16 | 4 | false | true | 2 | 364.0 | 12.8 | 51.7 | 57.0 | 2903 | ok |
| 3 | clahe_no_augmentation | 8 | 4 | false | true | 2 | 361.5 | 7.3 | 44.0 | 52.0 | 1835 | ok |
| 4 | clahe_no_augmentation | 16 | 4 | true | true | 2 | 356.9 | 14.8 | 48.6 | 56.0 | 2918 | ok |
| 5 | clahe_no_augmentation | 16 | 4 | false | false | 2 | 347.9 | 15.8 | 41.6 | 57.0 | 2906 | ok |
| 6 | clahe_no_augmentation | 8 | 4 | true | false | 2 | 347.3 | 6.4 | 44.8 | 52.0 | 1835 | ok |
| 7 | clahe_no_augmentation | 16 | 4 | true | false | 2 | 345.8 | 12.2 | 50.6 | 56.0 | 2900 | ok |
| 8 | clahe_small_rotation | 16 | 4 | true | true | 2 | 341.4 | 12.7 | 40.9 | 47.0 | 2726 | ok |
| 9 | clahe_no_augmentation | 8 | 4 | false | false | 2 | 338.5 | 7.6 | 34.8 | 48.0 | 1836 | ok |
| 10 | clahe_small_rotation | 16 | 4 | false | true | 2 | 335.9 | 15.0 | 41.9 | 48.0 | 2727 | ok |
| 11 | clahe_no_augmentation | 4 | 4 | true | true | 2 | 333.1 | 4.2 | 39.7 | 40.0 | 1482 | ok |
| 12 | clahe_small_rotation | 16 | 4 | true | false | 2 | 329.5 | 11.5 | 41.9 | 46.0 | 2727 | ok |
| 13 | clahe_small_rotation | 8 | 4 | true | true | 2 | 329.1 | 7.3 | 39.6 | 45.0 | 2894 | ok |
| 14 | clahe_small_rotation | 16 | 4 | false | false | 2 | 327.2 | 12.2 | 38.2 | 47.0 | 2726 | ok |
| 15 | clahe_no_augmentation | 4 | 4 | false | true | 2 | 318.3 | 4.3 | 37.0 | 40.0 | 1481 | ok |
| 16 | clahe_small_rotation | 8 | 4 | true | false | 2 | 314.4 | 7.2 | 39.0 | 44.0 | 2893 | ok |
| 17 | clahe_no_augmentation | 4 | 4 | true | false | 2 | 314.2 | 4.0 | 46.0 | 49.0 | 1481 | ok |
| 18 | clahe_small_rotation | 8 | 4 | false | true | 2 | 312.6 | 7.0 | 39.8 | 45.0 | 2895 | ok |
| 19 | clahe_small_rotation | 8 | 4 | false | false | 2 | 308.6 | 7.0 | 33.6 | 45.0 | 2895 | ok |
| 20 | clahe_no_augmentation | 4 | 4 | false | false | 2 | 306.3 | 4.3 | 31.0 | 31.0 | 1481 | ok |
| 21 | clahe_small_rotation | 4 | 4 | true | true | 2 | 291.9 | 4.6 | 38.7 | 40.0 | 2997 | ok |
| 22 | clahe_small_rotation | 4 | 4 | false | true | 2 | 289.9 | 4.7 | 37.3 | 39.0 | 2973 | ok |
| 23 | clahe_small_rotation | 4 | 4 | true | false | 2 | 274.9 | 4.3 | 46.7 | 50.0 | 2994 | ok |
| 24 | clahe_small_rotation | 4 | 4 | false | false | 2 | 256.0 | 5.0 | 33.3 | 34.0 | 2881 | ok |
| 25 | clahe_no_augmentation | 16 | 2 | false | true | 2 | 204.7 | 10.0 | 28.5 | 32.0 | 2793 | ok |
| 26 | clahe_no_augmentation | 16 | 2 | true | true | 2 | 203.0 | 11.2 | 27.3 | 32.0 | 2780 | ok |
| 27 | clahe_no_augmentation | 8 | 2 | true | true | 2 | 200.2 | 6.3 | 28.5 | 30.0 | 1836 | ok |
| 28 | clahe_no_augmentation | 8 | 2 | false | true | 2 | 197.4 | 5.2 | 28.8 | 30.0 | 1838 | ok |
| 29 | clahe_no_augmentation | 16 | 2 | false | false | 2 | 196.3 | 11.3 | 26.8 | 31.0 | 2793 | ok |
| 30 | clahe_no_augmentation | 4 | 2 | true | true | 2 | 195.3 | 2.3 | 29.8 | 31.0 | 1480 | ok |
| 31 | clahe_no_augmentation | 4 | 2 | false | true | 2 | 195.2 | 2.4 | 30.0 | 31.0 | 1484 | ok |
| 32 | clahe_no_augmentation | 8 | 2 | true | false | 2 | 194.5 | 6.2 | 28.2 | 30.0 | 1838 | ok |
| 33 | clahe_no_augmentation | 16 | 2 | true | false | 2 | 193.7 | 11.2 | 26.4 | 30.0 | 2778 | ok |
| 34 | clahe_no_augmentation | 8 | 2 | false | false | 2 | 188.0 | 6.2 | 24.9 | 30.0 | 1853 | ok |
| 35 | clahe_no_augmentation | 4 | 2 | false | false | 2 | 187.4 | 3.8 | 22.2 | 31.0 | 1483 | ok |
| 36 | clahe_no_augmentation | 4 | 2 | true | false | 2 | 185.2 | 3.8 | 31.0 | 32.0 | 1480 | ok |
| 37 | clahe_small_rotation | 16 | 2 | true | true | 2 | 183.0 | 2.7 | 26.0 | 28.0 | 2773 | ok |
| 38 | clahe_small_rotation | 16 | 2 | false | true | 2 | 181.1 | 11.8 | 26.8 | 29.0 | 2790 | ok |
| 39 | clahe_small_rotation | 8 | 2 | false | true | 2 | 179.2 | 6.0 | 26.1 | 29.0 | 2894 | ok |
| 40 | clahe_small_rotation | 8 | 2 | true | true | 2 | 176.5 | 6.6 | 25.8 | 29.0 | 2893 | ok |
| 41 | clahe_small_rotation | 16 | 2 | true | false | 2 | 174.9 | 11.8 | 25.9 | 28.0 | 2779 | ok |
| 42 | clahe_small_rotation | 8 | 2 | false | false | 2 | 174.0 | 6.5 | 23.9 | 29.0 | 2894 | ok |
| 43 | clahe_small_rotation | 8 | 2 | true | false | 2 | 172.8 | 6.6 | 26.1 | 28.0 | 2896 | ok |
| 44 | clahe_small_rotation | 16 | 2 | false | false | 2 | 171.8 | 11.8 | 26.6 | 32.0 | 2915 | ok |
| 45 | clahe_small_rotation | 4 | 2 | true | true | 2 | 170.4 | 3.2 | 31.2 | 34.0 | 2932 | ok |
| 46 | clahe_small_rotation | 4 | 2 | false | true | 2 | 165.0 | 2.8 | 32.0 | 37.0 | 2955 | ok |
| 47 | clahe_small_rotation | 4 | 2 | false | false | 2 | 163.6 | 4.0 | 22.6 | 29.0 | 2896 | ok |
| 48 | clahe_small_rotation | 4 | 2 | true | false | 2 | 160.1 | 4.2 | 32.0 | 34.0 | 2984 | ok |
| 49 | clahe_no_augmentation | 16 | 0 | false | false | — | 106.3 | 2.5 | 19.0 | 51.0 | 2816 | ok |
| 50 | clahe_no_augmentation | 4 | 0 | true | false | — | 105.8 | 2.0 | 15.9 | 17.0 | 1477 | ok |
| 51 | clahe_no_augmentation | 16 | 0 | true | false | — | 105.4 | 2.2 | 17.0 | 18.0 | 2794 | ok |
| 52 | clahe_no_augmentation | 8 | 0 | false | false | — | 105.1 | 2.6 | 20.7 | 39.0 | 1875 | ok |
| 53 | clahe_no_augmentation | 8 | 0 | true | false | — | 105.0 | 2.0 | 17.1 | 18.0 | 1868 | ok |
| 54 | clahe_no_augmentation | 4 | 0 | false | false | — | 98.0 | 4.3 | 15.0 | 19.0 | 1484 | ok |
| 55 | clahe_small_rotation | 8 | 0 | true | false | — | 95.1 | 2.1 | 15.9 | 17.0 | 2898 | ok |
| 56 | clahe_small_rotation | 16 | 0 | false | false | — | 94.8 | 2.5 | 17.3 | 44.0 | 2894 | ok |
| 57 | clahe_small_rotation | 16 | 0 | true | false | — | 91.8 | 2.3 | 17.1 | 23.0 | 2955 | ok |
| 58 | clahe_small_rotation | 8 | 0 | false | false | — | 91.6 | 2.4 | 20.1 | 46.0 | 3006 | ok |
| 59 | clahe_small_rotation | 4 | 0 | true | false | — | 89.7 | 2.1 | 18.1 | 21.0 | 2908 | ok |
| 60 | clahe_small_rotation | 4 | 0 | false | false | — | 88.8 | 2.3 | 21.2 | 55.0 | 2936 | ok |

## 6. GPU Utilization Analysis

- **Target range:** 70.0–80.0%
- **Configs in target:** 0 / 60 stable
- **Highest GPU util:** 51.7% (bs16_nw4_no-pm_pw_pf2 / clahe_no_augmentation)

## 7. Throughput Analysis

- **Baseline (bs=4, nw=0):** 105.8 samp/s
- **Best stable:** 371.1 samp/s
- **Speedup over baseline:** 3.51×

## 8. Stability and OOM Notes

- **Stable configs:** 60
- **OOM configs:** 0
- **Crash configs:** 0

## 9. Workstation Usability Notes

- `num_workers=8` excluded from grid to prevent CPU saturation on 12-core workstation
- `num_workers=4` with `persistent_workers=true` may cause mild CPU elevation
- Configs with `workstation_notes=ok` deemed safe for development use

## 10. Recommended DataLoader Profile

```
batch_size          = 8
num_workers         = 4
pin_memory          = true
persistent_workers  = true
prefetch_factor     = 2
```

- **Samples/second:** 371.1
- **GPU utilization (avg):** 45.8%
- **Workstation notes:** ok

## 11. Expected Runtime Reduction for Q40

- **Q40 baseline DataLoader:** batch_size=4, num_workers=0 → 105.8 samp/s
- **Recommended DataLoader:** 371.1 samp/s
- **Throughput improvement:** 3.51×

Training time scales approximately inversely with throughput. A 3.51× throughput gain translates to an ~3.51× reduction in training wall time per epoch, assuming GPU compute is the dominant cost.

## 12. PASS/FAIL Checklist

- [x] DataLoader throughput benchmark script created
- [x] Candidate grid executed
- [x] batch_size evaluated: {4, 8, 16}
- [x] num_workers evaluated: {0, 2, 4}
- [x] pin_memory evaluated
- [x] persistent_workers evaluated
- [x] samples_per_second recorded
- [x] avg_batch_time recorded
- [x] GPU utilization recorded
- [x] GPU memory recorded
- [x] OOM cases handled safely
- [x] Stable configs identified: 60
- [x] Recommended profile selected: bs=8 nw=4 pm=true pw=true
- [x] Expected Q40 runtime reduction estimated: 3.51×
- [x] Report generated
- [x] Leaderboard generated
- [x] JSON results generated
- [x] Roadmap updated
