# Q42 — Apply Optimised DataLoader Profile to Q39/Q40 Scripts

**Date:** 2026-05-30
**Slice:** Q42
**Branch:** `feature/q42-apply-optimized-dataloader-profile`

---

## 1. Objective

Apply the Q41-optimised DataLoader profile to the Q39 and Q40 execution scripts and
estimate the new runtime for full validation runs before executing expensive experiments.

---

## 2. Q41 Profile Summary

Q41 swept 60 configurations across batch\_size × num\_workers × pin\_memory ×
persistent\_workers for three Q39/Q40 candidates on RTX 2060 SUPER.

**Recommended (highest throughput, stable GPU util):**

| Parameter | Old (Q38A default) | New (Q41 recommended) |
|---|---|---|
| batch\_size | 4 | **8** |
| num\_workers | 0 | **4** |
| pin\_memory | False | **True** |
| persistent\_workers | False | **True** |
| prefetch\_factor | None | **2** |
| **Throughput (samp/s)** | 98.0 | **371.1** |
| **GPU util avg** | ~15% | ~46% |
| **Speedup** | — | **3.79×** |

The old profile (num\_workers=0) was CPU-bound on data loading: each batch spent
~36 ms waiting on synchronous I/O vs ~4 ms on compute. The optimised profile
prefetches data on four background workers, eliminating this bottleneck.

---

## 3. Scripts Updated

### 3.1 `scripts/run_q39_binary_augmentation_benchmark.py`

Changes applied:

- Added `_OPT_BATCH_SIZE=8`, `_OPT_NUM_WORKERS=4`, `_OPT_PIN_MEMORY=True`,
  `_OPT_PERSISTENT_WORKERS=True`, `_OPT_PREFETCH_FACTOR=2` constants.
- `run_variant_q39()` now accepts `batch_size` and `num_workers` kwargs (default to
  optimised values).
- All three `DataLoader` calls (train/val/test) updated to use the new profile:
  `pin_memory=True`, `persistent_workers=(num_workers>0)`, `prefetch_factor=2`.
- CUDA check softened: `--dry-run` mode falls back to CPU automatically (GPU required
  for full production run).
- CLI args added: `--batch-size` (default 8), `--num-workers` (default 4).

### 3.2 `scripts/run_q40_top_candidate_validation.py`

Changes applied (mirror of Q39 plus estimate-runtime mode):

- Same `_OPT_*` constants and `_Q38C_PER_VARIANT_S=744` reference added.
- `run_single()` now accepts `batch_size` and `num_workers` kwargs.
- All three `DataLoader` calls updated to use optimised profile.
- CLI args added: `--batch-size` (default 8), `--num-workers` (default 4),
  `--estimate-runtime`.
- `_print_runtime_estimates()` function prints Q40 and Q39 full-run estimates
  when `--estimate-runtime` is passed.

---

## 4. Runtime Estimates

Reference: Q38C actual wall time = 8935 s for 12 CLAHE variants × 4 epochs
→ **744 s per variant** (including eval splits, latency measurement, overhead).

Speedup applied: **3.79×** (measured samp/s ratio from Q41).

### 4.1 Q39 — Full Augmentation Benchmark (12 variants × 4 epochs)

| Track | Variants | Old est. (s) | New est. (s) | Old (min) | New (min) |
|---|---|---|---|---|---|
| Track A — raw (no CLAHE) | 6 | 2 271 | 599 | 37.9 | 10.0 |
| Track B — CLAHE + aug | 6 | 4 464 | 1 178 | 74.4 | 19.6 |
| **Total** | **12** | **6 735** | **1 777** | **112.3** | **29.6** |

**Time saved: ~82.7 min** per full Q39 run.

### 4.2 Q40 — Top Candidate Validation (3 candidates × 5 seeds × 4 epochs)

| | Runs | Old est. (s) | New est. (s) | Old (min) | New (min) |
|---|---|---|---|---|---|
| **Total** | 15 | 11 160 | 2 945 | 186.0 | 49.1 |

**Time saved: ~136.9 min** per full Q40 run.

### 4.3 Combined Savings

| Experiment | Old | New | Saved |
|---|---|---|---|
| Q39 full run | 112 min | 30 min | 82 min |
| Q40 full run | 186 min | 49 min | 137 min |
| **Combined** | **298 min** | **79 min** | **219 min** |

---

## 5. Validation

### Smoke validation — Q40

```
python scripts/run_q40_top_candidate_validation.py \
    --dry-run --batch-size 8 --num-workers 4 --estimate-runtime
```

Expected: runtime estimates printed, then 1-epoch dry-run completes for all 3
candidates × 5 seeds = 15 runs. Pass = `Q40_TOP_CANDIDATE_VALIDATION: PASS`.

### Smoke validation — Q39

```
python scripts/run_q39_binary_augmentation_benchmark.py \
    --dry-run --variants clahe_no_augmentation --batch-size 8 --num-workers 4
```

Expected: 1-epoch dry-run completes for `clahe_no_augmentation`. Pass =
`Q39C_AUGMENTATION_BENCHMARK: PASS`.

---

## 6. PASS/FAIL Checklist

- [x] Q40 script updated with optimised DataLoader profile
- [x] Q39 script updated with optimised DataLoader profile
- [x] CLI overrides added (`--batch-size`, `--num-workers`, `--estimate-runtime`)
- [x] Runtime estimate mode added to Q40 (`--estimate-runtime`)
- [x] Q40 updated runtime estimated: ~49 min (vs 186 min old)
- [x] Q39 updated runtime estimated: ~30 min (vs 112 min old)
- [x] Speedup vs old estimate computed: 3.79× throughput, ~3.5–3.8× practical
- [x] Smoke validation passes for Q40 (dry-run)
- [x] Smoke validation passes for Q39 (dry-run, single variant)
- [x] No full Q40 validation executed
- [x] No full Q39 benchmark executed
- [x] Report generated (`reports/q42_optimized_dataloader_application.md`)
- [x] Runtime estimate JSON generated (`experiments/results/q42_runtime_estimate.json`)
- [x] Roadmap updated (Q42 COMPLETE)
- [x] Local commit created
- [x] No push

---

## 7. Next Steps

- Q39 full run: execute on GPU with new profile (~30 min expected)
- Q40 full run: execute on GPU with new profile (~49 min expected)
- Q43: Partial Fine-Tuning Benchmark (blocked on Q42 ✓)
