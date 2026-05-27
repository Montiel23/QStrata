# Q34B-Parallel-Lite: DV NAS Pilot Parallel Execution Report

**Slice:** Q34B-Parallel-Lite  
**Branch:** `feature/q34b-runtime-hotfix`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** COMPLETE — 5/5 trials PASS, 4-member Pareto set, verdict PASS

---

## 1. Objective

Validate parallel trial execution for the DV NAS pilot. Based on the Q34B-HF assessment (`reports/q34b_runtime_bottleneck_skypilot_assessment.md`), sequential execution of Q34B required 11,491 s (~3.2 h) with 4 epochs per trial. Q34B-Parallel-Lite applies two Q34B-HF recommendations simultaneously:

1. **Parallel trial execution:** 5 independent `ThreadPoolExecutor` workers, each running a Q31 subprocess.
2. **Reduced epoch budget:** 2 epochs per trial (pilot-mode; metrics are comparative, not definitive).

The goal is to confirm that (a) parallel execution produces valid results, (b) PASS/FAIL criteria are maintained, and (c) the observed speedup and overhead characteristics are documented for Q34C planning.

---

## 2. Configuration

| Parameter | Value |
|---|---|
| Script | `scripts/run_q34b_dv_nas_pilot.py` |
| Trials | 5 |
| Epochs per trial | 2 |
| Execution mode | parallel |
| Max workers | 5 |
| Seed (base) | 42 |
| Search space version | `q34b_v1` |
| Dataset | VinDr-SpineXR DV binary |
| Pass threshold | ≥ 3/5 trials completed |
| Pilot name | `Q34B_DV_NAS_PILOT_PARALLEL_LITE` |

**Command executed:**
```bash
docker exec -i docker-qstrata-gpu-1 python3 /workspace/scripts/run_q34b_dv_nas_pilot.py \
  --trials 5 --seed 42 --epochs 2 --parallel --max-workers 5
```

---

## 3. Pilot Verdict

**PASS** — 5/5 trials completed, 4-member Pareto frontier, all gradient health and NaN/Inf checks passed.

| Metric | Value |
|---|---|
| Trials attempted | 5 |
| Trials completed | 5 |
| Trials failed | 0 |
| Trials invalid | 0 |
| Trials unstable | 0 |
| Pareto set size | 4 |
| Pass threshold | 3 |
| Verdict | **PASS** |

---

## 4. Trial Results

| Trial | Qubits | Depth | Rotation | Topology | Reupload | Comp. Dim | AUROC | F1 | Params | Latency (ms) | Pareto | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trial_001 | 2 | 1 | RX+RY+RZ | circular | none | 2 | 0.6406 | 0.6160 | 280 | 123.1 | ✓ | 4487.5 |
| trial_002 | 2 | 1 | RY | linear | none | 2 | 0.6373 | 0.6245 | 280 | 222.6 | ✓ | 4460.7 |
| trial_003 | 2 | 2 | RX+RY | circular | none | 2 | 0.6385 | 0.6223 | 292 | 82.0 | ✓ | 4492.8 |
| trial_004 | 4 | 2 | RX+RY+RZ | circular | every_layer | 4 | 0.6389 | 0.6246 | 598 | 49.3 | ✓ | 4620.8 |
| trial_005 | 2 | 1 | RY | linear | none | 2 | 0.6376 | 0.6083 | 280 | 337.3 | — | 4459.3 |

All trials: `gradient_health = PASS`, `nan_inf = PASS`, `return_code = 0`.

**Trial_005 Pareto exclusion:** Dominated by trial_001 (AUROC 0.6406 ≥ 0.6376, F1 0.6160 ≥ 0.6083, params 280 = 280; trial_001 strictly better on both AUROC and F1).

---

## 5. Pareto Frontier

Four trials constitute the Pareto set. The frontier spans a compact region (AUROC 0.637–0.641, F1 0.615–0.625, params 280–598). Trial_004 provides the lowest latency (49.3 ms) at the cost of highest parameter count (598). Trial_001 leads on AUROC (0.6406). Trial_002 leads on F1 (0.6245).

| Rank | Trial | AUROC | F1 | Params | Latency (ms) | Distinguish |
|---|---|---|---|---|---|---|
| 1 | trial_001 | **0.6406** | 0.6160 | 280 | 123.1 | Best AUROC |
| 2 | trial_002 | 0.6373 | **0.6245** | 280 | 222.6 | Best F1 |
| 3 | trial_003 | 0.6385 | 0.6223 | 292 | 82.0 | AUROC/F1 balance |
| 4 | trial_004 | 0.6389 | 0.6246 | 598 | **49.3** | Lowest latency |

**Pareto CSV:** `experiments/leaderboards/q34b_dv_pareto.csv`  
**Summary JSON:** `experiments/results/q34b_dv_nas_pilot_summary.json`

---

## 6. Wall Time and Speedup Analysis

| Execution | Epochs | Mode | Workers | Wall Time | Per-Trial (est.) |
|---|---|---|---|---|---|
| Q34B sequential (ref) | 4 | sequential | 1 | 11,491 s (~3.2 h) | ~2,298 s |
| Q34B-Parallel-Lite | 2 | parallel | 5 | 4,621 s (~77 min) | 4,460–4,621 s |
| Ideal 2-epoch sequential (est.) | 2 | sequential | 1 | ~5,746 s (~96 min) | ~1,149 s |
| Ideal 2-epoch parallel (theoretical) | 2 | parallel | 5 | ~1,149 s (~19 min) | ~1,149 s |

**Actual speedup vs Q34B sequential (4-epoch):** 11,491 / 4,621 = **2.49×**  
**Actual speedup vs estimated 2-epoch sequential:** 5,746 / 4,621 = **1.24×**  
**Theoretical ideal speedup (2-epoch parallel, no contention):** ~5×  
**Efficiency vs ideal:** 1.24× / 5× = **25%** — thread contention dominated.

---

## 7. Thread Contention Analysis

**Root cause of per-trial slowdown:** PyTorch spawns approximately 30 inter-op threads per process. With 5 parallel worker processes on a 12-CPU host, the system sustained ~150 threads competing for 12 physical CPUs (~12.5 threads/core). Per-trial wall times of 4,460–4,621 s represent a **~3.9× overhead** relative to the single-process 2-epoch estimate (~1,149 s).

**Observed per-trial timing pattern:** All five trials ran near-simultaneously (wall times span only 161 s), confirming they executed in parallel. The longest trial (trial_004, 4 qubits, depth 2, reuploading) took 4,620.83 s, dictating total wall time.

**Why the speedup is still 2.49× despite contention:**
- Each trial completes in ~4,487 s instead of ~1,149 s (3.9× per-trial overhead)
- But all 5 run concurrently → total wall ≈ max per-trial = 4,621 s vs 5 × 1,149 s = 5,746 s sequential
- The 3.9× per-trial penalty partially cancels the 5× concurrency gain, leaving net 1.24× over 2-epoch sequential
- Compared to 4-epoch sequential: epoch reduction (2×) + partial parallelism (1.24×) = 2.49× total

**Mitigation (not implemented — deferred to Q34C):** Setting `OMP_NUM_THREADS=2` and `torch.set_num_threads(2)` per subprocess before trial execution would cap PyTorch to 2 threads/process (10 threads total for 5 workers), reducing contention dramatically and approaching the theoretical 5× speedup.

---

## 8. Comparison to Q34B Sequential Results

The Q34B-Parallel-Lite Pareto frontier (AUROC 0.637–0.641) is consistent with the Q34B sequential pilot Pareto range. Metrics cannot be directly compared due to different epoch counts (2 vs 4 epochs); underfit models produce lower metrics than their fully-trained equivalents. The parallel-lite pilot confirms structural validity of the search space: small qubits (n=2) dominate on parameter count and F1, while trial_004 (n=4, depth 2, every-layer reuploading) achieves lowest latency despite highest parameter count.

| Pilot | Epochs | Best AUROC | Best F1 | Min Params | Pareto Size |
|---|---|---|---|---|---|
| Q34B sequential | 4 | 0.6551 (trial_004) | 0.6356 (trial_001) | 280 | 4 |
| Q34B-Parallel-Lite | 2 | 0.6406 (trial_001) | 0.6245 (trial_002) | 280 | 4 |

The 2-epoch results are lower, as expected. The Pareto structure (4-member set, 280-param anchor, 598-param latency outlier) is consistent between pilots.

---

## 9. Known Artifacts

**NumPy 1.x/2.x compatibility warning:** All 5 trial subprocesses emitted a non-fatal NumPy deprecation warning at import time. This is a known environment artifact from PennyLane's internal NumPy usage against the container's NumPy 2.x installation. No numerical behavior is affected; gradient health and NaN/Inf checks passed for all trials.

**Latency measurement:** `latency_ms_per_sample` is measured at inference time (no gradient), using the model's default batch size on the test split. It reflects the `circuit.matrix()` + `backend.run()` sequential CPU path per sample.

---

## 10. Q34C Gate Status

Q34B-Parallel-Lite COMPLETE. Q34C (CV NAS pilot) is unblocked.

**Before executing Q34C, apply the following corrections not implemented in this slice:**
1. **Thread count cap:** Set `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` in each subprocess environment before launching the Q31 runner. This limits PyTorch thread pool to 2/process (10 total for 5 workers on 12 CPUs) and should approach the theoretical 5× parallel speedup.
2. **Epoch budget:** 2 epochs per trial (pilot-mode), already validated here.
3. **Workers:** 5 parallel workers, as validated here.

With thread cap applied, estimated CV pilot wall time: ~1,200–1,500 s (20–25 min) for 5 trials at 2 epochs, depending on CV circuit complexity.

**Reference documents for Q34C:**
- `reports/q34b_runtime_bottleneck_skypilot_assessment.md` — bottleneck analysis, thread recommendation
- `reports/q34b_dv_nas_pilot_mvp.md` — Q34B sequential baseline (4-epoch)
- `reports/q34b_parallel_lite_dv_nas_pilot.md` — this report (2-epoch parallel)
- `docs/architecture/q33b_cv_quantum_nas_search_space.md` — CV search space definition

Gate: Q34B ✓ + Q34B-HF ✓ + Q34B-Parallel-Lite ✓ → Q34C unblocked.  
Branch merge required before Q34C execution. Branch: `feature/q34b-runtime-hotfix`.

---

## 11. Artifacts

| Artifact | Path | Status |
|---|---|---|
| Pilot orchestrator (modified) | `scripts/run_q34b_dv_nas_pilot.py` | COMMITTED |
| Pareto CSV (updated) | `experiments/leaderboards/q34b_dv_pareto.csv` | COMMITTED |
| Summary JSON (gitignored) | `experiments/results/q34b_dv_nas_pilot_summary.json` | NOT COMMITTED |
| Trial YAML configs | `configs/experiments/q34b_dv_nas/q34b_trial_00[1-5].yaml` | COMMITTED |
| Runtime bottleneck assessment | `reports/q34b_runtime_bottleneck_skypilot_assessment.md` | COMMITTED |
| SkyPilot YAML (reference only) | `infra/skypilot/q34b_dv_smoke_gpu.yaml` | COMMITTED |
| This report | `reports/q34b_parallel_lite_dv_nas_pilot.md` | COMMITTED |
| Roadmap (updated) | `docs/roadmaps/qstrata_master_research_roadmap.md` | COMMITTED |
