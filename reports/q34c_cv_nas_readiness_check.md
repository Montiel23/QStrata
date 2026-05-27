# Q34C CV NAS Readiness Check

**Slice:** Q34C-PREFLIGHT-CV-NAS-READINESS  
**Branch:** `feature/q34b-runtime-hotfix`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** CONDITIONAL READY — infrastructure complete; two scripts missing; pilot substitutions viable

---

## 1. Executive Summary

Q34C is **conditionally ready**. The execution infrastructure (parallel runner, thread cap, Q31 subprocess protocol) is fully validated and compatible. The CV backend (`GaussianVariationalAnsatz`, `GaussianBackend`) exists and supports the Q27-validated baseline architecture. However, two mandatory scripts do not exist yet:

1. **`scripts/train_q34c_cv_candidate.py`** — the per-trial CV training script (analog to `train_q34b_dv_candidate.py`)
2. **`scripts/run_q34c_cv_nas_pilot.py`** — the Q34C orchestrator (analog to `run_q34b_dv_nas_pilot.py` with Q33B search space)

No qcore changes are strictly required for a minimum viable pilot using Q34B-style substitutions (fixing unimplemented search dimensions to fixed pilot values). Q34C can proceed once these two scripts are written.

**Recommended next slice:** Write `train_q34c_cv_candidate.py` + `run_q34c_cv_nas_pilot.py`.

---

## 2. Execution Infrastructure — READY

| Component | Status | Notes |
|---|---|---|
| Parallel execution (`--parallel --max-workers 5`) | ✅ READY | Validated in Q34B-Parallel-Lite (5/5 PASS) |
| Thread cap (`--thread-cap 2`) | ✅ READY | Validated in EXP-005 preflight; OMP/MKL/OPENBLAS propagated |
| Epoch budget (`--epochs 2`) | ✅ READY | Validated in Q34B-Parallel-Lite |
| Q31 subprocess protocol | ✅ READY | Runner invokes subprocess, parses stdout lines, records result JSON |
| Per-trial YAML config generation | ✅ READY | Framework from Q34B orchestrator is fully reusable |
| Pareto computation logic | ✅ READY | `compute_pareto()` in DV orchestrator is reusable with stability filter added |
| Git + seed reproducibility | ✅ READY | Q31A validated; seed per trial, frozen YAML per trial |
| Docker container | ✅ READY | `docker-qstrata-gpu-1` used for Q34A, Q34B, Q34B-Parallel-Lite |

---

## 3. CV Backend (qcore) — PARTIALLY READY

### `GaussianVariationalAnsatz` (`qcore/ansatz/cv_spine_ansatz.py`, 81 lines)

| Parameter | Supported | Notes |
|---|---|---|
| `n_modes` | ✅ | Fully parameterized |
| `depth` | ✅ | Fully parameterized |
| `squeezing_cap` | ✅ | Bounded via `tanh(squeezing_raw) × squeezing_cap` |
| `beam_splitter_topology` | ⚠️ HARDCODED | Always circular (`(i+1) % n_modes`); no `linear` option |
| `displacement_cap` | ⚠️ MISSING | `disp_real` / `disp_imag` are unbounded raw `nn.Parameter`; no `tanh` cap |
| `encoding_strategy` | ⚠️ PARTIAL | `encoded_input` re-uploading path exists but no dispatching for `hybrid_displacement_phase_encoding` |
| `readout_strategy` | ⚠️ HARDCODED | Returns `(mu, cov)` — caller uses `mu_final` only (first_moments); no second_moments or concatenated path |
| `covariance_parameterization` | ✅ SYMPLECTIC ONLY | Always uses `apply_symplectic`; no `direct_covariance` path (this is correct by design) |

### `GaussianBackend` (`qcore/backends/cvBackend.py`, 24 lines)

| Method | Status | Notes |
|---|---|---|
| `get_vacuum()` | ✅ | Returns `(mu=0, cov=I×hbar/2)` |
| `apply_symplectic(mu, cov, S)` | ✅ | `mu = S@mu`, `cov = S@cov@Sᵀ` |
| PSD eigenvalue check | ❌ MISSING | No `check_psd()` method; caller must implement |
| Finite state check | ❌ MISSING | No NaN/inf monitor; `train_vindr_cv_binary.py` handles this inline |
| Symplectic condition check | ❌ MISSING | No `check_symplectic_condition()` method |
| `displacement()` method | ⚠️ MINOR BUG | Uses `torch.img()` (should be `torch.imag()`); not called in ansatz `apply()` path |

The stability checks required by Q33B Section 8 are not in `GaussianBackend`. They were implemented inline in `train_vindr_cv_binary.py` (Q27) and will need to be implemented inline in `train_q34c_cv_candidate.py` similarly.

---

## 4. Q33B Search Space vs. Current Implementation

| Dimension | Q33B Options | Current Implementation | Pilot Treatment |
|---|---|---|---|
| `n_modes` | [1, 2, 4, 6] | ✅ parameterized | Search all (recommend [1, 2, 4] for pilot) |
| `cv_depth` | [1, 2, 3] | ✅ parameterized | Search all |
| `squeezing_cap` | [0.5, 1.0, 1.5, 2.0] | ✅ parameterized | Search all |
| `displacement_cap` | [0.5, 1.0, 2.0, 4.0] | ❌ no cap — unbounded | **Pilot substitution**: fix to 2.0; log in config but don't search; add `tanh` cap in candidate script |
| `encoding_strategy` | [disp, reupload_disp, hybrid] | ⚠️ reuploading path exists; hybrid missing | **Pilot substitution**: always `displacement_encoding` |
| `beam_splitter_topology` | [linear, circular] | ⚠️ always circular | **Pilot substitution**: always `circular` |
| `readout_strategy` | [first_moments, second_moments, concatenated] | ⚠️ always first_moments | **Pilot substitution**: always `first_moments` |
| `compression_dim` | [2, 4, 8, 16] | ✅ via `Linear(128, compression_dim)` | Search [2, 4, 8] (constrained: must equal `2×n_modes` for displacement_encoding) |
| `covariance_parameterization` | [symplectic, direct_covariance] | ✅ symplectic only (correct) | **Pilot substitution**: always `symplectic_parameterization` |

**Variable dimensions for pilot (5 substitutions applied):** `n_modes`, `cv_depth`, `squeezing_cap`, `compression_dim` (with constraint `compression_dim = 2×n_modes`)

**Constraint enforcement required at config-generation time:** `compression_dim = 2 × n_modes` for `displacement_encoding`. This must be enforced in `run_q34c_cv_nas_pilot.py` before writing each trial YAML.

---

## 5. Missing Artifacts (Blockers)

### BLOCKER 1: `scripts/train_q34c_cv_candidate.py` — does not exist

The per-trial training script for Q34C. Required content:
- Reads YAML config with Q33B-style fields (`n_modes`, `cv_depth`, `squeezing_cap`, `displacement_cap`, `compression_dim`, `encoding_strategy`, `beam_splitter_topology`, `readout_strategy`)
- Applies pilot substitutions for unimplemented dimensions (beam_splitter_topology, readout_strategy, encoding_strategy, covariance_parameterization)
- Applies `displacement_cap` via `tanh(disp_raw) × displacement_cap` in the forward loop (workaround: handled in candidate script, not in ansatz)
- Per-epoch stability checks: covariance PSD, finite mu, finite cov, NaN/inf (inherited from Q27 inline checks)
- Records stability taxonomy label per trial
- Outputs `Q34C_TRIAL_*` machine-readable lines for orchestrator parsing:
  ```
  Q34C_TRIAL_AUROC: <float>
  Q34C_TRIAL_F1: <float>
  Q34C_TRIAL_ACCURACY: <float>
  Q34C_TRIAL_PARAMS: <int>
  Q34C_TRIAL_LATENCY_MS: <float>
  Q34C_TRIAL_GRADIENT_HEALTH: PASS|FAIL
  Q34C_TRIAL_NAN_INF: PASS|FAIL
  Q34C_TRIAL_STABILITY_TAXONOMY: valid|unstable_covariance|covariance_not_psd|nan_state|...
  Q34C_TRIAL_STATUS: completed
  ```
- Exit 0 on success, nonzero on failure

Complexity estimate: ~250–350 lines (Q27 training script is ~450 lines; Q34C candidate is simpler — no checkpoint saving, fewer epochs, no early stopping).

### BLOCKER 2: `scripts/run_q34c_cv_nas_pilot.py` — does not exist

The Q34C orchestrator. Required content:
- Q33B search space dict
- Config sampling with `compression_dim = 2×n_modes` constraint enforcement
- `build_trial_yaml()` for CV YAML fields
- `parse_trial_metrics()` that parses `Q34C_TRIAL_*` lines + `stability_taxonomy`
- `compute_pareto()` with stability filter (exclude `stability_taxonomy != "valid"` trials from Pareto)
- Q34C Pareto CSV with stability taxonomy column
- `--parallel`, `--max-workers`, `--thread-cap`, `--epochs` all inherited from Q34B orchestrator

Complexity estimate: ~400–450 lines (mirrors `run_q34b_dv_nas_pilot.py` with CV search space and stability filter).

### NON-BLOCKING (document only)

- `displacement_cap` not in `GaussianVariationalAnsatz` — handled in candidate script with `tanh` inline; no qcore change required for pilot
- `GaussianBackend.displacement()` has `torch.img()` typo (should be `torch.imag()`) — this method is not used in the `apply()` path; not blocking

---

## 6. CV Pareto Leaderboard Schema (new columns required)

Q34C leaderboard needs new columns vs Q34B:

| New Column | Type | Notes |
|---|---|---|
| `stability_taxonomy` | string | valid / unstable_covariance / covariance_not_psd / nan_state / invalid_symplectic / timeout / unstable_training |
| `n_modes` | int | Q33B dimension |
| `cv_depth` | int | Q33B dimension |
| `squeezing_cap` | float | Q33B dimension |
| `displacement_cap` | float | Q33B dimension (pilot-fixed at 2.0) |
| `encoding_strategy` | string | Q33B dimension (pilot-fixed) |
| `beam_splitter_topology` | string | Q33B dimension (pilot-fixed: circular) |
| `readout_strategy` | string | Q33B dimension (pilot-fixed: first_moments) |
| `covariance_parameterization` | string | Q33B dimension (pilot-fixed: symplectic) |

Output path: `experiments/leaderboards/q34c_cv_pareto.csv`

---

## 7. Thread Cap Applicability

**Thread cap fully applies to Q34C.** CV circuit simulation uses the same PyTorch tensor ops as DV simulation (symplectic matrix operations via `torch.mv`, matrix multiply). Each CV trial subprocess will spawn ~30 PyTorch threads by default. With 5 parallel workers: 150 threads on 12 CPUs.

**Recommended Q34C command:**
```bash
python3 scripts/run_q34c_cv_nas_pilot.py \
  --trials 5 --seed 42 --epochs 2 \
  --parallel --max-workers 5 --thread-cap 2
```

**Expected wall time** (with thread cap, vs Q34B-Parallel-Lite at 4621s): CV at n_modes=2,d=1 ran in ~82 s/epoch (Q27 benchmark). At 2 epochs, per-trial ~164s. With thread contention but thread cap reducing it: estimated total pilot wall time **300–600 s (~5–10 min)**. CV circuits are analytically simpler than DV circuits (covariance matrix ops on 4×4–12×12 matrices vs quantum state vector ops).

---

## 8. Readiness Summary

| Category | Status |
|---|---|
| Parallel + thread-cap infrastructure | ✅ READY |
| Q31 runner subprocess protocol | ✅ READY |
| `GaussianVariationalAnsatz` (n_modes, depth, squeezing_cap) | ✅ READY |
| `GaussianBackend` (apply_symplectic, get_vacuum) | ✅ READY |
| `train_q34c_cv_candidate.py` | ❌ MISSING — **BLOCKER 1** |
| `run_q34c_cv_nas_pilot.py` | ❌ MISSING — **BLOCKER 2** |
| displacement_cap in ansatz | ⚠️ workaround available in candidate script |
| Stability taxonomy checks | ⚠️ must be implemented inline in candidate script |
| Q33B variable dimensions (n_modes, cv_depth, squeezing_cap, compression_dim) | ✅ all parameterized |
| Q33B pilot-substituted dimensions (topology, readout, encoding, cov_param) | ✅ valid substitutions documented |

**Q34C is blocked only by the two missing scripts.** No qcore changes required for a 5-trial pilot with substitutions.

---

## 9. Recommended Next Slice

**Slice: Q34C-IMPL-CANDIDATE-SCRIPT**  
Write `scripts/train_q34c_cv_candidate.py` and `scripts/run_q34c_cv_nas_pilot.py` as a single atomic slice. Validate with Python syntax check and a `--trials 1 --epochs 1` dry-run (non-parallel, 1 trial, 1 epoch) to confirm Q34C_TRIAL_* output lines are parseable before the full pilot runs.

Pilot substitutions to document (not errors):
- `beam_splitter_topology`: always `circular` (hardcoded in ansatz)
- `readout_strategy`: always `first_moments` (always uses `mu_final`)
- `encoding_strategy`: always `displacement_encoding` (basic encoding; reuploading deferred)
- `covariance_parameterization`: always `symplectic_parameterization` (correct by design)
- `displacement_cap`: pilot-fixed at 2.0; `tanh(disp_raw) × 2.0` applied inline in candidate

After smoke validation: run full 5-trial pilot with `--parallel --max-workers 5 --epochs 2 --thread-cap 2`.
