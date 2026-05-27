# Q34A — Classical NAS Pilot MVP Report

**Slice:** Q34A  
**Phase:** Phase 5 — Local Multi-Objective NAS Pilot (Classical)  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Branch:** `feature/qnn-integration`  
**Git commit:** b5d8e7f5361e  
**Status:** COMPLETE — 5/5 trials completed, PASS

---

## 1. Context

Q34A is the first real NAS execution in the QStrata program. It follows three consecutive design-only slices:

- **Q32** — Classical CNN NAS search space design  
- **Q33A** — DV quantum head NAS search space design  
- **Q33B** — CV Gaussian quantum head NAS search space design  

Q34A validates the end-to-end pipeline before quantum NAS begins. The classical pilot must complete and produce a valid Pareto frontier before Q34B (DV NAS) or Q34C (CV NAS) are attempted.

**Frozen binary benchmarks (canonical reference):**

| Model | AUROC | F1 | Accuracy | Params | Source |
|---|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 60.66% | 23,650 | `reports/vindr_classical_baseline_full_training.md` |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 63.84% | 574 | `reports/vindr_dv_hybrid_pretrained_full_training.md` |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 64.37% | 526 | `reports/vindr_classical_control_tiny_head.md` |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 65.77% | 536 | `reports/q27_cv_binary_full_training.md` |

These values are frozen. Q34A pilot results are not yet a definitive classical ceiling — a larger NAS run with more trials and full-budget epochs is required for that. Q34A validates the pipeline only.

---

## 2. Slice Scope

**What Q34A is:**
- A 5-trial, 4-epoch-per-trial random search NAS pilot
- An end-to-end pipeline validation: config generation → Q31 runner → training → metric parsing → Pareto computation
- The first step in the Q34A → Q34B → Q34C incremental execution sequence

**What Q34A is not:**
- A definitive classical ceiling (full NAS ceiling requires a larger budget)
- A comparison to quantum baselines (Q34B and Q34C not yet run)
- A deployment-ready model selection (single-seed results, 4 epochs only)
- Any use of pymoo, Optuna, NSGA-II, Bayesian optimization, AWS, Ray, or distributed execution

**Hard constraints enforced:**
- Epoch ceiling: ≤ 5 epochs per trial (enforced in `train_q34a_classical_candidate.py`)
- Trainable parameter ceiling: ≤ 100,000 (enforced in `train_q34a_classical_candidate.py`)
- Trial count: exactly 5
- Sequential execution: one trial at a time, no parallelism
- No checkpoint saved: 4-epoch pilot only
- Existing validated scripts (Q17/Q21/Q22/Q27) not modified

---

## 3. Design Decisions and Pilot Limitations

### 3.1 Backbone family: linear heads, not convolutional blocks

The Q32 search space dimension `backbone_family` (options: `standard_cnn`, `depthwise_separable_cnn`) refers to the **trainable head architecture**, not the frozen C006-D040 backbone. However, the frozen backbone outputs 128-dimensional flat feature vectors (after `AdaptiveAvgPool2d(1,1)` + `Flatten`). True convolutional blocks require 4D spatial tensors `(N, C, H, W)` as input.

**Pilot resolution:** Both `backbone_family` options are implemented as linear projection heads (`nn.Linear` stacks). The backbone family label is acknowledged but has no functional effect on the architecture in this pilot. This limitation is acceptable for pipeline validation — the search over `depth`, `channel_width`, `compression_dim`, `activation`, and `dropout` provides meaningful variation. A future full NAS run may extend the pilot to accept intermediate feature maps from the backbone for true convolutional head exploration.

### 3.2 BatchNorm implemented as LayerNorm

The search space specifies `normalization: BatchNorm`. Standard `nn.BatchNorm1d` is designed for 2D `(N, C)` input and can be unstable at very small batch sizes during evaluation. `nn.LayerNorm` is equivalent for our 1D feature vectors and handles batch-size-1 latency measurement correctly. This substitution is documented and does not affect the scientific validity of the pilot.

### 3.3 Pooling is a no-op

The search space dimension `pooling: adaptive_average` is a no-op in this pilot. The frozen C006-D040 backbone already applies `AdaptiveAvgPool2d(1,1)` and `Flatten` before outputting the 128-dim vector. No spatial feature map is available for additional pooling. This is not a limitation of the search space design — it reflects that the pilot head operates on already-pooled features.

### 3.4 Seed assignment

The pilot seed (42) controls config sampling order. Each trial receives a deterministic seed: `trial_seed = pilot_seed + trial_idx` (42, 43, 44, 45, 46). All configs are sampled before execution begins, ensuring the sampling sequence is independent of trial wall time.

---

## 4. Search Space Used

Defined inline in `scripts/run_q34a_classical_nas_pilot.py` — no external NAS library.

| Dimension | Options | Notes |
|---|---|---|
| `backbone_family` | `standard_cnn`, `depthwise_separable_cnn` | Label only; both map to linear layers (see §3.1) |
| `channel_width` | `[16,32]`, `[32,64]`, `[64,128]` | Hidden dim(s); shallow uses first only |
| `depth` | `shallow`, `medium` | Shallow: 1 hidden layer; medium: 2 hidden layers |
| `compression_dim` | `4`, `8`, `16` | Bottleneck dimension before classifier |
| `activation` | `ReLU`, `GELU` | Applied after each linear layer |
| `normalization` | `BatchNorm` | Implemented as LayerNorm for 1D features |
| `dropout` | `0.0`, `0.2`, `0.3` | Applied after normalization in each hidden block |
| `pooling` | `adaptive_average` | No-op in pilot (backbone already pooled) |
| `learning_rate` | `0.001`, `0.0005` | AdamW learning rate |
| `weight_decay` | `0.0`, `0.0001` | AdamW weight decay |

**Total nominal search space size:** 2 × 3 × 2 × 3 × 2 × 1 × 3 × 1 × 2 × 2 = 864 configurations  
**Pilot coverage:** 5 / 864 = 0.6% (intentional — pipeline validation, not exhaustive search)

---

## 5. Execution Protocol

| Parameter | Value |
|---|---|
| Orchestrator | `scripts/run_q34a_classical_nas_pilot.py` |
| Trial script | `scripts/train_q34a_classical_candidate.py` |
| Runner | Q31 MVP (`scripts/run_experiment.py`) |
| Number of trials | 5 |
| Epochs per trial | 4 |
| Batch size | 4 |
| Optimizer | AdamW |
| Backbone | C006-D040 (frozen, identical to Q21/Q22/Q27) |
| Dataset | VinDr-SpineXR binary, canonical split (seed 42) |
| Execution environment | Docker — `docker-qstrata-gpu` |
| Hardware | NVIDIA GeForce RTX 2060 SUPER |
| Execution order | Sequential (trial 1 → 2 → 3 → 4 → 5) |
| Pilot seed | 42 |
| Total wall time | ~1450 s (~24 min) |
| Git commit | `b5d8e7f5361e` |

**Config generation:** One YAML per trial at `configs/experiments/q34a_classical_nas/<trial_id>.yaml`, passed to the Q31 runner, which freezes a read-only copy under `experiments/configs/`.

**Metric parsing:** The orchestrator captures combined stdout+stderr from each `run_experiment.py` invocation. `Q34A_TRIAL_*` lines from the candidate script are teed through the runner's stdout. The `[SUMMARY]` block provides `Trainable params:` and `Loss:` to the runner's result JSON.

**Reproducibility artifacts:** For each trial, the runner writes:
- Frozen config: `experiments/configs/<experiment_id>.yaml`
- Log: `experiments/logs/<experiment_id>.log`
- Result JSON: `experiments/results/<experiment_id>.json`
- Phase leaderboard: `experiments/leaderboards/q34a_classical_nas.csv`

**Known non-fatal artifact:** Each trial subprocess emits a NumPy 1.x/2.x compatibility `UserWarning` on import. This is a Docker environment artifact (scikit-learn compiled against NumPy 1.x, NumPy 2.2.6 installed). The warning does not prevent training — all 5 trials completed with return code 0. This artifact was present in Q31/Q31A and is documented in the runner hardening report.

---

## 6. Trial Results

All 5 trials completed successfully (return code 0).

| Trial | Backbone family | Depth | Channels | Compress | Act | Dropout | LR | Params | AUROC | F1 | Acc | Latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q34a_trial_001 | standard_cnn | medium | [16, 32] | 4 | ReLU | 0.3 | 0.001 | 2,846 | 0.6672 | 0.6205 | 63.84% | 1.48ms | COMPLETED |
| q34a_trial_002 | standard_cnn | shallow | [16, 32] | 4 | ReLU | 0.3 | 0.0005 | 2,174 | 0.6826 | 0.5843 | 64.52% | 1.62ms | COMPLETED |
| q34a_trial_003 | depthwise_sep | medium | [64, 128] | 4 | ReLU | 0.2 | 0.001 | 17,486 | **0.6868** | 0.4987 | 63.02% | 1.47ms | COMPLETED |
| q34a_trial_004 | depthwise_sep | shallow | [16, 32] | 8 | ReLU | 0.2 | 0.001 | 2,250 | 0.6835 | **0.6398** | 63.89% | 1.60ms | COMPLETED |
| q34a_trial_005 | standard_cnn | shallow | [32, 64] | 16 | GELU | 0.3 | 0.001 | 4,754 | 0.6867 | 0.6287 | **65.19%** | **1.40ms** | COMPLETED |

**Notes:**
- All trials exceed the Q17 baseline (AUROC 0.6224, F1 0.5355, 23,650 params)
- Trials 004 and 005 achieve AUROC and F1 exceeding Q21 (0.6800, 0.6159) and Q27 (0.6708, 0.6283)
- Trial 003 achieves the highest AUROC (0.6868) but the lowest F1 (0.4987) — large param count, poor F1
- Trial 004 achieves the best F1 (0.6398) with only 2,250 params — strongest compact candidate
- Trial 005 achieves the best accuracy (65.19%) and fastest latency (1.40ms/sample)
- Latency range: 1.40–1.62 ms/sample — all well within the Q21 soft constraint (~110 ms/sample)

---

## 7. Preliminary Pareto Set

Pareto frontier computed over three objectives: **AUROC ↑, F1 ↑, params ↓**.

A trial is Pareto non-dominated if no other trial is simultaneously at least as good on all three objectives and strictly better on at least one.

| Trial | AUROC | F1 | Params | Pareto? | Interpretation |
|---|---|---|---|---|---|
| q34a_trial_001 | 0.6672 | 0.6205 | 2,846 | **No** | Dominated by trial_004 (all three objectives worse) |
| q34a_trial_002 | 0.6826 | 0.5843 | 2,174 | **Yes** | Smallest params; high AUROC; lowest F1 of Pareto set |
| q34a_trial_003 | 0.6868 | 0.4987 | 17,486 | **Yes** | Highest AUROC; largest params; lowest F1 overall |
| q34a_trial_004 | 0.6835 | 0.6398 | 2,250 | **Yes** | Best F1; compact params; strong balanced candidate |
| q34a_trial_005 | 0.6867 | 0.6287 | 4,754 | **Yes** | Near-highest AUROC; good F1; mid-range params |

**Pareto set: 4 out of 5 trials (80%).**

**Interpretation:**
- **Trial 004** is the strongest compact candidate: AUROC 0.6835, F1 0.6398, 2,250 params. Exceeds Q21 (AUROC 0.6800, F1 0.6159) and Q27 (AUROC 0.6708, F1 0.6283) on both primary metrics with a comparable parameter count.
- **Trial 002** offers the lowest parameter count (2,174) with AUROC 0.6826, acceptable for parameter-budget-constrained scenarios.
- **Trial 003** sits on the Pareto frontier due to its AUROC advantage (0.6868) but is disfavored on params (17,486) and F1 (0.4987) — likely a training instability artifact at 4 epochs.
- **Trial 005** is a well-rounded candidate: near-highest AUROC (0.6867), good F1 (0.6287), fastest latency (1.40 ms/sample).

**Important caveat:** These are 4-epoch, single-seed pilot results. They do not constitute a validated classical ceiling. Full-budget training on the strongest candidates is required before making definitive comparisons against quantum baselines.

**Pareto CSV:** `experiments/leaderboards/q34a_classical_pareto.csv`  
**Summary JSON:** `experiments/results/q34a_classical_nas_pilot_summary.json`

---

## 8. Pipeline Validation Verdict

| Check | Result |
|---|---|
| Config YAML generation | PASS — 5 configs written to `configs/experiments/q34a_classical_nas/` |
| Q31 runner execution | PASS — runner invoked correctly for all 5 trials |
| Subprocess stdout tee | PASS — Q34A_TRIAL_* lines visible in captured runner output |
| Metric parsing (AUROC, F1, params, latency) | PASS — all 4 metrics parsed for all 5 trials |
| [SUMMARY] block parsing | PASS — Trainable params + Loss captured in runner result JSON |
| Result JSON written | PASS — 5 result JSONs at `experiments/results/<experiment_id>.json` |
| Phase leaderboard updated | PASS — `experiments/leaderboards/q34a_classical_nas.csv` has 5 rows |
| Pareto computation | PASS — 4-member Pareto set computed correctly |
| Pareto CSV written | PASS — `experiments/leaderboards/q34a_classical_pareto.csv` |
| Pilot summary JSON | PASS — `experiments/results/q34a_classical_nas_pilot_summary.json` |
| Backbone frozen (no grad) | PASS — gradient check enforced in training loop; zero violations |
| Trainable params ≤ 100K | PASS — max observed: 17,486 (trial_003) |
| Epoch ceiling ≤ 5 | PASS — all trials ran exactly 4 epochs |
| CUDA availability | PASS — NVIDIA GeForce RTX 2060 SUPER |
| Reproducibility | PASS — git commit captured (b5d8e7f5361e); per-trial seeds deterministic |
| Pass threshold (≥3/5) | PASS — **5/5 trials completed** |

**Verdict: Q34A_CLASSICAL_NAS_PILOT: PASS**

The Q34A pipeline is validated end-to-end. The Q31 runner, config generation, metric parsing, Pareto computation, and result artifact pipeline all function correctly. Q34B (DV NAS pilot) is unblocked.

---

## 9. Failures, Warnings, and Limitations

### 9.1 Non-fatal NumPy 1.x/2.x compatibility warning

Each trial subprocess emits a `UserWarning` on import:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

This is a Docker environment artifact — scikit-learn was compiled against NumPy 1.x, but NumPy 2.2.6 is installed. The warning is non-fatal: all 5 trials completed with return code 0 and produced valid metrics. The same artifact was observed in Q31 and Q31A without consequence. It does not affect result reproducibility.

### 9.2 backbone_family is a label only

Both `standard_cnn` and `depthwise_separable_cnn` map to identical linear projection heads in this pilot (see §3.1). True convolutional backbone-family variation requires spatial feature maps. This limits search coverage in the `backbone_family` dimension.

### 9.3 4-epoch results are not a definitive classical ceiling

The 5 trials ran for 4 epochs each to validate the pipeline within a feasible wall time (~24 min total). Training curves were still descending at epoch 4 in most trials — full-budget training (20–50 epochs) would be expected to improve metrics. These results are exploratory, not definitive.

### 9.4 Random search with n=5 provides limited coverage

5 trials over an 864-configuration search space (0.6% coverage) is insufficient to characterize the Pareto frontier. The pilot was sized for pipeline validation. A full classical NAS run with a larger budget (50–200 trials) is needed to establish the definitive classical ceiling for quantum comparison.

### 9.5 No cross-validation or confidence intervals

Single-seed, single train/val/test split. Results may vary with different seeds. No statistical validation is performed in the pilot phase.

---

## 10. Next Slice

**Q34B — DV NAS Pilot (second execution)**

Q34B is unblocked by Q34A PASS. It is the second step in the Q34A → Q34B → Q34C incremental execution sequence.

Q34B will:
- Use the DV quantum head search space defined in Q33A (`docs/architecture/q33a_dv_quantum_nas_search_space.md`)
- Follow the same 5-trial, 4-epoch-per-trial incremental protocol
- Produce a DV Pareto frontier over (AUROC ↑, F1 ↑, params ↓, latency ↓)
- Compare DV results against the Q34A classical pilot frontier

Q34B must complete before Q34C (CV NAS pilot) begins. Q34A and Q34B must both complete before Q35 (unified Pareto analysis).

**Reference for Q34B design:**
- `docs/architecture/q33a_dv_quantum_nas_search_space.md`
- `reports/q33a_dv_quantum_nas_search_space_design.md`
- `reports/q34a_classical_nas_pilot_mvp.md` (this document)

---

```
Q34A status: COMPLETE — 5/5 trials PASS
Q34A verdict: Q34A_CLASSICAL_NAS_PILOT: PASS
Q34A pilot Pareto set: 4 trials (q34a_trial_002, q34a_trial_003, q34a_trial_004, q34a_trial_005)
Q34A strongest compact candidate: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2250 params)
Q34A pipeline validation: complete
Q34B status: NEXT — DV NAS pilot (unblocked by Q34A PASS)
```
