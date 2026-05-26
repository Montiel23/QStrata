# Q31: Local GPU Experiment Runner MVP

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q31 — Local GPU Experiment Runner MVP  
**Status:** COMPLETE — smoke test PASS

---

## 1. Title

Q31: Local GPU Experiment Runner MVP

---

## 2. Context

Q30 (Slice, 2026-05-26) defined the experiment automation framework architecture and the YAML config schema that all future NAS and benchmarking work depends on. Q30 was design only — no orchestration code was written.

Q31 implements the minimal viable runner that proves the Q30 framework can execute one existing script end-to-end with reproducible metadata, a frozen config, captured logs, a result JSON, and a leaderboard CSV entry. The runner wraps the existing Q26 CV binary smoke test (`scripts/smoke_test_vindr_cv_binary.py`) as its first real execution target.

Q31 is a necessary precondition for all NAS work. Classical NAS (Q32) cannot proceed until the runner is validated. Q31's scope is deliberately minimal: prove the pipeline, confirm the artifact structure, and establish a working baseline before hardening and extending.

---

## 3. MVP Scope

### Implemented in Q31

1. **YAML config loading** — PyYAML `safe_load` via CLI argument
2. **Basic schema validation** — required fields checked by dot-notation path traversal; raises `ValueError` listing all missing fields
3. **`experiment_id` generation** — `YYYYMMDD_HHMMSS_<6-char hex>` from OS entropy; unique even for same-second starts
4. **Experiment directory creation** — `experiments/configs/`, `experiments/results/`, `experiments/logs/`, `experiments/leaderboards/`
5. **Frozen config copy** — validated YAML written to `experiments/configs/<experiment_id>.yaml`; `chmod 444` applied
6. **Git commit capture** — `git rev-parse HEAD` at run start; falls back to `"unknown"` if git unavailable (see Section 9)
7. **Git dirty flag capture** — checks unstaged and staged changes separately
8. **Hardware metadata capture** — GPU name, CUDA version, CPU fallback flag, GPU memory; graceful fallback if torch unavailable
9. **Sequential subprocess execution** — `subprocess.Popen` with `stdout=PIPE`, `stderr=STDOUT`
10. **stdout/stderr tee** — lines written simultaneously to log file and terminal
11. **Status tracking** — `"completed"` if `return_code == 0`, `"failed"` otherwise
12. **Result JSON creation** — `experiments/results/<experiment_id>.json` with all required fields (see Section 7)
13. **Leaderboard CSV append** — `experiments/leaderboards/<phase>.csv` created or appended after each run

### Explicitly Deferred from Q31

- Signal handler recovery (SIGINT, SIGTERM, SIGUSR1)
- Checkpoint-based interruption and resume
- Per-epoch partial metric export (atomic writes)
- Queue-based multi-experiment orchestration
- NAS trial generation or orchestration
- Distributed execution (AWS, Ray)
- Full training runs
- Advanced metric parsing beyond trainable_params and loss
- Reproducibility test (same config + seed → same result within tolerance)

---

## 4. Files Created

| File | Description |
|---|---|
| `qcore/experiments/__init__.py` | Package init for the experiments runner module |
| `qcore/experiments/schema.py` | Config validation — required field presence check |
| `qcore/experiments/metadata.py` | experiment_id generation, git commit capture, hardware capture |
| `qcore/experiments/leaderboard.py` | Phase leaderboard CSV append / create |
| `qcore/experiments/runner.py` | Main runner — orchestrates all modules; produces artifacts |
| `scripts/run_experiment.py` | CLI entry point; thin wrapper over runner.py |
| `configs/experiments/q31_smoke_vindr_cv_binary.yaml` | Q31 smoke experiment config |
| `reports/q31_local_gpu_experiment_runner_mvp.md` | This report |
| `docs/roadmaps/qstrata_master_research_roadmap.md` | Updated to reflect Q31 COMPLETE |

Experiment artifacts generated during smoke execution (not committed to source):
| Artifact | Path |
|---|---|
| Frozen config | `experiments/configs/20260526_222939_a508a2.yaml` |
| Result JSON | `experiments/results/20260526_222939_a508a2.json` |
| Log file | `experiments/logs/20260526_222939_a508a2.log` |
| Leaderboard CSV | `experiments/leaderboards/q31_runner_smoke.csv` |

---

## 5. Runner Architecture

**`runner.py`** is the primary execution engine. It accepts a parsed config dict, validates it, generates an `experiment_id`, creates experiment directories, freezes the config as a read-only YAML file, captures git and hardware metadata, builds the subprocess command from `config.command.executable` + `config.command.args`, executes it with stdout/stderr teed to both the log file and the terminal, records timing and return code, parses `smoke_pass` / `trainable_params` / `loss` from stdout, writes the result JSON, updates the leaderboard CSV, and returns the subprocess return code to the caller. All side effects are organized under `experiments/` and identified by `experiment_id`.

**`schema.py`** validates a config dict against six required fields using dot-notation path traversal. It does not use external validation libraries. Missing fields are collected and reported together in a single `ValueError`, so the caller sees all problems at once rather than one at a time.

**`metadata.py`** provides three stateless capture functions. `generate_experiment_id()` combines a timestamp with 3 bytes of OS entropy to produce a collision-resistant identifier. `capture_git_commit()` runs `git rev-parse HEAD` and two `git diff` checks, returning `{"commit": "unknown", "dirty": True}` on any failure (e.g., git not installed in container). `capture_hardware()` queries torch CUDA properties and returns safe defaults if torch is unavailable.

**`leaderboard.py`** maintains a per-phase CSV at `experiments/leaderboards/<phase>.csv`. The `update_leaderboard()` function creates the file with a header row on first call and appends one data row on every subsequent call. It uses Python's built-in `csv.DictWriter` with `extrasaction="ignore"` to tolerate extra fields in the row dict.

**`run_experiment.py`** is a thin CLI script. It parses `--config`, loads the YAML with `yaml.safe_load`, imports `run_experiment` from `qcore.experiments.runner`, calls it with the config dict and config path, and exits with the subprocess return code. All execution logic resides in `runner.py`.

---

## 6. Config Schema Use

`configs/experiments/q31_smoke_vindr_cv_binary.yaml` conforms to the Q30 schema defined in `docs/specs/qstrata_experiment_config_schema.md` with the following notes:

**Standard fields used:** `experiment.phase`, `experiment.description`, `dataset.name`, `dataset.split`, `dataset.preprocessing`, `model.architecture`, `model.backbone`, `model.backbone_frozen`, `model.head`, `model.parameters`, `training.*`, `metrics.primary`, `metrics.secondary`, `artifacts.*`, `reproducibility.seed`.

**Q31-specific addition:** The `command` block (`command.executable`, `command.args`) is not in the Q30 schema's core sections — Q30 designed around training configs that directly call model training. Q31's MVP wraps arbitrary subprocess commands, so the `command` block was added as a required extension. This is the correct approach for an MVP: the runner does not need to know the internal structure of what it executes.

**Simplified fields:** `training.epochs: 0` because the smoke test is not a training run. `early_stopping.enabled: false` for the same reason. `artifacts.*: false` because the smoke test produces no model artifacts.

**`executable: python3`:** The Q31 design spec referenced `executable: python`, but the GPU container (`docker-qstrata-gpu`) only has `python3` in PATH. The config was updated to `python3` to match the actual container environment.

---

## 7. Experiment Artifact Output

All artifacts are identified by `experiment_id: 20260526_222939_a508a2`.

| Artifact | Path |
|---|---|
| Frozen config (read-only) | `experiments/configs/20260526_222939_a508a2.yaml` |
| Result JSON | `experiments/results/20260526_222939_a508a2.json` |
| Execution log | `experiments/logs/20260526_222939_a508a2.log` |
| Phase leaderboard | `experiments/leaderboards/q31_runner_smoke.csv` |

**Config immutability confirmed:** Frozen config permissions are `-r--r--r--` (`0o444`). Modification attempt would fail with `Permission denied`.

**Result JSON note:** `*.json` is listed in `.gitignore`. The result JSON is generated correctly on disk but is not committed to the repository. The frozen config YAML and leaderboard CSV are not gitignored and are staged.

---

## 8. Smoke Execution Result

| Field | Value |
|---|---|
| Command | `python3 scripts/smoke_test_vindr_cv_binary.py --root data/processed/vindr_binary_roi_224 --checkpoint checkpoints/c006_d040_classical_anchor.pt --batch-size 4 --seed 42` |
| Return code | `0` |
| Status | `completed` |
| `smoke_pass` | `True` |
| Duration | `2.854` seconds |
| `experiment_id` | `20260526_222939_a508a2` |
| Trainable params | `536` |
| Loss | `0.781537` |
| All 14 health checks | PASS |

Execution was run inside the `docker-qstrata-gpu` container (GPU: NVIDIA GeForce RTX 2060 SUPER, CUDA 12.1) via `docker compose exec`.

---

## 9. Reproducibility Metadata

| Field | Value |
|---|---|
| `git_commit` | `unknown` |
| `git_dirty` | `true` |
| `seed` | `42` |
| `hardware.gpu_model` | `NVIDIA GeForce RTX 2060 SUPER` |
| `hardware.cuda_version` | `12.1` |
| `hardware.cpu_fallback` | `false` |
| `hardware.gpu_memory_mb` | `7783` |

**Git capture note:** The `docker-qstrata-gpu` container does not have `git` installed. `capture_git_commit()` caught the subprocess exception and returned the documented fallback `{"commit": "unknown", "dirty": True}`. This is the correct behavior per the Q30 design specification ("on failure return `{"commit": "unknown", "dirty": True}`"). Adding `git` to the Docker image or capturing the commit from the host before exec is a Q31A hardening item.

**Dirty flag note:** The working tree had uncommitted Q31 implementation files at the time of the smoke run. This is expected — the Q31 files were not yet committed when the smoke test was executed.

---

## 10. Limitations

The Q31 MVP is intentionally narrow. The following are known limitations, not defects:

**No NAS support.** The runner executes one sequential subprocess per call. It does not generate, queue, or orchestrate NAS trial configs. NAS support begins in Q32 (search space design) using this runner as the execution backend.

**No full resume.** If a subprocess is interrupted mid-run, the runner exits with a non-zero return code and writes `status: failed`. There is no checkpoint recovery or partial-metric preservation for interrupted runs. Full resume logic is deferred to Q31A hardening.

**No signal handler recovery.** `SIGINT`, `SIGTERM`, and `SIGUSR1` are not intercepted. The runner does not write `status: interrupted` before process exit. Signal handling is deferred to Q31A.

**No distributed execution.** Single-process, single-GPU, single-machine only. AWS and Ray remain blocked until Q34 validates local NAS.

**Limited metric parsing.** `smoke_pass`, `trainable_params`, and `loss` are extracted from stdout via simple string matching. All other metrics present in the raw stdout (e.g., AUROC, F1, per-check results) are stored in the log file but not parsed into structured form. Full metric parsing is deferred.

**Git commit unavailable in container.** `git` is not installed in `docker-qstrata-gpu`. The git commit field records `"unknown"` for all container-executed experiments. Installing `git` in the container or pre-capturing the commit on the host are both viable Q31A fixes.

**No reproducibility test.** The runner does not yet automatically verify that re-running the same config produces identical results within tolerance. This test is required before Q32 NAS begins and is the primary deliverable of Q31A.

---

## 11. Next Slice

**Result: PASS**

Q31 smoke test passed (`return_code: 0`, `smoke_pass: True`, all 14 health checks PASS).

**Q31A — Runner Reproducibility Test and Hardening**

Q31A is recommended before Q32 proceeds. Q31A should address:
1. Reproducibility test: run the Q31 smoke config twice with seed 42; confirm `smoke_pass=True` and `loss` within ±0.0001 on both runs
2. Git commit capture: resolve container git availability (install git in image or pre-capture on host)
3. Signal handler skeleton: register SIGINT handler that sets `status: interrupted` and flushes log on abnormal exit
4. Schema extension: document `command` block in `docs/specs/qstrata_experiment_config_schema.md`
5. Consider whether `.gitignore` should be amended to allow tracking small result JSON files under `experiments/results/`

State: Q31A is recommended before Q32 regardless of result quality.

```
Q31 status: COMPLETE — smoke PASS
Q31A status: NEXT — Runner Reproducibility Test and Hardening
Q32 status: PLANNED — gated after Q31A
```

---

```
Q31 status: COMPLETE — b9c8fcf → Q31 commit pending
Smoke test PASS: experiment_id 20260526_222939_a508a2
smoke_pass: True | return_code: 0 | duration: 2.854s
Hardware: NVIDIA GeForce RTX 2060 SUPER, CUDA 12.1
Q31A status: NEXT
Q32 status: PLANNED (gated after Q31A)
NAS: GATED — requires Q31A
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 local NAS validated
```
