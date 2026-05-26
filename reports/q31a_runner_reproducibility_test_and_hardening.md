# Q31A: Runner Reproducibility Test and Hardening

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q31A — Runner Reproducibility Test and Hardening  
**Status:** COMPLETE — Q31A_REPRODUCIBILITY_TEST: PASS

---

## 1. Title

Q31A: Runner Reproducibility Test and Hardening

---

## 2. Context

Q31 implemented the local GPU experiment runner MVP and validated it with a single execution of the Q26 CV binary smoke test. Q31 identified three known limitations requiring resolution before NAS work begins:

1. Git commit capture returned `"unknown"` inside the Docker container (no `git` installed)
2. The `command` block was present in `schema.py` but not documented in the schema specification
3. No signal handler was registered; interrupted runs would exit without writing `status: interrupted`

Q31A validates that the runner produces reproducible results across identical runs, and hardens the three identified gaps. A runner that is not reproducible cannot be trusted to produce meaningful NAS trial comparisons. Reproducibility is not aspirational — it is a precondition for any search that produces result tables with comparative claims.

---

## 3. Scope

### Implemented in Q31A

1. **Reproducibility test script** — `scripts/test_runner_reproducibility.py`: runs the same config twice sequentially, compares `status`, `return_code`, `smoke_pass`, `trainable_params`, and `loss` (within tolerance), writes `experiments/results/q31a_reproducibility_test.json`, prints `Q31A_REPRODUCIBILITY_TEST: PASS/FAIL`
2. **Sequential double-run comparison** — same config, same seed, both runs compared across five fields
3. **Git metadata env var fallback** — `QSTRATA_GIT_COMMIT` and `QSTRATA_GIT_DIRTY` env vars added to `capture_git_commit()`; git commit now resolves correctly when injected via Docker exec
4. **Command block formalization** — `command.executable` and `command.args` confirmed in `REQUIRED_FIELDS`; dedicated Section 8 added to `docs/specs/qstrata_experiment_config_schema.md` documenting purpose, NAS compatibility, and container note
5. **SIGINT/SIGTERM handler skeleton** — `_signal_handler()`, `_register_signal_handlers()`, `_interrupted` flag, and `KeyboardInterrupt` safety net added to `runner.py`; interrupted experiments write `status: interrupted`
6. **Q31A report** — this document
7. **Roadmap update** — Q31A COMPLETE, Q32 NEXT

### Explicitly Deferred from Q31A

- NAS trial generation or orchestration
- Full checkpoint-based resume (on interruption, resume training from last epoch)
- Per-epoch partial metric accumulation (write metrics at end of each epoch atomically)
- Experiment scheduler / queue daemon
- Distributed execution (AWS, Ray)
- Full training runs (not in Q31A scope)
- Advanced metric parsing beyond `smoke_pass`, `trainable_params`, `loss`

---

## 4. Reproducibility Test Design

**Script:** `scripts/test_runner_reproducibility.py`

**Protocol:**
- Load the config YAML from `--config`
- Call `scripts/run_experiment.py --config <path>` twice sequentially as a subprocess
- Parse each run's result JSON path from the `[RUNNER] result JSON:` line in stdout
- Load both result JSONs from `experiments/results/`
- Compare five fields:

| Field | Comparison | Failure Condition |
|---|---|---|
| `status` | Both must equal `"completed"` | Either not `"completed"` |
| `return_code` | Both must equal `0` | Either non-zero |
| `metrics.smoke_pass` | Both must be `True` | Either `False` or `null` |
| `metrics.trainable_params` | Must be equal | Different values (skip if either `null`) |
| `metrics.loss` | Absolute delta ≤ tolerance | Delta > tolerance (skip if either `null`) |

**Tolerance:** `0.0001` (matches the Q30 design requirement: AUROC within ±0.0001, F1 within ±0.0001)

**Output:** `experiments/results/q31a_reproducibility_test.json`

**Note:** The reproducibility test is deliberately narrow — it tests the smoke test pipeline, not full training. Full training reproducibility (multi-epoch, AUROC/F1 matching) is a property to verify in Q32 once NAS training scripts exist.

---

## 5. Reproducibility Test Results

**Execution:** 2026-05-26, inside `docker-qstrata-gpu` container with git env vars injected.

| Field | Run 1 | Run 2 |
|---|---|---|
| `experiment_id` | `20260526_232753_1f1b26` | `20260526_232757_a4a4aa` |
| `status` | `completed` | `completed` |
| `return_code` | `0` | `0` |
| `smoke_pass` | `True` | `True` |
| `trainable_params` | `536` | `536` |
| `loss` | `0.781537` | `0.781537` |

**Comparison results:**

| Check | Result |
|---|---|
| `status_match` | PASS — both `completed` |
| `return_code_match` | PASS — both `0` |
| `smoke_pass_match` | PASS — both `True` |
| `trainable_params_match` | PASS — both `536` |
| `loss_delta` | `0.00000000` (tolerance = `0.0001`) |
| `loss_within_tolerance` | PASS |

**Verdict:** `Q31A_REPRODUCIBILITY_TEST: PASS`

Loss delta is exactly 0.0 — both runs produce bitwise-identical results. This is expected given the deterministic seed (42), frozen backbone weights, and the fact that the smoke test performs a single forward+backward+optimizer step with no stochastic augmentation.

---

## 6. Git Metadata Hardening

**Problem in Q31:** The `docker-qstrata-gpu` container does not have `git` installed. Every container-run experiment recorded `git_commit: "unknown"` and `git_dirty: true`, making all container-run experiments non-traceable to a specific code state.

**Solution in Q31A:** Added environment variable priority to `capture_git_commit()` in `qcore/experiments/metadata.py`.

**Priority order for commit SHA:**
1. `QSTRATA_GIT_COMMIT` environment variable — used if present and non-empty
2. `git rev-parse HEAD` subprocess — used if git is available
3. `"unknown"` — final fallback

**Priority order for dirty flag:**
1. `QSTRATA_GIT_DIRTY` environment variable — accepts `"true"` or `"false"` (case-insensitive)
2. `git diff --quiet` + `git diff --cached --quiet` subprocess checks
3. `True` — conservative fallback

**Recommended Docker invocation (with git capture):**
```bash
docker compose -f infra/docker/docker-compose.gpu.yml exec \
  -e QSTRATA_GIT_COMMIT=$(git rev-parse HEAD) \
  -e QSTRATA_GIT_DIRTY=false \
  qstrata-gpu \
  python3 scripts/run_experiment.py \
    --config configs/experiments/<config>.yaml
```

**Validation:** In the Q31A reproducibility test, both runs recorded `git_commit: 156302eecec6` (the first 12 chars of `156302eecec674e4888d662260800988fa9eeb1c`) and `git_dirty: False`. The leaderboard CSV confirms the resolved values. Git capture is now functional in the container when env vars are provided.

**Limitation:** Git env vars must be passed manually by the caller. If a run is executed without them, `"unknown"` still results. A Makefile wrapper or shell alias that always injects the git vars is a simple operational fix (not a framework code change).

---

## 7. Command Schema Formalization

**Problem in Q31:** The `command` block (`command.executable`, `command.args`) was in `REQUIRED_FIELDS` in `schema.py` and in the smoke config, but was not documented in `docs/specs/qstrata_experiment_config_schema.md`.

**Resolution in Q31A:**
- `docs/specs/qstrata_experiment_config_schema.md` Section 2 updated to list `command` as the seventh top-level section
- Full YAML schema updated to include the `command` block with field-level comments
- New Section 8 ("Command Block — Formalized in Q31A") added, covering:
  - Purpose: decouples runner from model logic
  - Required fields: `command.executable`, `command.args`
  - NAS compatibility example
  - Container note: use `python3`, not `python`
- Schema version bumped from 1.0 → 1.1
- `qcore/experiments/schema.py` module docstring updated to reflect Q31A formalization

**Significance for NAS:** NAS trial generation (Q32, Q33, Q34) will produce `command` blocks programmatically. The runner executes them identically to hand-crafted configs, with no runner modification required.

---

## 8. Signal Handler Skeleton

**Scope in Q31A:** Minimal skeleton only. No checkpoint recovery, no partial epoch metric accumulation.

**Implementation in `runner.py`:**

```python
_interrupted: bool = False   # module-level flag; reset per run_experiment() call

def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    sys.stderr.write(f"\n[RUNNER] {sig_name} received — experiment will be marked interrupted\n")

def _register_signal_handlers():
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
```

**Behavior on signal:**
- `_interrupted` flag set to `True`
- Signal name written to stderr (does not corrupt tee'd stdout)
- Subprocess continues to run (it may also receive SIGINT from Ctrl+C if in same process group)
- After subprocess exits, runner checks `_interrupted` and sets `status = "interrupted"`, `return_code = 130`
- Result JSON written with `status: interrupted`
- Leaderboard entry appended with `status: interrupted`

**KeyboardInterrupt safety net:** A `try/except KeyboardInterrupt` block wraps the subprocess execution loop. If the signal handler was not registered (e.g., non-main thread), `KeyboardInterrupt` is caught, the flag is set, the subprocess is terminated, and the interrupted result is written.

**`_register_signal_handlers()` failure handling:** The registration call is wrapped in `try/except (OSError, ValueError)` to handle non-main thread and platform-specific limitations. Failure is logged to stderr and execution continues — signal handling is best-effort in Q31A.

**What is deferred:**
- Active subprocess termination on signal (`proc.terminate()` mid-run)
- Preserving partial epoch metrics (requires per-epoch callback hooks in training scripts)
- Checkpoint-based resume from last completed epoch

**Platform note:** `signal.signal()` for `SIGTERM` is not supported on Windows. The safety net `KeyboardInterrupt` handler works on all platforms for SIGINT.

---

## 9. Artifacts Generated

| Artifact | Path | Notes |
|---|---|---|
| Reproducibility test report | `experiments/results/q31a_reproducibility_test.json` | Summary of both run comparisons; `pass: true` |
| Run 1 result JSON | `experiments/results/20260526_232753_1f1b26.json` | gitignored; not committed |
| Run 2 result JSON | `experiments/results/20260526_232757_a4a4aa.json` | gitignored; not committed |
| Run 1 frozen config | `experiments/configs/20260526_232753_1f1b26.yaml` | read-only YAML; committed |
| Run 2 frozen config | `experiments/configs/20260526_232757_a4a4aa.yaml` | read-only YAML; committed |
| Leaderboard CSV (updated) | `experiments/leaderboards/q31_runner_smoke.csv` | now has 3 entries (Q31 run + 2 Q31A runs) |

**Leaderboard CSV after Q31A (3 entries):**

| experiment_id | status | smoke_pass | git_commit | git_dirty |
|---|---|---|---|---|
| `20260526_222939_a508a2` | `completed` | `True` | `unknown` | `True` |
| `20260526_232753_1f1b26` | `completed` | `True` | `156302eecec6` | `False` |
| `20260526_232757_a4a4aa` | `completed` | `True` | `156302eecec6` | `False` |

Row 1 is the Q31 smoke run (git unavailable in container). Rows 2 and 3 are the Q31A reproducibility runs with git env vars injected. The git_commit field progression from `unknown` → `156302eecec6` confirms the env var fallback works correctly.

---

## 10. Limitations

**No NAS.** Q31A does not implement any search, trial generation, or optimization. The runner wraps sequential single-experiment execution only.

**No AWS/Ray.** All execution is local, single-GPU. Distributed infrastructure remains BLOCKED until Q34 validates local NAS.

**No queue daemon.** Experiments are started explicitly by the caller. There is no background service polling a queue directory.

**No checkpoint recovery.** An interrupted experiment cannot be resumed from its last checkpoint. The `status: interrupted` JSON is written, but no resume path exists yet.

**No distributed execution.** Single-process, single-GPU, single-machine only.

**Limited metric parsing.** `smoke_pass`, `trainable_params`, and `loss` are extracted from stdout via string matching. All other metrics in the raw stdout are stored in the log but not parsed into the result JSON.

**Git env vars are caller's responsibility.** The runner resolves git metadata from env vars if present, but does not fail if they are absent. Experiments run without env vars in a git-less container still record `"unknown"`.

**No full training.** Q31A validates only the smoke test pipeline. Full training reproducibility (multi-epoch AUROC/F1 matching within tolerance) is not tested and is a property to verify in Q32.

**Signal handler activates post-subprocess only.** The Q31A skeleton does not pre-emptively kill the subprocess when a signal arrives. For long-running training, the signal is only acted on after the subprocess completes naturally. Full pre-emption is deferred to Q32+.

---

## 11. Next Slice

**Result: PASS** — `Q31A_REPRODUCIBILITY_TEST: PASS`

**Q32 — NAS Search Space Design: Classical Feature Extractors**

Q32 is a design-only slice (no NAS execution). It will define the classical CNN search space (block types, channel counts, depths, pooling strategies) and produce the YAML template that Q31A's runner will use to execute NAS trials.

Q32 gate: Q31A complete ✓

**Q32 is design only.** No NAS search is executed in Q32. No new training runs are started. The Q32 output is a search space specification document and a config template. NAS execution begins in Q34.

```
Q31A status: COMPLETE
Q32 status: NEXT — design only; no NAS execution
Q31A_REPRODUCIBILITY_TEST: PASS
Run 1: 20260526_232753_1f1b26 | loss=0.781537 | smoke_pass=True
Run 2: 20260526_232757_a4a4aa | loss=0.781537 | smoke_pass=True
loss_delta: 0.00000000 (tolerance: 0.0001)
```

---

```
Q31A status: COMPLETE — reproducibility PASS
Q32 status: NEXT — NAS Search Space Design, Classical (design only)
NAS execution: BLOCKED until Q33 + Q34
Multiclass: BLOCKED — requires Phase 3 + 4 + 5
AWS/Ray: BLOCKED — requires Q34 local NAS validated
Object detection: BLOCKED — out of current roadmap scope
```
