# NAS Evaluation Module Interfaces v1

- **Status:** Draft
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Informed by:** docs/design/nas_evaluation_plan_v1.md, Slice 23

---

## Purpose

This document defines the interfaces between the v1 NAS evaluation modules before any implementation begins, establishing clear contracts that each module must satisfy. The design is intentionally minimal and terminal-first — no dashboards, no MLflow, no monitoring stack, no UI; all output is written to stdout or plain local files. Defining interfaces first prevents implementation coupling and makes each module independently testable: `pareto.py` can be tested against synthetic result objects before the evaluator exists, and the evaluator can be validated in isolation before the orchestration script exists. No code is created in this slice; this document is the sole deliverable.

---

## Module — `qcore/nas/evaluator.py`

**Responsibility:** Train and evaluate a single candidate using the stable benchmark protocol v1. Return a structured result. Fail fast on any error.

**Inputs:**

| Input | Type | Description |
|---|---|---|
| `candidate_id` | `str` | Candidate identifier (e.g. `C001`) |
| `config_path` | `str` | Path to the candidate's YAML config file |
| `seed` | `int` | Random seed — must be 42 for all v1 evaluations |

**Responsibilities:**
- Load the YAML config from `config_path`
- Apply the fixed seed to Python, NumPy, and PyTorch before any data loading or model instantiation
- Call the stable benchmark protocol v1: train the model, track validation accuracy every epoch, save best-validation checkpoint in memory, evaluate test accuracy at best-validation checkpoint
- Collect all required metrics
- Return a structured result object
- On any failure: stop immediately and surface a clear terminal error — no silent fallback, no partial result

**Output — structured result object:**

| Field | Type | Description |
|---|---|---|
| `candidate_id` | `str` | Candidate identifier |
| `block_type` | `str` | Block type from config |
| `conv_channels` | `list[int]` | Channel configuration from config |
| `params` | `int` | Total trainable parameter count |
| `best_val_acc` | `float` | Best validation accuracy across all epochs (%) |
| `best_epoch` | `int` | Epoch at which best validation accuracy occurred (1-based) |
| `final_train_acc` | `float` | Training accuracy at the last epoch (%) |
| `test_acc_analysis_only` | `float` | Test accuracy at best-validation checkpoint (%) — analysis only, must not be used as fitness |
| `mean_epoch_time` | `float` | Mean wall-clock time per training epoch (seconds) |
| `latency_ms` | `float` | Mean inference latency (ms/batch) on the test set at best-validation checkpoint |

**Failure behavior:** If the evaluator encounters any error during config loading, model building, training, or evaluation — stop immediately, print a clear error to stderr, and raise an exception. Do not return a partial result. Do not silently continue to the next candidate.

---

## Module — `qcore/nas/pareto.py`

**Responsibility:** Accept a list of evaluator result objects, compute Pareto dominance, identify the Pareto front, rank all candidates, and produce a markdown summary table.

**Inputs:**

| Input | Type | Description |
|---|---|---|
| `results` | `list[EvaluatorResult]` | List of structured result objects from the evaluator |

**Objectives used for Pareto ranking:**
- Maximize `best_val_acc`
- Minimize `params`
- Minimize `latency_ms`

Note: `test_acc_analysis_only` must not be used in Pareto dominance calculations.

**Responsibilities:**
- For each candidate, determine whether it is dominated by any other candidate on all three objectives simultaneously
- Identify the Pareto front — the set of candidates not dominated by any other
- Rank all candidates: Pareto-front candidates first, then dominated candidates
- Produce a markdown table summarising all candidates with their metrics, Pareto status, and rank

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pareto_front` | `list[str]` | Candidate IDs on the Pareto front |
| `ranked_candidates` | `list[EvaluatorResult]` | All candidates ordered by Pareto rank |
| `markdown_table` | `str` | Rendered markdown table for inclusion in the report |

**Failure behavior:** If the input list is empty or malformed — stop and raise a clear error. Do not produce a partial or empty table silently.

---

## Module — `scripts/run_nas_v1_eval.py`

**Responsibility:** Orchestrate the full sequential evaluation sweep. Load candidates, run the evaluator for each, collect results, call the Pareto ranker, print a terminal summary, and save output files.

**Inputs:**

| Input | Description |
|---|---|
| `--config` | Path to the YAML config file defining the candidate list, seed, and output paths |

**Responsibilities:**
- Load the candidate list and evaluation settings from the YAML config
- Run the evaluator sequentially for each candidate in order — C001, C004, C006
- Collect all result objects into a results list
- Pass the results list to the Pareto ranker
- Print a formatted summary table to stdout
- Save a markdown report to the configured output path
- Optionally save a CSV of raw results to the configured output path
- On any candidate failure: stop the sweep immediately and report the error — do not skip and continue

**Outputs:**

| Output | Description |
|---|---|
| stdout | Formatted terminal summary — candidate metrics table and Pareto front |
| `reports/nas_v1_eval.md` | Markdown report — full results table, Pareto front, interpretation |
| `reports/nas_v1_eval.csv` | Optional — raw metrics for all candidates in CSV format |

No dashboards. No databases. No tracking servers. No MLflow. No monitoring stack.

**Failure behavior:** Fail fast. If any candidate evaluation fails, stop the sweep, print the error clearly to stderr, and exit with a non-zero status code. Do not produce a partial report.

---

## Data Flow

The explicit execution sequence:

```
1. run_nas_v1_eval.py loads YAML config
2. Candidate list is extracted from config (C001, C004, C006 in order)
3. For each candidate:
   a. evaluator.py is called with candidate_id, config_path, seed=42
   b. evaluator trains the model, tracks validation accuracy per epoch
   c. evaluator selects best-validation checkpoint
   d. evaluator evaluates test accuracy at best-validation checkpoint
   e. evaluator returns structured result object
4. All result objects are collected into a results list
5. pareto.py receives the results list
6. pareto.py computes Pareto dominance across best_val_acc, params, latency_ms
7. pareto.py identifies Pareto front and ranks all candidates
8. pareto.py produces markdown table
9. run_nas_v1_eval.py prints terminal summary to stdout
10. run_nas_v1_eval.py saves markdown report
11. run_nas_v1_eval.py optionally saves CSV
```

Any failure at step 3 halts the entire sequence immediately; no further candidates are evaluated and no report is written.

---

## Output Files

The following output files will be produced by a future implementation. They are not created in this slice:

- `reports/nas_v1_eval.md` — full markdown report: results table, Pareto front, short interpretation
- `reports/nas_v1_eval.csv` — optional raw metrics CSV for all candidates

No dashboards, no databases, no tracking servers, no MLflow, no monitoring stack, no visual UI.

---

## Error Handling

- Fail fast — any error in any module stops execution immediately
- No silent fallback — a failed candidate must not be silently skipped or replaced with a default value
- Clear terminal error — all errors printed to stderr with enough context to diagnose the failure
- No partial success — if the sweep does not complete for all candidates, no report is saved
- Non-zero exit code on failure — the entry-point script must exit with a non-zero status code if any step fails

---

## Out of Scope

- Any code implementation in this slice
- Dashboards or visual UI of any kind
- MLflow, W&B, TensorBoard, or any experiment tracking framework
- Databases or tracking servers
- Ray or any distributed execution framework
- AWS, SkyPilot, or SageMaker
- Docker changes
- Notebook modifications
- NSGA-II or evolutionary search
- pymoo or any optimisation library
- Training runs or experiments

---

## Next Step Recommendation

The next human-approved slice may create minimal code implementations for the three modules defined in this document — `evaluator.py`, `pareto.py`, and `run_nas_v1_eval.py`; any YAML config file required by the entry-point script must be approved explicitly before creation. Implementation must not begin until this interface design has been reviewed and explicitly approved by the human architect. A full evaluation sweep must not be run until the implementation is complete and separately approved in a subsequent slice.
