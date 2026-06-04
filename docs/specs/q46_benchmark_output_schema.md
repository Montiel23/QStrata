# Q46 Feature Extractor Benchmark — Output Schema

**Slice:** Q46F-BENCHMARK-OUTPUT-SCHEMA  
**Date:** 2026-06-01  
**Branch:** feature/q46f_benchmark_output_schema  
**Script:** `scripts/run_q46b_feature_extractor_benchmark.py`  
**Config:** `configs/q46_feature_extractor_benchmark.yaml`

---

## 1. Overview

The Q46 benchmark produces three output artifacts:

| Artifact | Path | Phase | Format |
|---|---|---|---|
| Smoke leaderboard | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` | Phase 1 (1 seed) | CSV |
| Full leaderboard | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` | Phase 2 (3 seeds) | CSV |
| Extended leaderboard | `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv` | Phase 3 (5 seeds, winner only) | CSV |
| Results JSON | `experiments/results/q46b_feature_extractor_benchmark.json` | Post-run summary | JSON |
| Benchmark report | `reports/q46b_feature_extractor_benchmark.md` | Final narrative | Markdown |

Output directories are created automatically by the script if they do not exist.

---

## 2. Leaderboard CSV — Per-Seed Rows

### 2.1 File paths

- Phase 1 (smoke): `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv`
- Phase 2 (full): `experiments/leaderboards/q46b_extractor_full_leaderboard.csv`
- Phase 3 (extended): `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv`

### 2.2 Sorting and ranking

Rows are sorted by `auroc` descending. `rank` is assigned 1-based after sorting. All rows in the file are per-seed observations; summary aggregation rows are defined separately in §3.

### 2.3 Schema

| Column | Type | Description |
|---|---|---|
| `rank` | int | Row rank by `auroc` (descending), 1-based |
| `candidate_id` | str | Backbone identifier: `baseline`, `efficientnet_b0`, `mobilenetv3_small`, `mobilenetv3_large`, `convnext_tiny` |
| `seed` | int | Random seed used for this run |
| `auroc` | float64 | ROC-AUC on the validation split; rounded to 6 decimal places |
| `f1` | float64 | F1 score on the validation split; rounded to 6 decimal places |
| `accuracy` | float64 | Accuracy on the validation split; rounded to 6 decimal places |
| `params_backbone` | int | Parameter count of the frozen backbone (exact) |
| `params_head` | int | Parameter count of the classification head (exact) |
| `params_total` | int | `params_backbone + params_head` |
| `latency_ms_per_batch` | float64 | Wall-time per training batch in milliseconds (total wall_time_s / train_loader steps × 1000); rounded to 2 decimal places |
| `wall_time_s` | float64 | Total wall time for training + eval on this run in seconds; rounded to 2 decimal places |
| `delta_auroc_vs_q45a` | float64 | `auroc − 0.7196` (Q45A 3-seed mean); rounded to 6 decimal places |
| `delta_auroc_vs_q38c` | float64 | `auroc − 0.723922` (Q38C best single-run); rounded to 6 decimal places |
| `delta_f1_vs_q38c` | float64 | `f1 − 0.677858` (Q38C best single-run); rounded to 6 decimal places |

### 2.4 Example row

```csv
rank,candidate_id,seed,auroc,f1,accuracy,params_backbone,params_head,params_total,latency_ms_per_batch,wall_time_s,delta_auroc_vs_q45a,delta_auroc_vs_q38c,delta_f1_vs_q38c
1,efficientnet_b0,42,0.731450,0.682100,0.763000,5288548,2250,5290798,142.30,187.45,0.011850,0.007528,0.004242
```

---

## 3. Summary Aggregation Row Schema (Phase 2+)

After Phase 2 (3-seed) and Phase 3 (5-seed) runs, a summary row is appended to or written alongside the per-seed leaderboard to aggregate across seeds per candidate. The summary row set uses `SUMMARY_FIELDS` defined in the script.

| Column | Type | Description |
|---|---|---|
| `rank` | int | Candidate rank by `mean_auroc` descending |
| `candidate_id` | str | Backbone identifier (same values as per-seed rows) |
| `row_type` | str | Literal `"summary"` (distinguishes from per-seed rows if mixed) |
| `mean_auroc` | float64 | Mean AUROC across all seeds for this candidate |
| `std_auroc` | float64 | Standard deviation of AUROC across seeds |
| `ci95_lo_auroc` | float64 | 95% confidence interval lower bound for AUROC |
| `ci95_hi_auroc` | float64 | 95% confidence interval upper bound for AUROC |
| `mean_f1` | float64 | Mean F1 across all seeds |
| `std_f1` | float64 | Standard deviation of F1 across seeds |
| `ci95_lo_f1` | float64 | 95% confidence interval lower bound for F1 |
| `ci95_hi_f1` | float64 | 95% confidence interval upper bound for F1 |
| `mean_accuracy` | float64 | Mean accuracy across all seeds |
| `delta_mean_auroc_vs_q45a` | float64 | `mean_auroc − 0.7196` |
| `delta_mean_auroc_vs_q38c` | float64 | `mean_auroc − 0.723922` |
| `delta_mean_f1_vs_q38c` | float64 | `mean_f1 − 0.677858` |
| `seeds_auroc_beat_q45a` | int | Count of seeds where `auroc > 0.7196` |
| `seeds_auroc_beat_q38c` | int | Count of seeds where `auroc > 0.723922` |
| `decision` | str | `WINNER` if `mean_auroc > 0.7196` and `seeds_auroc_beat_q45a >= 2`, else `NEGATIVE` |

**Decision rule (from config `decision_rule` block):**
- `WINNER`: candidate `mean_auroc > 0.7196` (Q45A baseline) AND beats baseline in `≥ 2/3` seeds
- `NEGATIVE`: no candidate meets both conditions
- `EXTENDED_TRIGGER`: candidate is within `0.005` AUROC of the ceiling → Phase 3 (5-seed) run triggered

---

## 4. Results JSON

**Path:** `experiments/results/q46b_feature_extractor_benchmark.json`

The JSON file consolidates benchmark metadata, all per-seed rows, and per-candidate summaries into a single machine-readable record. Declared schema:

```json
{
  "slice_id": "Q46",
  "phase": "smoke | full | extended",
  "run_timestamp": "<ISO 8601 UTC>",
  "git_commit": "<SHA>",
  "config_path": "configs/q46_feature_extractor_benchmark.yaml",
  "baselines": {
    "q38c_auroc": 0.723922,
    "q38c_f1": 0.677858,
    "q45a_mean_auroc": 0.7196,
    "q45a_mean_f1": 0.6360
  },
  "training": {
    "epochs": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "loss": "BCEWithLogitsLoss",
    "optimizer": "Adam",
    "batch_size": 8
  },
  "preprocessing": {
    "clahe_clip": 3.0,
    "clahe_tile": [4, 4],
    "image_size": 224
  },
  "candidates": [
    {
      "candidate_id": "baseline | efficientnet_b0 | mobilenetv3_small | mobilenetv3_large | convnext_tiny",
      "backbone_name": "<torchvision model name or c006_d040_frozen_resnet>",
      "approx_params": "<int>",
      "feature_dim": "<int>",
      "requires_projection": "<bool>"
    }
  ],
  "seeds": ["<list of ints>"],
  "rows": [
    {
      "rank": "<int>",
      "candidate_id": "<str>",
      "seed": "<int>",
      "auroc": "<float>",
      "f1": "<float>",
      "accuracy": "<float>",
      "params_backbone": "<int>",
      "params_head": "<int>",
      "params_total": "<int>",
      "latency_ms_per_batch": "<float>",
      "wall_time_s": "<float>",
      "delta_auroc_vs_q45a": "<float>",
      "delta_auroc_vs_q38c": "<float>",
      "delta_f1_vs_q38c": "<float>"
    }
  ],
  "summary": [
    {
      "rank": "<int>",
      "candidate_id": "<str>",
      "row_type": "summary",
      "mean_auroc": "<float>",
      "std_auroc": "<float>",
      "ci95_lo_auroc": "<float>",
      "ci95_hi_auroc": "<float>",
      "mean_f1": "<float>",
      "std_f1": "<float>",
      "ci95_lo_f1": "<float>",
      "ci95_hi_f1": "<float>",
      "mean_accuracy": "<float>",
      "delta_mean_auroc_vs_q45a": "<float>",
      "delta_mean_auroc_vs_q38c": "<float>",
      "delta_mean_f1_vs_q38c": "<float>",
      "seeds_auroc_beat_q45a": "<int>",
      "seeds_auroc_beat_q38c": "<int>",
      "decision": "WINNER | NEGATIVE | EXTENDED_TRIGGER"
    }
  ],
  "verdict": "WINNER | NEGATIVE | EXTENDED_TRIGGER",
  "winner_candidate_id": "<str or null>",
  "wall_time_total_s": "<float>"
}
```

> **Note:** The JSON output path is declared in the script and config (`RESULTS_JSON_PATH`) but the write call is not yet implemented in the `main()` function as of the Q46C scaffold. The CSV leaderboard is the authoritative output of the current implementation. The JSON schema above is the declared contract for the post-execution write step.

---

## 5. Candidate Reference

| `candidate_id` | Backbone | Backbone Params | Feature Dim | Projection Layer |
|---|---|---|---|---|
| `baseline` | C006-D040 frozen ResNet | ~11,000,000 | 512 | No |
| `efficientnet_b0` | EfficientNet-B0 (IMAGENET1K_V1) | 5,288,548 | 1280 | Yes (linear → head input dim) |
| `mobilenetv3_small` | MobileNetV3-Small (IMAGENET1K_V1) | 2,542,856 | 576 | Yes |
| `mobilenetv3_large` | MobileNetV3-Large (IMAGENET1K_V1) | 5,483,032 | 960 | Yes |
| `convnext_tiny` | ConvNeXt-Tiny (IMAGENET1K_V1) | 28,589,128 | 768 | Yes |

Head: `q34a_trial_004` — 2,250 parameters. Backbone weights are frozen in all Q46B runs.

---

## 6. Reference Baselines

| Metric | Value | Source |
|---|---|---|
| `Q45A_MEAN_AUROC` | 0.7196 | Q45A 3-seed mean — primary decision threshold |
| `Q45A_MEAN_F1` | 0.6360 | Q45A 3-seed mean |
| `Q38C_BEST_AUROC` | 0.723922 | Q38C single-run best (CLAHE clip=3.0 tile=4×4) |
| `Q38C_BEST_F1` | 0.677858 | Q38C single-run best |

---

## 7. Python Load Snippets

### 7.1 Load per-seed leaderboard CSV

```python
import pandas as pd

# Phase 1 (smoke) — single seed
smoke_lb = pd.read_csv("experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv")

# Phase 2 (full) — 3 seeds
full_lb = pd.read_csv("experiments/leaderboards/q46b_extractor_full_leaderboard.csv")

# Verify dtypes
print(full_lb.dtypes)
# rank                    int64
# candidate_id           object
# seed                    int64
# auroc                 float64
# f1                    float64
# accuracy              float64
# params_backbone         int64
# params_head             int64
# params_total            int64
# latency_ms_per_batch  float64
# wall_time_s           float64
# delta_auroc_vs_q45a   float64
# delta_auroc_vs_q38c   float64
# delta_f1_vs_q38c      float64
```

### 7.2 Compute per-candidate summary (Phase 2+)

```python
import pandas as pd
import numpy as np
from scipy import stats

Q45A_MEAN_AUROC = 0.7196
Q38C_BEST_AUROC = 0.723922
Q38C_BEST_F1    = 0.677858

df = pd.read_csv("experiments/leaderboards/q46b_extractor_full_leaderboard.csv")

def summarize(group):
    n = len(group)
    auroc_vals = group["auroc"].values
    f1_vals = group["f1"].values
    se_auroc = stats.sem(auroc_vals)
    se_f1 = stats.sem(f1_vals)
    t = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    mean_auroc = auroc_vals.mean()
    beats_q45a = int((auroc_vals > Q45A_MEAN_AUROC).sum())
    decision = (
        "WINNER"
        if mean_auroc > Q45A_MEAN_AUROC and beats_q45a >= 2
        else "NEGATIVE"
    )
    return pd.Series({
        "mean_auroc":               round(mean_auroc, 6),
        "std_auroc":                round(auroc_vals.std(ddof=1), 6),
        "ci95_lo_auroc":            round(mean_auroc - t * se_auroc, 6),
        "ci95_hi_auroc":            round(mean_auroc + t * se_auroc, 6),
        "mean_f1":                  round(f1_vals.mean(), 6),
        "std_f1":                   round(f1_vals.std(ddof=1), 6),
        "ci95_lo_f1":               round(f1_vals.mean() - t * se_f1, 6),
        "ci95_hi_f1":               round(f1_vals.mean() + t * se_f1, 6),
        "mean_accuracy":            round(group["accuracy"].mean(), 6),
        "delta_mean_auroc_vs_q45a": round(mean_auroc - Q45A_MEAN_AUROC, 6),
        "delta_mean_auroc_vs_q38c": round(mean_auroc - Q38C_BEST_AUROC, 6),
        "delta_mean_f1_vs_q38c":    round(f1_vals.mean() - Q38C_BEST_F1, 6),
        "seeds_auroc_beat_q45a":    beats_q45a,
        "seeds_auroc_beat_q38c":    int((auroc_vals > Q38C_BEST_AUROC).sum()),
        "decision":                 decision,
    })

summary = (
    df.groupby("candidate_id")
      .apply(summarize)
      .reset_index()
      .sort_values("mean_auroc", ascending=False)
      .reset_index(drop=True)
)
summary.insert(0, "rank", range(1, len(summary) + 1))
summary.insert(2, "row_type", "summary")

print(summary[["rank", "candidate_id", "mean_auroc", "std_auroc", "decision"]])
```

### 7.3 Load results JSON

```python
import json

with open("experiments/results/q46b_feature_extractor_benchmark.json") as f:
    results = json.load(f)

verdict = results["verdict"]
winner  = results["winner_candidate_id"]  # None if NEGATIVE

print(f"Verdict: {verdict}  Winner: {winner}")

# Access per-seed rows
rows = results["rows"]
# Access summary
summary = results["summary"]
```

### 7.4 Quick winner check from full leaderboard CSV

```python
import pandas as pd

Q45A_MEAN_AUROC = 0.7196

df = pd.read_csv("experiments/leaderboards/q46b_extractor_full_leaderboard.csv")
mean_by_candidate = df.groupby("candidate_id")["auroc"].mean()
winners = mean_by_candidate[mean_by_candidate > Q45A_MEAN_AUROC]

if winners.empty:
    print("NEGATIVE — no candidate clears Q45A baseline")
else:
    print("WINNER(s):")
    print(winners.sort_values(ascending=False))
```

---

## 8. Phase-to-File Mapping

| Phase | CLI flag | Seeds | Output CSV |
|---|---|---|---|
| Phase 1 — Smoke | `--smoke` | `[42]` | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` |
| Phase 2 — Full | `--full` | `[42, 7, 123]` | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` |
| Phase 3 — Extended | *(manual, winner only)* | `[42, 7, 123, 999, 2025]` | `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv` |

Custom seed/candidate subsets (via `--seed` / `--candidate` flags) write to the same path for the active phase, or to `--output-dir` if overridden.

---

## 9. Implementation Notes

- The script writes the CSV leaderboard in one pass at the end of all runs for a given invocation. If the wall-time cap fires mid-run, only completed rows are written.
- `latency_ms_per_batch` is derived as `wall_time_s / max(len(train_loader), 1) * 1000`. This measures total training time normalized to batches and is an approximation; it includes DataLoader overhead and is not a pure inference latency measurement.
- `params_backbone` is the exact count of all parameters in the backbone module (frozen + unfrozen), not just trainable ones. Backbone weights are always frozen in Q46B runs.
- The `rank` column reflects order within the file at time of write. If multiple phase runs are concatenated manually, re-sort by `auroc` and re-rank before analysis.
- The JSON output path (`experiments/results/q46b_feature_extractor_benchmark.json`) is declared in the script constant `RESULTS_JSON_PATH` but the write step is not yet implemented in the Q46C scaffold. This spec documents the intended schema for when that write step is added.

---

```
Schema version: 1.0
Defined: Q46F (output schema documentation)
Script source: scripts/run_q46b_feature_extractor_benchmark.py (Q46C scaffold)
Config source: configs/q46_feature_extractor_benchmark.yaml
```
