"""
scripts/run_q34a_classical_nas_pilot.py

Q34A Classical NAS Pilot Orchestrator.

Runs a small random-search NAS pilot over the classical CNN search space defined
in Q32 (docs/architecture/q32_classical_nas_search_space.md).

This is the FIRST execution step in the three-part Q34 NAS pipeline:
  Q34A (classical) → Q34B (DV quantum) → Q34C (CV quantum)

Intentionally minimal:
  - 5 trials, 3–5 epochs each
  - Inline random search (no pymoo, Optuna, or external NAS library)
  - Sequential execution (no parallelism, no AWS, no Ray)
  - Uses existing Q31 runner (scripts/run_experiment.py) for each trial
  - Pareto computed inline over (AUROC, F1, params)
  - Pass threshold: ≥ 3/5 trials completed without error

CLI:
    python3 scripts/run_q34a_classical_nas_pilot.py --trials 5 --seed 42

Outputs:
    configs/experiments/q34a_classical_nas/<trial_id>.yaml   — per-trial frozen configs
    experiments/results/q34a_classical_nas_pilot_summary.json — summary with all trial results
    experiments/leaderboards/q34a_classical_pareto.csv        — Pareto frontier CSV

Exit code:
    0  if ≥ 3 trials completed
    1  if < 3 trials completed (pilot FAIL)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR     = os.path.join(PROJECT_ROOT, "configs", "experiments", "q34a_classical_nas")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "experiments", "results")
LEADERBOARD_DIR = os.path.join(PROJECT_ROOT, "experiments", "leaderboards")
SUMMARY_PATH    = os.path.join(RESULTS_DIR, "q34a_classical_nas_pilot_summary.json")
PARETO_CSV_PATH = os.path.join(LEADERBOARD_DIR, "q34a_classical_pareto.csv")

RUNNER_SCRIPT   = os.path.join(PROJECT_ROOT, "scripts", "run_experiment.py")
CANDIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_q34a_classical_candidate.py")

DATASET_ROOT    = "data/processed/vindr_binary_roi_224"
CHECKPOINT_PATH = "checkpoints/c006_d040_classical_anchor.pt"

# ── Search space (Q32 classical CNN space, inline — no external NAS library) ──
#
# Matches the dimensions defined in:
#   docs/architecture/q32_classical_nas_search_space.md
#
# backbone_family: Both options map to linear projection heads in this pilot
#   because the frozen C006-D040 backbone outputs flat 128-dim vectors. True
#   convolutional blocks require spatial feature maps; this pilot limitation is
#   documented in reports/q34a_classical_nas_pilot_mvp.md Section 3.
#
# channel_width: List of [hidden_dim_1, hidden_dim_2] for shallow/medium depth.
# normalization: Only "BatchNorm" in pilot (applied as LayerNorm for 1D features).
# pooling: "adaptive_average" is a no-op — backbone already pooled.

SEARCH_SPACE: dict = {
    "backbone_family": ["standard_cnn", "depthwise_separable_cnn"],
    "channel_width":   [[16, 32], [32, 64], [64, 128]],
    "depth":           ["shallow", "medium"],
    "compression_dim": [4, 8, 16],
    "activation":      ["ReLU", "GELU"],
    "normalization":   ["BatchNorm"],
    "dropout":         [0.0, 0.2, 0.3],
    "pooling":         ["adaptive_average"],
    "learning_rate":   [0.001, 0.0005],
    "weight_decay":    [0.0, 0.0001],
}

# Hard pilot constraints
EPOCHS_PER_TRIAL = 4          # within [3, 5] ceiling
BATCH_SIZE       = 4
PASS_THRESHOLD   = 3          # minimum completed trials for PASS verdict


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q34A Classical NAS Pilot — random search orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/run_q34a_classical_nas_pilot.py --trials 5 --seed 42\n"
        ),
    )
    p.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of NAS trials to run (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for config sampling (default: 42)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_PER_TRIAL,
        help=f"Epochs per trial (default: {EPOCHS_PER_TRIAL}, max: 5)",
    )
    return p.parse_args()


# ── Config sampling ────────────────────────────────────────────────────────────

def sample_config(trial_seed: int) -> dict:
    """
    Sample one config from SEARCH_SPACE using random.choice per dimension.

    The trial seed is set before sampling to ensure each trial's config
    is deterministic given the trial index. The global pilot seed (set
    in main) determines the sequence of per-trial seeds.
    """
    sampled: dict = {}
    for key, options in SEARCH_SPACE.items():
        sampled[key] = random.choice(options)
    return sampled


def build_trial_yaml(trial_id: str, trial_idx: int, sampled: dict,
                     pilot_seed: int, epochs: int) -> dict:
    """
    Build a runner-compatible YAML config dict from a sampled config.

    The trial seed is pilot_seed + trial_idx to ensure deterministic but
    distinct seeds across trials. The YAML includes both the runner-required
    fields and the model-specific search dimensions used by
    train_q34a_classical_candidate.py.
    """
    trial_seed = pilot_seed + trial_idx

    config = {
        "experiment": {
            "phase":       "q34a_classical_nas",
            "description": (
                f"Q34A classical NAS pilot trial {trial_idx + 1} — "
                f"backbone_family={sampled['backbone_family']} "
                f"depth={sampled['depth']} "
                f"channels={sampled['channel_width']} "
                f"compress={sampled['compression_dim']} "
                f"act={sampled['activation']} "
                f"dropout={sampled['dropout']} "
                f"lr={sampled['learning_rate']} "
                f"wd={sampled['weight_decay']}"
            ),
        },
        "dataset": {
            "name": "vindr_binary_roi_224",
            "root": DATASET_ROOT,
        },
        "model": {
            "architecture":   "classical_nas_candidate",
            "backbone":       "c006_d040_classical_anchor",
            "backbone_frozen": True,
            "checkpoint_path": CHECKPOINT_PATH,
            # Search space dimensions (consumed by train_q34a_classical_candidate.py)
            "backbone_family": sampled["backbone_family"],
            "channel_width":   sampled["channel_width"],
            "depth":           sampled["depth"],
            "compression_dim": sampled["compression_dim"],
            "activation":      sampled["activation"],
            "normalization":   sampled["normalization"],
            "dropout":         sampled["dropout"],
            "pooling":         sampled["pooling"],
        },
        "training": {
            "epochs":        epochs,
            "batch_size":    BATCH_SIZE,
            "optimizer":     "AdamW",
            "learning_rate": sampled["learning_rate"],
            "weight_decay":  sampled["weight_decay"],
            "loss":          "CrossEntropyLoss",
        },
        "reproducibility": {
            "seed": trial_seed,
        },
        "command": {
            "executable": "python3",
            "args": [
                "scripts/train_q34a_classical_candidate.py",
                "--config",
                os.path.join("configs", "experiments", "q34a_classical_nas", f"{trial_id}.yaml"),
            ],
        },
    }
    return config


# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_trial_metrics(stdout: str) -> dict:
    """
    Parse Q34A_TRIAL_* lines from captured subprocess + runner stdout.

    The runner tees train_q34a_classical_candidate.py's stdout to its own
    stdout, so Q34A_TRIAL_* lines appear in the combined output of
    'python3 scripts/run_experiment.py --config ...'.

    Returns a dict with keys:
        auroc, f1, accuracy, params, latency_ms, status
    All values are None if the line was not found or could not be parsed.
    """
    metrics: dict = {
        "auroc":      None,
        "f1":         None,
        "accuracy":   None,
        "params":     None,
        "latency_ms": None,
        "status":     None,
    }
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Q34A_TRIAL_AUROC:"):
            try:
                metrics["auroc"] = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("Q34A_TRIAL_F1:"):
            try:
                metrics["f1"] = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("Q34A_TRIAL_ACCURACY:"):
            try:
                metrics["accuracy"] = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("Q34A_TRIAL_PARAMS:"):
            try:
                metrics["params"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("Q34A_TRIAL_LATENCY_MS:"):
            try:
                metrics["latency_ms"] = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("Q34A_TRIAL_STATUS:"):
            metrics["status"] = stripped.split(":", 1)[1].strip()
    return metrics


def parse_result_json_path(stdout: str) -> str | None:
    """
    Extract the result JSON path from runner stdout.
    Looks for: [RUNNER] result JSON:      <path>
    """
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[RUNNER] result JSON:"):
            path = stripped.split(":", 1)[1].strip()
            return path
    return None


# ── Pareto computation ─────────────────────────────────────────────────────────

def compute_pareto(trials: list[dict]) -> list[bool]:
    """
    Compute Pareto non-dominance over (AUROC ↑, F1 ↑, params ↓).

    Trials with None metrics are excluded from Pareto (marked False).
    Trial A dominates trial B if:
      A.auroc >= B.auroc AND A.f1 >= B.f1 AND A.params <= B.params
      AND at least one inequality is strict.

    Returns a list of booleans (same length as trials):
      True  = Pareto non-dominated
      False = dominated or ineligible (None metrics)
    """
    n = len(trials)
    eligible = []
    for i, t in enumerate(trials):
        m = t.get("metrics", {})
        if (m.get("auroc") is not None
                and m.get("f1") is not None
                and m.get("params") is not None
                and not (isinstance(m.get("auroc"), float) and np.isnan(m["auroc"]))
                and not (isinstance(m.get("f1"), float) and np.isnan(m["f1"]))):
            eligible.append(i)

    is_pareto = [False] * n
    for i in eligible:
        mi = trials[i]["metrics"]
        dominated = False
        for j in eligible:
            if i == j:
                continue
            mj = trials[j]["metrics"]
            # j dominates i?
            j_ge_auroc = mj["auroc"] >= mi["auroc"]
            j_ge_f1    = mj["f1"]    >= mi["f1"]
            j_le_params = mj["params"] <= mi["params"]
            j_strict    = (mj["auroc"] > mi["auroc"]
                           or mj["f1"] > mi["f1"]
                           or mj["params"] < mi["params"])
            if j_ge_auroc and j_ge_f1 and j_le_params and j_strict:
                dominated = True
                break
        if not dominated:
            is_pareto[i] = True

    return is_pareto


# ── Pareto CSV writer ──────────────────────────────────────────────────────────

PARETO_CSV_FIELDS = [
    "trial_id",
    "trial_idx",
    "is_pareto",
    "auroc",
    "f1",
    "accuracy",
    "params",
    "latency_ms",
    "status",
    "return_code",
    "experiment_id",
    "backbone_family",
    "channel_width",
    "depth",
    "compression_dim",
    "activation",
    "normalization",
    "dropout",
    "pooling",
    "learning_rate",
    "weight_decay",
    "epochs",
    "seed",
    "config_path",
    "result_json_path",
]


def write_pareto_csv(trials: list[dict], is_pareto: list[bool]) -> None:
    """Write Pareto frontier CSV to PARETO_CSV_PATH."""
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    with open(PARETO_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARETO_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, (trial, pareto) in enumerate(zip(trials, is_pareto)):
            m  = trial.get("metrics", {})
            sc = trial.get("sampled_config", {})
            row = {
                "trial_id":        trial.get("trial_id", ""),
                "trial_idx":       trial.get("trial_idx", i),
                "is_pareto":       pareto,
                "auroc":           m.get("auroc", ""),
                "f1":              m.get("f1", ""),
                "accuracy":        m.get("accuracy", ""),
                "params":          m.get("params", ""),
                "latency_ms":      m.get("latency_ms", ""),
                "status":          m.get("status", ""),
                "return_code":     trial.get("return_code", ""),
                "experiment_id":   trial.get("experiment_id", ""),
                "backbone_family": sc.get("backbone_family", ""),
                "channel_width":   str(sc.get("channel_width", "")),
                "depth":           sc.get("depth", ""),
                "compression_dim": sc.get("compression_dim", ""),
                "activation":      sc.get("activation", ""),
                "normalization":   sc.get("normalization", ""),
                "dropout":         sc.get("dropout", ""),
                "pooling":         sc.get("pooling", ""),
                "learning_rate":   sc.get("learning_rate", ""),
                "weight_decay":    sc.get("weight_decay", ""),
                "epochs":          trial.get("epochs", ""),
                "seed":            trial.get("seed", ""),
                "config_path":     trial.get("config_path", ""),
                "result_json_path": trial.get("result_json_path", ""),
            }
            writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.epochs > 5:
        print(f"ERROR: --epochs {args.epochs} exceeds Q34A ceiling of 5", flush=True)
        sys.exit(1)
    if args.trials < 1:
        print(f"ERROR: --trials must be >= 1", flush=True)
        sys.exit(1)

    # ── Set pilot-level random seeds ───────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Create output directories ──────────────────────────────────────────────
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print("Q34A Classical NAS Pilot", flush=True)
    print("=" * 60, flush=True)
    print(f"  trials:        {args.trials}", flush=True)
    print(f"  epochs/trial:  {args.epochs}", flush=True)
    print(f"  seed:          {args.seed}", flush=True)
    print(f"  pass_threshold:{PASS_THRESHOLD} / {args.trials}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # ── Sample configs ─────────────────────────────────────────────────────────
    # Sample all configs before execution so the random state is deterministic
    # regardless of trial wall-time or OS scheduling.
    sampled_configs: list[dict] = []
    trial_ids: list[str] = []

    for i in range(args.trials):
        trial_id = f"q34a_trial_{i + 1:03d}"
        sc = sample_config(trial_seed=args.seed + i)
        sampled_configs.append(sc)
        trial_ids.append(trial_id)

    print(f"[PILOT] Sampled {args.trials} configs:", flush=True)
    for i, (tid, sc) in enumerate(zip(trial_ids, sampled_configs)):
        print(
            f"  [{i+1}] {tid}: "
            f"family={sc['backbone_family']} depth={sc['depth']} "
            f"channels={sc['channel_width']} compress={sc['compression_dim']} "
            f"act={sc['activation']} dropout={sc['dropout']} "
            f"lr={sc['learning_rate']} wd={sc['weight_decay']}",
            flush=True,
        )
    print(flush=True)

    # ── Execute trials ─────────────────────────────────────────────────────────
    all_trials: list[dict] = []
    pilot_start = time.monotonic()

    for i, (trial_id, sampled) in enumerate(zip(trial_ids, sampled_configs)):
        print("-" * 60, flush=True)
        print(f"[PILOT] Starting trial {i + 1}/{args.trials}: {trial_id}", flush=True)
        print("-" * 60, flush=True)

        trial_seed = args.seed + i
        config_yaml = build_trial_yaml(
            trial_id=trial_id,
            trial_idx=i,
            sampled=sampled,
            pilot_seed=args.seed,
            epochs=args.epochs,
        )

        # Write YAML config
        config_path = os.path.join(CONFIGS_DIR, f"{trial_id}.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"[PILOT] Config written: {config_path}", flush=True)

        # Execute via Q31 runner
        cmd = [sys.executable, RUNNER_SCRIPT, "--config", config_path]
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            stdout_combined = proc.stdout or ""
        except Exception as exc:
            print(f"[PILOT] ERROR running trial {trial_id}: {exc}", flush=True)
            stdout_combined = ""
            proc = type("FakeProc", (), {"returncode": -1})()

        # Echo captured output to our own stdout
        if stdout_combined:
            sys.stdout.write(stdout_combined)
            sys.stdout.flush()

        t_trial = time.monotonic() - t0

        # Parse metrics from tee'd output
        metrics   = parse_trial_metrics(stdout_combined)
        rj_path   = parse_result_json_path(stdout_combined)
        exp_id    = None

        # Try to extract experiment_id from result JSON
        if rj_path and os.path.isfile(rj_path):
            try:
                with open(rj_path, "r", encoding="utf-8") as jf:
                    rj = json.load(jf)
                exp_id = rj.get("experiment_id")
                # Fill in status from result JSON if not from stdout
                if metrics["status"] is None:
                    metrics["status"] = rj.get("status")
            except Exception:
                pass

        trial_record = {
            "trial_id":      trial_id,
            "trial_idx":     i,
            "sampled_config": sampled,
            "config_path":   config_path,
            "result_json_path": rj_path or "",
            "experiment_id": exp_id or "",
            "return_code":   proc.returncode,
            "wall_time_s":   round(t_trial, 2),
            "epochs":        args.epochs,
            "seed":          trial_seed,
            "metrics":       metrics,
        }
        all_trials.append(trial_record)

        completed = metrics.get("status") == "completed" and proc.returncode == 0
        status_str = "COMPLETED" if completed else f"FAILED (rc={proc.returncode})"
        print(
            f"\n[PILOT] Trial {i + 1}/{args.trials} {status_str} | "
            f"auroc={metrics['auroc']} f1={metrics['f1']} "
            f"params={metrics['params']} lat={metrics['latency_ms']}ms | "
            f"wall={t_trial:.1f}s",
            flush=True,
        )
        print(flush=True)

    # ── Tally completed trials ─────────────────────────────────────────────────
    pilot_elapsed = time.monotonic() - pilot_start
    n_completed = sum(
        1 for t in all_trials
        if t["metrics"].get("status") == "completed" and t["return_code"] == 0
    )

    # ── Compute Pareto frontier ────────────────────────────────────────────────
    is_pareto = compute_pareto(all_trials)
    pareto_indices = [i for i, p in enumerate(is_pareto) if p]

    # ── Write pilot summary JSON ───────────────────────────────────────────────
    verdict = "PASS" if n_completed >= PASS_THRESHOLD else "FAIL"
    summary = {
        "pilot":         "Q34A_CLASSICAL_NAS_PILOT",
        "verdict":       verdict,
        "pilot_seed":    args.seed,
        "n_trials":      args.trials,
        "n_completed":   n_completed,
        "pass_threshold": PASS_THRESHOLD,
        "epochs_per_trial": args.epochs,
        "wall_time_s":   round(pilot_elapsed, 2),
        "search_space":  SEARCH_SPACE,
        "trials":        all_trials,
        "pareto_indices": pareto_indices,
        "pareto_csv":    PARETO_CSV_PATH,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Write Pareto CSV ───────────────────────────────────────────────────────
    write_pareto_csv(all_trials, is_pareto)

    # ── Final report ───────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print(f"Q34A CLASSICAL NAS PILOT RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Trials run:      {args.trials}", flush=True)
    print(f"  Trials completed:{n_completed}", flush=True)
    print(f"  Pass threshold:  {PASS_THRESHOLD}", flush=True)
    print(f"  Wall time:       {pilot_elapsed:.1f}s", flush=True)
    print(flush=True)

    print("  Trial results:", flush=True)
    for t in all_trials:
        m = t["metrics"]
        s = "✓" if (m.get("status") == "completed" and t["return_code"] == 0) else "✗"
        print(
            f"    {s} {t['trial_id']}: "
            f"auroc={m.get('auroc')} f1={m.get('f1')} "
            f"params={m.get('params')} lat={m.get('latency_ms')}ms",
            flush=True,
        )
    print(flush=True)

    if pareto_indices:
        print(f"  Preliminary Pareto set ({len(pareto_indices)} trials):", flush=True)
        for i in pareto_indices:
            t = all_trials[i]
            m = t["metrics"]
            print(
                f"    {t['trial_id']}: "
                f"auroc={m.get('auroc')} f1={m.get('f1')} params={m.get('params')}",
                flush=True,
            )
    else:
        print("  Preliminary Pareto set: empty (no eligible trials)", flush=True)
    print(flush=True)

    print(f"  Summary JSON:    {SUMMARY_PATH}", flush=True)
    print(f"  Pareto CSV:      {PARETO_CSV_PATH}", flush=True)
    print(flush=True)

    # Machine-readable verdict line (parsed by Step 7 report and CI checks)
    print(f"Q34A_CLASSICAL_NAS_PILOT: {verdict}", flush=True)

    if verdict == "PASS":
        sys.exit(0)
    else:
        print(
            f"[PILOT] FAIL: only {n_completed}/{args.trials} trials completed "
            f"(threshold: {PASS_THRESHOLD})",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
