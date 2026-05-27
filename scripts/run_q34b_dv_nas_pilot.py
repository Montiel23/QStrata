"""
scripts/run_q34b_dv_nas_pilot.py

Q34B DV NAS Pilot Orchestrator.

Runs a small random-search NAS pilot over the DV quantum head search space
defined in Q33A (docs/architecture/q33a_dv_quantum_nas_search_space.md).

This is the SECOND execution step in the three-part Q34 NAS pipeline:
  Q34A (classical) → Q34B (DV quantum) → Q34C (CV quantum)

Intentionally minimal:
  - 5 trials, 4 epochs each
  - Inline random search (no pymoo, Optuna, or external NAS library)
  - Sequential execution (no parallelism, no AWS, no Ray)
  - Uses existing Q31 runner (scripts/run_experiment.py) for each trial
  - Pareto computed inline over (AUROC, F1, params; latency if available)
  - Pass threshold: ≥ 3/5 trials completed
  - Hard exclusions: params > 100K, NaN/inf, severe gradient collapse

Note on DV pilot search space vs Q33A full space:
  The Q34B pilot constrains the Q33A search space to simulator-tractable
  options for a 5-trial feasibility pilot:
  - qubit_count: [2, 4] (Q33A supports up to 8; 6 and 8 deferred)
  - ansatz_depth: [1, 2] (Q33A supports up to 4; 3 and 4 deferred)
  - rotation_family: all three options listed but mapped to RX+RY+RZ (see substitutions)
  - entanglement_topology: [linear, circular] (full deferred; nearest_neighbor deferred)
  - Other dimensions: fixed in medical_ansatz (see train_q34b_dv_candidate.py)

CLI:
    python3 scripts/run_q34b_dv_nas_pilot.py --trials 5 --seed 42

Outputs:
    configs/experiments/q34b_dv_nas/<trial_id>.yaml     — per-trial frozen configs
    experiments/results/q34b_dv_nas_pilot_summary.json  — pilot summary
    experiments/leaderboards/q34b_dv_pareto.csv         — Pareto frontier CSV

Exit code:
    0  if ≥ 3 trials completed
    1  if < 3 trials completed (pilot FAIL)
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CONFIGS_DIR     = os.path.join(PROJECT_ROOT, "configs", "experiments", "q34b_dv_nas")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "experiments", "results")
LEADERBOARD_DIR = os.path.join(PROJECT_ROOT, "experiments", "leaderboards")
SUMMARY_PATH    = os.path.join(RESULTS_DIR,     "q34b_dv_nas_pilot_summary.json")
PARETO_CSV_PATH = os.path.join(LEADERBOARD_DIR, "q34b_dv_pareto.csv")

RUNNER_SCRIPT    = os.path.join(PROJECT_ROOT, "scripts", "run_experiment.py")
CANDIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_q34b_dv_candidate.py")

DATASET_ROOT     = "data/processed/vindr_binary_roi_224"
CHECKPOINT_PATH  = "checkpoints/c006_d040_classical_anchor.pt"

# ── Q34B DV search space ────────────────────────────────────────────────────────
#
# Constrained pilot subset of the Q33A full DV search space.
# See docs/architecture/q33a_dv_quantum_nas_search_space.md for full space.
#
# IMPLEMENTED (varies between trials):
#   qubit_count:   [2, 4]          (Q33A allows up to 8; 6,8 deferred for pilot)
#   ansatz_depth:  [1, 2]          (Q33A allows up to 4; 3,4 deferred for pilot)
#   learning_rate: [0.001, 0.0005]
#   weight_decay:  [0.0, 0.0001]
#
# PILOT SUBSTITUTIONS (listed for completeness; value does not affect architecture):
#   rotation_family:           all map to RX+RY+RZ (medical_ansatz fixed)
#   entanglement_topology:     all map to circular (medical_ansatz fixed)
#   encoding_strategy:         angle_encoding only (medical_ansatz fixed)
#   reuploading_frequency:     all map to every_layer (medical_ansatz fixed)
#   measurement_strategy:      pauli_z_per_qubit only (full prob vector → linear readout)
#   compression_dim:           maps to n_qubits (no independent bottleneck)
#   classical_projection_layer: linear only

SEARCH_SPACE: dict = {
    "qubit_count":               [2, 4],
    "ansatz_depth":              [1, 2],
    "rotation_family":           ["RY", "RX+RY", "RX+RY+RZ"],
    "entanglement_topology":     ["linear", "circular"],
    "encoding_strategy":         ["angle_encoding"],
    "reuploading_frequency":     ["none", "every_layer"],
    "measurement_strategy":      ["pauli_z_per_qubit"],
    "compression_dim":           [2, 4],
    "classical_projection_layer": ["linear"],
    "learning_rate":             [0.001, 0.0005],
    "weight_decay":              [0.0, 0.0001],
}

SEARCH_SPACE_VERSION = "q34b_v1"

# Hard pilot constraints
EPOCHS_PER_TRIAL    = 4
BATCH_SIZE          = 4
PASS_THRESHOLD      = 3
MAX_TRAINABLE_PARAMS = 100_000


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q34B DV NAS Pilot — random search orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/run_q34b_dv_nas_pilot.py --trials 5 --seed 42\n"
        ),
    )
    p.add_argument("--trials",      type=int,  default=5)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--epochs",      type=int,  default=EPOCHS_PER_TRIAL)
    p.add_argument("--parallel",    action="store_true", default=False,
                   help="Run trials concurrently via ThreadPoolExecutor")
    p.add_argument("--max-workers", type=int,  default=5,
                   help="Max concurrent workers in parallel mode (default: 5)")
    p.add_argument("--thread-cap",  type=int,  default=0,
                   help=(
                       "Per-subprocess thread cap: sets OMP_NUM_THREADS, MKL_NUM_THREADS, "
                       "and OPENBLAS_NUM_THREADS in each trial subprocess environment. "
                       "0 = no cap (default). Recommended: 2 for 5-worker parallel runs."
                   ))
    return p.parse_args()


# ── Config sampling ────────────────────────────────────────────────────────────

def sample_config() -> dict:
    """Sample one config using random.choice per dimension."""
    return {key: random.choice(options) for key, options in SEARCH_SPACE.items()}


def build_trial_yaml(trial_id: str, trial_idx: int, sampled: dict,
                     pilot_seed: int, epochs: int) -> dict:
    """Build a Q31-compatible YAML config dict from a sampled DV config."""
    trial_seed = pilot_seed + trial_idx
    config_rel = os.path.join(
        "configs", "experiments", "q34b_dv_nas", f"{trial_id}.yaml"
    )
    return {
        "experiment": {
            "phase":       "q34b_dv_nas_pilot",
            "description": (
                f"Q34B DV NAS pilot trial {trial_idx + 1} — "
                f"qubits={sampled['qubit_count']} depth={sampled['ansatz_depth']} "
                f"rot={sampled['rotation_family']} "
                f"entangle={sampled['entanglement_topology']} "
                f"reup={sampled['reuploading_frequency']} "
                f"compress={sampled['compression_dim']} "
                f"lr={sampled['learning_rate']} wd={sampled['weight_decay']}"
            ),
            "trial_id": trial_id,
        },
        "dataset": {
            "name":           "vindr_binary_roi_224",
            "split":          "canonical_42",
            "preprocessing":  "roi_224_grayscale",
            "root":           DATASET_ROOT,
        },
        "model": {
            "architecture":              "dv_quantum_head",
            "backbone":                  "c006_d040_frozen",
            "checkpoint_path":           CHECKPOINT_PATH,
            "qubit_count":               sampled["qubit_count"],
            "ansatz_depth":              sampled["ansatz_depth"],
            "rotation_family":           sampled["rotation_family"],
            "entanglement_topology":     sampled["entanglement_topology"],
            "encoding_strategy":         sampled["encoding_strategy"],
            "reuploading_frequency":     sampled["reuploading_frequency"],
            "measurement_strategy":      sampled["measurement_strategy"],
            "compression_dim":           sampled["compression_dim"],
            "classical_projection_layer": sampled["classical_projection_layer"],
        },
        "training": {
            "epochs":         epochs,
            "batch_size":     BATCH_SIZE,
            "optimizer":      "AdamW",
            "learning_rate":  sampled["learning_rate"],
            "weight_decay":   sampled["weight_decay"],
            "early_stopping": {"enabled": False},
        },
        "metrics": {
            "primary":   "AUROC",
            "secondary": [
                "F1",
                "accuracy",
                "parameter_count",
                "latency_ms_per_sample",
                "gradient_health",
                "nan_inf_status",
            ],
        },
        "reproducibility": {
            "seed": trial_seed,
        },
        "command": {
            "executable": "python3",
            "args": [
                "scripts/train_q34b_dv_candidate.py",
                "--config",
                config_rel,
            ],
        },
    }


# ── Metric parsing ─────────────────────────────────────────────────────────────

def parse_trial_metrics(stdout: str) -> dict:
    """Parse Q34B_TRIAL_* lines from combined runner+subprocess stdout."""
    metrics: dict = {
        "auroc":           None,
        "f1":              None,
        "accuracy":        None,
        "params":          None,
        "latency_ms":      None,
        "gradient_health": None,
        "nan_inf":         None,
        "status":          None,
    }
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Q34B_TRIAL_AUROC:"):
            try:    metrics["auroc"]    = float(s.split(":", 1)[1].strip())
            except  ValueError: pass
        elif s.startswith("Q34B_TRIAL_F1:"):
            try:    metrics["f1"]       = float(s.split(":", 1)[1].strip())
            except  ValueError: pass
        elif s.startswith("Q34B_TRIAL_ACCURACY:"):
            try:    metrics["accuracy"] = float(s.split(":", 1)[1].strip())
            except  ValueError: pass
        elif s.startswith("Q34B_TRIAL_PARAMS:"):
            try:    metrics["params"]   = int(s.split(":", 1)[1].strip())
            except  ValueError: pass
        elif s.startswith("Q34B_TRIAL_LATENCY_MS:"):
            try:    metrics["latency_ms"] = float(s.split(":", 1)[1].strip())
            except  ValueError: pass
        elif s.startswith("Q34B_TRIAL_GRADIENT_HEALTH:"):
            metrics["gradient_health"] = s.split(":", 1)[1].strip()
        elif s.startswith("Q34B_TRIAL_NAN_INF:"):
            metrics["nan_inf"] = s.split(":", 1)[1].strip()
        elif s.startswith("Q34B_TRIAL_STATUS:"):
            metrics["status"] = s.split(":", 1)[1].strip()
    return metrics


def parse_result_json_path(stdout: str) -> str | None:
    """Extract result JSON path from runner stdout."""
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("[RUNNER] result JSON:"):
            return s.split(":", 1)[1].strip()
    return None


def classify_trial_status(metrics: dict, return_code: int) -> str:
    """
    Classify trial into: completed | failed | invalid | unstable

    completed: return_code=0 and required metrics available
    failed:    return_code ≠ 0
    invalid:   params > 100K or NaN/inf = FAIL
    unstable:  gradient_health = FAIL (severe) without actual NaN/inf
    """
    if return_code != 0:
        return "failed"
    if metrics.get("nan_inf") == "FAIL":
        return "invalid"
    if metrics.get("params") and metrics["params"] > MAX_TRAINABLE_PARAMS:
        return "invalid"
    if metrics.get("gradient_health") == "FAIL":
        return "unstable"
    if metrics.get("status") == "completed":
        return "completed"
    return "failed"


# ── Pareto computation ─────────────────────────────────────────────────────────

def compute_pareto(trials: list[dict]) -> list[bool]:
    """
    Compute Pareto non-dominance over (AUROC ↑, F1 ↑, params ↓).
    Latency included if available for all eligible trials.

    Hard exclusions (is_pareto=False regardless):
    - status ≠ completed
    - nan_inf = FAIL
    - params > 100K
    - gradient_health = FAIL (severe instability)

    Trial A dominates B if:
      A.auroc >= B.auroc AND A.f1 >= B.f1 AND A.params <= B.params
      AND strictly better on at least one.
    """
    n = len(trials)

    def eligible(t: dict) -> bool:
        m = t.get("metrics", {})
        if t.get("trial_status", "") != "completed":
            return False
        if m.get("auroc") is None or m.get("f1") is None or m.get("params") is None:
            return False
        if isinstance(m.get("auroc"), float) and np.isnan(m["auroc"]):
            return False
        if isinstance(m.get("f1"), float) and np.isnan(m["f1"]):
            return False
        if m.get("nan_inf") == "FAIL":
            return False
        if m.get("gradient_health") == "FAIL":
            return False
        if m.get("params", 0) > MAX_TRAINABLE_PARAMS:
            return False
        return True

    elig_idx = [i for i, t in enumerate(trials) if eligible(t)]
    is_pareto = [False] * n

    for i in elig_idx:
        mi = trials[i]["metrics"]
        dominated = False
        for j in elig_idx:
            if i == j:
                continue
            mj = trials[j]["metrics"]
            j_ge_auroc  = mj["auroc"]  >= mi["auroc"]
            j_ge_f1     = mj["f1"]     >= mi["f1"]
            j_le_params = mj["params"] <= mi["params"]
            j_strict    = (
                mj["auroc"]  > mi["auroc"]
                or mj["f1"]  > mi["f1"]
                or mj["params"] < mi["params"]
            )
            if j_ge_auroc and j_ge_f1 and j_le_params and j_strict:
                dominated = True
                break
        if not dominated:
            is_pareto[i] = True

    return is_pareto


# ── Pareto CSV ─────────────────────────────────────────────────────────────────

PARETO_FIELDS = [
    "trial_id", "trial_idx", "is_pareto", "status",
    "auroc", "f1", "accuracy", "parameter_count", "latency_ms_per_sample",
    "gradient_health", "nan_inf_status",
    "qubit_count", "ansatz_depth", "rotation_family", "entanglement_topology",
    "reuploading_frequency", "compression_dim",
    "learning_rate", "weight_decay", "epochs", "seed",
    "experiment_id", "config_path", "result_json_path",
]


def write_pareto_csv(trials: list[dict], is_pareto: list[bool]) -> None:
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    with open(PARETO_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARETO_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, (t, pareto) in enumerate(zip(trials, is_pareto)):
            m  = t.get("metrics", {})
            sc = t.get("sampled_config", {})
            writer.writerow({
                "trial_id":             t.get("trial_id", ""),
                "trial_idx":            t.get("trial_idx", i),
                "is_pareto":            pareto,
                "status":               t.get("trial_status", ""),
                "auroc":                m.get("auroc", ""),
                "f1":                   m.get("f1", ""),
                "accuracy":             m.get("accuracy", ""),
                "parameter_count":      m.get("params", ""),
                "latency_ms_per_sample": m.get("latency_ms", ""),
                "gradient_health":      m.get("gradient_health", ""),
                "nan_inf_status":       m.get("nan_inf", ""),
                "qubit_count":          sc.get("qubit_count", ""),
                "ansatz_depth":         sc.get("ansatz_depth", ""),
                "rotation_family":      sc.get("rotation_family", ""),
                "entanglement_topology": sc.get("entanglement_topology", ""),
                "reuploading_frequency": sc.get("reuploading_frequency", ""),
                "compression_dim":       sc.get("compression_dim", ""),
                "learning_rate":         sc.get("learning_rate", ""),
                "weight_decay":          sc.get("weight_decay", ""),
                "epochs":               t.get("epochs", ""),
                "seed":                 t.get("seed", ""),
                "experiment_id":        t.get("experiment_id", ""),
                "config_path":          t.get("config_path", ""),
                "result_json_path":     t.get("result_json_path", ""),
            })


# ── Per-trial execution (factored out; called by sequential loop and parallel pool) ──

def run_one_trial(
    trial_id: str,
    trial_idx: int,
    sampled: dict,
    pilot_seed: int,
    epochs: int,
    thread_cap: int = 0,
) -> tuple[dict, str]:
    """
    Execute one DV NAS trial. Returns (trial_record, stdout_text).
    Thread-safe: each trial writes to a distinct config path and experiment_id.

    thread_cap: when > 0, sets OMP_NUM_THREADS / MKL_NUM_THREADS /
    OPENBLAS_NUM_THREADS in the child process environment to limit PyTorch's
    thread pool size. Recommended: 2 for 5-worker parallel runs on a 12-CPU host
    (10 total threads vs 150 observed in Q34B-Parallel-Lite with no cap).
    """
    config_yaml = build_trial_yaml(trial_id, trial_idx, sampled, pilot_seed, epochs)
    config_path = os.path.join(CONFIGS_DIR, f"{trial_id}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)

    # Build subprocess environment: inherit parent, apply thread cap if requested.
    env = os.environ.copy()
    if thread_cap > 0:
        cap_str = str(thread_cap)
        env["OMP_NUM_THREADS"]      = cap_str
        env["MKL_NUM_THREADS"]      = cap_str
        env["OPENBLAS_NUM_THREADS"] = cap_str

    cmd = [sys.executable, RUNNER_SCRIPT, "--config", config_path]
    t0  = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )
        stdout_combined = proc.stdout or ""
    except Exception as exc:
        stdout_combined = f"[ERROR] {exc}\n"
        proc = type("FakeProc", (), {"returncode": -1})()

    t_trial     = time.monotonic() - t0
    trial_seed  = pilot_seed + trial_idx
    metrics     = parse_trial_metrics(stdout_combined)
    rj_path     = parse_result_json_path(stdout_combined)
    exp_id      = None

    if rj_path and os.path.isfile(rj_path):
        try:
            with open(rj_path, "r", encoding="utf-8") as jf:
                rj = json.load(jf)
            exp_id = rj.get("experiment_id")
            if metrics["status"] is None:
                metrics["status"] = rj.get("status") or metrics["status"]
        except Exception:
            pass

    trial_status = classify_trial_status(metrics, proc.returncode)
    record = {
        "trial_id":         trial_id,
        "trial_idx":        trial_idx,
        "sampled_config":   sampled,
        "config_path":      config_path,
        "result_json_path": rj_path or "",
        "experiment_id":    exp_id or "",
        "return_code":      proc.returncode,
        "wall_time_s":      round(t_trial, 2),
        "epochs":           epochs,
        "seed":             trial_seed,
        "trial_status":     trial_status,
        "metrics":          metrics,
    }
    return record, stdout_combined


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.epochs > 5:
        print(f"ERROR: --epochs {args.epochs} exceeds Q34B ceiling of 5", flush=True)
        sys.exit(1)
    if args.trials < 1:
        print(f"ERROR: --trials must be >= 1", flush=True)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(CONFIGS_DIR,     exist_ok=True)
    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print("Q34B DV NAS Pilot", flush=True)
    print("=" * 60, flush=True)
    exec_mode = f"parallel (max_workers={args.max_workers})" if args.parallel else "sequential"
    thread_cap_display = str(args.thread_cap) if args.thread_cap > 0 else "none"
    print(f"  trials:        {args.trials}", flush=True)
    print(f"  epochs/trial:  {args.epochs}", flush=True)
    print(f"  seed:          {args.seed}", flush=True)
    print(f"  pass_threshold:{PASS_THRESHOLD} / {args.trials}", flush=True)
    print(f"  search_space:  {SEARCH_SPACE_VERSION}", flush=True)
    print(f"  execution:     {exec_mode}", flush=True)
    print(f"  thread_cap:    {thread_cap_display} (OMP/MKL/OPENBLAS per subprocess)", flush=True)
    print(f"  device:        CPU (DV quantum circuit is CPU-only)", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # ── Sample all configs before execution (deterministic given seed) ────────
    sampled_configs: list[dict] = []
    trial_ids: list[str] = []
    for i in range(args.trials):
        trial_ids.append(f"q34b_trial_{i + 1:03d}")
        sampled_configs.append(sample_config())

    print(f"[PILOT] Sampled {args.trials} configs:", flush=True)
    for i, (tid, sc) in enumerate(zip(trial_ids, sampled_configs)):
        print(
            f"  [{i+1}] {tid}: "
            f"qubits={sc['qubit_count']} depth={sc['ansatz_depth']} "
            f"rot={sc['rotation_family']} "
            f"entangle={sc['entanglement_topology']} "
            f"reup={sc['reuploading_frequency']} "
            f"compress={sc['compression_dim']} "
            f"lr={sc['learning_rate']} wd={sc['weight_decay']}",
            flush=True,
        )
    print(flush=True)

    # ── Execute trials ─────────────────────────────────────────────────────────
    all_trials: list[dict] = []
    pilot_start = time.monotonic()

    if not args.parallel:
        # ── Sequential mode (default) ──────────────────────────────────────────
        for i, (trial_id, sampled) in enumerate(zip(trial_ids, sampled_configs)):
            print("-" * 60, flush=True)
            print(f"[PILOT] Starting trial {i + 1}/{args.trials}: {trial_id}", flush=True)
            print("-" * 60, flush=True)
            record, stdout_combined = run_one_trial(
                trial_id, i, sampled, args.seed, args.epochs,
                thread_cap=args.thread_cap,
            )
            if stdout_combined:
                sys.stdout.write(stdout_combined)
                sys.stdout.flush()
            all_trials.append(record)
            m = record["metrics"]
            status_str = (
                "COMPLETED" if record["trial_status"] == "completed" else
                "UNSTABLE"  if record["trial_status"] == "unstable"  else
                "INVALID"   if record["trial_status"] == "invalid"   else
                f"FAILED (rc={record['return_code']})"
            )
            print(
                f"\n[PILOT] Trial {i + 1}/{args.trials} {status_str} | "
                f"auroc={m.get('auroc')} f1={m.get('f1')} "
                f"params={m.get('params')} lat={m.get('latency_ms')}ms | "
                f"grad={m.get('gradient_health')} nan_inf={m.get('nan_inf')} | "
                f"wall={record['wall_time_s']}s",
                flush=True,
            )
            print(flush=True)

    else:
        # ── Parallel mode (--parallel) ─────────────────────────────────────────
        print(
            f"[PILOT] Submitting {args.trials} trials to ThreadPoolExecutor "
            f"(max_workers={args.max_workers})",
            flush=True,
        )
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for i, (trial_id, sampled) in enumerate(zip(trial_ids, sampled_configs)):
                fut = pool.submit(run_one_trial, trial_id, i, sampled, args.seed, args.epochs,
                                  args.thread_cap)
                futures[fut] = (i, trial_id)

            for fut in as_completed(futures):
                i, trial_id = futures[fut]
                try:
                    record, stdout_combined = fut.result()
                except Exception as exc:
                    print(f"[PILOT] Trial {trial_id} raised exception: {exc}", flush=True)
                    continue

                print("-" * 60, flush=True)
                print(f"[PILOT] Trial {i + 1}/{args.trials} completed: {trial_id}", flush=True)
                print("-" * 60, flush=True)
                if stdout_combined:
                    sys.stdout.write(stdout_combined)
                    sys.stdout.flush()

                all_trials.append(record)
                m = record["metrics"]
                status_str = (
                    "COMPLETED" if record["trial_status"] == "completed" else
                    "UNSTABLE"  if record["trial_status"] == "unstable"  else
                    "INVALID"   if record["trial_status"] == "invalid"   else
                    f"FAILED (rc={record['return_code']})"
                )
                print(
                    f"\n[PILOT] {trial_id} {status_str} | "
                    f"auroc={m.get('auroc')} f1={m.get('f1')} "
                    f"params={m.get('params')} lat={m.get('latency_ms')}ms | "
                    f"grad={m.get('gradient_health')} nan_inf={m.get('nan_inf')} | "
                    f"wall={record['wall_time_s']}s",
                    flush=True,
                )
                print(flush=True)

        # Restore deterministic order by trial_idx for downstream processing
        all_trials.sort(key=lambda t: t["trial_idx"])

    pilot_elapsed = time.monotonic() - pilot_start

    # ── Tally ──────────────────────────────────────────────────────────────────
    n_completed = sum(1 for t in all_trials if t["trial_status"] == "completed")
    n_failed    = sum(1 for t in all_trials if t["trial_status"] == "failed")
    n_invalid   = sum(1 for t in all_trials if t["trial_status"] == "invalid")
    n_unstable  = sum(1 for t in all_trials if t["trial_status"] == "unstable")

    # ── Pareto ─────────────────────────────────────────────────────────────────
    is_pareto     = compute_pareto(all_trials)
    pareto_indices = [i for i, p in enumerate(is_pareto) if p]

    # ── Summary JSON ───────────────────────────────────────────────────────────
    verdict    = "PASS" if n_completed >= PASS_THRESHOLD else "FAIL"
    pilot_name = "Q34B_DV_NAS_PILOT_PARALLEL_LITE" if args.parallel else "Q34B_DV_NAS_PILOT"
    summary = {
        "pilot":              pilot_name,
        "verdict":            verdict,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials_attempted":   args.trials,
        "trials_completed":   n_completed,
        "trials_failed":      n_failed,
        "trials_invalid":     n_invalid,
        "trials_unstable":    n_unstable,
        "seed":               args.seed,
        "search_space_version": SEARCH_SPACE_VERSION,
        "pass_threshold":     PASS_THRESHOLD,
        "epochs_per_trial":   args.epochs,
        "execution_mode":     "parallel" if args.parallel else "sequential",
        "max_workers":        args.max_workers if args.parallel else 1,
        "thread_cap":         args.thread_cap,
        "wall_time_s":        round(pilot_elapsed, 2),
        "search_space":       SEARCH_SPACE,
        "trial_results":      all_trials,
        "pareto_indices":     pareto_indices,
        "pareto_csv":         PARETO_CSV_PATH,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_pareto_csv(all_trials, is_pareto)

    # ── Final report ───────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Q34B DV NAS PILOT RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Trials run:      {args.trials}", flush=True)
    print(f"  Completed:       {n_completed}", flush=True)
    print(f"  Failed:          {n_failed}", flush=True)
    print(f"  Invalid:         {n_invalid}", flush=True)
    print(f"  Unstable:        {n_unstable}", flush=True)
    print(f"  Pass threshold:  {PASS_THRESHOLD}", flush=True)
    print(f"  Wall time:       {pilot_elapsed:.1f}s", flush=True)
    print(flush=True)

    print("  Trial results:", flush=True)
    for t in all_trials:
        m  = t["metrics"]
        s  = "✓" if t["trial_status"] == "completed" else "✗"
        sc = t["sampled_config"]
        print(
            f"    {s} {t['trial_id']} [{t['trial_status']}]: "
            f"qubits={sc.get('qubit_count')} depth={sc.get('ansatz_depth')} "
            f"auroc={m.get('auroc')} f1={m.get('f1')} "
            f"params={m.get('params')} "
            f"grad={m.get('gradient_health')}",
            flush=True,
        )
    print(flush=True)

    if pareto_indices:
        print(f"  Preliminary DV Pareto set ({len(pareto_indices)} trials):", flush=True)
        for i in pareto_indices:
            t = all_trials[i]
            m = t["metrics"]
            sc = t["sampled_config"]
            print(
                f"    {t['trial_id']}: auroc={m.get('auroc')} f1={m.get('f1')} "
                f"params={m.get('params')} "
                f"qubits={sc.get('qubit_count')} depth={sc.get('ansatz_depth')}",
                flush=True,
            )
    else:
        print("  Preliminary DV Pareto set: empty (no eligible trials)", flush=True)
    print(flush=True)

    print(f"  Summary JSON:    {SUMMARY_PATH}", flush=True)
    print(f"  Pareto CSV:      {PARETO_CSV_PATH}", flush=True)
    print(flush=True)

    print(f"{pilot_name}: {verdict}", flush=True)

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
