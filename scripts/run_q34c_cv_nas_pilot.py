"""
scripts/run_q34c_cv_nas_pilot.py

Q34C CV NAS Pilot Orchestrator.

Runs a random-search NAS pilot over the CV quantum head search space
defined in Q33B (docs/architecture/q33b_cv_quantum_nas_search_space.md).

Third execution step in the Q34 NAS pipeline: Q34A (classical) -> Q34B (DV) -> Q34C (CV).

Pilot substitutions (fixed for this pilot):
    beam_splitter_topology:     circular
    readout_strategy:           first_moments
    encoding_strategy:          displacement_encoding
    covariance_parameterization: symplectic_parameterization
    displacement_cap:           2.0 (not searched; applied via tanh in candidate script)

Search constraint:
    compression_dim = 2 * n_modes  (displacement encoding requires n_modes complex amplitudes)

CV-specific Pareto filter:
    Only trials with stability_taxonomy == "valid" enter Pareto computation.
    Trials with any other taxonomy are recorded but excluded from the frontier.

CLI:
    python3 scripts/run_q34c_cv_nas_pilot.py --trials 5 --seed 42 --epochs 2
    python3 scripts/run_q34c_cv_nas_pilot.py --trials 1 --seed 42 --epochs 1  # smoke
    python3 scripts/run_q34c_cv_nas_pilot.py --trials 5 --parallel --max-workers 5 --thread-cap 2

Outputs:
    configs/experiments/q34c_cv_nas/<trial_id>.yaml
    experiments/results/q34c_cv_nas_pilot_summary.json
    experiments/leaderboards/q34c_cv_pareto.csv
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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR      = os.path.join(PROJECT_ROOT, "configs", "experiments", "q34c_cv_nas")
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "experiments", "results")
LEADERBOARD_DIR  = os.path.join(PROJECT_ROOT, "experiments", "leaderboards")
SUMMARY_PATH     = os.path.join(RESULTS_DIR,     "q34c_cv_nas_pilot_summary.json")
PARETO_CSV_PATH  = os.path.join(LEADERBOARD_DIR, "q34c_cv_pareto.csv")
CANDIDATE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_q34c_cv_candidate.py")

DATASET_ROOT    = "data/processed/vindr_binary_roi_224"
CHECKPOINT_PATH = "checkpoints/c006_d040_classical_anchor.pt"

# ── Q34C CV search space (Q33B pilot subset) ───────────────────────────────────

SEARCH_SPACE: dict = {
    "n_modes":       [1, 2, 4],
    "cv_depth":      [1, 2, 3],
    "squeezing_cap": [0.5, 1.0, 1.5, 2.0],
    "learning_rate": [0.001, 0.0005],
    "weight_decay":  [0.0, 0.0001],
}
SEARCH_SPACE_VERSION = "q34c_v1"
DISPLACEMENT_CAP     = 2.0     # pilot-fixed (per Q34C-Preflight readiness check)
PASS_THRESHOLD       = 3
MAX_TRAINABLE_PARAMS = 100_000
EPOCHS_PER_TRIAL     = 2


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q34C CV NAS Pilot — random search orchestrator")
    p.add_argument("--trials",      type=int,  default=5)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--epochs",      type=int,  default=EPOCHS_PER_TRIAL)
    p.add_argument("--parallel",    action="store_true", default=False)
    p.add_argument("--max-workers", type=int,  default=5)
    p.add_argument("--thread-cap",  type=int,  default=0,
                   help="Per-subprocess thread cap: sets OMP/MKL/OPENBLAS_NUM_THREADS (0=no cap)")
    return p.parse_args()


# ── Config sampling ────────────────────────────────────────────────────────────

def sample_config() -> dict:
    sampled = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    # Enforce compression_dim = 2 * n_modes (displacement encoding constraint)
    sampled["compression_dim"]           = 2 * sampled["n_modes"]
    sampled["displacement_cap"]          = DISPLACEMENT_CAP
    sampled["encoding_strategy"]         = "displacement_encoding"
    sampled["beam_splitter_topology"]    = "circular"
    sampled["readout_strategy"]          = "first_moments"
    sampled["covariance_parameterization"] = "symplectic_parameterization"
    return sampled


def build_trial_yaml(trial_id, trial_idx, sampled, pilot_seed, epochs) -> dict:
    trial_seed = pilot_seed + trial_idx
    config_rel = os.path.join("configs", "experiments", "q34c_cv_nas", f"{trial_id}.yaml")
    return {
        "experiment": {
            "phase": "q34c_cv_nas_pilot",
            "trial_id": trial_id,
            "description": (
                f"Q34C trial {trial_idx+1} n_modes={sampled['n_modes']} "
                f"depth={sampled['cv_depth']} sq_cap={sampled['squeezing_cap']} "
                f"lr={sampled['learning_rate']} wd={sampled['weight_decay']}"
            ),
        },
        "dataset": {
            "name": "vindr_binary_roi_224",
            "split": "canonical_42",
            "root": DATASET_ROOT,
        },
        "model": {
            "architecture":               "cv_quantum_head",
            "backbone":                   "c006_d040_frozen",
            "checkpoint_path":            CHECKPOINT_PATH,
            "n_modes":                    sampled["n_modes"],
            "cv_depth":                   sampled["cv_depth"],
            "squeezing_cap":              sampled["squeezing_cap"],
            "displacement_cap":           sampled["displacement_cap"],
            "compression_dim":            sampled["compression_dim"],
            "encoding_strategy":          sampled["encoding_strategy"],
            "beam_splitter_topology":     sampled["beam_splitter_topology"],
            "readout_strategy":           sampled["readout_strategy"],
            "covariance_parameterization": sampled["covariance_parameterization"],
        },
        "training": {
            "epochs":       epochs,
            "batch_size":   4,
            "optimizer":    "AdamW",
            "learning_rate": sampled["learning_rate"],
            "weight_decay":  sampled["weight_decay"],
            "early_stopping": {"enabled": False},
        },
        "reproducibility": {"seed": trial_seed},
        "command": {
            "executable": "python3",
            "args": ["scripts/train_q34c_cv_candidate.py", "--config", config_rel],
        },
    }


# ── Metric parsing ─────────────────────────────────────────────────────────────

def parse_trial_metrics(stdout: str) -> dict:
    m: dict = {k: None for k in
               ("auroc", "f1", "accuracy", "params", "latency_ms",
                "gradient_health", "nan_inf", "stability_taxonomy", "status")}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Q34C_TRIAL_AUROC:"):
            try: m["auroc"]    = float(s.split(":", 1)[1])
            except ValueError: pass
        elif s.startswith("Q34C_TRIAL_F1:"):
            try: m["f1"]       = float(s.split(":", 1)[1])
            except ValueError: pass
        elif s.startswith("Q34C_TRIAL_ACCURACY:"):
            try: m["accuracy"] = float(s.split(":", 1)[1])
            except ValueError: pass
        elif s.startswith("Q34C_TRIAL_PARAMS:"):
            try: m["params"]   = int(s.split(":", 1)[1])
            except ValueError: pass
        elif s.startswith("Q34C_TRIAL_LATENCY_MS:"):
            try: m["latency_ms"] = float(s.split(":", 1)[1])
            except ValueError: pass
        elif s.startswith("Q34C_TRIAL_GRADIENT_HEALTH:"):
            m["gradient_health"]    = s.split(":", 1)[1].strip()
        elif s.startswith("Q34C_TRIAL_NAN_INF:"):
            m["nan_inf"]            = s.split(":", 1)[1].strip()
        elif s.startswith("Q34C_TRIAL_STABILITY_TAXONOMY:"):
            m["stability_taxonomy"] = s.split(":", 1)[1].strip()
        elif s.startswith("Q34C_TRIAL_STATUS:"):
            m["status"]             = s.split(":", 1)[1].strip()
    return m


def classify_trial_status(m: dict, rc: int) -> str:
    if rc != 0:
        return "failed"
    if m.get("nan_inf") == "FAIL" or m.get("stability_taxonomy") == "nan_state":
        return "invalid"
    if m.get("params") and m["params"] > MAX_TRAINABLE_PARAMS:
        return "invalid"
    if m.get("stability_taxonomy") not in ("valid", None) and \
       m.get("stability_taxonomy") != "unstable_training":
        return "invalid"
    if m.get("gradient_health") == "FAIL":
        return "unstable"
    if m.get("status") == "completed":
        return "completed"
    return "failed"


# ── Pareto (stability-aware) ───────────────────────────────────────────────────

def compute_pareto(trials: list[dict]) -> list[bool]:
    """Pareto over (AUROC ↑, F1 ↑, params ↓). Only stability_taxonomy='valid' trials eligible."""
    n = len(trials)

    def eligible(t):
        m = t.get("metrics", {})
        return (t.get("trial_status") == "completed"
                and m.get("stability_taxonomy") == "valid"
                and m.get("auroc") is not None
                and m.get("f1") is not None
                and m.get("params") is not None
                and not (isinstance(m.get("auroc"), float) and np.isnan(m["auroc"]))
                and m.get("params", 0) <= MAX_TRAINABLE_PARAMS)

    elig = [i for i, t in enumerate(trials) if eligible(t)]
    is_pareto = [False] * n
    for i in elig:
        mi = trials[i]["metrics"]
        dominated = any(
            j != i
            and trials[j]["metrics"]["auroc"]  >= mi["auroc"]
            and trials[j]["metrics"]["f1"]     >= mi["f1"]
            and trials[j]["metrics"]["params"] <= mi["params"]
            and (trials[j]["metrics"]["auroc"]  > mi["auroc"]
                 or trials[j]["metrics"]["f1"]  > mi["f1"]
                 or trials[j]["metrics"]["params"] < mi["params"])
            for j in elig
        )
        if not dominated:
            is_pareto[i] = True
    return is_pareto


# ── CSV ────────────────────────────────────────────────────────────────────────

PARETO_FIELDS = [
    "trial_id", "trial_idx", "is_pareto", "status", "stability_taxonomy",
    "auroc", "f1", "accuracy", "parameter_count", "latency_ms_per_sample",
    "gradient_health", "nan_inf_status",
    "n_modes", "cv_depth", "squeezing_cap", "displacement_cap",
    "encoding_strategy", "beam_splitter_topology", "readout_strategy",
    "covariance_parameterization", "compression_dim",
    "learning_rate", "weight_decay", "epochs", "seed", "experiment_id", "config_path",
]


def write_pareto_csv(trials, is_pareto):
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    with open(PARETO_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PARETO_FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, (t, pareto) in enumerate(zip(trials, is_pareto)):
            m  = t.get("metrics", {})
            sc = t.get("sampled_config", {})
            w.writerow({
                "trial_id": t.get("trial_id", ""),  "trial_idx": t.get("trial_idx", i),
                "is_pareto": pareto,                 "status": t.get("trial_status", ""),
                "stability_taxonomy": m.get("stability_taxonomy", ""),
                "auroc": m.get("auroc", ""),         "f1": m.get("f1", ""),
                "accuracy": m.get("accuracy", ""),   "parameter_count": m.get("params", ""),
                "latency_ms_per_sample": m.get("latency_ms", ""),
                "gradient_health": m.get("gradient_health", ""),
                "nan_inf_status": m.get("nan_inf", ""),
                "n_modes": sc.get("n_modes", ""),    "cv_depth": sc.get("cv_depth", ""),
                "squeezing_cap": sc.get("squeezing_cap", ""),
                "displacement_cap": sc.get("displacement_cap", ""),
                "encoding_strategy": sc.get("encoding_strategy", ""),
                "beam_splitter_topology": sc.get("beam_splitter_topology", ""),
                "readout_strategy": sc.get("readout_strategy", ""),
                "covariance_parameterization": sc.get("covariance_parameterization", ""),
                "compression_dim": sc.get("compression_dim", ""),
                "learning_rate": sc.get("learning_rate", ""),
                "weight_decay": sc.get("weight_decay", ""),
                "epochs": t.get("epochs", ""),       "seed": t.get("seed", ""),
                "experiment_id": t.get("experiment_id", ""),
                "config_path": t.get("config_path", ""),
            })


# ── Per-trial execution ────────────────────────────────────────────────────────

def run_one_trial(trial_id, trial_idx, sampled, pilot_seed, epochs, thread_cap=0):
    config_yaml = build_trial_yaml(trial_id, trial_idx, sampled, pilot_seed, epochs)
    config_path = os.path.join(CONFIGS_DIR, f"{trial_id}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)

    env = os.environ.copy()
    if thread_cap > 0:
        cap = str(thread_cap)
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = cap

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, CANDIDATE_SCRIPT, "--config", config_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=PROJECT_ROOT, env=env,
        )
        stdout = proc.stdout or ""
    except Exception as exc:
        stdout = f"[ERROR] {exc}\n"
        proc = type("P", (), {"returncode": -1})()

    wall    = time.monotonic() - t0
    seed    = pilot_seed + trial_idx
    metrics = parse_trial_metrics(stdout)
    exp_id  = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    status  = classify_trial_status(metrics, proc.returncode)

    record = {
        "trial_id": trial_id,        "trial_idx": trial_idx,
        "sampled_config": sampled,   "config_path": config_path,
        "experiment_id": exp_id,     "return_code": proc.returncode,
        "wall_time_s": round(wall, 2), "epochs": epochs, "seed": seed,
        "trial_status": status,      "metrics": metrics,
    }
    return record, stdout


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.epochs > 5:
        print(f"ERROR: --epochs {args.epochs} exceeds Q34C ceiling of 5", flush=True); sys.exit(1)

    random.seed(args.seed); np.random.seed(args.seed)
    for d in (CONFIGS_DIR, RESULTS_DIR, LEADERBOARD_DIR):
        os.makedirs(d, exist_ok=True)

    exec_mode   = f"parallel (max_workers={args.max_workers})" if args.parallel else "sequential"
    pilot_name  = "Q34C_CV_NAS_PILOT_PARALLEL" if args.parallel else "Q34C_CV_NAS_PILOT"
    thread_disp = str(args.thread_cap) if args.thread_cap > 0 else "none"
    print("=" * 60, flush=True)
    print("Q34C CV NAS Pilot", flush=True)
    print("=" * 60, flush=True)
    print(f"  trials:        {args.trials}", flush=True)
    print(f"  epochs/trial:  {args.epochs}", flush=True)
    print(f"  seed:          {args.seed}", flush=True)
    print(f"  execution:     {exec_mode}", flush=True)
    print(f"  thread_cap:    {thread_disp}", flush=True)
    print(f"  search_space:  {SEARCH_SPACE_VERSION}", flush=True)
    print(f"  disp_cap:      {DISPLACEMENT_CAP} (pilot-fixed, not searched)", flush=True)
    print("=" * 60, flush=True)

    trial_ids      = [f"q34c_trial_{i+1:03d}" for i in range(args.trials)]
    sampled_configs = [sample_config() for _ in range(args.trials)]

    for i, (tid, sc) in enumerate(zip(trial_ids, sampled_configs)):
        print(f"  [{i+1}] {tid}: n_modes={sc['n_modes']} depth={sc['cv_depth']} "
              f"sq={sc['squeezing_cap']} comp={sc['compression_dim']} "
              f"lr={sc['learning_rate']} wd={sc['weight_decay']}", flush=True)
    print(flush=True)

    all_trials: list[dict] = []
    pilot_start = time.monotonic()

    if not args.parallel:
        for i, (tid, sc) in enumerate(zip(trial_ids, sampled_configs)):
            print("-" * 60, flush=True)
            print(f"[PILOT] Starting trial {i+1}/{args.trials}: {tid}", flush=True)
            record, stdout = run_one_trial(tid, i, sc, args.seed, args.epochs, args.thread_cap)
            if stdout: sys.stdout.write(stdout); sys.stdout.flush()
            all_trials.append(record)
            m = record["metrics"]
            print(f"\n[PILOT] {tid} {record['trial_status'].upper()} | "
                  f"auroc={m.get('auroc')} f1={m.get('f1')} params={m.get('params')} "
                  f"stability={m.get('stability_taxonomy')} wall={record['wall_time_s']}s\n",
                  flush=True)
    else:
        print(f"[PILOT] Submitting {args.trials} trials to ThreadPoolExecutor "
              f"(max_workers={args.max_workers})", flush=True)
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for i, (tid, sc) in enumerate(zip(trial_ids, sampled_configs)):
                fut = pool.submit(run_one_trial, tid, i, sc, args.seed, args.epochs, args.thread_cap)
                futures[fut] = (i, tid)
            for fut in as_completed(futures):
                i, tid = futures[fut]
                try:
                    record, stdout = fut.result()
                except Exception as exc:
                    print(f"[PILOT] {tid} exception: {exc}", flush=True); continue
                print("-" * 60, flush=True)
                if stdout: sys.stdout.write(stdout); sys.stdout.flush()
                all_trials.append(record)
                m = record["metrics"]
                print(f"\n[PILOT] {tid} {record['trial_status'].upper()} | "
                      f"auroc={m.get('auroc')} stability={m.get('stability_taxonomy')} "
                      f"wall={record['wall_time_s']}s\n", flush=True)
        all_trials.sort(key=lambda t: t["trial_idx"])

    wall_total = time.monotonic() - pilot_start

    n_completed = sum(1 for t in all_trials if t["trial_status"] == "completed")
    n_failed    = sum(1 for t in all_trials if t["trial_status"] == "failed")
    n_invalid   = sum(1 for t in all_trials if t["trial_status"] == "invalid")
    n_unstable  = sum(1 for t in all_trials if t["trial_status"] == "unstable")

    is_pareto      = compute_pareto(all_trials)
    pareto_indices = [i for i, p in enumerate(is_pareto) if p]
    verdict        = "PASS" if n_completed >= PASS_THRESHOLD else "FAIL"

    summary = {
        "pilot":            pilot_name,
        "verdict":          verdict,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials_attempted": args.trials,
        "trials_completed": n_completed,
        "trials_failed":    n_failed,
        "trials_invalid":   n_invalid,
        "trials_unstable":  n_unstable,
        "seed":             args.seed,
        "search_space_version": SEARCH_SPACE_VERSION,
        "pass_threshold":   PASS_THRESHOLD,
        "epochs_per_trial": args.epochs,
        "execution_mode":   "parallel" if args.parallel else "sequential",
        "max_workers":      args.max_workers if args.parallel else 1,
        "thread_cap":       args.thread_cap,
        "displacement_cap": DISPLACEMENT_CAP,
        "wall_time_s":      round(wall_total, 2),
        "search_space":     SEARCH_SPACE,
        "trial_results":    all_trials,
        "pareto_indices":   pareto_indices,
        "pareto_csv":       PARETO_CSV_PATH,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_pareto_csv(all_trials, is_pareto)

    print("=" * 60, flush=True)
    print("Q34C CV NAS PILOT RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  Completed:  {n_completed} / {args.trials}", flush=True)
    print(f"  Failed:     {n_failed}", flush=True)
    print(f"  Invalid:    {n_invalid}", flush=True)
    print(f"  Unstable:   {n_unstable}", flush=True)
    print(f"  Pareto set: {len(pareto_indices)} (stability-filtered)", flush=True)
    print(f"  Wall time:  {wall_total:.1f}s", flush=True)
    for t in all_trials:
        m  = t["metrics"]; sc = t["sampled_config"]
        st = "✓" if t["trial_status"] == "completed" else "✗"
        print(f"    {st} {t['trial_id']}: n_modes={sc.get('n_modes')} "
              f"depth={sc.get('cv_depth')} auroc={m.get('auroc')} "
              f"stability={m.get('stability_taxonomy')}", flush=True)
    print(flush=True)
    print(f"  Summary JSON: {SUMMARY_PATH}", flush=True)
    print(f"  Pareto CSV:   {PARETO_CSV_PATH}", flush=True)
    print(flush=True)
    print(f"{pilot_name}: {verdict}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
