"""
qcore/experiments/runner.py

Local GPU experiment runner for the QStrata automation framework (Q31 MVP).

Implements the minimal viable runner defined in Q31:
  - YAML config loading and schema validation
  - experiment_id generation
  - Experiment directory creation
  - Config freezing (copy + chmod 444)
  - Git commit and hardware metadata capture
  - Sequential subprocess execution with stdout/stderr tee to log file
  - Result JSON creation at experiments/results/<experiment_id>.json
  - Leaderboard CSV append at experiments/leaderboards/<phase>.csv

OUT OF SCOPE for Q31 MVP:
  - Signal handler recovery (SIGINT, SIGTERM, SIGUSR1)
  - Checkpoint-based resume logic
  - Per-epoch partial metric export
  - NAS trial generation or orchestration
  - Distributed execution (AWS, Ray)
  - Full training runs (use smoke configs only in Q31)

Reference:
  docs/architecture/qstrata_experiment_automation_framework.md
  docs/specs/qstrata_experiment_config_schema.md
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time

import yaml

from qcore.experiments.leaderboard import update_leaderboard
from qcore.experiments.metadata import (
    capture_git_commit,
    capture_hardware,
    generate_experiment_id,
)
from qcore.experiments.schema import validate_config

# Root directory for all experiment artifacts.
# Relative to the working directory (project root) when the runner executes.
EXPERIMENTS_ROOT = "experiments"


# ─────────────────────────────────────────────────────────────────────────────
# Stdout metric parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_smoke_pass(stdout: str, return_code: int, phase: str) -> object:
    """
    Determine smoke_pass from subprocess stdout and return code.

    Returns True  if return_code == 0 and "SMOKE TEST RESULT: PASS" is in stdout.
    Returns False if the config phase looks like a smoke test but the above is not met.
    Returns None  if the phase is not a smoke test (non-smoke experiments do not
                  have a meaningful smoke_pass value).

    The string "SMOKE TEST RESULT: PASS" matches the output of
    scripts/smoke_test_vindr_cv_binary.py (Q26) and related smoke scripts.

    Note: The Q31 design spec referenced "CV_BINARY_SMOKE_TEST: PASS" but the
    actual Q26 script outputs "SMOKE TEST RESULT: PASS". This parser uses the
    actual script output.
    """
    smoke_keywords = ["smoke", "q26", "q31_runner_smoke"]
    phase_lower = phase.lower()
    is_smoke = any(kw in phase_lower for kw in smoke_keywords)

    if not is_smoke:
        return None

    if return_code == 0 and "SMOKE TEST RESULT: PASS" in stdout:
        return True
    return False


def _parse_trainable_params(stdout: str) -> object:
    """
    Extract trainable parameter count from the [SUMMARY] block.

    Looks for a line containing "Trainable params:" in the stdout summary.
    Returns int if found and parseable, None otherwise.
    """
    in_summary = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "[SUMMARY]":
            in_summary = True
            continue
        if in_summary and stripped.startswith("Trainable params:"):
            val = stripped.split(":")[-1].strip()
            try:
                return int(val)
            except ValueError:
                pass
    return None


def _parse_loss(stdout: str) -> object:
    """
    Extract final loss value from the [SUMMARY] block.

    Looks for a line containing "Loss:" after the "[SUMMARY]" marker.
    Returns float if found and parseable, None otherwise.
    """
    in_summary = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "[SUMMARY]":
            in_summary = True
            continue
        if in_summary and stripped.startswith("Loss:"):
            val = stripped.split(":")[-1].strip()
            try:
                return float(val)
            except ValueError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(config: dict, config_path: str = "") -> int:
    """
    Execute one experiment from a parsed config dict.

    Args:
        config:      Parsed YAML config dict (from yaml.safe_load).
        config_path: Path to the original YAML file (for result JSON record only).

    Returns:
        int: Subprocess return code (0 = success, non-zero = failure).

    Side effects:
        - Creates experiment directories under EXPERIMENTS_ROOT
        - Writes frozen config YAML (read-only)
        - Writes stdout + stderr to log file
        - Writes result JSON
        - Appends one row to phase leaderboard CSV
    """

    # ── 1. Validate config ────────────────────────────────────────────────────
    validate_config(config)

    # ── 2. Generate experiment_id ─────────────────────────────────────────────
    experiment_id   = generate_experiment_id()
    timestamp_start = datetime.datetime.now().isoformat()

    # ── 3. Resolve fields from config ────────────────────────────────────────
    phase      = config.get("experiment", {}).get("phase", "unknown")
    dataset    = config.get("dataset",    {}).get("name", "unknown")
    model_arch = config.get("model",      {}).get("architecture", "unknown")
    seed       = config.get("reproducibility", {}).get("seed", None)

    executable = config["command"]["executable"]
    cmd_args   = [str(a) for a in config["command"]["args"]]
    command    = [executable] + cmd_args

    # ── 4. Create experiment directories ─────────────────────────────────────
    dir_configs      = os.path.join(EXPERIMENTS_ROOT, "configs")
    dir_results      = os.path.join(EXPERIMENTS_ROOT, "results")
    dir_logs         = os.path.join(EXPERIMENTS_ROOT, "logs")
    dir_leaderboards = os.path.join(EXPERIMENTS_ROOT, "leaderboards")

    for d in (dir_configs, dir_results, dir_logs, dir_leaderboards):
        os.makedirs(d, exist_ok=True)

    # ── 5. Paths ──────────────────────────────────────────────────────────────
    frozen_config_path = os.path.join(dir_configs,      f"{experiment_id}.yaml")
    result_path        = os.path.join(dir_results,      f"{experiment_id}.json")
    log_path           = os.path.join(dir_logs,         f"{experiment_id}.log")
    leaderboard_path   = os.path.join(dir_leaderboards, f"{phase}.csv")

    # ── 6. Freeze config ──────────────────────────────────────────────────────
    with open(frozen_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    try:
        os.chmod(frozen_config_path, 0o444)
    except OSError as e:
        print(f"[RUNNER] WARNING: Could not set frozen config read-only: {e}", flush=True)

    # ── 7. Capture metadata ───────────────────────────────────────────────────
    git_info = capture_git_commit()
    hardware = capture_hardware()

    # ── 8. Runner header ──────────────────────────────────────────────────────
    print(f"[RUNNER] ═══════════════════════════════════════════════", flush=True)
    print(f"[RUNNER] QStrata Experiment Runner — Q31 MVP", flush=True)
    print(f"[RUNNER] ───────────────────────────────────────────────", flush=True)
    print(f"[RUNNER] experiment_id:    {experiment_id}", flush=True)
    print(f"[RUNNER] phase:            {phase}", flush=True)
    print(f"[RUNNER] dataset:          {dataset}", flush=True)
    print(f"[RUNNER] model:            {model_arch}", flush=True)
    print(f"[RUNNER] seed:             {seed}", flush=True)
    git_short = git_info["commit"][:12] if git_info["commit"] != "unknown" else "unknown"
    dirty_flag = " (DIRTY)" if git_info["dirty"] else ""
    print(f"[RUNNER] git_commit:       {git_short}{dirty_flag}", flush=True)
    print(f"[RUNNER] hardware:         {hardware['gpu_model']}", flush=True)
    print(f"[RUNNER] frozen config:    {frozen_config_path}", flush=True)
    print(f"[RUNNER] log:              {log_path}", flush=True)
    print(f"[RUNNER] command:          {' '.join(command)}", flush=True)
    print(f"[RUNNER] ───────────────────────────────────────────────", flush=True)

    # ── 9. Execute subprocess with tee ────────────────────────────────────────
    t_start      = time.monotonic()
    stdout_lines: list[str] = []

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            stdout_lines.append(line)
        proc.wait()

    t_end          = time.monotonic()
    duration       = round(t_end - t_start, 3)
    return_code    = proc.returncode
    timestamp_end  = datetime.datetime.now().isoformat()
    status         = "completed" if return_code == 0 else "failed"
    stdout_full    = "".join(stdout_lines)

    # ── 10. Runner footer ─────────────────────────────────────────────────────
    print(f"[RUNNER] ───────────────────────────────────────────────", flush=True)
    print(f"[RUNNER] return_code:      {return_code}", flush=True)
    print(f"[RUNNER] status:           {status}", flush=True)
    print(f"[RUNNER] duration:         {duration}s", flush=True)

    # ── 11. Parse metrics from stdout ─────────────────────────────────────────
    smoke_pass       = _parse_smoke_pass(stdout_full, return_code, phase)
    trainable_params = _parse_trainable_params(stdout_full)
    loss_val         = _parse_loss(stdout_full)

    print(f"[RUNNER] smoke_pass:       {smoke_pass}", flush=True)
    print(f"[RUNNER] trainable_params: {trainable_params}", flush=True)
    print(f"[RUNNER] loss:             {loss_val}", flush=True)

    # ── 12. Write result JSON ─────────────────────────────────────────────────
    result: dict = {
        "experiment_id":      experiment_id,
        "timestamp_start":    timestamp_start,
        "timestamp_end":      timestamp_end,
        "status":             status,
        "return_code":        return_code,
        "duration_seconds":   duration,
        "config_path":        config_path,
        "frozen_config_path": frozen_config_path,
        "log_path":           log_path,
        "phase":              phase,
        "dataset":            dataset,
        "model":              model_arch,
        "seed":               seed,
        "git_commit":         git_info["commit"],
        "git_dirty":          git_info["dirty"],
        "hardware":           hardware,
        "command":            command,
        "metrics": {
            "smoke_pass":       smoke_pass,
            "trainable_params": trainable_params,
            "loss":             loss_val,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ── 13. Update leaderboard ────────────────────────────────────────────────
    leaderboard_row: dict = {
        "experiment_id":   experiment_id,
        "timestamp_start": timestamp_start,
        "phase":           phase,
        "status":          status,
        "smoke_pass":      smoke_pass,
        "return_code":     return_code,
        "duration_seconds": duration,
        "git_commit":      git_short,
        "git_dirty":       git_info["dirty"],
    }
    update_leaderboard(leaderboard_path, leaderboard_row)

    print(f"[RUNNER] result JSON:      {result_path}", flush=True)
    print(f"[RUNNER] leaderboard:      {leaderboard_path}", flush=True)
    print(f"[RUNNER] ═══════════════════════════════════════════════", flush=True)

    return return_code
