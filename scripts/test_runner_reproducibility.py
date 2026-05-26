"""
scripts/test_runner_reproducibility.py

Q31A Reproducibility Test for the QStrata experiment runner.

Runs the same experiment config twice sequentially and verifies that both
executions produce consistent results within the defined tolerance.

A runner that is not reproducible cannot be trusted to produce meaningful NAS
trial comparisons. This test is a required gate before Q32 (classical NAS) begins.

Usage:
    python scripts/test_runner_reproducibility.py \\
        --config configs/experiments/q31_smoke_vindr_cv_binary.yaml \\
        --runs 2 \\
        --tolerance 0.0001

Arguments:
    --config     Path to the experiment YAML config to run (required)
    --runs       Number of sequential runs (default: 2; only 2 is tested)
    --tolerance  Absolute tolerance for loss comparison (default: 0.0001)

Exit code:
    0 if all comparisons pass (Q31A_REPRODUCIBILITY_TEST: PASS)
    1 if any comparison fails (Q31A_REPRODUCIBILITY_TEST: FAIL)

Output:
    experiments/results/q31a_reproducibility_test.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys


RESULT_PATH = "experiments/results/q31a_reproducibility_test.json"


# ─────────────────────────────────────────────────────────────────────────────
# Single run helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_once(config_path: str) -> tuple[int, str | None, str]:
    """
    Execute the runner once via subprocess.

    Calls scripts/run_experiment.py with the given config. Inherits the current
    environment (so QSTRATA_GIT_COMMIT and QSTRATA_GIT_DIRTY env vars pass through
    automatically if they are set in the calling environment).

    Returns:
        (return_code, result_json_path, stdout_combined)

    result_json_path is parsed from the [RUNNER] result JSON line in stdout.
    Returns None for result_json_path if the line is not found.
    """
    cmd = [sys.executable, "scripts/run_experiment.py", "--config", config_path]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    stdout = proc.stdout or ""

    # Print subprocess output to our stdout for visibility
    sys.stdout.write(stdout)
    sys.stdout.flush()

    # Parse result JSON path from runner footer
    result_json_path: str | None = None
    for line in stdout.splitlines():
        if "[RUNNER] result JSON:" in line:
            candidate = line.split("[RUNNER] result JSON:")[-1].strip()
            if candidate:
                result_json_path = candidate
            break

    return proc.returncode, result_json_path, stdout


# ─────────────────────────────────────────────────────────────────────────────
# Field comparison
# ─────────────────────────────────────────────────────────────────────────────

def _compare_results(
    r1: dict,
    r2: dict,
    tolerance: float,
) -> tuple[dict, bool]:
    """
    Compare two result dicts on the required reproducibility fields.

    Returns:
        (comparison_dict, overall_pass)
    """
    comparison: dict = {}
    all_pass = True

    # status — both must be "completed"
    s1 = r1.get("status")
    s2 = r2.get("status")
    comparison["status_match"] = (s1 == s2 == "completed")
    if not comparison["status_match"]:
        all_pass = False

    # return_code — both must be 0
    rc1 = r1.get("return_code")
    rc2 = r2.get("return_code")
    comparison["return_code_match"] = (rc1 == rc2 == 0)
    if not comparison["return_code_match"]:
        all_pass = False

    # smoke_pass — both must be True
    sp1 = r1.get("metrics", {}).get("smoke_pass")
    sp2 = r2.get("metrics", {}).get("smoke_pass")
    comparison["smoke_pass_match"] = (sp1 is True and sp2 is True)
    if not comparison["smoke_pass_match"]:
        all_pass = False

    # trainable_params — must be equal (if both present)
    tp1 = r1.get("metrics", {}).get("trainable_params")
    tp2 = r2.get("metrics", {}).get("trainable_params")
    if tp1 is None or tp2 is None:
        comparison["trainable_params_match"] = None  # skipped — one or both null
    else:
        comparison["trainable_params_match"] = (tp1 == tp2)
        if not comparison["trainable_params_match"]:
            all_pass = False

    # loss — absolute delta must be <= tolerance (if both present)
    l1 = r1.get("metrics", {}).get("loss")
    l2 = r2.get("metrics", {}).get("loss")
    if l1 is None or l2 is None:
        comparison["loss_delta"] = None
        comparison["loss_within_tolerance"] = None  # skipped
    else:
        delta = abs(float(l1) - float(l2))
        comparison["loss_delta"] = round(delta, 8)
        comparison["loss_within_tolerance"] = (delta <= tolerance)
        if not comparison["loss_within_tolerance"]:
            all_pass = False

    return comparison, all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q31A reproducibility test: run same config twice, compare results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config to test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of sequential runs (default: 2)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0001,
        help="Absolute tolerance for loss comparison (default: 0.0001)",
    )
    args = parser.parse_args()

    if args.runs != 2:
        print(
            f"[Q31A] WARNING: --runs={args.runs} requested; only 2 is supported in Q31A. "
            f"Running 2 runs.",
            flush=True,
        )

    print("=" * 70, flush=True)
    print("Q31A — Reproducibility Test", flush=True)
    print(f"Config:    {args.config}", flush=True)
    print(f"Runs:      2", flush=True)
    print(f"Tolerance: {args.tolerance}", flush=True)
    print("=" * 70, flush=True)

    # ── Run 1 ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}", flush=True)
    print(f"[Q31A] RUN 1 / 2", flush=True)
    print(f"{'─' * 60}", flush=True)

    rc1, path1, stdout1 = _run_once(args.config)

    print(f"\n[Q31A] Run 1 complete — return_code={rc1}, result_path={path1}", flush=True)

    # ── Run 2 ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}", flush=True)
    print(f"[Q31A] RUN 2 / 2", flush=True)
    print(f"{'─' * 60}", flush=True)

    rc2, path2, stdout2 = _run_once(args.config)

    print(f"\n[Q31A] Run 2 complete — return_code={rc2}, result_path={path2}", flush=True)

    # ── Load result JSONs ─────────────────────────────────────────────────────
    print(f"\n{'─' * 60}", flush=True)
    print(f"[Q31A] Loading result files...", flush=True)

    missing: list[str] = []

    def _load_json(path: str | None, run_label: str) -> dict:
        if path is None:
            missing.append(f"{run_label}: path not found in runner output")
            return {}
        if not os.path.exists(path):
            missing.append(f"{run_label}: file not found at {path}")
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    r1 = _load_json(path1, "Run 1")
    r2 = _load_json(path2, "Run 2")

    if missing:
        print("[Q31A] ERROR: Could not load result files:", flush=True)
        for m in missing:
            print(f"  {m}", flush=True)
        notes = "Result JSON files could not be loaded: " + "; ".join(missing)
        _write_report(
            config=args.config,
            tolerance=args.tolerance,
            result_files=[path1 or "", path2 or ""],
            comparison={},
            overall_pass=False,
            notes=notes,
        )
        print(f"\nQ31A_REPRODUCIBILITY_TEST: FAIL", flush=True)
        sys.exit(1)

    # ── Compare ───────────────────────────────────────────────────────────────
    comparison, overall_pass = _compare_results(r1, r2, args.tolerance)

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n[Q31A] Comparison:", flush=True)
    print(f"  {'Field':<30} {'Run 1':<20} {'Run 2':<20} {'Match'}", flush=True)
    print(f"  {'─'*30} {'─'*20} {'─'*20} {'─'*10}", flush=True)

    def _val(result: dict, field: str) -> str:
        parts = field.split(".")
        v = result
        for p in parts:
            v = v.get(p, None) if isinstance(v, dict) else None
        return str(v)

    rows = [
        ("status",                 "status",          "status"),
        ("return_code",            "return_code",     "return_code"),
        ("metrics.smoke_pass",     "metrics.smoke_pass", "metrics.smoke_pass"),
        ("metrics.trainable_params", "metrics.trainable_params", "metrics.trainable_params"),
        ("metrics.loss",           "metrics.loss",    "metrics.loss"),
    ]

    for label, f1key, f2key in rows:
        v1 = _val(r1, f1key)
        v2 = _val(r2, f2key)
        # Get check key
        check_key_map = {
            "status":               "status_match",
            "return_code":          "return_code_match",
            "metrics.smoke_pass":   "smoke_pass_match",
            "metrics.trainable_params": "trainable_params_match",
            "metrics.loss":         "loss_within_tolerance",
        }
        ck = check_key_map.get(label, "")
        result_flag = comparison.get(ck)
        if result_flag is None:
            flag_str = "SKIP"
        elif result_flag:
            flag_str = "PASS"
        else:
            flag_str = "FAIL"
        print(f"  {label:<30} {v1:<20} {v2:<20} {flag_str}", flush=True)

    if comparison.get("loss_delta") is not None:
        print(
            f"\n  loss_delta = {comparison['loss_delta']:.8f} "
            f"(tolerance = {args.tolerance})",
            flush=True,
        )

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes_parts: list[str] = []
    if r1.get("git_commit") == "unknown":
        notes_parts.append(
            "git_commit=unknown (git not available in container; "
            "pass QSTRATA_GIT_COMMIT env var to inject commit SHA)"
        )
    if not overall_pass:
        failed_checks = [k for k, v in comparison.items() if v is False]
        notes_parts.append(f"Failed checks: {', '.join(failed_checks)}")

    notes = ". ".join(notes_parts) if notes_parts else "All checks passed."

    # ── Write reproducibility report ──────────────────────────────────────────
    _write_report(
        config=args.config,
        tolerance=args.tolerance,
        result_files=[path1 or "", path2 or ""],
        comparison=comparison,
        overall_pass=overall_pass,
        notes=notes,
    )

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    if overall_pass:
        print("Q31A_REPRODUCIBILITY_TEST: PASS", flush=True)
        print(
            "Both runs completed with matching status, smoke_pass, "
            "trainable_params, and loss within tolerance.",
            flush=True,
        )
    else:
        print("Q31A_REPRODUCIBILITY_TEST: FAIL", flush=True)
        print(
            "One or more comparison checks failed. See report for details.",
            flush=True,
        )
    print("=" * 70, flush=True)

    sys.exit(0 if overall_pass else 1)


def _write_report(
    config: str,
    tolerance: float,
    result_files: list,
    comparison: dict,
    overall_pass: bool,
    notes: str,
) -> None:
    """Write the Q31A reproducibility report JSON."""
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    report = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "config":       config,
        "runs":         2,
        "tolerance":    tolerance,
        "result_files": result_files,
        "comparison":   comparison,
        "pass":         overall_pass,
        "notes":        notes,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Q31A] Reproducibility report: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
