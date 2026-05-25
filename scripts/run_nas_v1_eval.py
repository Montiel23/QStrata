#!/usr/bin/env python3
"""
scripts/run_nas_v1_eval.py

NAS v1 evaluation entry point — Slice 25.

Orchestrates the full sequential evaluation sweep for the v1 candidate set.
Loads the sweep YAML config, evaluates each candidate in order using
evaluate_candidate(), passes all results to the Pareto ranker, prints a
terminal summary, and saves the markdown report (and optionally a CSV).

Usage
-----
    python scripts/run_nas_v1_eval.py --config <path/to/sweep.yaml>

Sweep YAML schema
-----------------
    seed: 42                             # int — must be 42 for v1

    candidates:
      - id: C001                         # str — candidate identifier
        config_path: experiments/configs/...yaml  # str — path to candidate YAML

    output:
      markdown: reports/nas_v1_eval.md   # str — markdown report path
      csv: reports/nas_v1_eval.csv       # str, optional — raw CSV path

Behaviour
---------
  - Candidates are evaluated strictly in list order (C001 → C004 → C006).
  - On any candidate failure the sweep stops immediately and exits 1.
  - No partial report is written on failure.
  - All errors are written to stderr; normal progress to stdout.
  - No MLflow, no dashboards, no databases, no monitoring stack.

Output files
------------
  reports/nas_v1_eval.md  — full markdown report: results table + Pareto front
  reports/nas_v1_eval.csv — optional raw metrics CSV (all candidates)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

import yaml

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.nas.evaluator import EvaluatorResult, evaluate_candidate
from qcore.nas.pareto import (
    compute_pareto_front,
    rank_candidates,
    render_markdown_table,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NAS v1 sequential evaluation sweep.\n\n"
            "Evaluates all candidates defined in the sweep YAML config, "
            "computes Pareto ranking, prints a terminal summary, and saves "
            "the markdown report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the sweep YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate the sweep config without running any evaluation. "
            "Checks that all candidate config_path files exist and that "
            "required output keys are present. Exits 0 on success."
        ),
    )
    return parser.parse_args()


# ── Config loading ────────────────────────────────────────────────────────────

def _load_sweep_config(config_path: str) -> dict:
    """Load and minimally validate the sweep YAML config."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        print(
            f"ERROR: failed to load sweep config '{config_path}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Required keys
    missing = [k for k in ("seed", "candidates", "output") if k not in cfg]
    if missing:
        print(
            f"ERROR: sweep config '{config_path}' is missing required keys: "
            f"{missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not cfg["candidates"]:
        print(
            f"ERROR: sweep config '{config_path}' has an empty candidates list.",
            file=sys.stderr,
        )
        sys.exit(1)

    for entry in cfg["candidates"]:
        for field in ("id", "config_path"):
            if field not in entry:
                print(
                    f"ERROR: candidate entry missing '{field}' field: {entry}",
                    file=sys.stderr,
                )
                sys.exit(1)

    if "markdown" not in cfg["output"]:
        print(
            "ERROR: sweep config 'output' section must include 'markdown' key.",
            file=sys.stderr,
        )
        sys.exit(1)

    return cfg


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_terminal_summary(
    ranked:       list,
    pareto_front: list,
) -> None:
    """Print a formatted summary table and Pareto front to stdout."""
    col_w = {
        "rank": 5, "id": 9, "block": 14, "channels": 12,
        "params": 8, "val": 12, "train": 12, "latency": 16, "pareto": 6,
    }

    def _header() -> str:
        return (
            f"  {'Rank':>{col_w['rank']}}  "
            f"{'Candidate':<{col_w['id']}}  "
            f"{'Block type':<{col_w['block']}}  "
            f"{'Channels':<{col_w['channels']}}  "
            f"{'Params':>{col_w['params']}}  "
            f"{'Best val acc':>{col_w['val']}}  "
            f"{'Train acc':>{col_w['train']}}  "
            f"{'Latency ms/b':>{col_w['latency']}}  "
            f"{'Pareto':>{col_w['pareto']}}"
        )

    front_set = set(pareto_front)

    print(f"\n{'='*90}")
    print("  NAS v1 Evaluation Results")
    print(f"{'='*90}")
    print(_header())
    print(f"  {'-'*85}")

    for rank, r in enumerate(ranked, start=1):
        marker = "✓" if r.candidate_id in front_set else ""
        print(
            f"  {rank:>{col_w['rank']}}  "
            f"{r.candidate_id:<{col_w['id']}}  "
            f"{r.block_type:<{col_w['block']}}  "
            f"{str(r.conv_channels):<{col_w['channels']}}  "
            f"{r.params:>{col_w['params']},}  "
            f"{r.best_val_acc:>{col_w['val']}.2f}%  "
            f"{r.final_train_acc:>{col_w['train']}.2f}%  "
            f"{r.latency_ms:>{col_w['latency']}.3f}  "
            f"{marker:>{col_w['pareto']}}"
        )

    print(f"\n  Pareto front: {', '.join(pareto_front)}")
    print(f"  (Objectives: maximize best_val_acc, minimize params, minimize latency_ms)")
    print(f"  test_acc_analysis_only not shown — for post-hoc analysis only.")
    print(f"{'='*90}\n")


# ── Report writing ────────────────────────────────────────────────────────────

def _write_markdown_report(
    output_path:  str,
    markdown_table: str,
    pareto_front: list,
    ranked:       list,
    seed:         int,
) -> None:
    """Write the full markdown report to output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    front_lines = "\n".join(
        f"- **{cid}** — "
        + next(
            f"{r.block_type} {r.conv_channels}, "
            f"best val acc {r.best_val_acc:.2f}%, "
            f"{r.params:,} params, "
            f"{r.latency_ms:.3f} ms/batch"
            for r in ranked
            if r.candidate_id == cid
        )
        for cid in pareto_front
    )

    report = textwrap.dedent(f"""\
        # NAS v1 Evaluation Report

        - **Protocol:** nas_benchmark_protocol_v1 — seed {seed}, best-val checkpoint selection
        - **Candidates:** {", ".join(r.candidate_id for r in ranked)}
        - **Objectives:** maximize best_val_acc, minimize params, minimize latency_ms

        ---

        ## Results

        {markdown_table}

        ---

        ## Pareto Front

        The following candidates are not dominated by any other on all three objectives:

        {front_lines}

        ---

        ## Notes

        - test_acc_analysis_only is reported in the table for post-hoc analysis only.
          It must not be used as a fitness signal or in Pareto dominance calculations.
        - Evaluation order: {", ".join(r.candidate_id for r in ranked)} (sequential, single GPU).
        - No dashboards, no databases, no tracking servers, no MLflow, no monitoring stack.
    """)

    try:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Markdown report saved → {output_path}")
    except Exception as exc:
        print(
            f"ERROR: failed to write markdown report to '{output_path}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_csv(output_path: str, results: list) -> None:
    """Write raw metrics CSV to output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_id", "block_type", "conv_channels", "params",
        "best_val_acc", "best_epoch", "final_train_acc",
        "test_acc_analysis_only", "mean_epoch_time", "latency_ms",
    ]

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "candidate_id":           r.candidate_id,
                    "block_type":             r.block_type,
                    "conv_channels":          str(r.conv_channels),
                    "params":                 r.params,
                    "best_val_acc":           f"{r.best_val_acc:.4f}",
                    "best_epoch":             r.best_epoch,
                    "final_train_acc":        f"{r.final_train_acc:.4f}",
                    "test_acc_analysis_only": f"{r.test_acc_analysis_only:.4f}",
                    "mean_epoch_time":        f"{r.mean_epoch_time:.4f}",
                    "latency_ms":             f"{r.latency_ms:.4f}",
                })
        print(f"CSV saved → {output_path}")
    except Exception as exc:
        print(
            f"ERROR: failed to write CSV to '{output_path}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Dry-run validation ────────────────────────────────────────────────────────

def _run_dry_run(config_path: str) -> None:
    """
    Validate the sweep YAML config without executing any evaluation.

    Checks:
      1. YAML loads successfully
      2. 'seed' key is present
      3. Each candidate has 'id' and 'config_path' and the config_path file exists
      4. 'output.markdown' key is present

    Prints a success line per candidate, then 'DRY RUN COMPLETE — config is valid',
    and exits with status 0.

    Does not call evaluate_candidate() under any condition.
    Does not write any output files.
    Does not train any models.
    """
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        print(
            f"ERROR: failed to load sweep config '{config_path}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if "seed" not in cfg:
        print("ERROR: sweep config missing 'seed' key.", file=sys.stderr)
        sys.exit(1)

    if not cfg.get("candidates"):
        print("ERROR: sweep config missing or empty 'candidates' list.", file=sys.stderr)
        sys.exit(1)

    for entry in cfg["candidates"]:
        cid = entry.get("id")
        if not cid:
            print(f"ERROR: candidate entry missing 'id' field: {entry}", file=sys.stderr)
            sys.exit(1)
        cp = entry.get("config_path")
        if not cp:
            print(
                f"ERROR: candidate '{cid}' missing 'config_path' field.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not os.path.isfile(cp):
            print(
                f"ERROR: candidate '{cid}' config_path not found on disk: '{cp}'",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[DRY RUN] {cid} OK — config_path: {cp}")

    if "output" not in cfg or "markdown" not in cfg.get("output", {}):
        print(
            "ERROR: sweep config 'output' section must include 'markdown' key.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("DRY RUN COMPLETE — config is valid")
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.dry_run:
        _run_dry_run(args.config)
        return  # _run_dry_run always calls sys.exit(); this line is unreachable

    sweep_cfg  = _load_sweep_config(args.config)

    seed       = int(sweep_cfg["seed"])
    candidates = sweep_cfg["candidates"]
    report_path = sweep_cfg["output"]["markdown"]
    csv_path    = sweep_cfg["output"].get("csv")   # optional

    print(f"\nNAS v1 sweep — {len(candidates)} candidate(s), seed={seed}")
    print(f"Evaluation order: {', '.join(c['id'] for c in candidates)}")

    # ── Sequential evaluation ─────────────────────────────────────────────────
    results: list[EvaluatorResult] = []

    for entry in candidates:
        cid         = entry["id"]
        config_path = entry["config_path"]

        try:
            result = evaluate_candidate(cid, config_path, seed=seed)
        except Exception as exc:
            print(
                f"\nERROR: evaluation of candidate '{cid}' failed:\n  {exc}",
                file=sys.stderr,
            )
            print(
                "Sweep halted — no report written.",
                file=sys.stderr,
            )
            sys.exit(1)

        results.append(result)

    # ── Pareto ranking ────────────────────────────────────────────────────────
    try:
        pareto_front = compute_pareto_front(results)
        ranked       = rank_candidates(results)
        md_table     = render_markdown_table(results, pareto_front)
    except Exception as exc:
        print(
            f"\nERROR: Pareto ranking failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Terminal summary ──────────────────────────────────────────────────────
    _print_terminal_summary(ranked, pareto_front)

    # ── Output files ──────────────────────────────────────────────────────────
    _write_markdown_report(report_path, md_table, pareto_front, ranked, seed)

    if csv_path:
        _write_csv(csv_path, results)

    print("Sweep complete.")


if __name__ == "__main__":
    main()
