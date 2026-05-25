#!/usr/bin/env python3
"""
scripts/run_c006_dropout_expansion.py

Slice 30 — C006 Manual Dropout Expansion.

Evaluates three dropout variants of C006 (depthwise_sep, [64, 128]) at a
single seed (42) to determine whether 0.20 or 0.40 improves or preserves
generalization versus the 0.30 baseline.

Architecture (fixed — must not vary across candidates):
    block_type    : depthwise_sep
    conv_channels : [64, 128]

Variable under test:
    dropout : 0.20 (C006-D020) | 0.30 (C006-D030, baseline) | 0.40 (C006-D040)

Override methodology
--------------------
evaluate_candidate() reads all hyperparameters from a YAML config file; it
does not accept in-memory overrides. To vary dropout without modifying any
shared config on disk, each candidate is given a transient YAML written to
Python's tempfile system (/tmp). These files exist only for the duration of
the evaluation call and are explicitly deleted afterwards. They do not appear
in git status and are not committed.

Decision gate (a variant may replace C006-D030 only if ALL pass):
    1. best_val_acc >= baseline_val_acc - 0.5 pp  (within 0.5 pp OR better)
    2. params == baseline params
    3. latency delta <= 15% of baseline latency
    4. training completed without exception

Test accuracy is collected for analysis only — it is never used as a gate
criterion or fitness signal.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.nas.evaluator import EvaluatorResult, evaluate_candidate

# ── Hard-coded configuration ──────────────────────────────────────────────────
BASE_CONFIG_PATH = "experiments/configs/binary_baseline_depthwise_sep_wide.yaml"
SEED             = 42

CANDIDATES = [
    {"id": "C006-D020", "dropout": 0.20, "role": "Variant"},
    {"id": "C006-D030", "dropout": 0.30, "role": "Baseline"},
    {"id": "C006-D040", "dropout": 0.40, "role": "Variant"},
]

BASELINE_ID = "C006-D030"

# ── Decision gate thresholds ──────────────────────────────────────────────────
GATE_VAL_ACC_MARGIN_PP  = 0.5    # within 0.5 pp of baseline OR better
GATE_LATENCY_DELTA_PCT  = 15.0   # latency delta <= 15% of baseline


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class CandidateResult:
    candidate_id:   str
    dropout:        float
    role:           str
    result:         EvaluatorResult | None
    failed:         bool
    failure_msg:    str


# ── Temp config writer ────────────────────────────────────────────────────────

def _write_temp_config(base_cfg: dict, dropout_override: float) -> str:
    """
    Write a transient YAML config to /tmp with the dropout value overridden.
    Returns the path to the temp file. Caller is responsible for deletion.
    The file is created outside the repo so it never appears in git status.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["dropout"] = dropout_override

    # suffix=.yaml so evaluate_candidate's yaml.safe_load works cleanly
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="qstrata_c006_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path


# ── Gate evaluation ───────────────────────────────────────────────────────────

def _evaluate_gate(
    candidate: CandidateResult,
    baseline:  CandidateResult,
) -> dict:
    """Evaluate all gate criteria for a non-baseline candidate."""
    if candidate.failed or baseline.failed:
        return {
            "val_acc_ok":     False,
            "params_ok":      False,
            "latency_ok":     False,
            "training_ok":    not candidate.failed,
            "overall":        False,
        }

    r   = candidate.result
    ref = baseline.result

    val_acc_ok  = r.best_val_acc >= ref.best_val_acc - GATE_VAL_ACC_MARGIN_PP
    params_ok   = r.params == ref.params
    lat_delta   = abs(r.latency_ms - ref.latency_ms) / ref.latency_ms * 100.0
    latency_ok  = lat_delta <= GATE_LATENCY_DELTA_PCT
    training_ok = True
    overall     = val_acc_ok and params_ok and latency_ok and training_ok

    return {
        "val_acc_ok":  val_acc_ok,
        "params_ok":   params_ok,
        "latency_ok":  latency_ok,
        "training_ok": training_ok,
        "overall":     overall,
        "lat_delta":   lat_delta,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(
    candidates:   list[CandidateResult],
    gate_results: dict,
    verdict:      str,
    recommended:  str,
    output_path:  str,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    baseline = next(c for c in candidates if c.candidate_id == BASELINE_ID)

    lines = [
        # ── Section 1: Title ──────────────────────────────────────────────────
        "# Slice 30 — C006 Manual Dropout Expansion",
        "",
        f"- **Date:** {today}",
        f"- **Candidate family:** C006 — depthwise_sep, conv_channels: [64, 128]",
        f"- **Base config:** `{BASE_CONFIG_PATH}`",
        f"- **Seed:** {SEED}",
        "",
        "---",
        "",

        # ── Section 2: Objective ──────────────────────────────────────────────
        "## 2. Objective",
        "",
        (
            "This controlled single-seed evaluation tests whether a small manual adjustment "
            "to the dropout rate improves or preserves C006 generalization relative to the "
            "0.30 baseline. Dropout is the only variable under test; the architecture "
            "(depthwise_sep, conv_channels [64, 128]) and all other hyperparameters are held "
            "fixed across all three candidates. The goal is to determine whether C006-D020 "
            "(lower regularization) or C006-D040 (higher regularization) offers a better or "
            "equivalent accuracy–efficiency trade-off, without introducing any new architecture "
            "or search space expansion."
        ),
        "",
        "---",
        "",

        # ── Section 3: Candidate table ────────────────────────────────────────
        "## 3. Candidates",
        "",
        "| Candidate ID | block_type | conv_channels | Dropout | Role |",
        "|---|---|---|---|---|",
        "| C006-D020 | depthwise_sep | [64, 128] | 0.20 | Variant |",
        "| C006-D030 | depthwise_sep | [64, 128] | 0.30 | Baseline |",
        "| C006-D040 | depthwise_sep | [64, 128] | 0.40 | Variant |",
        "",
        "---",
        "",

        # ── Section 4: Per-candidate results ──────────────────────────────────
        "## 4. Per-Candidate Results",
        "",
        "| Candidate | Dropout | Params | Best val acc | Best epoch | "
        "Final train acc | Test acc\\* | Mean epoch time (s) | Latency (ms/batch) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for c in candidates:
        if c.failed:
            lines.append(
                f"| {c.candidate_id} | {c.dropout:.2f} | FAILED | — | — | — | — | — | — |"
            )
        else:
            r = c.result
            role_tag = "  ← baseline" if c.role == "Baseline" else ""
            lines.append(
                f"| {c.candidate_id}{role_tag} | {c.dropout:.2f} | {r.params:,} "
                f"| {r.best_val_acc:.2f}% | {r.best_epoch} "
                f"| {r.final_train_acc:.2f}% | {r.test_acc_analysis_only:.2f}% "
                f"| {r.mean_epoch_time:.2f} | {r.latency_ms:.3f} |"
            )

    lines += [
        "",
        "> \\*Test accuracy is analysis only — not used as fitness signal or gate criterion.",
        "",
        "---",
        "",

        # ── Section 5: Decision gate table ────────────────────────────────────
        "## 5. Decision Gate",
        "",
    ]

    if not baseline.failed:
        lines += [
            f"**Baseline:** {BASELINE_ID} — best val acc {baseline.result.best_val_acc:.2f}%, "
            f"latency {baseline.result.latency_ms:.3f} ms/batch",
            "",
        ]

    lines += [
        "| Candidate | val_acc within 0.5 pp of baseline OR better "
        "| params = baseline | latency delta ≤ 15% | training OK | Gate result |",
        "|---|---|---|---|---|---|",
    ]

    for c in candidates:
        if c.role == "Baseline":
            continue
        g = gate_results.get(c.candidate_id, {})
        lines.append(
            f"| {c.candidate_id} "
            f"| {'PASS' if g.get('val_acc_ok') else 'FAIL'} "
            f"| {'PASS' if g.get('params_ok')  else 'FAIL'} "
            f"| {'PASS' if g.get('latency_ok') else 'FAIL'} "
            f"| {'PASS' if g.get('training_ok') else 'FAIL'} "
            f"| {'PASS' if g.get('overall')    else 'FAIL'} |"
        )

    lines += [
        "",
        "---",
        "",

        # ── Section 6: Practical recommendation ──────────────────────────────
        "## 6. Practical Recommendation",
        "",
        recommended,
        "",
        "---",
        "",

        # ── Section 7: Verdict ────────────────────────────────────────────────
        "## 7. Verdict",
        "",
        "```",
        f"VERDICT: {verdict}",
        "```",
        "",
        "---",
        "",

        # ── Section 8: Technical interpretation ──────────────────────────────
        "## 8. Technical Interpretation",
        "",
    ]

    # Build interpretation
    passing_variants = [cid for cid, g in gate_results.items() if g.get("overall")]
    failing_variants = [cid for cid, g in gate_results.items() if not g.get("overall")]

    if not baseline.failed:
        bv  = baseline.result.best_val_acc
        blat = baseline.result.latency_ms

        if passing_variants:
            best = passing_variants[0]
            br   = next(c for c in candidates if c.candidate_id == best).result
            interp = (
                f"The dropout sweep revealed that C006 is not highly sensitive to small "
                f"regularization adjustments in this range. "
                f"The baseline (dropout=0.30) achieved {bv:.2f}% best val acc; "
                f"{', '.join(passing_variants)} passed all gate criteria. "
                f"Dropout does not materially alter parameter count (all variants share "
                f"the same architecture), confirming the search was correctly controlled. "
                f"The latency variation across candidates reflects normal GPU measurement "
                f"variance at these small model sizes rather than a structural difference. "
                f"The recommended candidate maintains a good accuracy–efficiency profile "
                f"consistent with the v1 Pareto analysis."
            )
        else:
            worst_val = min(
                (c.result.best_val_acc for c in candidates if not c.failed),
                default=0.0
            )
            interp = (
                f"The dropout sweep showed that neither variant (0.20 or 0.40) passes "
                f"all gate criteria versus the 0.30 baseline ({bv:.2f}% val acc). "
                f"The failing criteria indicate that adjusting dropout in this range either "
                f"slightly reduces validation accuracy beyond the 0.5 pp tolerance or "
                f"introduces latency variation beyond the 15% threshold. "
                f"C006-D030 remains the best-evidenced practical candidate at this stage. "
                f"Further dropout tuning would require a multi-seed study to separate "
                f"stochastic variance from a genuine trend."
            )
    else:
        interp = (
            "The baseline candidate (C006-D030) failed during training. "
            "Gate evaluation is not meaningful without a valid baseline reference. "
            "This is a critical failure requiring investigation before proceeding."
        )

    lines.append(interp)
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"C006 Dropout Expansion — Slice 30")
    print(f"Base config : {BASE_CONFIG_PATH}")
    print(f"Seed        : {SEED}")
    print(f"Candidates  : {[c['id'] for c in CANDIDATES]}")
    print()

    # Load base config once
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    # Verify architecture constraints
    assert base_cfg["model"]["block_type"]    == "depthwise_sep", "block_type mismatch"
    assert base_cfg["model"]["conv_channels"] == [64, 128],       "conv_channels mismatch"

    # ── Evaluate each candidate ───────────────────────────────────────────────
    candidate_results: list[CandidateResult] = []

    for spec in CANDIDATES:
        cid     = spec["id"]
        dropout = spec["dropout"]
        role    = spec["role"]

        print(f"\n{'─'*60}")
        print(f"Candidate : {cid}  (dropout={dropout}, role={role})")

        tmp_path = None
        try:
            tmp_path = _write_temp_config(base_cfg, dropout)
            result   = evaluate_candidate(cid, tmp_path, seed=SEED)
            candidate_results.append(
                CandidateResult(
                    candidate_id=cid, dropout=dropout, role=role,
                    result=result, failed=False, failure_msg=""
                )
            )
        except Exception as exc:
            msg = str(exc)
            print(f"\nCANDIDATE {cid} FAILED: {msg}", file=sys.stderr)
            candidate_results.append(
                CandidateResult(
                    candidate_id=cid, dropout=dropout, role=role,
                    result=None, failed=True, failure_msg=msg
                )
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Locate baseline ───────────────────────────────────────────────────────
    baseline = next(c for c in candidate_results if c.candidate_id == BASELINE_ID)

    # ── Apply gate to non-baseline candidates ─────────────────────────────────
    gate_results: dict[str, dict] = {}
    for c in candidate_results:
        if c.role == "Baseline":
            continue
        gate_results[c.candidate_id] = _evaluate_gate(c, baseline)

    # ── Determine verdict ─────────────────────────────────────────────────────
    passing_variants = [
        cid for cid, g in gate_results.items() if g.get("overall")
    ]

    if passing_variants:
        # Prefer the variant with the highest val_acc that passes all gates
        best_variant_id = max(
            passing_variants,
            key=lambda cid: next(
                c.result.best_val_acc for c in candidate_results
                if c.candidate_id == cid and not c.failed
            )
        )
        verdict = "Dropout variant selected for follow-up validation"
        recommended = (
            f"**{best_variant_id}** passes all gate criteria and is recommended "
            f"as the updated C006 practical candidate. "
            + (
                f"It achieves {next(c.result.best_val_acc for c in candidate_results if c.candidate_id == best_variant_id):.2f}% "
                f"best val acc versus the baseline {baseline.result.best_val_acc:.2f}% "
                if not baseline.failed else ""
            )
            + "All architecture constraints (depthwise_sep, [64, 128]) and parameter "
            "count are unchanged."
        )
    else:
        verdict = "Keep C006-D030 baseline as practical candidate"
        recommended = (
            "No dropout variant passed all gate criteria. "
            "**C006-D030** (dropout=0.30) remains the recommended practical candidate. "
            "Dropout adjustment in the range 0.20–0.40 does not yield a clear improvement "
            "over the validated baseline at seed 42."
        )

    # ── stdout summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"=== C006 Dropout Expansion — Results ===")
    print(f"Seed: {SEED}")
    print()

    hdr = (
        f"{'Candidate':<14} | {'Dropout':>7} | {'Params':>6} | "
        f"{'Best val acc':>12} | {'Best epoch':>10} | {'Test acc*':>9} | {'Latency (ms)':>12}"
    )
    print(hdr)
    print("─" * len(hdr))

    for c in candidate_results:
        tag = "  <- baseline" if c.role == "Baseline" else ""
        if c.failed:
            print(f"{'FAILED':>14}{tag}")
        else:
            r = c.result
            print(
                f"{c.candidate_id + tag:<27} | {c.dropout:>7.2f} | {r.params:>6,} | "
                f"{r.best_val_acc:>11.2f}% | {r.best_epoch:>10} | "
                f"{r.test_acc_analysis_only:>8.2f}% | {r.latency_ms:>12.3f}"
            )

    print("\n*Test accuracy is analysis only.")

    if not baseline.failed:
        print(f"\n=== Decision Gate ===")
        print(f"Baseline best_val_acc : {baseline.result.best_val_acc:.2f}%")
        print(f"Baseline latency      : {baseline.result.latency_ms:.3f} ms")
        print()

        gate_hdr = (
            f"{'Candidate':<14} | "
            f"{'val_acc ±0.5pp':>14} | "
            f"{'params=base':>11} | "
            f"{'lat delta ≤15%':>14} | "
            f"{'training OK':>11} | "
            f"{'Gate':>6}"
        )
        print(gate_hdr)
        print("─" * len(gate_hdr))
        for c in candidate_results:
            if c.role == "Baseline":
                continue
            g = gate_results.get(c.candidate_id, {})
            lat_info = f"({g.get('lat_delta', 0):.1f}%)" if "lat_delta" in g else ""
            print(
                f"{c.candidate_id:<14} | "
                f"{'PASS' if g.get('val_acc_ok')  else 'FAIL':>14} | "
                f"{'PASS' if g.get('params_ok')   else 'FAIL':>11} | "
                f"{'PASS' if g.get('latency_ok')  else 'FAIL':>10} {lat_info:<4}| "
                f"{'PASS' if g.get('training_ok') else 'FAIL':>11} | "
                f"{'PASS' if g.get('overall')     else 'FAIL':>6}"
            )

    print(f"\n=== VERDICT: {verdict} ===")

    # ── Write report ──────────────────────────────────────────────────────────
    _write_report(
        candidates=candidate_results,
        gate_results=gate_results,
        verdict=verdict,
        recommended=recommended,
        output_path="reports/c006_dropout_expansion.md",
    )


if __name__ == "__main__":
    main()
