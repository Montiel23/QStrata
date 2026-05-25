#!/usr/bin/env python3
"""
scripts/run_c006_stability_validation.py

Slice 29 — C006 Targeted Stability Validation.

Runs evaluate_candidate() for C006 across four fixed seeds, aggregates
results, applies the hard-coded decision gate, and writes the markdown
report to reports/c006_stability_validation.md.

Hard-coded inputs (not exposed as CLI arguments):
    SEEDS       : [42, 7, 123, 999]
    CONFIG_PATH : experiments/configs/binary_baseline_depthwise_sep_wide.yaml
    CANDIDATE   : C006

Decision gate thresholds:
    std(best_val_acc)          <= 1.0%
    No seed > 2.5% below mean val acc
    Latency std <= 15% of mean
    No training failures

Test accuracy is collected and reported for analysis only. It must not be
used as a gate criterion or fitness signal.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import date
from pathlib import Path

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.nas.evaluator import EvaluatorResult, evaluate_candidate

# ── Hard-coded configuration ──────────────────────────────────────────────────
CANDIDATE_ID = "C006"
CONFIG_PATH  = "experiments/configs/binary_baseline_depthwise_sep_wide.yaml"
SEEDS        = [42, 7, 123, 999]

# ── Decision gate thresholds ──────────────────────────────────────────────────
GATE_VAL_STD_MAX      = 1.0    # std(best_val_acc) <= 1.0%
GATE_BELOW_MEAN_MAX   = 2.5    # no seed > 2.5% below mean val acc
GATE_LATENCY_STD_PCT  = 15.0   # latency std <= 15% of mean


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

def _min(values: list[float]) -> float:
    return min(values)

def _max(values: list[float]) -> float:
    return max(values)


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(
    results:       list[EvaluatorResult],
    failures:      list[tuple[int, str]],   # (seed, error_message)
    agg:           dict,
    gate_results:  dict,
    verdict:       str,
    output_path:   str,
) -> None:
    """Write the full markdown stability validation report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        "# C006 Stability Validation Report",
        "",
        f"- **Slice:** 29",
        f"- **Date:** {today}",
        f"- **Candidate:** C006 — depthwise_sep, conv_channels: [64, 128], 9,870 params",
        f"- **Config:** `{CONFIG_PATH}`",
        f"- **Seeds:** {', '.join(str(s) for s in SEEDS)}",
        f"- **Protocol:** nas_benchmark_protocol_v1 — best-val checkpoint, seed applied before all ops",
        "",
        "---",
        "",
    ]

    # ── Per-seed results table ────────────────────────────────────────────────
    lines += [
        "## Per-Seed Results",
        "",
        "| Seed | Params | Best val acc | Best epoch | Final train acc | "
        "Test acc\\* | Mean epoch time (s) | Latency (ms/batch) |",
        "|------|-------:|-------------:|-----------:|----------------:|"
        "----------:|--------------------:|-------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.candidate_id.split('_s')[1] if '_s' in r.candidate_id else SEEDS[results.index(r)]} "
            f"| {r.params:,} "
            f"| {r.best_val_acc:.2f}% "
            f"| {r.best_epoch} "
            f"| {r.final_train_acc:.2f}% "
            f"| {r.test_acc_analysis_only:.2f}% "
            f"| {r.mean_epoch_time:.2f} "
            f"| {r.latency_ms:.3f} |"
        )
    for seed, err in failures:
        lines.append(f"| {seed} | — | FAILED | — | — | — | — | — |")

    lines += [
        "",
        "> \\*Test accuracy is analysis only — not used as fitness signal or gate criterion.",
        "",
        "---",
        "",
    ]

    # ── Aggregate statistics table ────────────────────────────────────────────
    lines += [
        "## Aggregate Statistics",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|-----:|----:|----:|----:|",
        f"| Best val acc (%) | {agg['val_mean']:.2f}% | {agg['val_std']:.2f}% "
        f"| {agg['val_min']:.2f}% | {agg['val_max']:.2f}% |",
        f"| Test acc (%) [analysis only] | {agg['test_mean']:.2f}% | {agg['test_std']:.2f}% "
        f"| {agg['test_min']:.2f}% | {agg['test_max']:.2f}% |",
        f"| Latency (ms/batch) | {agg['lat_mean']:.3f} | {agg['lat_std']:.3f} "
        f"| {agg['lat_min']:.3f} | {agg['lat_max']:.3f} |",
        f"| Mean epoch time (s) | {agg['epoch_mean']:.2f} | {agg['epoch_std']:.2f} "
        f"| {agg['epoch_min']:.2f} | {agg['epoch_max']:.2f} |",
        "",
        "---",
        "",
    ]

    # ── Decision gate table ───────────────────────────────────────────────────
    lines += [
        "## Decision Gate",
        "",
        "| Gate Criterion | Threshold | Actual | Result |",
        "|----------------|-----------|--------|--------|",
        f"| std(best_val_acc) | ≤ {GATE_VAL_STD_MAX:.1f}% | {agg['val_std']:.2f}% "
        f"| {'PASS' if gate_results['val_std_ok'] else 'FAIL'} |",
        f"| No seed > 2.5% below mean val acc | ≤ {GATE_BELOW_MEAN_MAX:.1f}% gap "
        f"| {agg['max_below_mean']:.2f}% max gap "
        f"| {'PASS' if gate_results['below_mean_ok'] else 'FAIL'} |",
        f"| Latency std % of mean | ≤ {GATE_LATENCY_STD_PCT:.0f}% "
        f"| {agg['lat_std_pct']:.1f}% "
        f"| {'PASS' if gate_results['latency_ok'] else 'FAIL'} |",
        f"| No training failures | 0 failures | {len(failures)} failure(s) "
        f"| {'PASS' if gate_results['no_failures'] else 'FAIL'} |",
        "",
        "---",
        "",
    ]

    # ── Stability interpretation ──────────────────────────────────────────────
    passing = sum(1 for v in gate_results.values() if v)
    total   = len(gate_results)
    failed_names = [k for k, v in gate_results.items() if not v]

    if all(gate_results.values()):
        interp = (
            f"All {total} gate criteria passed. "
            f"C006 demonstrated consistent best validation accuracy across all four seeds "
            f"(mean {agg['val_mean']:.2f}%, std {agg['val_std']:.2f}%), "
            f"with no individual seed falling more than {agg['max_below_mean']:.2f}% below the mean. "
            f"Inference latency was stable with a coefficient of variation of {agg['lat_std_pct']:.1f}%, "
            f"well within the {GATE_LATENCY_STD_PCT:.0f}% threshold. "
            f"All four training runs completed without exception. "
            f"These results support proceeding with C006 as the recommended practical candidate."
        )
    else:
        gate_label_map = {
            "val_std_ok":    "std(best_val_acc) ≤ 1.0%",
            "below_mean_ok": "no seed > 2.5% below mean",
            "latency_ok":    "latency std ≤ 15% of mean",
            "no_failures":   "no training failures",
        }
        failed_labels = [gate_label_map.get(k, k) for k in failed_names]
        interp = (
            f"{passing} of {total} gate criteria passed. "
            f"The following criteria failed: {'; '.join(failed_labels)}. "
            f"C006 achieved a mean best validation accuracy of {agg['val_mean']:.2f}% "
            f"(std {agg['val_std']:.2f}%) across the four seeds. "
            f"The failed gate(s) indicate instability that should be investigated "
            f"before committing to C006 for follow-up work."
        )

    lines += [
        "## Stability Interpretation",
        "",
        interp,
        "",
        "---",
        "",
    ]

    # ── Explicit verdict ──────────────────────────────────────────────────────
    lines += [
        "## Verdict",
        "",
        f"```",
        f"VERDICT: {verdict}",
        f"```",
        "",
    ]

    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"C006 Stability Validation — Slice 29")
    print(f"Candidate : {CANDIDATE_ID}")
    print(f"Config    : {CONFIG_PATH}")
    print(f"Seeds     : {SEEDS}")
    print()

    results:  list[EvaluatorResult] = []
    failures: list[tuple[int, str]] = []

    for seed in SEEDS:
        cid = f"C006_s{seed}"
        try:
            result = evaluate_candidate(cid, CONFIG_PATH, seed=seed)
            results.append(result)
        except Exception as exc:
            msg = str(exc)
            print(f"\nSEED {seed} FAILED: {msg}", file=sys.stderr)
            failures.append((seed, msg))

    if not results:
        print(
            "\nERROR: all seeds failed — no results to aggregate.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    val_accs    = [r.best_val_acc           for r in results]
    test_accs   = [r.test_acc_analysis_only for r in results]
    latencies   = [r.latency_ms             for r in results]
    epoch_times = [r.mean_epoch_time        for r in results]

    val_mean  = _mean(val_accs)
    val_std   = _std(val_accs)
    val_min   = _min(val_accs)
    val_max   = _max(val_accs)

    test_mean = _mean(test_accs)
    test_std  = _std(test_accs)
    test_min  = _min(test_accs)
    test_max  = _max(test_accs)

    lat_mean  = _mean(latencies)
    lat_std   = _std(latencies)
    lat_min   = _min(latencies)
    lat_max   = _max(latencies)
    lat_std_pct = (lat_std / lat_mean * 100.0) if lat_mean > 0 else 0.0

    epoch_mean = _mean(epoch_times)
    epoch_std  = _std(epoch_times)
    epoch_min  = _min(epoch_times)
    epoch_max  = _max(epoch_times)

    max_below_mean = max(val_mean - v for v in val_accs)

    agg = {
        "val_mean":       val_mean,
        "val_std":        val_std,
        "val_min":        val_min,
        "val_max":        val_max,
        "test_mean":      test_mean,
        "test_std":       test_std,
        "test_min":       test_min,
        "test_max":       test_max,
        "lat_mean":       lat_mean,
        "lat_std":        lat_std,
        "lat_min":        lat_min,
        "lat_max":        lat_max,
        "lat_std_pct":    lat_std_pct,
        "epoch_mean":     epoch_mean,
        "epoch_std":      epoch_std,
        "epoch_min":      epoch_min,
        "epoch_max":      epoch_max,
        "max_below_mean": max_below_mean,
    }

    # ── Decision gate ─────────────────────────────────────────────────────────
    gate_results = {
        "val_std_ok":    val_std        <= GATE_VAL_STD_MAX,
        "below_mean_ok": max_below_mean <= GATE_BELOW_MEAN_MAX,
        "latency_ok":    lat_std_pct    <= GATE_LATENCY_STD_PCT,
        "no_failures":   len(failures)  == 0,
    }

    all_pass = all(gate_results.values())
    verdict = (
        "Stable enough for follow-up validation"
        if all_pass
        else "Not stable enough; stop and investigate"
    )

    # ── stdout summary ────────────────────────────────────────────────────────
    seeds_run = [str(r.candidate_id.split("_s")[1]) if "_s" in r.candidate_id else str(SEEDS[results.index(r)])
                 for r in results]

    print(f"\n{'='*55}")
    print(f"=== C006 Stability Validation — Aggregate ===")
    print(f"{'='*55}")
    print(f"Seeds run       : {', '.join(str(s) for s in SEEDS)}")
    print(f"Mean val acc    : {val_mean:.2f}%  (std: {val_std:.2f}%)")
    print(f"Min  val acc    : {val_min:.2f}%")
    print(f"Max  val acc    : {val_max:.2f}%")
    print(f"Mean test acc   : {test_mean:.2f}%  (std: {test_std:.2f}%)  [analysis only]")
    print(f"Mean latency    : {lat_mean:.3f} ms/batch  (std: {lat_std:.3f} ms/batch)")
    print(f"Latency std %   : {lat_std_pct:.1f}% of mean")
    print(f"Mean epoch time : {epoch_mean:.2f}s  (std: {epoch_std:.2f}s)")
    print()
    print(f"=== Decision Gate ===")
    print(f"std(best_val_acc) <= 1.0%          : {'PASS' if gate_results['val_std_ok']    else 'FAIL'}")
    print(f"No seed > 2.5% below mean val acc  : {'PASS' if gate_results['below_mean_ok'] else 'FAIL'}")
    print(f"Latency std <= 15% of mean         : {'PASS' if gate_results['latency_ok']    else 'FAIL'}")
    print(f"No training failures               : {'PASS' if gate_results['no_failures']   else 'FAIL'}")
    print()
    print(f"=== VERDICT: {verdict} ===")

    # ── Write markdown report ─────────────────────────────────────────────────
    _write_report(
        results=results,
        failures=failures,
        agg=agg,
        gate_results=gate_results,
        verdict=verdict,
        output_path="reports/c006_stability_validation.md",
    )


if __name__ == "__main__":
    main()
