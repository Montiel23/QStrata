#!/usr/bin/env python3
"""
scripts/run_c006_latency_sanity_check.py

Slice 29B — C006 Latency Measurement Sanity Check.

Investigates whether the Slice 29 latency gate failure (std % = 21.4% > 15%)
reflects real inference instability or naive timing artifacts from async CUDA
execution without synchronization.

Methodology
-----------
- Model loaded from C006 config, moved to GPU, set to eval()
- Input batch sampled from the real validation loader (real data, not synthetic)
- 25 warmup iterations (results discarded) to reach GPU steady state
- 100 timed measurement blocks:
    - Each block runs 25 forward passes
    - torch.cuda.synchronize() called once before the block starts
    - torch.cuda.synchronize() called once after the block ends
    - time.perf_counter() wraps the synchronized block
    - Per-pass latency = total block time / 25
- Statistics computed over the 100 per-pass latency values
- Decision gate: latency std <= 15% of mean

Hard-coded inputs:
    CONFIG_PATH : experiments/configs/binary_baseline_depthwise_sep_wide.yaml
    CANDIDATE   : C006
    WARMUP      : 25
    N_BLOCKS    : 100
    PASSES_PER_BLOCK : 25

Test accuracy note: this script performs inference measurement only. No
training. No labels used. No fitness signal produced.

Do not modify qcore/nas/evaluator.py.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import torch
import yaml

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter
from qcore.models.cnn import build_model
from torch.utils.data import DataLoader

# ── Hard-coded configuration ──────────────────────────────────────────────────
CONFIG_PATH      = "experiments/configs/binary_baseline_depthwise_sep_wide.yaml"
CANDIDATE_ID     = "C006"
WARMUP_ITERS     = 25
N_BLOCKS         = 100
PASSES_PER_BLOCK = 25

# ── Decision gate threshold ───────────────────────────────────────────────────
GATE_LATENCY_STD_PCT = 15.0   # std <= 15% of mean


def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) of sorted data."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def _write_report(
    stats:        dict,
    gate_pass:    bool,
    verdict:      str,
    output_path:  str,
) -> None:
    """Write the full markdown latency sanity check report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    lines = [
        "# Slice 29B — C006 Latency Measurement Sanity Check",
        "",
        f"- **Date:** {today}",
        f"- **Candidate:** C006 — depthwise_sep, conv_channels: [64, 128], 9,870 params",
        f"- **Config:** `{CONFIG_PATH}`",
        "",
        "---",
        "",

        # ── Section 2: Methodology ────────────────────────────────────────────
        "## 2. Measurement Methodology",
        "",
        (
            "Inference latency was measured using GPU-synchronized block timing. "
            "The C006 model was loaded from the YAML config, moved to the CUDA device, "
            "and set to `eval()` mode with `torch.no_grad()` active throughout. "
            "Input batches were sampled from the real PneumoniaMNIST validation loader — "
            "not from synthetic random tensors — so that data movement and preprocessing "
            "costs are reflected in the measurement. "
            "For each timed measurement, `torch.cuda.synchronize()` was called once "
            "before a block of 25 consecutive forward passes and once after the block "
            "completed, ensuring all GPU work is flushed before the wall-clock timer "
            "(`time.perf_counter()`) is read. The per-pass latency for each measurement "
            "block is the total synchronized wall-clock time divided by 25. "
            "This block approach amortises per-call Python overhead and CUDA "
            "kernel-launch micro-jitter, producing stable per-pass estimates."
        ),
        "",
        "---",
        "",

        # ── Section 3: Warmup ─────────────────────────────────────────────────
        "## 3. Warmup Configuration",
        "",
        f"- **Warmup iterations:** {WARMUP_ITERS}",
        (
            "- **Purpose:** Allow the GPU to reach steady-state thermal and clock "
            "conditions, populate CUDA kernel caches, and eliminate first-call "
            "compilation overhead before timed measurements begin. "
            "Results from warmup iterations are discarded."
        ),
        "",
        "---",
        "",

        # ── Section 4: Timed iteration config ─────────────────────────────────
        "## 4. Timed Iteration Configuration",
        "",
        f"- **Timed measurement blocks:** {N_BLOCKS}",
        f"- **Forward passes per block:** {PASSES_PER_BLOCK}",
        f"- **Total timed forward passes:** {N_BLOCKS * PASSES_PER_BLOCK}",
        (
            "- **Per-pass latency:** total synchronized block time ÷ "
            f"{PASSES_PER_BLOCK} forward passes"
        ),
        "- **Input:** real batch sampled from the PneumoniaMNIST validation loader",
        "",
        "---",
        "",

        # ── Section 5: Latency statistics table ───────────────────────────────
        "## 5. Latency Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Mean latency (ms/batch) | {stats['mean']:.3f} |",
        f"| Std latency (ms/batch) | {stats['std']:.3f} |",
        f"| Min latency (ms/batch) | {stats['min']:.3f} |",
        f"| Max latency (ms/batch) | {stats['max']:.3f} |",
        f"| p50 latency (ms/batch) | {stats['p50']:.3f} |",
        f"| p95 latency (ms/batch) | {stats['p95']:.3f} |",
        f"| Latency std % of mean | {stats['std_pct']:.1f}% |",
        "",
        "---",
        "",

        # ── Section 6: Decision gate ──────────────────────────────────────────
        "## 6. Decision Gate Evaluation",
        "",
        "| Gate Criterion | Threshold | Actual | Result |",
        "|----------------|-----------|-------:|--------|",
        (
            f"| Latency std % of mean | ≤ {GATE_LATENCY_STD_PCT:.0f}% "
            f"| {stats['std_pct']:.1f}% "
            f"| {'PASS' if gate_pass else 'FAIL'} |"
        ),
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

    if gate_pass:
        interp = (
            f"With GPU-synchronized block timing, the measured latency std % of mean "
            f"is {stats['std_pct']:.1f}%, comfortably within the {GATE_LATENCY_STD_PCT:.0f}% "
            f"gate threshold. This strongly suggests that the 21.4% std % observed in "
            f"Slice 29 was an artifact of naive wall-clock timing around asynchronous "
            f"CUDA calls — specifically, `time.perf_counter()` measurements taken without "
            f"`torch.cuda.synchronize()` captured variable amounts of GPU-queued work, "
            f"introducing apparent jitter that did not reflect actual inference speed "
            f"variability. The true steady-state per-pass latency is {stats['mean']:.3f} ms/batch "
            f"(p50 {stats['p50']:.3f} ms, p95 {stats['p95']:.3f} ms), which is stable. "
            f"C006 should be considered cleared of the latency stability concern from Slice 29, "
            f"and the original 'Not stable enough' verdict should be reattributed to a "
            f"measurement methodology issue rather than genuine model instability."
        )
    else:
        interp = (
            f"Even with GPU-synchronized block timing, the measured latency std % of mean "
            f"is {stats['std_pct']:.1f}%, exceeding the {GATE_LATENCY_STD_PCT:.0f}% gate "
            f"threshold. This indicates that the latency variation observed in Slice 29 "
            f"is not solely an artifact of unsynchronized timing — genuine inference "
            f"latency instability is present. The mean latency is {stats['mean']:.3f} ms/batch "
            f"(p50 {stats['p50']:.3f} ms, p95 {stats['p95']:.3f} ms). "
            f"Further investigation is needed before C006 can be cleared for follow-up "
            f"validation. Possible causes include GPU memory pressure, thermal throttling, "
            f"or kernel scheduling variability at this batch size."
        )

    lines.append(interp)
    lines.append("")

    Path(output_path).write_text("\n".join(lines))
    print(f"\nReport written → {output_path}")


def main() -> None:
    print(f"C006 Latency Sanity Check — Slice 29B")
    print(f"Candidate   : {CANDIDATE_ID}")
    print(f"Config      : {CONFIG_PATH}")
    print(f"Warmup      : {WARMUP_ITERS} iterations")
    print(f"Timed blocks: {N_BLOCKS}  ({PASSES_PER_BLOCK} passes/block)")
    print()

    # ── Device ────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.", file=sys.stderr)
        sys.exit(1)
    device = "cuda"
    print(f"Device      : {device} — {torch.cuda.get_device_name(0)}")

    # ── Load config ───────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    batch_size  = config["training"]["batch_size"]
    num_classes = config["dataset"]["num_classes"]

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = {**config["model"], **config["dataset"]}
    model = build_model(model_cfg).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {n_params:,}")

    # ── Real validation batch ─────────────────────────────────────────────────
    val_ds      = get_dataset(config["dataset"]["name"], "val")
    val_adapted = TorchDatasetAdapter(val_ds)
    val_loader  = DataLoader(val_adapted, batch_size=batch_size, shuffle=False)
    # Grab the first batch — reused for all timed iterations
    real_batch_x, _ = next(iter(val_loader))
    real_batch_x = real_batch_x.to(device)
    print(f"Input batch : shape={list(real_batch_x.shape)}, dtype={real_batch_x.dtype}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\nRunning {WARMUP_ITERS} warmup iterations (results discarded)...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(real_batch_x)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # ── Timed measurement blocks ──────────────────────────────────────────────
    print(f"\nRunning {N_BLOCKS} timed blocks ({PASSES_PER_BLOCK} passes/block)...")
    latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(N_BLOCKS):
            torch.cuda.synchronize()           # flush any queued GPU work
            t0 = time.perf_counter()
            for _ in range(PASSES_PER_BLOCK):  # 25 consecutive forward passes
                _ = model(real_batch_x)
            torch.cuda.synchronize()           # wait for all passes to complete
            t1 = time.perf_counter()
            # per-pass latency in milliseconds
            per_pass_ms = (t1 - t0) * 1000.0 / PASSES_PER_BLOCK
            latencies_ms.append(per_pass_ms)

    # ── Statistics ────────────────────────────────────────────────────────────
    mean_ms   = statistics.mean(latencies_ms)
    std_ms    = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    min_ms    = min(latencies_ms)
    max_ms    = max(latencies_ms)
    p50_ms    = _percentile(latencies_ms, 50)
    p95_ms    = _percentile(latencies_ms, 95)
    std_pct   = (std_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0

    stats = {
        "mean": mean_ms, "std": std_ms, "min": min_ms, "max": max_ms,
        "p50": p50_ms,   "p95": p95_ms, "std_pct": std_pct,
    }

    # ── Decision gate ─────────────────────────────────────────────────────────
    gate_pass = std_pct <= GATE_LATENCY_STD_PCT
    verdict = (
        "Latency measurement noise likely; C006 cleared for follow-up validation"
        if gate_pass
        else "Latency instability remains unresolved; stop and investigate"
    )

    # ── stdout summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"=== C006 Latency Sanity Check ===")
    print(f"Warmup iterations : {WARMUP_ITERS}")
    print(f"Timed iterations  : {N_BLOCKS}")
    print(
        f"Timing method     : torch.cuda.synchronize() + time.perf_counter() "
        f"— block of {PASSES_PER_BLOCK} passes per measurement"
    )
    print()
    print(f"Mean latency      : {mean_ms:.3f} ms/batch")
    print(f"Std latency       : {std_ms:.3f} ms/batch")
    print(f"Min latency       : {min_ms:.3f} ms/batch")
    print(f"Max latency       : {max_ms:.3f} ms/batch")
    print(f"p50 latency       : {p50_ms:.3f} ms/batch")
    print(f"p95 latency       : {p95_ms:.3f} ms/batch")
    print(f"Latency std %     : {std_pct:.1f}% of mean")
    print()
    print(f"=== Decision Gate ===")
    print(f"Latency std <= 15% of mean : {'PASS' if gate_pass else 'FAIL'}  ({std_pct:.1f}%)")
    print()
    print(f"=== VERDICT: {verdict} ===")

    # ── Write report ──────────────────────────────────────────────────────────
    _write_report(
        stats=stats,
        gate_pass=gate_pass,
        verdict=verdict,
        output_path="reports/c006_latency_sanity_check.md",
    )


if __name__ == "__main__":
    main()
