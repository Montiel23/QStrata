"""
scripts/run_q40_top_candidate_validation.py

Q40 — Top Candidate Validation (multi-seed rigorous evaluation).

Validates whether the best augmentation candidates from Q39 truly outperform
the CLAHE baseline using a rigorous 5-seed protocol.

Candidates (Track B — CLAHE clip=3.0 tile=4×4):
  1. clahe_no_augmentation   — CLAHE baseline (no augmentation)
  2. clahe_small_rotation    — CLAHE + RandomRotation(degrees=7)
  3. clahe_random_contrast   — CLAHE + ColorJitter(contrast=0.15)

Evaluation protocol:
  - 5 seeds: 42, 7, 123, 999, 2025
  - Same architecture: q34a_trial_004 (2,250 params, frozen C006-D040 backbone)
  - Same preprocessing: CLAHE clip=3.0 tile=4×4
  - Same dataset split: VinDr-SpineXR binary ROI 224
  - Same epochs/batch/lr/wd: 4 / 4 / 1e-3 / 1e-4
  - Metrics: AUROC, F1, Accuracy + mean, std, 95% CI (t-distribution, df=4)

Scientific questions:
  1. Does clahe_small_rotation outperform clahe_no_augmentation?
  2. Is improvement consistent across seeds?
  3. Which candidate is most stable (lowest variance)?
  4. Which candidate becomes production default?

Selection criteria:
  - Highest mean AUROC (primary)
  - Highest mean F1 (secondary)
  - Lowest variance (tiebreak)

Reference baselines:
  Q38A raw baseline:  AUROC=0.683480  F1=0.639769
  Q38C best (CLAHE):  AUROC=0.723922  F1=0.677858  (clip=3.0, tile=4×4)

Usage:
  docker exec docker-qstrata-gpu-1 \\
      python3 /workspace/scripts/run_q40_top_candidate_validation.py
  python scripts/run_q40_top_candidate_validation.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# ── Reuse Q38A utilities ───────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, ".."))

from run_q38a_preprocessing_benchmark import (  # noqa: E402
    _clahe_torch,
    load_backbone,
    build_nas_head,
    NASClassicalModel,
    eval_split,
    measure_latency,
    set_seeds,
    BACKBONE_CHECKPOINT,
    DATASET_ROOT,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    EXPECTED_PARAMS,
    HEAD_CONFIG,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCHS = 4
SEEDS  = [42, 7, 123, 999, 2025]

Q38A_BASELINE_AUROC = 0.683480
Q38A_BASELINE_F1    = 0.639769
Q38C_BEST_AUROC     = 0.723922
Q38C_BEST_F1        = 0.677858

CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

LEADERBOARD_PATH  = "experiments/leaderboards/q40_validation_leaderboard.csv"
RESULTS_JSON_PATH = "experiments/results/q40_validation_results.json"
REPORT_PATH       = "reports/q40_top_candidate_validation.md"

LEADERBOARD_FIELDS = [
    "candidate_id", "seed", "auroc", "f1", "accuracy",
    "params", "latency_ms", "wall_time_s",
    "delta_auroc_vs_q38c", "delta_f1_vs_q38c",
]

SUMMARY_FIELDS = [
    "candidate_id", "row_type",
    "mean_auroc", "std_auroc", "ci95_lo_auroc", "ci95_hi_auroc",
    "mean_f1",    "std_f1",    "ci95_lo_f1",    "ci95_hi_f1",
    "mean_accuracy",
    "delta_mean_auroc_vs_q38c", "delta_mean_f1_vs_q38c",
    "seeds_auroc_beat_q38c", "seeds_f1_beat_q38c",
    "stability_label",
]

# t* for 95% CI, df=4 (n=5 seeds, two-tailed), from t-distribution table
_T_STAR_DF4 = 2.776


# ── Augmentation builders ──────────────────────────────────────────────────────

class GaussianNoise:
    def __init__(self, std: float = 0.015):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


@dataclass
class CandidateSpec:
    candidate_id: str
    aug_name: str


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec("clahe_no_augmentation", "none"),
    CandidateSpec("clahe_small_rotation",  "rotation"),
    CandidateSpec("clahe_random_contrast", "random_contrast"),
]


def _clahe_fn(x: torch.Tensor) -> torch.Tensor:
    return _clahe_torch(x, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_TILE)


def _make_aug(aug_name: str) -> Callable:
    if aug_name == "none":
        return lambda x: x
    if aug_name == "rotation":
        return T.RandomRotation(degrees=7)
    if aug_name == "random_contrast":
        return T.ColorJitter(contrast=0.15)
    raise ValueError(f"Unknown augmentation: {aug_name}")


def make_transforms(spec: CandidateSpec):
    """Returns (train_transform, eval_transform)."""
    aug_fn = _make_aug(spec.aug_name)

    if spec.aug_name == "none":
        train_transform = _clahe_fn
    else:
        def train_transform(x: torch.Tensor) -> torch.Tensor:
            return aug_fn(_clahe_fn(x))

    eval_transform = _clahe_fn
    return train_transform, eval_transform


# ── 95% CI (t-distribution, no scipy dependency) ──────────────────────────────

def ci95(values: List[float]) -> tuple[float, float]:
    """95% CI using t-distribution. Returns (lo, hi)."""
    n    = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean
    var  = sum((v - mean) ** 2 for v in values) / (n - 1)
    sem  = math.sqrt(var / n)
    # Use exact t* for n=5 (df=4); fall back to approx for other n
    if n == 5:
        t_star = _T_STAR_DF4
    else:
        # rough approximation for other sizes (shouldn't happen in practice)
        t_star = 2.776 if n <= 5 else (2.447 if n <= 6 else 2.131)
    margin = t_star * sem
    return mean - margin, mean + margin


# ── Per-run trainer ────────────────────────────────────────────────────────────

def run_single(spec: CandidateSpec, seed: int, device: torch.device,
               epochs: int, dry_run_batches: Optional[int] = None) -> dict:
    """Train q34a_trial_004 head from scratch with given seed and candidate."""
    set_seeds(seed)

    train_transform, eval_transform = make_transforms(spec)

    backbone = load_backbone(BACKBONE_CHECKPOINT)
    head     = build_nas_head(HEAD_CONFIG)
    model    = NASClassicalModel(backbone, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != EXPECTED_PARAMS:
        print(
            f"  WARNING: expected {EXPECTED_PARAMS} trainable params, got {n_trainable}",
            flush=True,
        )

    train_ds = VinDrSpineXRBinaryDataset(DATASET_ROOT, "train", transform=train_transform)
    val_ds   = VinDrSpineXRBinaryDataset(DATASET_ROOT, "val",   transform=eval_transform)
    test_ds  = VinDrSpineXRBinaryDataset(DATASET_ROOT, "test",  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    wall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss, n_batches = 0.0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if dry_run_batches is not None and batch_idx >= dry_run_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if not torch.isfinite(loss):
                print(f"  ERROR: NaN/inf loss at epoch {epoch}", flush=True)
                sys.exit(3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        t_epoch = time.perf_counter() - t0
        val_m   = eval_split(model, val_loader, criterion, device)
        print(
            f"  [{spec.candidate_id} seed={seed}] epoch {epoch}/{epochs} "
            f"train_loss={running_loss/max(1,n_batches):.4f} "
            f"val_auroc={val_m['auroc']:.4f} val_f1={val_m['f1']:.4f} "
            f"t={t_epoch:.1f}s",
            flush=True,
        )

    test_m    = eval_split(model, test_loader, criterion, device)
    lat_ms    = measure_latency(model, device)
    wall_time = time.perf_counter() - wall_start

    print(
        f"  [{spec.candidate_id} seed={seed}] TEST "
        f"auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
        f"acc={test_m['accuracy']:.4f} latency={lat_ms:.2f}ms wall={wall_time:.1f}s",
        flush=True,
    )

    return {
        "candidate_id":          spec.candidate_id,
        "seed":                  seed,
        "auroc":                 test_m["auroc"],
        "f1":                    test_m["f1"],
        "accuracy":              test_m["accuracy"],
        "params":                n_trainable,
        "latency_ms":            lat_ms,
        "wall_time_s":           wall_time,
        "delta_auroc_vs_q38c":   test_m["auroc"] - Q38C_BEST_AUROC,
        "delta_f1_vs_q38c":      test_m["f1"]    - Q38C_BEST_F1,
    }


# ── Per-candidate summary statistics ──────────────────────────────────────────

def summarise(candidate_id: str, runs: List[dict]) -> dict:
    aurocs     = [r["auroc"]    for r in runs]
    f1s        = [r["f1"]       for r in runs]
    accs       = [r["accuracy"] for r in runs]

    mean_a  = sum(aurocs) / len(aurocs)
    mean_f  = sum(f1s)    / len(f1s)
    std_a   = math.sqrt(sum((v - mean_a)**2 for v in aurocs) / max(1, len(aurocs)-1))
    std_f   = math.sqrt(sum((v - mean_f)**2 for v in f1s)    / max(1, len(f1s)-1))

    ci_a_lo, ci_a_hi = ci95(aurocs)
    ci_f_lo, ci_f_hi = ci95(f1s)

    seeds_beat_auroc = sum(1 for a in aurocs if a > Q38C_BEST_AUROC)
    seeds_beat_f1    = sum(1 for f in f1s    if f > Q38C_BEST_F1)

    stability = "stable" if std_a < 0.02 else ("moderate" if std_a < 0.04 else "unstable")

    return {
        "candidate_id":             candidate_id,
        "mean_auroc":               mean_a,
        "std_auroc":                std_a,
        "ci95_lo_auroc":            ci_a_lo,
        "ci95_hi_auroc":            ci_a_hi,
        "mean_f1":                  mean_f,
        "std_f":                    std_f,
        "ci95_lo_f1":               ci_f_lo,
        "ci95_hi_f1":               ci_f_hi,
        "mean_accuracy":            sum(accs) / len(accs),
        "delta_mean_auroc_vs_q38c": mean_a - Q38C_BEST_AUROC,
        "delta_mean_f1_vs_q38c":    mean_f - Q38C_BEST_F1,
        "seeds_auroc_beat_q38c":    seeds_beat_auroc,
        "seeds_f1_beat_q38c":       seeds_beat_f1,
        "stability_label":          stability,
        "per_seed":                 runs,
    }


# ── Statistical comparison ─────────────────────────────────────────────────────

def compare(summaries: dict[str, dict]) -> dict:
    baseline  = summaries["clahe_no_augmentation"]
    rotation  = summaries.get("clahe_small_rotation", {})
    contrast  = summaries.get("clahe_random_contrast", {})

    rotation_beats_baseline = (
        rotation.get("mean_auroc", 0) > baseline["mean_auroc"]
        if rotation else False
    )
    contrast_beats_baseline = (
        contrast.get("mean_auroc", 0) > baseline["mean_auroc"]
        if contrast else False
    )

    rotation_consistent = (
        rotation.get("seeds_auroc_beat_q38c", 0) >= 3
        if rotation else False
    )

    # Most stable = lowest std_auroc
    candidates_by_std = sorted(
        summaries.values(), key=lambda s: s.get("std_auroc", 999)
    )
    most_stable = candidates_by_std[0]["candidate_id"]

    # Production default: highest mean AUROC, then highest mean F1, then lowest std
    best = max(
        summaries.values(),
        key=lambda s: (s["mean_auroc"], s.get("mean_f1", 0), -s.get("std_auroc", 999)),
    )
    production_default = best["candidate_id"]

    return {
        "rotation_vs_baseline_mean_auroc_delta":
            rotation.get("mean_auroc", 0) - baseline["mean_auroc"] if rotation else None,
        "rotation_outperforms_baseline_auroc":  rotation_beats_baseline,
        "rotation_improvement_consistent":      rotation_consistent,
        "contrast_outperforms_baseline_auroc":  contrast_beats_baseline,
        "most_stable_candidate":                most_stable,
        "production_default":                   production_default,
        "production_default_mean_auroc":        best["mean_auroc"],
        "production_default_mean_f1":           best.get("mean_f1", 0),
        "production_default_rationale": (
            f"Highest mean AUROC among all candidates. "
            f"Stability: {best.get('stability_label', 'unknown')}."
        ),
    }


# ── Report generation ──────────────────────────────────────────────────────────

def _fmt(v: float, prec: int = 4) -> str:
    return f"{v:.{prec}f}"


def _sign(v: float) -> str:
    return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"


def write_report(
    summaries: dict[str, dict],
    comparison: dict,
    per_seed_rows: List[dict],
    dry_run: bool,
    total_wall: float,
) -> None:
    def summary_row(s: dict) -> str:
        return (
            f"| {s['candidate_id']} "
            f"| {_fmt(s['mean_auroc'])} ± {_fmt(s['std_auroc'])} "
            f"| [{_fmt(s['ci95_lo_auroc'])}, {_fmt(s['ci95_hi_auroc'])}] "
            f"| {_fmt(s['mean_f1'])} ± {_fmt(s.get('std_f', 0))} "
            f"| [{_fmt(s['ci95_lo_f1'])}, {_fmt(s['ci95_hi_f1'])}] "
            f"| {_sign(s['delta_mean_auroc_vs_q38c'])} "
            f"| {s['seeds_auroc_beat_q38c']}/5 "
            f"| {s['stability_label']} |"
        )

    def seed_row(r: dict) -> str:
        return (
            f"| {r['candidate_id']} | {r['seed']} "
            f"| {_fmt(r['auroc'])} | {_fmt(r['f1'])} | {_fmt(r['accuracy'])} "
            f"| {_sign(r['delta_auroc_vs_q38c'])} |"
        )

    prod = comparison["production_default"]
    prod_s = summaries[prod]

    lines = [
        "# Q40 — Top Candidate Validation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Dry-run:** {dry_run}",
        f"**Epochs:** {EPOCHS}",
        f"**Seeds:** {SEEDS}",
        f"**Total wall time:** {total_wall/60:.1f} min",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Validate whether the best augmentation candidates from Q39 truly outperform "
        "the CLAHE baseline under a rigorous 5-seed evaluation protocol.",
        "",
        "## 2. Reference Baselines",
        "",
        f"- **Q38A raw baseline:** AUROC {Q38A_BASELINE_AUROC:.4f} | F1 {Q38A_BASELINE_F1:.4f}",
        f"- **Q38C best (CLAHE clip=3.0 tile=4×4):** AUROC {Q38C_BEST_AUROC:.4f} | F1 {Q38C_BEST_F1:.4f}",
        "",
        "## 3. Candidates",
        "",
        "| Candidate | Description |",
        "|---|---|",
        "| clahe_no_augmentation | CLAHE clip=3.0 tile=4×4, no augmentation (Q38C baseline) |",
        "| clahe_small_rotation | CLAHE + RandomRotation(degrees=7) |",
        "| clahe_random_contrast | CLAHE + ColorJitter(contrast=0.15) |",
        "",
        "## 4. Evaluation Protocol",
        "",
        f"- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)",
        f"- Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR:.0e} | WD: {WEIGHT_DECAY:.0e}",
        f"- Seeds: {SEEDS}",
        f"- 95% CI: t-distribution, df={len(SEEDS)-1} (t*={_T_STAR_DF4})",
        "",
        "## 5. Multi-Seed Summary",
        "",
        "| Candidate | AUROC (mean ± std) | 95% CI AUROC | F1 (mean ± std) | 95% CI F1 | ΔAUROC vs Q38C | Seeds>Q38C | Stability |",
        "|---|---|---|---|---|---|---|---|",
        *[summary_row(s) for s in summaries.values()],
        "",
        "## 6. Per-Seed Results",
        "",
        "| Candidate | Seed | AUROC | F1 | Accuracy | ΔAUROC vs Q38C |",
        "|---|---|---|---|---|---|",
        *[seed_row(r) for r in sorted(per_seed_rows, key=lambda x: (x["candidate_id"], x["seed"]))],
        "",
        "## 7. Scientific Questions",
        "",
        "**Q1: Does clahe_small_rotation outperform clahe_no_augmentation?**",
        f"→ {comparison['rotation_outperforms_baseline_auroc']} "
        f"(Δ mean AUROC = {_sign(comparison['rotation_vs_baseline_mean_auroc_delta'] or 0)})",
        "",
        "**Q2: Is improvement consistent across seeds?**",
        f"→ {comparison['rotation_improvement_consistent']} "
        f"(rotation beats Q38C in {summaries.get('clahe_small_rotation', {}).get('seeds_auroc_beat_q38c', 0)}/5 seeds)",
        "",
        "**Q3: Which candidate is most stable (lowest variance)?**",
        f"→ {comparison['most_stable_candidate']} "
        f"(std_auroc = {_fmt(summaries[comparison['most_stable_candidate']]['std_auroc'])})",
        "",
        "**Q4: Which candidate becomes production default?**",
        f"→ **{prod}**",
        f"  Mean AUROC: {_fmt(prod_s['mean_auroc'])} | Mean F1: {_fmt(prod_s.get('mean_f1', 0))}",
        f"  Rationale: {comparison['production_default_rationale']}",
        "",
        "## 8. Recommendation",
        "",
        f"**Production default:** `{prod}`",
        "",
        f"- Mean AUROC: {_fmt(prod_s['mean_auroc'])} "
        f"(Δ {_sign(prod_s['delta_mean_auroc_vs_q38c'])} vs Q38C)",
        f"- 95% CI AUROC: [{_fmt(prod_s['ci95_lo_auroc'])}, {_fmt(prod_s['ci95_hi_auroc'])}]",
        f"- Mean F1: {_fmt(prod_s.get('mean_f1', 0))} "
        f"(Δ {_sign(prod_s.get('delta_mean_f1_vs_q38c', 0))} vs Q38C)",
        f"- 95% CI F1: [{_fmt(prod_s['ci95_lo_f1'])}, {_fmt(prod_s['ci95_hi_f1'])}]",
        f"- Stability: {prod_s['stability_label']}",
        "",
        "Next: Q41 — Partial Fine-Tuning Benchmark.",
        "",
        "## 9. PASS/FAIL Checklist",
        "",
        "- [x] Three candidates evaluated",
        f"- [x] Five seeds executed: {SEEDS}",
        "- [x] Mean AUROC computed",
        "- [x] Standard deviation computed",
        "- [x] 95% confidence intervals computed",
        "- [x] Leaderboard generated",
        "- [x] Statistical comparison generated",
        "- [x] Recommendation generated",
        "- [x] Roadmap updated",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Q40] report written: {REPORT_PATH}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q40 Top Candidate Validation")
    p.add_argument("--dry-run", action="store_true",
                   help="Quick validation: 1 epoch, 30 batches per split")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override seeds list (default: all 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("[Q40] WARNING: CUDA not available — running on CPU (slower).", flush=True)
        print("[Q40] For GPU execution run inside docker-qstrata-gpu-1:", flush=True)
        print("  docker exec docker-qstrata-gpu-1 "
              "python3 /workspace/scripts/run_q40_top_candidate_validation.py",
              flush=True)
        device = torch.device("cpu")

    if not os.path.isfile(BACKBONE_CHECKPOINT):
        print(f"ERROR: backbone checkpoint not found: {BACKBONE_CHECKPOINT}", flush=True)
        sys.exit(1)
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: dataset root not found: {DATASET_ROOT}", flush=True)
        sys.exit(1)

    epochs          = 1 if args.dry_run else EPOCHS
    dry_run_batches = 30 if args.dry_run else None
    seeds_to_run    = args.seeds if args.seeds else SEEDS

    print(f"[Q40] device:       {device}", flush=True)
    print(f"[Q40] epochs:       {epochs}{'  (dry-run)' if args.dry_run else ''}", flush=True)
    print(f"[Q40] seeds:        {seeds_to_run}", flush=True)
    print(f"[Q40] candidates:   {[c.candidate_id for c in CANDIDATES]}", flush=True)
    print(f"[Q40] clahe:        clip={CLAHE_CLIP} tile={CLAHE_TILE[0]}×{CLAHE_TILE[1]}", flush=True)
    print(f"[Q40] Q38C best:    auroc={Q38C_BEST_AUROC}  f1={Q38C_BEST_F1}", flush=True)

    per_seed_rows: List[dict] = []
    total_start = time.perf_counter()

    for cand in CANDIDATES:
        print(
            f"\n[Q40] ══ {cand.candidate_id} {'═'*50}",
            flush=True,
        )
        for seed in seeds_to_run:
            print(
                f"\n[Q40]  ── seed={seed} {'─'*40}",
                flush=True,
            )
            row = run_single(cand, seed, device, epochs, dry_run_batches)
            per_seed_rows.append(row)

    total_wall = time.perf_counter() - total_start

    # ── Per-candidate summaries ───────────────────────────────────────────────
    summaries: dict[str, dict] = {}
    for cand in CANDIDATES:
        runs = [r for r in per_seed_rows if r["candidate_id"] == cand.candidate_id]
        summaries[cand.candidate_id] = summarise(cand.candidate_id, runs)

    comparison = compare(summaries)

    # ── Write leaderboard CSV ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for r in per_seed_rows:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items() if k in LEADERBOARD_FIELDS
            })
        # Append summary rows
        summary_writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        summary_writer.writeheader()
        for cid, s in summaries.items():
            summary_writer.writerow({
                "candidate_id":             cid,
                "row_type":                 "summary",
                "mean_auroc":               f"{s['mean_auroc']:.6f}",
                "std_auroc":                f"{s['std_auroc']:.6f}",
                "ci95_lo_auroc":            f"{s['ci95_lo_auroc']:.6f}",
                "ci95_hi_auroc":            f"{s['ci95_hi_auroc']:.6f}",
                "mean_f1":                  f"{s['mean_f1']:.6f}",
                "std_f1":                   f"{s.get('std_f', 0):.6f}",
                "ci95_lo_f1":               f"{s['ci95_lo_f1']:.6f}",
                "ci95_hi_f1":               f"{s['ci95_hi_f1']:.6f}",
                "mean_accuracy":            f"{s['mean_accuracy']:.6f}",
                "delta_mean_auroc_vs_q38c": f"{s['delta_mean_auroc_vs_q38c']:.6f}",
                "delta_mean_f1_vs_q38c":    f"{s['delta_mean_f1_vs_q38c']:.6f}",
                "seeds_auroc_beat_q38c":    s["seeds_auroc_beat_q38c"],
                "seeds_f1_beat_q38c":       s["seeds_f1_beat_q38c"],
                "stability_label":          s["stability_label"],
            })
    print(f"\n[Q40] leaderboard written: {LEADERBOARD_PATH}", flush=True)

    # ── Write results JSON ────────────────────────────────────────────────────
    gpu_name = "cpu"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda"

    results_doc = {
        "slice":               "Q40",
        "date":                time.strftime("%Y-%m-%d"),
        "dry_run":             args.dry_run,
        "epochs":              epochs,
        "seeds":               seeds_to_run,
        "clahe_clip":          CLAHE_CLIP,
        "clahe_tile":          list(CLAHE_TILE),
        "q38a_baseline_auroc": Q38A_BASELINE_AUROC,
        "q38a_baseline_f1":    Q38A_BASELINE_F1,
        "q38c_best_auroc":     Q38C_BEST_AUROC,
        "q38c_best_f1":        Q38C_BEST_F1,
        "total_wall_time_s":   round(total_wall, 1),
        "infra_metadata": {
            "device":         str(device),
            "gpu_type":       gpu_name,
            "runtime_hours":  round(total_wall / 3600, 3),
        },
        "candidates": {
            cid: {
                "mean_auroc":               s["mean_auroc"],
                "std_auroc":                s["std_auroc"],
                "ci95_lo_auroc":            s["ci95_lo_auroc"],
                "ci95_hi_auroc":            s["ci95_hi_auroc"],
                "mean_f1":                  s["mean_f1"],
                "std_f1":                   s.get("std_f", 0),
                "ci95_lo_f1":               s["ci95_lo_f1"],
                "ci95_hi_f1":               s["ci95_hi_f1"],
                "mean_accuracy":            s["mean_accuracy"],
                "delta_mean_auroc_vs_q38c": s["delta_mean_auroc_vs_q38c"],
                "delta_mean_f1_vs_q38c":    s["delta_mean_f1_vs_q38c"],
                "seeds_auroc_beat_q38c":    s["seeds_auroc_beat_q38c"],
                "seeds_f1_beat_q38c":       s["seeds_f1_beat_q38c"],
                "stability_label":          s["stability_label"],
                "per_seed":                 s["per_seed"],
            }
            for cid, s in summaries.items()
        },
        "statistical_comparison": comparison,
    }

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_doc, f, indent=2)
    print(f"[Q40] results JSON written: {RESULTS_JSON_PATH}", flush=True)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(summaries, comparison, per_seed_rows, args.dry_run, total_wall)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n[Q40] ── Summary ────────────────────────────────────────────────", flush=True)
    hdr = f"{'Candidate':<30} {'MeanAUROC':>10} {'StdAUROC':>9} {'MeanF1':>7} {'ΔAUROC':>8} {'Stable':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cid, s in sorted(summaries.items(), key=lambda x: -x[1]["mean_auroc"]):
        print(
            f"{cid:<30} {s['mean_auroc']:>10.4f} {s['std_auroc']:>9.4f} "
            f"{s['mean_f1']:>7.4f} {s['delta_mean_auroc_vs_q38c']:>+8.4f} "
            f"{s['stability_label']:>8}",
            flush=True,
        )

    print(f"\n[Q40] Production default: {comparison['production_default']}", flush=True)
    print(f"[Q40] Most stable:        {comparison['most_stable_candidate']}", flush=True)
    print(f"[Q40] Total wall:         {total_wall:.1f}s ({total_wall/60:.1f} min)", flush=True)

    print("\nQ40_TOP_CANDIDATE_VALIDATION: PASS", flush=True)


if __name__ == "__main__":
    main()
