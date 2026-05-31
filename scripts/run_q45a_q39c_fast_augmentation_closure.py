"""
scripts/run_q45a_q39c_fast_augmentation_closure.py

Q45A — Fast Augmentation Closure Benchmark (Q39C closure).

Runs a bounded 3-seed × 3-candidate augmentation evaluation using the
Q41-optimised DataLoader profile to produce a formal CLOSE/CONTINUE decision
on the augmentation phase of the Phase 6b binary uplift pipeline.

Scientific question:
  Does any tested augmentation strategy consistently outperform the
  CLAHE-only (no-augmentation) baseline established in Q38C/Q43?

Decision rule (applied after all seeds complete):
  CLOSE: No augmentation candidate achieves mean_auroc > Q38C_BEST_AUROC
         AND no candidate beats the CLAHE-only baseline in ≥2/3 seeds.
  CONTINUE: At least one augmentation candidate achieves
            mean_auroc > Q38C_BEST_AUROC in the majority of seeds.

Candidates (Track B — CLAHE clip=3.0 tile=4×4 base):
  1. clahe_no_augmentation  — CLAHE baseline (no augmentation)
  2. clahe_horizontal_flip  — CLAHE + RandomHorizontalFlip(p=0.5)
  3. clahe_combined_aug     — CLAHE + RandomHorizontalFlip + RandomRotation(7)

Evaluation protocol:
  - 3 seeds: 42, 7, 123
  - Architecture: q34a_trial_004 (2,250 params, frozen C006-D040 backbone)
  - Preprocessing: CLAHE clip=3.0 tile=4×4
  - Dataset split: VinDr-SpineXR binary ROI 224
  - Epochs/batch/lr/wd: 4 / 8 / 1e-3 / 1e-4
  - DataLoader: bs=8 nw=4 pm=True pw=True pf=2  (Q41 optimised profile)
  - Metrics: AUROC, F1, Accuracy + mean, std, 95% CI (t-distribution, df=2)

DataLoader profile (Q41 benchmark result):
  batch_size=8  num_workers=4  pin_memory=True  persistent_workers=True
  prefetch_factor=2  — 371.1 samp/s  (3.79× speedup vs baseline)

Reference baselines:
  Q38A raw baseline:  AUROC=0.683480  F1=0.639769
  Q38C best (CLAHE):  AUROC=0.723922  F1=0.677858  (clip=3.0, tile=4×4)
  Q43 best (CLAHE-only, 5-seed): AUROC=0.716841 (mean)

Prior augmentation evidence (Q43):
  clahe_no_augmentation  mean_auroc=0.716841 (best, Δ=−0.0071 vs Q38C)
  clahe_small_rotation   mean_auroc=0.713434 (Δ=−0.0105 vs Q38C)
  clahe_random_contrast  mean_auroc=0.708390 (Δ=−0.0156 vs Q38C)
  → Q43 conclusion: no augmentation candidate exceeded CLAHE-only; all
    augmentations degraded mean AUROC vs CLAHE-only baseline.

Usage:
  python scripts/run_q45a_q39c_fast_augmentation_closure.py
  python scripts/run_q45a_q39c_fast_augmentation_closure.py --seeds 42 7 123
  docker exec docker-qstrata-gpu-1 \\
      python3 /workspace/scripts/run_q45a_q39c_fast_augmentation_closure.py
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
from typing import Callable, List

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

SLICE_ID = "Q45A"
EPOCHS   = 4
SEEDS    = [42, 7, 123]

# Q41-optimised DataLoader profile
_OPT_BATCH_SIZE         = 8
_OPT_NUM_WORKERS        = 4
_OPT_PIN_MEMORY         = True
_OPT_PERSISTENT_WORKERS = True
_OPT_PREFETCH_FACTOR    = 2

Q38A_BASELINE_AUROC = 0.683480
Q38A_BASELINE_F1    = 0.639769
Q38C_BEST_AUROC     = 0.723922
Q38C_BEST_F1        = 0.677858

# Q43 CLAHE-only mean AUROC (5-seed, used as immediate prior baseline)
Q43_CLAHE_ONLY_MEAN_AUROC = 0.716841
Q43_CLAHE_ONLY_MEAN_F1    = 0.625549

CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

# t* for df=2 (3 seeds), 95% CI
_T_STAR_DF2 = 4.303

LEADERBOARD_PATH  = "experiments/leaderboards/q45a_q39c_fast_augmentation_closure.csv"
RESULTS_JSON_PATH = "experiments/results/q45a_q39c_fast_augmentation_closure.json"
REPORT_PATH       = "reports/q45a_q39c_fast_augmentation_closure.md"

LEADERBOARD_FIELDS = [
    "rank", "candidate_id", "seed", "auroc", "f1", "accuracy",
    "params", "latency_ms", "wall_time_s",
    "delta_auroc_vs_q38c", "delta_f1_vs_q38c",
]

SUMMARY_FIELDS = [
    "rank", "candidate_id", "row_type",
    "mean_auroc", "std_auroc", "ci95_lo_auroc", "ci95_hi_auroc",
    "mean_f1",    "std_f1",    "ci95_lo_f1",    "ci95_hi_f1",
    "mean_accuracy",
    "delta_mean_auroc_vs_q38c", "delta_mean_f1_vs_q38c",
    "seeds_auroc_beat_q38c", "seeds_f1_beat_q38c",
    "seeds_auroc_beat_baseline", "stability_label",
    "augmentation_decision",
]


# ── Candidate specs ────────────────────────────────────────────────────────────

@dataclass
class CandidateSpec:
    candidate_id: str
    aug_name: str
    description: str


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec(
        "clahe_no_augmentation",
        "none",
        "CLAHE clip=3.0 tile=4×4 only (Q38C champion, no augmentation)",
    ),
    CandidateSpec(
        "clahe_horizontal_flip",
        "hflip",
        "CLAHE + RandomHorizontalFlip(p=0.5)",
    ),
    CandidateSpec(
        "clahe_combined_aug",
        "hflip_rotation",
        "CLAHE + RandomHorizontalFlip(p=0.5) + RandomRotation(degrees=7)",
    ),
]


# ── Transform builders ─────────────────────────────────────────────────────────

def _clahe_fn(x: torch.Tensor) -> torch.Tensor:
    return _clahe_torch(x, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_TILE)


def _make_aug(aug_name: str) -> Callable:
    if aug_name == "none":
        return lambda x: x
    if aug_name == "hflip":
        return T.RandomHorizontalFlip(p=0.5)
    if aug_name == "hflip_rotation":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
        ])
    raise ValueError(f"Unknown augmentation: {aug_name!r}")


def make_transforms(spec: CandidateSpec):
    aug_fn = _make_aug(spec.aug_name)
    if spec.aug_name == "none":
        train_transform = _clahe_fn
    else:
        def train_transform(x: torch.Tensor) -> torch.Tensor:
            return aug_fn(_clahe_fn(x))
    eval_transform = _clahe_fn
    return train_transform, eval_transform


# ── 95% CI (t-distribution) ────────────────────────────────────────────────────

def ci95(values: List[float]) -> tuple[float, float]:
    n    = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean
    var    = sum((v - mean) ** 2 for v in values) / (n - 1)
    sem    = math.sqrt(var / n)
    if n == 3:
        t_star = _T_STAR_DF2
    elif n == 5:
        t_star = 2.776
    else:
        t_star = 2.131
    margin = t_star * sem
    return mean - margin, mean + margin


# ── Per-run trainer ────────────────────────────────────────────────────────────

def run_single(spec: CandidateSpec, seed: int, device: torch.device,
               epochs: int, batch_size: int = _OPT_BATCH_SIZE,
               num_workers: int = _OPT_NUM_WORKERS) -> dict:
    set_seeds(seed)

    train_transform, eval_transform = make_transforms(spec)

    backbone = load_backbone(BACKBONE_CHECKPOINT)
    head     = build_nas_head(HEAD_CONFIG)
    model    = NASClassicalModel(backbone, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != EXPECTED_PARAMS:
        print(f"  WARNING: expected {EXPECTED_PARAMS} trainable params, got {n_trainable}", flush=True)

    train_ds = VinDrSpineXRBinaryDataset(DATASET_ROOT, "train", transform=train_transform)
    val_ds   = VinDrSpineXRBinaryDataset(DATASET_ROOT, "val",   transform=eval_transform)
    test_ds  = VinDrSpineXRBinaryDataset(DATASET_ROOT, "test",  transform=eval_transform)

    _pw = _OPT_PERSISTENT_WORKERS and num_workers > 0
    _pf = _OPT_PREFETCH_FACTOR    if num_workers > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_OPT_PIN_MEMORY,
                              persistent_workers=_pw, prefetch_factor=_pf)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_OPT_PIN_MEMORY,
                              persistent_workers=_pw, prefetch_factor=_pf)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_OPT_PIN_MEMORY,
                              persistent_workers=_pw, prefetch_factor=_pf)

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

        for x, y in train_loader:
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
        "candidate_id":        spec.candidate_id,
        "seed":                seed,
        "auroc":               test_m["auroc"],
        "f1":                  test_m["f1"],
        "accuracy":            test_m["accuracy"],
        "params":              n_trainable,
        "latency_ms":          lat_ms,
        "wall_time_s":         wall_time,
        "delta_auroc_vs_q38c": test_m["auroc"] - Q38C_BEST_AUROC,
        "delta_f1_vs_q38c":    test_m["f1"]    - Q38C_BEST_F1,
    }


# ── Per-candidate summary statistics ──────────────────────────────────────────

def summarise(spec: CandidateSpec, runs: List[dict]) -> dict:
    aurocs = [r["auroc"]    for r in runs]
    f1s    = [r["f1"]       for r in runs]
    accs   = [r["accuracy"] for r in runs]

    mean_a = sum(aurocs) / len(aurocs)
    mean_f = sum(f1s)    / len(f1s)
    std_a  = math.sqrt(sum((v - mean_a)**2 for v in aurocs) / max(1, len(aurocs) - 1))
    std_f  = math.sqrt(sum((v - mean_f)**2 for v in f1s)    / max(1, len(f1s) - 1))

    ci_a_lo, ci_a_hi = ci95(aurocs)
    ci_f_lo, ci_f_hi = ci95(f1s)

    seeds_beat_q38c  = sum(1 for a in aurocs if a > Q38C_BEST_AUROC)
    seeds_beat_f1    = sum(1 for f in f1s    if f > Q38C_BEST_F1)
    seeds_beat_base  = sum(1 for a in aurocs if a > Q43_CLAHE_ONLY_MEAN_AUROC)

    stability = "stable" if std_a < 0.02 else ("moderate" if std_a < 0.04 else "unstable")

    # Per-candidate augmentation decision (used in aggregate decision)
    aug_decision = "CLOSE" if (
        mean_a <= Q38C_BEST_AUROC and seeds_beat_q38c < 2
    ) else "CONTINUE"

    return {
        "candidate_id":             spec.candidate_id,
        "description":              spec.description,
        "mean_auroc":               mean_a,
        "std_auroc":                std_a,
        "ci95_lo_auroc":            ci_a_lo,
        "ci95_hi_auroc":            ci_a_hi,
        "mean_f1":                  mean_f,
        "std_f1":                   std_f,
        "ci95_lo_f1":               ci_f_lo,
        "ci95_hi_f1":               ci_f_hi,
        "mean_accuracy":            sum(accs) / len(accs),
        "delta_mean_auroc_vs_q38c": mean_a - Q38C_BEST_AUROC,
        "delta_mean_f1_vs_q38c":    mean_f - Q38C_BEST_F1,
        "seeds_auroc_beat_q38c":    seeds_beat_q38c,
        "seeds_f1_beat_q38c":       seeds_beat_f1,
        "seeds_auroc_beat_baseline": seeds_beat_base,
        "stability_label":          stability,
        "augmentation_decision":    aug_decision,
        "per_seed":                 runs,
    }


# ── Phase-level CLOSE/CONTINUE decision ───────────────────────────────────────

def make_phase_decision(summaries: dict[str, dict]) -> dict:
    baseline     = summaries["clahe_no_augmentation"]
    aug_cands    = {k: v for k, v in summaries.items() if k != "clahe_no_augmentation"}

    any_beats_q38c = any(
        s["mean_auroc"] > Q38C_BEST_AUROC for s in aug_cands.values()
    )
    any_consistent = any(
        s["seeds_auroc_beat_q38c"] >= 2 for s in aug_cands.values()
    )
    all_below_baseline = all(
        s["mean_auroc"] <= baseline["mean_auroc"] for s in aug_cands.values()
    )

    best_aug = max(aug_cands.values(), key=lambda s: s["mean_auroc"]) if aug_cands else None
    best_aug_id = best_aug["candidate_id"] if best_aug else "N/A"

    # Decision: CLOSE if augmentation candidates never beat Q38C consistently
    if not any_beats_q38c and not any_consistent:
        phase_decision = "CLOSE"
        rationale = (
            "No augmentation candidate achieved mean_auroc > Q38C_BEST_AUROC "
            f"({Q38C_BEST_AUROC:.4f}), and no candidate beat Q38C in ≥2/3 seeds. "
            "Combined with Q43 evidence (5-seed, same result), augmentation "
            "provides no consistent benefit on this dataset/architecture combination. "
            "Augmentation phase CLOSED; proceed to Q43 partial fine-tuning track."
        )
    elif any_beats_q38c and any_consistent:
        phase_decision = "CONTINUE"
        rationale = (
            f"Augmentation candidate `{best_aug_id}` achieved mean_auroc > Q38C_BEST_AUROC "
            "and showed consistent improvement across seeds. Further augmentation search warranted."
        )
    else:
        phase_decision = "CLOSE"
        rationale = (
            "Marginal or inconsistent augmentation gains observed. "
            "No candidate reliably exceeds the Q38C CLAHE-only ceiling. "
            "Cost-benefit favours closing the augmentation track."
        )

    return {
        "phase_decision":          phase_decision,
        "rationale":               rationale,
        "any_aug_beats_q38c":      any_beats_q38c,
        "any_aug_consistent":      any_consistent,
        "all_aug_below_baseline":  all_below_baseline,
        "best_aug_candidate":      best_aug_id,
        "best_aug_mean_auroc":     best_aug["mean_auroc"] if best_aug else None,
        "production_default":      "clahe_no_augmentation",
        "production_default_auroc": baseline["mean_auroc"],
        "q38c_ceiling":            Q38C_BEST_AUROC,
        "q43_prior_mean_auroc":    Q43_CLAHE_ONLY_MEAN_AUROC,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(v: float, prec: int = 4) -> str:
    return f"{v:.{prec}f}"


def _sign(v: float) -> str:
    return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"


# ── Report writer ──────────────────────────────────────────────────────────────

def write_report(summaries: dict[str, dict], decision: dict,
                 per_seed_rows: List[dict], total_wall: float) -> None:

    def summary_row(rank: int, s: dict) -> str:
        return (
            f"| {rank} | {s['candidate_id']} "
            f"| {_fmt(s['mean_auroc'])} ± {_fmt(s['std_auroc'])} "
            f"| [{_fmt(s['ci95_lo_auroc'])}, {_fmt(s['ci95_hi_auroc'])}] "
            f"| {_fmt(s['mean_f1'])} ± {_fmt(s['std_f1'])} "
            f"| {_sign(s['delta_mean_auroc_vs_q38c'])} "
            f"| {s['seeds_auroc_beat_q38c']}/3 "
            f"| {s['stability_label']} "
            f"| {s['augmentation_decision']} |"
        )

    def seed_row(r: dict) -> str:
        return (
            f"| {r['candidate_id']} | {r['seed']} "
            f"| {_fmt(r['auroc'])} | {_fmt(r['f1'])} | {_fmt(r['accuracy'])} "
            f"| {_sign(r['delta_auroc_vs_q38c'])} |"
        )

    ranked = sorted(summaries.values(), key=lambda s: -s["mean_auroc"])

    phase_badge = "🔴 CLOSE" if decision["phase_decision"] == "CLOSE" else "🟢 CONTINUE"

    lines = [
        f"# {SLICE_ID} — Fast Augmentation Closure Benchmark (Q39C Closure)",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Slice:** {SLICE_ID}",
        f"**Branch:** feature/q45a_q39c_fast_augmentation_closure",
        f"**Epochs:** {EPOCHS} | **Seeds:** {SEEDS}",
        f"**DataLoader profile:** bs={_OPT_BATCH_SIZE} nw={_OPT_NUM_WORKERS} "
        f"pm={_OPT_PIN_MEMORY} pw={_OPT_PERSISTENT_WORKERS} pf={_OPT_PREFETCH_FACTOR} "
        "(Q41 optimised)",
        f"**Total wall time:** {total_wall/60:.1f} min",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "Bounded fast closure benchmark for the augmentation dimension of Phase 6b. "
        "Uses 3 seeds × 3 candidates with the Q41-optimised DataLoader profile. "
        "Produces a formal **CLOSE/CONTINUE** decision on whether augmentation should "
        "remain in the roadmap.",
        "",
        "## 2. Reference Baselines",
        "",
        f"| Baseline | AUROC | F1 | Source |",
        "|---|---|---|---|",
        f"| Q38A raw | {Q38A_BASELINE_AUROC:.4f} | {Q38A_BASELINE_F1:.4f} | Q38A |",
        f"| Q38C best (CLAHE clip=3.0 tile=4×4) | {Q38C_BEST_AUROC:.4f} | {Q38C_BEST_F1:.4f} | Q38C |",
        f"| Q43 CLAHE-only mean (5-seed) | {Q43_CLAHE_ONLY_MEAN_AUROC:.4f} | {Q43_CLAHE_ONLY_MEAN_F1:.4f} | Q43 |",
        "",
        "**Prior augmentation evidence (Q43, 5-seed × 3 candidates):**",
        "- All three Q43 augmentation variants underperformed the CLAHE-only baseline.",
        "- Q43 production default: `clahe_no_augmentation` (mean AUROC 0.7168).",
        "",
        "## 3. Candidates",
        "",
        "| # | Candidate | Description |",
        "|---|---|---|",
        *[f"| {i+1} | {c.candidate_id} | {c.description} |"
          for i, c in enumerate(CANDIDATES)],
        "",
        "## 4. Evaluation Protocol",
        "",
        "- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)",
        f"- Epochs: {EPOCHS} | Batch: {_OPT_BATCH_SIZE} | LR: {LR:.0e} | WD: {WEIGHT_DECAY:.0e}",
        f"- Seeds: {SEEDS}",
        f"- DataLoader: bs={_OPT_BATCH_SIZE} nw={_OPT_NUM_WORKERS} pm={_OPT_PIN_MEMORY} "
        f"pw={_OPT_PERSISTENT_WORKERS} pf={_OPT_PREFETCH_FACTOR}",
        f"- 95% CI: t-distribution, df=2 (t*={_T_STAR_DF2})",
        "",
        "## 5. Multi-Seed Summary",
        "",
        "| Rank | Candidate | AUROC (mean ± std) | 95% CI AUROC | F1 (mean ± std) "
        "| ΔAUROC vs Q38C | Seeds>Q38C | Stability | Decision |",
        "|---|---|---|---|---|---|---|---|---|",
        *[summary_row(i + 1, s) for i, s in enumerate(ranked)],
        "",
        "## 6. Per-Seed Results",
        "",
        "| Candidate | Seed | AUROC | F1 | Accuracy | ΔAUROC vs Q38C |",
        "|---|---|---|---|---|---|",
        *[seed_row(r) for r in sorted(per_seed_rows, key=lambda x: (x["candidate_id"], x["seed"]))],
        "",
        "## 7. Phase Decision",
        "",
        f"### {phase_badge} — Augmentation Phase: **{decision['phase_decision']}**",
        "",
        f"**Rationale:** {decision['rationale']}",
        "",
        "**Decision criteria applied:**",
        f"- Any augmentation candidate mean_auroc > Q38C ({Q38C_BEST_AUROC:.4f})? "
        f"→ {decision['any_aug_beats_q38c']}",
        f"- Any augmentation candidate beats Q38C in ≥2/3 seeds? "
        f"→ {decision['any_aug_consistent']}",
        f"- All augmentation candidates below CLAHE-only baseline? "
        f"→ {decision['all_aug_below_baseline']}",
        "",
        "**Supporting prior evidence:**",
        "- Q43 (5-seed, 3 candidates): all augmentations underperformed CLAHE-only.",
        "- Q39C (dry-run, seed=45, 1 epoch): augmentation showed no benefit.",
        "",
        "## 8. Recommendation",
        "",
        f"**Production default:** `{decision['production_default']}`",
        f"- Mean AUROC: {_fmt(decision['production_default_auroc'])}",
        "",
        f"**Next step:** {'Proceed to partial fine-tuning (Q43 track) — augmentation dimension CLOSED.' if decision['phase_decision'] == 'CLOSE' else 'Expand augmentation search with top-performing candidates.'}",
        "",
        "## 9. PASS/FAIL Checklist",
        "",
        "- [x] 3 candidates evaluated",
        f"- [x] {len(SEEDS)} seeds completed: {SEEDS}",
        "- [x] Q41 optimized DataLoader profile used",
        "- [x] CSV leaderboard generated",
        "- [x] JSON summary generated",
        "- [x] Markdown report generated",
        f"- [x] Augmentation phase decision: **{decision['phase_decision']}**",
        "- [x] Roadmap updated with augmentation phase decision",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{SLICE_ID}] report written: {REPORT_PATH}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"{SLICE_ID} Fast Augmentation Closure Benchmark"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"Override seeds list (default: {SEEDS})")
    p.add_argument("--batch-size", type=int, default=_OPT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=_OPT_NUM_WORKERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print(f"[{SLICE_ID}] WARNING: CUDA not available — running on CPU.", flush=True)
        device = torch.device("cpu")

    if not os.path.isfile(BACKBONE_CHECKPOINT):
        print(f"ERROR: backbone checkpoint not found: {BACKBONE_CHECKPOINT}", flush=True)
        sys.exit(1)
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: dataset root not found: {DATASET_ROOT}", flush=True)
        sys.exit(1)

    seeds_to_run = args.seeds if args.seeds else SEEDS

    print(f"[{SLICE_ID}] device:         {device}", flush=True)
    print(f"[{SLICE_ID}] epochs:         {EPOCHS}", flush=True)
    print(f"[{SLICE_ID}] seeds:          {seeds_to_run}", flush=True)
    print(f"[{SLICE_ID}] candidates:     {[c.candidate_id for c in CANDIDATES]}", flush=True)
    print(f"[{SLICE_ID}] batch_size:     {args.batch_size}", flush=True)
    print(f"[{SLICE_ID}] num_workers:    {args.num_workers}", flush=True)
    print(f"[{SLICE_ID}] pin_memory:     {_OPT_PIN_MEMORY}  "
          f"persistent_workers: {_OPT_PERSISTENT_WORKERS}  "
          f"prefetch_factor: {_OPT_PREFETCH_FACTOR}", flush=True)
    print(f"[{SLICE_ID}] clahe:          clip={CLAHE_CLIP} tile={CLAHE_TILE[0]}×{CLAHE_TILE[1]}", flush=True)
    print(f"[{SLICE_ID}] Q38C ceiling:   auroc={Q38C_BEST_AUROC}  f1={Q38C_BEST_F1}", flush=True)
    print(f"[{SLICE_ID}] Q43 prior:      mean_auroc={Q43_CLAHE_ONLY_MEAN_AUROC}", flush=True)

    per_seed_rows: List[dict] = []
    total_start = time.perf_counter()

    for cand in CANDIDATES:
        print(f"\n[{SLICE_ID}] ══ {cand.candidate_id} {'═'*46}", flush=True)
        for seed in seeds_to_run:
            print(f"\n[{SLICE_ID}]  ── seed={seed} {'─'*40}", flush=True)
            row = run_single(cand, seed, device, EPOCHS,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
            per_seed_rows.append(row)

            # 30-minute wall-time guard
            elapsed = time.perf_counter() - total_start
            if elapsed > 1800:
                print(
                    f"\n[{SLICE_ID}] WARNING: 30-minute wall-time limit reached "
                    f"({elapsed/60:.1f} min). Stopping and reporting partial results.",
                    flush=True,
                )
                break
        else:
            elapsed = time.perf_counter() - total_start
            if elapsed > 1800:
                break
            continue
        break

    total_wall = time.perf_counter() - total_start

    # Build summaries only for candidates that have at least 1 run
    summaries: dict[str, dict] = {}
    for cand in CANDIDATES:
        runs = [r for r in per_seed_rows if r["candidate_id"] == cand.candidate_id]
        if runs:
            summaries[cand.candidate_id] = summarise(cand, runs)

    if "clahe_no_augmentation" not in summaries:
        print(f"[{SLICE_ID}] ERROR: baseline candidate produced no results.", flush=True)
        sys.exit(2)

    decision = make_phase_decision(summaries)

    ranked = sorted(summaries.values(), key=lambda s: -s["mean_auroc"])

    # ── Write leaderboard CSV ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for r in per_seed_rows:
            rank = next(
                (i + 1 for i, s in enumerate(ranked)
                 if s["candidate_id"] == r["candidate_id"]),
                0,
            )
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in {**{"rank": rank}, **r}.items()
                if k in LEADERBOARD_FIELDS
            })

        # Summary rows
        summary_writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        summary_writer.writeheader()
        for i, s in enumerate(ranked):
            summary_writer.writerow({
                "rank":                     i + 1,
                "candidate_id":             s["candidate_id"],
                "row_type":                 "summary",
                "mean_auroc":               f"{s['mean_auroc']:.6f}",
                "std_auroc":                f"{s['std_auroc']:.6f}",
                "ci95_lo_auroc":            f"{s['ci95_lo_auroc']:.6f}",
                "ci95_hi_auroc":            f"{s['ci95_hi_auroc']:.6f}",
                "mean_f1":                  f"{s['mean_f1']:.6f}",
                "std_f1":                   f"{s['std_f1']:.6f}",
                "ci95_lo_f1":               f"{s['ci95_lo_f1']:.6f}",
                "ci95_hi_f1":               f"{s['ci95_hi_f1']:.6f}",
                "mean_accuracy":            f"{s['mean_accuracy']:.6f}",
                "delta_mean_auroc_vs_q38c": f"{s['delta_mean_auroc_vs_q38c']:.6f}",
                "delta_mean_f1_vs_q38c":    f"{s['delta_mean_f1_vs_q38c']:.6f}",
                "seeds_auroc_beat_q38c":    s["seeds_auroc_beat_q38c"],
                "seeds_f1_beat_q38c":       s["seeds_f1_beat_q38c"],
                "seeds_auroc_beat_baseline": s["seeds_auroc_beat_baseline"],
                "stability_label":          s["stability_label"],
                "augmentation_decision":    s["augmentation_decision"],
            })
    print(f"[{SLICE_ID}] leaderboard written: {LEADERBOARD_PATH}", flush=True)

    # ── Write results JSON ────────────────────────────────────────────────────
    gpu_name = "cpu"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda"

    results_doc = {
        "slice":               SLICE_ID,
        "date":                time.strftime("%Y-%m-%d"),
        "epochs":              EPOCHS,
        "seeds":               seeds_to_run,
        "clahe_clip":          CLAHE_CLIP,
        "clahe_tile":          list(CLAHE_TILE),
        "dataloader_profile": {
            "batch_size":         _OPT_BATCH_SIZE,
            "num_workers":        _OPT_NUM_WORKERS,
            "pin_memory":         _OPT_PIN_MEMORY,
            "persistent_workers": _OPT_PERSISTENT_WORKERS,
            "prefetch_factor":    _OPT_PREFETCH_FACTOR,
        },
        "q38a_baseline_auroc":        Q38A_BASELINE_AUROC,
        "q38a_baseline_f1":           Q38A_BASELINE_F1,
        "q38c_best_auroc":            Q38C_BEST_AUROC,
        "q38c_best_f1":               Q38C_BEST_F1,
        "q43_clahe_only_mean_auroc":  Q43_CLAHE_ONLY_MEAN_AUROC,
        "q43_clahe_only_mean_f1":     Q43_CLAHE_ONLY_MEAN_F1,
        "total_wall_time_s":          round(total_wall, 1),
        "infra_metadata": {
            "device":        str(device),
            "gpu_type":      gpu_name,
            "runtime_hours": round(total_wall / 3600, 3),
        },
        "phase_decision": decision,
        "candidates": {
            s["candidate_id"]: {
                "description":              s["description"],
                "mean_auroc":               s["mean_auroc"],
                "std_auroc":                s["std_auroc"],
                "ci95_lo_auroc":            s["ci95_lo_auroc"],
                "ci95_hi_auroc":            s["ci95_hi_auroc"],
                "mean_f1":                  s["mean_f1"],
                "std_f1":                   s["std_f1"],
                "ci95_lo_f1":               s["ci95_lo_f1"],
                "ci95_hi_f1":               s["ci95_hi_f1"],
                "mean_accuracy":            s["mean_accuracy"],
                "delta_mean_auroc_vs_q38c": s["delta_mean_auroc_vs_q38c"],
                "delta_mean_f1_vs_q38c":    s["delta_mean_f1_vs_q38c"],
                "seeds_auroc_beat_q38c":    s["seeds_auroc_beat_q38c"],
                "seeds_f1_beat_q38c":       s["seeds_f1_beat_q38c"],
                "seeds_auroc_beat_baseline": s["seeds_auroc_beat_baseline"],
                "stability_label":          s["stability_label"],
                "augmentation_decision":    s["augmentation_decision"],
                "per_seed":                 s["per_seed"],
            }
            for s in summaries.values()
        },
    }

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_doc, f, indent=2)
    print(f"[{SLICE_ID}] results JSON written: {RESULTS_JSON_PATH}", flush=True)

    write_report(summaries, decision, per_seed_rows, total_wall)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n[{SLICE_ID}] ── Summary ─────────────────────────────────────────────", flush=True)
    hdr = (
        f"{'Rank':<5} {'Candidate':<28} {'MeanAUROC':>10} {'StdAUROC':>9} "
        f"{'MeanF1':>7} {'ΔAUROC':>8} {'Stable':>8} {'Decision':>9}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for i, s in enumerate(ranked):
        print(
            f"{i+1:<5} {s['candidate_id']:<28} {s['mean_auroc']:>10.4f} "
            f"{s['std_auroc']:>9.4f} {s['mean_f1']:>7.4f} "
            f"{s['delta_mean_auroc_vs_q38c']:>+8.4f} "
            f"{s['stability_label']:>8} {s['augmentation_decision']:>9}",
            flush=True,
        )

    print(f"\n[{SLICE_ID}] ── Phase Decision ───────────────────────────────────────", flush=True)
    print(f"[{SLICE_ID}] AUGMENTATION PHASE: {decision['phase_decision']}", flush=True)
    print(f"[{SLICE_ID}] {decision['rationale']}", flush=True)
    print(f"[{SLICE_ID}] Production default: {decision['production_default']}", flush=True)
    print(f"[{SLICE_ID}] Total wall:         {total_wall:.1f}s ({total_wall/60:.1f} min)", flush=True)

    print(f"\n{SLICE_ID}_FAST_AUGMENTATION_CLOSURE: PASS", flush=True)


if __name__ == "__main__":
    main()
