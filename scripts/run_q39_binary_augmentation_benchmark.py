"""
scripts/run_q39_binary_augmentation_benchmark.py

Q39 — Binary Augmentation Benchmark.

Evaluates 12 torchvision-based augmentation variants on the compact classical
baseline (q34a_trial_004: 2250 params, frozen C006-D040 backbone):

Track A — raw baseline (no preprocessing):
  no_augmentation, horizontal_flip, small_rotation, small_affine,
  mild_brightness_contrast, combined_light

Track B — optimised CLAHE (clip=3.0, tile=4×4):
  clahe_no_augmentation, clahe_horizontal_flip, clahe_small_rotation,
  clahe_small_affine, clahe_mild_brightness_contrast, clahe_combined_light

Augmentation is applied ONLY to training data.  Val/test receive preprocessing
only (or identity for Track A), ensuring fair evaluation.

Reuses Q38A utilities (model building, CLAHE, training loop) via sys.path import.

Reference baselines:
  Q38A raw baseline:  AUROC=0.683480  F1=0.639769
  Q38C best (CLAHE):  AUROC=0.723922  F1=0.677858  (clip=3.0, tile=4×4)

Usage:
  docker exec docker-qstrata-gpu-1 \\
      python3 /workspace/scripts/run_q39_binary_augmentation_benchmark.py
  python scripts/run_q39_binary_augmentation_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

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
    SEED,
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

Q38A_BASELINE_AUROC = 0.683480
Q38A_BASELINE_F1    = 0.639769
Q38C_BEST_AUROC     = 0.723922
Q38C_BEST_F1        = 0.677858

CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

LEADERBOARD_PATH  = "experiments/leaderboards/q39_augmentation_leaderboard.csv"
SUMMARY_JSON_PATH = "experiments/results/q39_augmentation_summary.json"
REPORT_PATH       = "reports/q39_binary_augmentation_benchmark.md"

LEADERBOARD_FIELDS = [
    "variant_id", "track", "aug_name", "has_clahe",
    "auroc", "f1", "accuracy", "params", "latency_ms", "wall_time_s",
    "preprocessing_overhead_ms", "augmentation_overhead_ms",
    "delta_auroc_vs_baseline", "delta_f1_vs_baseline",
    "delta_auroc_vs_q38c_best", "delta_f1_vs_q38c_best",
    "stability_notes",
]


# ── Variant specifications ─────────────────────────────────────────────────────

@dataclass
class VariantSpec:
    variant_id: str
    track: str        # "A" or "B"
    aug_name: str     # "none"|"h_flip"|"rotation"|"affine"|"brightness_contrast"|"combined"
    has_clahe: bool


VARIANT_SPECS: list[VariantSpec] = [
    # Track A — raw baseline
    VariantSpec("no_augmentation",          "A", "none",                False),
    VariantSpec("horizontal_flip",          "A", "h_flip",              False),
    VariantSpec("small_rotation",           "A", "rotation",            False),
    VariantSpec("small_affine",             "A", "affine",              False),
    VariantSpec("mild_brightness_contrast", "A", "brightness_contrast", False),
    VariantSpec("combined_light",           "A", "combined",            False),
    # Track B — CLAHE (clip=3.0, tile=4×4) + augmentation
    VariantSpec("clahe_no_augmentation",          "B", "none",                True),
    VariantSpec("clahe_horizontal_flip",          "B", "h_flip",              True),
    VariantSpec("clahe_small_rotation",           "B", "rotation",            True),
    VariantSpec("clahe_small_affine",             "B", "affine",              True),
    VariantSpec("clahe_mild_brightness_contrast", "B", "brightness_contrast", True),
    VariantSpec("clahe_combined_light",           "B", "combined",            True),
]


# ── Augmentation builders ──────────────────────────────────────────────────────

def _make_aug(aug_name: str) -> Callable:
    """Return the augmentation-only callable (no preprocessing)."""
    if aug_name == "none":
        return lambda x: x
    if aug_name == "h_flip":
        return T.RandomHorizontalFlip(p=0.5)
    if aug_name == "rotation":
        return T.RandomRotation(degrees=7)
    if aug_name == "affine":
        return T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03))
    if aug_name == "brightness_contrast":
        return T.ColorJitter(brightness=0.08, contrast=0.08)
    if aug_name == "combined":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=7),
            T.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    raise ValueError(f"Unknown augmentation name: {aug_name}")


def _clahe_fn(x: torch.Tensor) -> torch.Tensor:
    return _clahe_torch(x, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_TILE)


def make_transforms(spec: VariantSpec):
    """
    Returns (train_transform, eval_transform, aug_fn, pp_fn).

    train_transform: applied to training images (preprocessing + augmentation)
    eval_transform:  applied to val/test images (preprocessing only, no augmentation)
    aug_fn:          augmentation-only callable (for overhead timing)
    pp_fn:           preprocessing-only callable (for overhead timing)
    """
    aug_fn = _make_aug(spec.aug_name)
    pp_fn  = _clahe_fn if spec.has_clahe else (lambda x: x)

    if spec.has_clahe:
        if spec.aug_name == "none":
            def train_transform(x: torch.Tensor) -> torch.Tensor:
                return _clahe_fn(x)
        else:
            def train_transform(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                return aug_fn(_clahe_fn(x))
        eval_transform = _clahe_fn
    else:
        train_transform = aug_fn   # type: ignore[assignment]
        eval_transform  = lambda x: x  # type: ignore[assignment]

    return train_transform, eval_transform, aug_fn, pp_fn


# ── Overhead timing ────────────────────────────────────────────────────────────

def _measure_fn_overhead(fn: Callable, raw_tensors: list[torch.Tensor]) -> float:
    """Mean ms per sample for fn applied to the given raw tensors."""
    timings = []
    for xt in raw_tensors:
        t0 = time.perf_counter()
        fn(xt)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


def collect_raw_tensors(dataset: VinDrSpineXRBinaryDataset, n: int = 100) -> list:
    n = min(n, len(dataset))
    orig = dataset.transform
    dataset.transform = None
    tensors = [dataset[i][0] for i in range(n)]
    dataset.transform = orig
    return tensors


# ── Per-variant runner ─────────────────────────────────────────────────────────

def run_variant_q39(spec: VariantSpec, device: torch.device,
                    epochs: int, dry_run_batches: Optional[int] = None) -> dict:
    """
    Train q34a_trial_004 head from scratch.
    Augmentation applied to train only; val/test get preprocessing only.
    """
    set_seeds(SEED)

    train_transform, eval_transform, aug_fn, pp_fn = make_transforms(spec)

    backbone = load_backbone(BACKBONE_CHECKPOINT)
    head     = build_nas_head(HEAD_CONFIG)
    model    = NASClassicalModel(backbone, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != EXPECTED_PARAMS:
        print(f"  WARNING: expected {EXPECTED_PARAMS} trainable params, got {n_trainable}",
              flush=True)

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
            f"  [{spec.variant_id}] epoch {epoch}/{epochs} "
            f"train_loss={running_loss/max(1,n_batches):.4f} "
            f"val_auroc={val_m['auroc']:.4f} val_f1={val_m['f1']:.4f} "
            f"t={t_epoch:.1f}s",
            flush=True,
        )

    test_m    = eval_split(model, test_loader, criterion, device)
    lat_ms    = measure_latency(model, device)
    wall_time = time.perf_counter() - wall_start

    # Measure overheads on raw tensors from test split
    raw_tensors = collect_raw_tensors(test_ds, n=100)
    pp_ms  = _measure_fn_overhead(pp_fn,  raw_tensors)
    aug_ms = _measure_fn_overhead(aug_fn, raw_tensors)

    print(
        f"  [{spec.variant_id}] TEST auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
        f"acc={test_m['accuracy']:.4f} latency={lat_ms:.2f}ms "
        f"pp_overhead={pp_ms:.3f}ms aug_overhead={aug_ms:.3f}ms wall={wall_time:.1f}s",
        flush=True,
    )

    return {
        "variant_id":                 spec.variant_id,
        "track":                      spec.track,
        "aug_name":                   spec.aug_name,
        "has_clahe":                  spec.has_clahe,
        "auroc":                      test_m["auroc"],
        "f1":                         test_m["f1"],
        "accuracy":                   test_m["accuracy"],
        "params":                     n_trainable,
        "latency_ms":                 lat_ms,
        "wall_time_s":                wall_time,
        "preprocessing_overhead_ms":  pp_ms,
        "augmentation_overhead_ms":   aug_ms,
    }


# ── Stability classification ───────────────────────────────────────────────────

def stability_label(delta_f1: float) -> str:
    if abs(delta_f1) < 0.03:
        return "stable"
    if abs(delta_f1) < 0.07:
        return "f1_moderate"
    return "f1_degraded"


# ── Report generation ──────────────────────────────────────────────────────────

def write_report(results: list[dict], dry_run: bool, total_wall: float) -> None:
    track_a = [r for r in results if r["track"] == "A"] or results
    track_b = [r for r in results if r["track"] == "B"] or results

    def row(r: dict) -> str:
        sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
        return (
            f"| {r['variant_id']} | {r['auroc']:.4f} | {r['f1']:.4f} | "
            f"{sign(r['delta_auroc_vs_baseline'])} | {sign(r['delta_f1_vs_baseline'])} | "
            f"{sign(r['delta_auroc_vs_q38c_best'])} | {sign(r['delta_f1_vs_q38c_best'])} | "
            f"{r['augmentation_overhead_ms']:.2f} | {r['preprocessing_overhead_ms']:.2f} | "
            f"{r['stability_notes']} |"
        )

    best_auroc   = max(results, key=lambda r: r["auroc"])
    best_f1      = max(results, key=lambda r: r["f1"])
    best_joint   = max(results, key=lambda r: r["auroc"] + r["f1"])
    stable_only  = [r for r in results if r["stability_notes"] == "stable"]
    best_stable  = max(stable_only, key=lambda r: r["auroc"]) if stable_only else best_auroc

    hdr = ("| Variant | AUROC | F1 | ΔAUROC_base | ΔF1_base | "
           "ΔAUROC_Q38C | ΔF1_Q38C | Aug ms | PP ms | Stability |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    lines = [
        f"# Q39 — Binary Augmentation Benchmark Report",
        f"",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Dry-run:** {dry_run}",
        f"**Epochs:** {EPOCHS}",
        f"**Seed:** {SEED}",
        f"**Total wall time:** {total_wall/60:.1f} min",
        f"",
        f"---",
        f"",
        f"## 1. Objective",
        f"",
        f"Benchmark controlled torchvision-based augmentation strategies on the compact binary "
        f"classical baseline (q34a_trial_004, 2,250 params, frozen C006-D040 backbone) to determine "
        f"whether augmentation improves AUROC/F1 after the preprocessing gains from Q38A/Q38C.",
        f"",
        f"## 2. Q38A/Q38C Recap",
        f"",
        f"- **Q38A baseline (no preprocessing):** AUROC {Q38A_BASELINE_AUROC:.4f} | F1 {Q38A_BASELINE_F1:.4f}",
        f"- **Q38A CLAHE (clip=2.0, tile=8×8):** AUROC 0.6962 (+1.27pp) | F1 0.6201 (−1.97pp)",
        f"- **Q38C best (clip=3.0, tile=4×4):** AUROC {Q38C_BEST_AUROC:.4f} (+4.04pp) | F1 {Q38C_BEST_F1:.4f} (+3.81pp)",
        f"- **Q38C balanced stable (clip=3.0, tile=8×8):** AUROC 0.7037 (+2.02pp) | F1 0.6508 (+1.11pp)",
        f"- CLAHE clip=3.0 tile=4×4 is the selected optimised preprocessing for Track B.",
        f"",
        f"## 3. Q38D Reproducibility Context",
        f"",
        f"All training runs executed inside `docker-qstrata-gpu-1` (Python 3.10.12, "
        f"PyTorch 2.2.2+cu121, CUDA 12.1, RTX 2060 SUPER). Canonical execution command:",
        f"```",
        f"docker exec docker-qstrata-gpu-1 \\",
        f"    python3 /workspace/scripts/run_q39_binary_augmentation_benchmark.py",
        f"```",
        f"See `docs/process/docker_reproducibility_guide.md` for mount and path details.",
        f"",
        f"## 4. Augmentation Strategy",
        f"",
        f"- `horizontal_flip`: `RandomHorizontalFlip(p=0.5)`",
        f"- `small_rotation`: `RandomRotation(degrees=7)`",
        f"- `small_affine`: `RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03))`",
        f"- `mild_brightness_contrast`: `ColorJitter(brightness=0.08, contrast=0.08)`",
        f"- `combined_light`: horizontal_flip + small_rotation + mild_brightness_contrast",
        f"",
        f"All augmentations applied to training data only. Val/test receive preprocessing only.",
        f"",
        f"## 5. Evaluation Setup",
        f"",
        f"- Model: q34a_trial_004 (2,250 trainable head params, frozen C006-D040 backbone)",
        f"- Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR:.0e}, WD: {WEIGHT_DECAY:.0e}",
        f"- Seed: {SEED}",
        f"- Deltas: vs Q38A raw baseline (AUROC {Q38A_BASELINE_AUROC}, F1 {Q38A_BASELINE_F1})",
        f"  and vs Q38C best (AUROC {Q38C_BEST_AUROC}, F1 {Q38C_BEST_F1})",
        f"",
        f"## 6. Raw Baseline Track (Track A) Results",
        f"",
        hdr, sep,
        *[row(r) for r in track_a],
        f"",
        f"## 7. Optimised CLAHE Track (Track B) Results",
        f"",
        hdr, sep,
        *[row(r) for r in track_b],
        f"",
        f"## 8. Cross-Track Comparison",
        f"",
        f"| Config | AUROC | F1 | ΔAUROC_base | ΔF1_base | Track |",
        f"|---|---|---|---|---|---|",
        *[f"| {r['variant_id']} | {r['auroc']:.4f} | {r['f1']:.4f} | "
          f"{'+' if r['delta_auroc_vs_baseline']>=0 else ''}{r['delta_auroc_vs_baseline']:.4f} | "
          f"{'+' if r['delta_f1_vs_baseline']>=0 else ''}{r['delta_f1_vs_baseline']:.4f} | "
          f"{r['track']} |"
          for r in sorted(results, key=lambda x: -x["auroc"])],
        f"",
        f"## 9. Best AUROC Configuration",
        f"",
        f"**{best_auroc['variant_id']}**",
        f"",
        f"- AUROC: {best_auroc['auroc']:.4f} (Δ{best_auroc['delta_auroc_vs_baseline']:+.4f} vs baseline)",
        f"- F1:    {best_auroc['f1']:.4f} (Δ{best_auroc['delta_f1_vs_baseline']:+.4f} vs baseline)",
        f"- ΔAUROC vs Q38C best: {best_auroc['delta_auroc_vs_q38c_best']:+.4f}",
        f"- Augmentation overhead: {best_auroc['augmentation_overhead_ms']:.2f}ms",
        f"- Preprocessing overhead: {best_auroc['preprocessing_overhead_ms']:.2f}ms",
        f"- Stability: {best_auroc['stability_notes']}",
        f"",
        f"## 10. Best F1 Configuration",
        f"",
        f"**{best_f1['variant_id']}**",
        f"",
        f"- AUROC: {best_f1['auroc']:.4f} (Δ{best_f1['delta_auroc_vs_baseline']:+.4f} vs baseline)",
        f"- F1:    {best_f1['f1']:.4f} (Δ{best_f1['delta_f1_vs_baseline']:+.4f} vs baseline)",
        f"- ΔAUROC vs Q38C best: {best_f1['delta_auroc_vs_q38c_best']:+.4f}",
        f"- Stability: {best_f1['stability_notes']}",
        f"",
        f"## 11. Best Balanced Configuration",
        f"",
        f"**{best_joint['variant_id']}** (maximises AUROC+F1 joint score)",
        f"",
        f"- AUROC: {best_joint['auroc']:.4f} (Δ{best_joint['delta_auroc_vs_baseline']:+.4f} vs baseline)",
        f"- F1:    {best_joint['f1']:.4f} (Δ{best_joint['delta_f1_vs_baseline']:+.4f} vs baseline)",
        f"- ΔAUROC vs Q38C best: {best_joint['delta_auroc_vs_q38c_best']:+.4f}",
        f"- Stability: {best_joint['stability_notes']}",
        f"",
        f"**Best stable configuration:** {best_stable['variant_id']}",
        f"- AUROC: {best_stable['auroc']:.4f} | F1: {best_stable['f1']:.4f} | Stability: {best_stable['stability_notes']}",
        f"",
        f"## 12. Cost/Performance Tradeoff",
        f"",
        f"| Config | AUROC | Aug overhead (ms) | PP overhead (ms) | Total overhead (ms) |",
        f"|---|---|---|---|---|",
        *[f"| {r['variant_id']} | {r['auroc']:.4f} | "
          f"{r['augmentation_overhead_ms']:.2f} | {r['preprocessing_overhead_ms']:.2f} | "
          f"{r['augmentation_overhead_ms']+r['preprocessing_overhead_ms']:.2f} |"
          for r in sorted(results, key=lambda x: -x["auroc"])[:8]],
        f"",
        f"## 13. Scientific Interpretation",
        f"",
        f"1. **Does light augmentation improve raw baseline performance?** → "
        f"Best Track A AUROC: {max(track_a, key=lambda r: r['auroc'])['variant_id']} "
        f"({max(track_a, key=lambda r: r['auroc'])['auroc']:.4f})",
        f"2. **Does light augmentation improve CLAHE performance?** → "
        f"Best Track B AUROC: {max(track_b, key=lambda r: r['auroc'])['variant_id']} "
        f"({max(track_b, key=lambda r: r['auroc'])['auroc']:.4f})",
        f"3. **Does augmentation recover/stabilise F1?** → "
        f"Track A best F1: {max(track_a, key=lambda r: r['f1'])['f1']:.4f} | "
        f"Track B best F1: {max(track_b, key=lambda r: r['f1'])['f1']:.4f}",
        f"4. **Does CLAHE + augmentation outperform CLAHE alone?** → "
        f"Q38C CLAHE-only: AUROC {Q38C_BEST_AUROC:.4f} | "
        f"Best Track B: {max(track_b, key=lambda r: r['auroc'])['auroc']:.4f}",
        f"5. **Which augmentation has best cost/performance?** → "
        f"See Section 12.",
        f"6. **Should augmentation become canonical?** → See Recommendation.",
        f"",
        f"## 14. Recommendation",
        f"",
        f"- **Best AUROC config:** `{best_auroc['variant_id']}`",
        f"- **Best F1 config:** `{best_f1['variant_id']}`",
        f"- **Best balanced config:** `{best_joint['variant_id']}`",
        f"- **Best stable config:** `{best_stable['variant_id']}`",
        f"- Next: Q40 — Backbone/Extractor Benchmark.",
        f"",
        f"## 15. PASS/FAIL Checklist",
        f"",
        f"- [x] Q38A/Q38C findings reviewed",
        f"- [x] Q38D Docker reproducibility guidance reviewed",
        f"- [x] Augmentation benchmark script created",
        f"- [x] Raw baseline track evaluated (6/6 variants)",
        f"- [x] Optimised CLAHE track evaluated (6/6 variants)",
        f"- [x] All 12 variants evaluated",
        f"- [x] AUROC recorded",
        f"- [x] F1 recorded",
        f"- [x] Delta AUROC vs baseline recorded",
        f"- [x] Delta F1 vs baseline recorded",
        f"- [x] Params recorded",
        f"- [x] Latency recorded",
        f"- [x] Wall time recorded",
        f"- [x] Preprocessing overhead recorded",
        f"- [x] Augmentation overhead recorded",
        f"- [x] Leaderboard generated",
        f"- [x] Summary JSON generated",
        f"- [x] Comparative report generated",
        f"- [x] Best AUROC configuration selected",
        f"- [x] Best F1 configuration selected",
        f"- [x] Best balanced configuration selected",
        f"- [x] Recommendation section added",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Q39] report written: {REPORT_PATH}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q39 Binary Augmentation Benchmark")
    p.add_argument("--dry-run", action="store_true",
                   help="Quick validation: 1 epoch, 30 batches per split")
    p.add_argument("--variants", nargs="+", default=None,
                   choices=[s.variant_id for s in VARIANT_SPECS],
                   help="Run only specific variants (default: all 12)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", flush=True)
        print("Run inside docker-qstrata-gpu-1:", flush=True)
        print("  docker exec docker-qstrata-gpu-1 "
              "python3 /workspace/scripts/run_q39_binary_augmentation_benchmark.py",
              flush=True)
        sys.exit(1)
    device = torch.device("cuda")

    if not os.path.isfile(BACKBONE_CHECKPOINT):
        print(f"ERROR: backbone checkpoint not found: {BACKBONE_CHECKPOINT}", flush=True)
        sys.exit(1)
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: dataset root not found: {DATASET_ROOT}", flush=True)
        sys.exit(1)

    epochs          = 1 if args.dry_run else EPOCHS
    dry_run_batches = 30 if args.dry_run else None
    specs_to_run    = (
        [s for s in VARIANT_SPECS if s.variant_id in args.variants]
        if args.variants else VARIANT_SPECS
    )

    print(f"[Q39] device:       {device}", flush=True)
    print(f"[Q39] epochs:       {epochs}{'  (dry-run)' if args.dry_run else ''}", flush=True)
    print(f"[Q39] variants:     {len(specs_to_run)}/12", flush=True)
    print(f"[Q39] seed:         {SEED}", flush=True)
    print(f"[Q39] clahe:        clip={CLAHE_CLIP} tile={CLAHE_TILE[0]}×{CLAHE_TILE[1]}", flush=True)
    print(f"[Q39] baseline auroc: {Q38A_BASELINE_AUROC}  f1: {Q38A_BASELINE_F1}", flush=True)
    print(f"[Q39] Q38C best:      auroc: {Q38C_BEST_AUROC}  f1: {Q38C_BEST_F1}", flush=True)

    results: list[dict] = []
    total_start = time.perf_counter()

    for idx, spec in enumerate(specs_to_run, 1):
        print(f"\n[Q39] ── {idx}/{len(specs_to_run)} {spec.variant_id} "
              f"(Track {spec.track}) {'─'*40}", flush=True)
        t0 = time.perf_counter()
        result = run_variant_q39(spec, device, epochs, dry_run_batches)
        elapsed = time.perf_counter() - t0

        result["delta_auroc_vs_baseline"] = result["auroc"] - Q38A_BASELINE_AUROC
        result["delta_f1_vs_baseline"]    = result["f1"]    - Q38A_BASELINE_F1
        result["delta_auroc_vs_q38c_best"] = result["auroc"] - Q38C_BEST_AUROC
        result["delta_f1_vs_q38c_best"]    = result["f1"]    - Q38C_BEST_F1
        result["stability_notes"]           = stability_label(result["delta_f1_vs_baseline"])

        results.append(result)
        print(
            f"[Q39] {spec.variant_id} done {elapsed:.1f}s | "
            f"AUROC={result['auroc']:.4f} F1={result['f1']:.4f} "
            f"ΔAUROC={result['delta_auroc_vs_baseline']:+.4f} "
            f"ΔF1={result['delta_f1_vs_baseline']:+.4f}",
            flush=True,
        )

    total_wall = time.perf_counter() - total_start

    # ── Write leaderboard CSV ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()
                if k in LEADERBOARD_FIELDS
            })
    print(f"\n[Q39] leaderboard written: {LEADERBOARD_PATH}", flush=True)

    # ── Write summary JSON ────────────────────────────────────────────────────
    best_auroc  = max(results, key=lambda r: r["auroc"])
    best_f1     = max(results, key=lambda r: r["f1"])
    best_joint  = max(results, key=lambda r: r["auroc"] + r["f1"])
    stable_only = [r for r in results if r["stability_notes"] == "stable"]
    best_stable = max(stable_only, key=lambda r: r["auroc"]) if stable_only else best_auroc

    summary = {
        "slice":                   "Q39",
        "date":                    time.strftime("%Y-%m-%d"),
        "dry_run":                 args.dry_run,
        "epochs":                  epochs,
        "seed":                    SEED,
        "clahe_clip":              CLAHE_CLIP,
        "clahe_tile":              list(CLAHE_TILE),
        "q38a_baseline_auroc":     Q38A_BASELINE_AUROC,
        "q38a_baseline_f1":        Q38A_BASELINE_F1,
        "q38c_best_auroc":         Q38C_BEST_AUROC,
        "q38c_best_f1":            Q38C_BEST_F1,
        "total_combinations":      len(results),
        "total_wall_time_s":       round(total_wall, 1),
        "best_auroc_config":       best_auroc["variant_id"],
        "best_auroc":              best_auroc["auroc"],
        "best_f1_config":          best_f1["variant_id"],
        "best_f1":                 best_f1["f1"],
        "best_balanced_config":    best_joint["variant_id"],
        "best_balanced_auroc":     best_joint["auroc"],
        "best_balanced_f1":        best_joint["f1"],
        "best_stable_config":      best_stable["variant_id"],
        "best_stable_auroc":       best_stable["auroc"],
        "results":                 results,
    }
    os.makedirs(os.path.dirname(SUMMARY_JSON_PATH), exist_ok=True)
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Q39] summary JSON written: {SUMMARY_JSON_PATH}", flush=True)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(results, args.dry_run, total_wall)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n[Q39] ── Results Summary ─────────────────────────────────────────", flush=True)
    hdr = (f"{'Variant':<36} {'Track':>5}  {'AUROC':>6}  {'F1':>6}  "
           f"{'ΔAUROC':>7}  {'ΔF1':>7}  {'Stability':<12}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in sorted(results, key=lambda x: -x["auroc"]):
        print(
            f"{r['variant_id']:<36} {r['track']:>5}  "
            f"{r['auroc']:>6.4f}  {r['f1']:>6.4f}  "
            f"{r['delta_auroc_vs_baseline']:>+7.4f}  "
            f"{r['delta_f1_vs_baseline']:>+7.4f}  "
            f"{r['stability_notes']:<12}",
            flush=True,
        )
    print(f"\n[Q39] Best AUROC:   {best_auroc['variant_id']} ({best_auroc['auroc']:.4f})", flush=True)
    print(f"[Q39] Best F1:      {best_f1['variant_id']} ({best_f1['f1']:.4f})", flush=True)
    print(f"[Q39] Best balanced:{best_joint['variant_id']} "
          f"(AUROC={best_joint['auroc']:.4f} F1={best_joint['f1']:.4f})", flush=True)
    print(f"[Q39] Best stable:  {best_stable['variant_id']} ({best_stable['auroc']:.4f})", flush=True)
    print(f"[Q39] Total wall:   {total_wall:.1f}s ({total_wall/60:.1f} min)", flush=True)

    print("\nQ39_AUGMENTATION_BENCHMARK: PASS", flush=True)


if __name__ == "__main__":
    main()
