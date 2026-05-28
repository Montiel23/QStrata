"""
scripts/run_q38a_preprocessing_benchmark.py

Q38A — Binary Preprocessing Benchmark.

Evaluates 5 preprocessing strategies on the compact classical baseline
(q34a_trial_004: 2250 trainable params, frozen C006-D040 backbone) to
determine whether preprocessing improves AUROC/F1 on VinDr-SpineXR binary ROI.

Variants:
  baseline               — identity (raw tensor, no preprocessing)
  clahe                  — tile-based CLAHE (8×8 tiles, clip_limit=2.0)
  histogram_equalization — global histogram equalization
  contrast_normalization — percentile clipping (p2–p98) + min-max rescale
  clahe_plus_normalization — CLAHE → contrast_normalization (chained)

Implementation note: all transforms are pure PyTorch (no cv2, no .numpy() call)
to avoid the numpy-2.x/torch C-API incompatibility in docker-qstrata-gpu-1.

Fixed hyperparameters (q34a_trial_004):
  epochs=4, batch_size=4, lr=0.001, wd=0.0001, seed=45

Outputs:
  experiments/leaderboards/q38a_preprocessing_leaderboard.csv
  experiments/results/q38a_preprocessing_summary.json

Usage:
  python3 scripts/run_q38a_preprocessing_benchmark.py
  python3 scripts/run_q38a_preprocessing_benchmark.py --dry-run  # 1 epoch, 30 batches/split

Hard constraints (Q38A):
  - No NAS, no CV quantum, no DV quantum, no augmentation
  - No multiclass, no Ray, no Optuna, no distributed execution
  - No cv2, no .numpy() call (torch-numpy bridge unavailable in container)
  - Standard library only for new dependencies
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.cnn import build_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Constants ──────────────────────────────────────────────────────────────────

BACKBONE_CHECKPOINT  = "checkpoints/c006_d040_classical_anchor.pt"
DATASET_ROOT         = "data/processed/vindr_binary_roi_224"
LEADERBOARD_PATH     = "experiments/leaderboards/q38a_preprocessing_leaderboard.csv"
SUMMARY_JSON_PATH    = "experiments/results/q38a_preprocessing_summary.json"

BACKBONE_OUT_DIM     = 128
EXPECTED_PARAMS      = 2250

# Fixed q34a_trial_004 hyperparameters
EPOCHS       = 4
BATCH_SIZE   = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 45

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

HEAD_CONFIG = {
    "backbone_family": "depthwise_separable_cnn",
    "channel_width":   [16, 32],
    "depth":           "shallow",
    "compression_dim": 8,
    "activation":      "ReLU",
    "normalization":   "BatchNorm",
    "dropout":         0.2,
}

VARIANTS = [
    "baseline",
    "clahe",
    "histogram_equalization",
    "contrast_normalization",
    "clahe_plus_normalization",
]

LEADERBOARD_FIELDNAMES = [
    "variant",
    "auroc",
    "f1",
    "accuracy",
    "params",
    "latency_ms",
    "wall_time_s",
    "preprocessing_overhead_ms",
    "delta_auroc_vs_baseline",
    "delta_f1_vs_baseline",
]


# ── Preprocessing transforms (pure PyTorch) ────────────────────────────────────
#
# All transforms: Tensor[1, H, W] float32 [0,1] → Tensor[1, H, W] float32 [0,1]
# Operations are CPU-side (transforms called in DataLoader on CPU tensors).
# No .numpy() or torch.from_numpy() — avoids numpy-2.x/torch C-API conflict.

def _clahe_torch(x: torch.Tensor, clip_limit: float = 2.0,
                 tile_grid: tuple = (8, 8)) -> torch.Tensor:
    """
    Tile-based CLAHE on Tensor[1, H, W] float32 [0,1].
    Returns Tensor[1, H, W] float32 [0,1]. Pure PyTorch, no cv2.

    Algorithm:
      1. Discretise to uint8 (256 bins).
      2. Per tile: clip histogram, redistribute excess, compute equalisation LUT.
      3. Bilinear interpolation between tile LUTs (no blocking artifacts).
    """
    H, W    = x.shape[1], x.shape[2]
    ny, nx  = tile_grid
    tile_h  = H // ny
    tile_w  = W // nx

    img_int = (x.squeeze(0) * 255.0).round().clamp(0, 255).long()  # [H, W]

    # Per-tile equalisation LUTs
    tile_luts = torch.zeros(ny, nx, 256, dtype=torch.float32)
    for i in range(ny):
        for j in range(nx):
            tile_flat = img_int[i*tile_h:(i+1)*tile_h,
                                j*tile_w:(j+1)*tile_w].reshape(-1)
            hist = torch.bincount(tile_flat, minlength=256).float()

            total_px  = float(tile_h * tile_w)
            clip_val  = max(1.0, clip_limit * total_px / 256.0)
            excess    = (hist - clip_val).clamp(min=0.0).sum()
            hist      = hist.clamp(max=clip_val) + excess / 256.0

            cdf       = hist.cumsum(0)
            nz_mask   = hist > 0
            cdf_min   = cdf[nz_mask][0] if nz_mask.any() else torch.zeros(1)
            cdf_range = cdf[-1] - cdf_min
            if cdf_range > 0:
                lut = ((cdf - cdf_min) / cdf_range * 255.0).clamp(0, 255)
            else:
                lut = torch.arange(256, dtype=torch.float32)
            tile_luts[i, j] = lut

    # Vectorised bilinear interpolation across all pixels
    fy = torch.arange(H, dtype=torch.float32) / tile_h - 0.5
    fx = torch.arange(W, dtype=torch.float32) / tile_w - 0.5

    i0 = fy.floor().long().clamp(0, ny - 1)   # (H,)
    j0 = fx.floor().long().clamp(0, nx - 1)   # (W,)
    i1 = (i0 + 1).clamp(0, ny - 1)
    j1 = (j0 + 1).clamp(0, nx - 1)

    dy = (fy - fy.floor()).clamp(0, 1)[:, None]   # (H, 1)
    dx = (fx - fx.floor()).clamp(0, 1)[None, :]   # (1, W)
    pv = img_int                                   # (H, W)

    # Advanced indexing: tile_luts[tile_row, tile_col, pixel_val] → (H, W)
    v00 = tile_luts[i0[:, None], j0[None, :], pv]
    v01 = tile_luts[i0[:, None], j1[None, :], pv]
    v10 = tile_luts[i1[:, None], j0[None, :], pv]
    v11 = tile_luts[i1[:, None], j1[None, :], pv]

    result = ((1-dy)*(1-dx)*v00 + (1-dy)*dx*v01 +
              dy*(1-dx)*v10    + dy*dx*v11)
    return (result / 255.0).clamp(0.0, 1.0).unsqueeze(0)


def make_transforms() -> dict[str, Callable]:
    """Returns variant_name → transform callable (all pure PyTorch)."""

    def baseline(x: torch.Tensor) -> torch.Tensor:
        return x

    def clahe(x: torch.Tensor) -> torch.Tensor:
        return _clahe_torch(x, clip_limit=2.0, tile_grid=(8, 8))

    def histogram_equalization(x: torch.Tensor) -> torch.Tensor:
        flat_int = (x.reshape(-1) * 255.0).round().clamp(0, 255).long()
        hist     = torch.bincount(flat_int, minlength=256).float()
        cdf      = hist.cumsum(0)
        nz_mask  = hist > 0
        cdf_min  = cdf[nz_mask][0] if nz_mask.any() else torch.zeros(1)
        cdf_range = cdf[-1] - cdf_min
        if cdf_range > 0:
            lut = ((cdf - cdf_min) / cdf_range * 255.0).clamp(0, 255)
        else:
            lut = torch.arange(256, dtype=torch.float32)
        result = lut[flat_int].reshape(x.shape) / 255.0
        return result.clamp(0.0, 1.0)

    def contrast_normalization(x: torch.Tensor) -> torch.Tensor:
        lo = torch.quantile(x, 0.02)
        hi = torch.quantile(x, 0.98)
        if hi > lo:
            x = (x - lo) / (hi - lo)
        return x.clamp(0.0, 1.0)

    def clahe_plus_normalization(x: torch.Tensor) -> torch.Tensor:
        x  = _clahe_torch(x, clip_limit=2.0, tile_grid=(8, 8))
        lo = torch.quantile(x, 0.02)
        hi = torch.quantile(x, 0.98)
        if hi > lo:
            x = (x - lo) / (hi - lo)
        return x.clamp(0.0, 1.0)

    return {
        "baseline":                baseline,
        "clahe":                   clahe,
        "histogram_equalization":  histogram_equalization,
        "contrast_normalization":  contrast_normalization,
        "clahe_plus_normalization": clahe_plus_normalization,
    }


# ── Seeds ──────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Model utilities ────────────────────────────────────────────────────────────

def build_nas_head(cfg: dict) -> nn.Sequential:
    """Build q34a_trial_004 MLP head (2250 trainable params). Mirrors train_q34a_classical_candidate.py."""
    channel_width = cfg.get("channel_width",   [16, 32])
    depth         = cfg.get("depth",           "shallow")
    compress_dim  = cfg.get("compression_dim", 8)
    act_name      = cfg.get("activation",      "ReLU")
    norm_name     = cfg.get("normalization",   "BatchNorm")
    dropout_rate  = float(cfg.get("dropout",   0.0))

    hidden_dims = [channel_width[0]] if depth == "shallow" else list(channel_width)

    layers: list[nn.Module] = []
    in_dim = BACKBONE_OUT_DIM
    for hd in hidden_dims:
        layers.append(nn.Linear(in_dim, hd))
        if norm_name in ("BatchNorm", "LayerNorm"):
            layers.append(nn.LayerNorm(hd))
        layers.append(nn.ReLU() if act_name == "ReLU" else nn.GELU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = hd
    layers.append(nn.Linear(in_dim, compress_dim))
    layers.append(nn.ReLU() if act_name == "ReLU" else nn.GELU())
    layers.append(nn.Linear(compress_dim, 2))
    return nn.Sequential(*layers)


class NASClassicalModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Sequential) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def train(self, mode: bool = True) -> "NASClassicalModel":
        super().train(mode)
        self.backbone.eval()   # backbone always frozen in eval mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)   # (B, 128)
        return self.head(features)


def load_backbone(ck_path: str) -> nn.Module:
    full_model = build_model(CNN_CONFIG)
    backbone   = nn.Sequential(*list(full_model.children())[:4])

    ck     = torch.load(ck_path, map_location="cpu", weights_only=False)
    src_sd = ck.get("model_state_dict") or ck.get("state_dict")
    if src_sd is None:
        print("ERROR: unrecognised checkpoint format", flush=True)
        sys.exit(1)

    remapped = {f"backbone.{k}": v for k, v in src_sd.items()
                if k.startswith("0.") or k.startswith("1.")}
    tmp          = nn.Module()
    tmp.backbone = backbone
    missing, _   = tmp.load_state_dict(remapped, strict=False)
    missing_bb   = [k for k in missing if k.startswith("backbone.")]
    if missing_bb:
        print(f"ERROR: {len(missing_bb)} backbone keys missing", flush=True)
        sys.exit(1)

    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


# ── Evaluation & latency ───────────────────────────────────────────────────────

def eval_split(model: NASClassicalModel, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = criterion(logits, y)
            total_loss += loss.item()
            probs   = torch.softmax(logits, dim=1)[:, 1]
            preds   = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_prob   = np.array(y_prob)
    has_both = len(np.unique(y_true)) > 1
    return {
        "loss":     total_loss / max(1, len(loader)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


def measure_latency(model: NASClassicalModel, device: torch.device,
                    n_warmup: int = 10, n_samples: int = 50) -> float:
    """Mean ms per single-sample forward pass."""
    model.eval()
    x = torch.randn(1, 1, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        timings = []
        for _ in range(n_samples):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


def measure_preprocessing_overhead(transform: Callable,
                                    dataset: VinDrSpineXRBinaryDataset,
                                    n_samples: int = 100) -> float:
    """
    Mean ms per sample for the transform callable.
    Temporarily disables the dataset's transform to collect raw tensors first.
    """
    n = min(n_samples, len(dataset))

    # Collect raw tensors without transform
    orig_transform = dataset.transform
    dataset.transform = None
    raw_tensors = [dataset[i][0] for i in range(n)]
    dataset.transform = orig_transform

    timings = []
    for xt in raw_tensors:
        t0 = time.perf_counter()
        transform(xt)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


# ── Per-variant runner ─────────────────────────────────────────────────────────

def run_variant(variant: str, transform: Callable, device: torch.device,
                epochs: int, dry_run_batches: Optional[int] = None) -> dict:
    """Train trial_004 head from scratch with the given preprocessing transform."""
    set_seeds(SEED)

    backbone = load_backbone(BACKBONE_CHECKPOINT)
    head     = build_nas_head(HEAD_CONFIG)
    model    = NASClassicalModel(backbone, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != EXPECTED_PARAMS:
        print(f"  WARNING: expected {EXPECTED_PARAMS} trainable params, got {n_trainable}",
              flush=True)

    train_ds = VinDrSpineXRBinaryDataset(root=DATASET_ROOT, split="train", transform=transform)
    val_ds   = VinDrSpineXRBinaryDataset(root=DATASET_ROOT, split="val",   transform=transform)
    test_ds  = VinDrSpineXRBinaryDataset(root=DATASET_ROOT, split="test",  transform=transform)

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
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0

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
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.size(0)
            n_batches    += 1

        t_epoch = time.perf_counter() - t0
        val_m   = eval_split(model, val_loader, criterion, device)
        print(
            f"  [{variant}] epoch {epoch}/{epochs} "
            f"train_loss={running_loss/max(1,n_batches):.4f} "
            f"val_auroc={val_m['auroc']:.4f} val_f1={val_m['f1']:.4f} "
            f"t={t_epoch:.1f}s",
            flush=True,
        )

    test_m    = eval_split(model, test_loader, criterion, device)
    lat_ms    = measure_latency(model, device)
    wall_time = time.perf_counter() - wall_start

    # Preprocessing overhead: timed on raw CPU tensors from test split
    pp_ms = measure_preprocessing_overhead(transform, test_ds, n_samples=100)

    print(
        f"  [{variant}] TEST auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
        f"acc={test_m['accuracy']:.4f} latency={lat_ms:.2f}ms "
        f"pp_overhead={pp_ms:.3f}ms wall={wall_time:.1f}s",
        flush=True,
    )

    return {
        "variant":                   variant,
        "auroc":                     test_m["auroc"],
        "f1":                        test_m["f1"],
        "accuracy":                  test_m["accuracy"],
        "params":                    n_trainable,
        "latency_ms":                lat_ms,
        "wall_time_s":               wall_time,
        "preprocessing_overhead_ms": pp_ms,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q38A Binary Preprocessing Benchmark")
    p.add_argument("--dry-run", action="store_true",
                   help="Quick validation: 1 epoch, 30 batches per split")
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANTS,
                   help="Run only specific variants (default: all 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — Q38A requires GPU", flush=True)
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
    variants_to_run = args.variants if args.variants else VARIANTS

    print(f"[Q38A] device:   {device}",                          flush=True)
    print(f"[Q38A] epochs:   {epochs}{'  (dry-run)' if args.dry_run else ''}", flush=True)
    print(f"[Q38A] variants: {variants_to_run}",                 flush=True)
    print(f"[Q38A] seed:     {SEED}",                            flush=True)

    transforms   = make_transforms()
    results: list[dict] = []
    total_start  = time.perf_counter()

    for variant in variants_to_run:
        print(f"\n[Q38A] ── variant: {variant} ────────────────────────────────", flush=True)
        t0     = time.perf_counter()
        result = run_variant(
            variant         = variant,
            transform       = transforms[variant],
            device          = device,
            epochs          = epochs,
            dry_run_batches = dry_run_batches,
        )
        results.append(result)
        print(f"[Q38A] variant {variant} done in {time.perf_counter()-t0:.1f}s", flush=True)

    # Delta vs baseline
    baseline_row = next((r for r in results if r["variant"] == "baseline"), None)
    for r in results:
        if baseline_row is not None:
            r["delta_auroc_vs_baseline"] = r["auroc"] - baseline_row["auroc"]
            r["delta_f1_vs_baseline"]    = r["f1"]    - baseline_row["f1"]
        else:
            r["delta_auroc_vs_baseline"] = float("nan")
            r["delta_f1_vs_baseline"]    = float("nan")

    # ── Write leaderboard CSV ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    with open(LEADERBOARD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()
                if k in LEADERBOARD_FIELDNAMES
            })
    print(f"\n[Q38A] leaderboard written: {LEADERBOARD_PATH}", flush=True)

    # ── Write summary JSON ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SUMMARY_JSON_PATH), exist_ok=True)
    summary = {
        "slice":               "Q38A",
        "date":                time.strftime("%Y-%m-%d"),
        "dry_run":             args.dry_run,
        "epochs":              epochs,
        "seed":                SEED,
        "fixed_params":        EXPECTED_PARAMS,
        "backbone_checkpoint": BACKBONE_CHECKPOINT,
        "dataset_root":        DATASET_ROOT,
        "variants":            results,
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Q38A] summary JSON written: {SUMMARY_JSON_PATH}", flush=True)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n[Q38A] ── Results Summary ──────────────────────────────────────────",
          flush=True)
    hdr = f"{'Variant':<28} {'AUROC':>6}  {'F1':>6}  {'Acc':>6}  {'ΔAUROC':>7}  {'ΔF1':>7}  {'PP ms':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        print(
            f"{r['variant']:<28} "
            f"{r['auroc']:>6.4f}  "
            f"{r['f1']:>6.4f}  "
            f"{r['accuracy']:>6.4f}  "
            f"{r['delta_auroc_vs_baseline']:>+7.4f}  "
            f"{r['delta_f1_vs_baseline']:>+7.4f}  "
            f"{r['preprocessing_overhead_ms']:>6.3f}",
            flush=True,
        )

    best    = max(results, key=lambda r: r["auroc"])
    best_f1 = max(results, key=lambda r: r["f1"])
    print(f"\n[Q38A] Best by AUROC: {best['variant']} (AUROC={best['auroc']:.4f})",    flush=True)
    print(f"[Q38A] Best by F1:    {best_f1['variant']} (F1={best_f1['f1']:.4f})",      flush=True)
    print(f"[Q38A] Total wall:    {time.perf_counter()-total_start:.1f}s",             flush=True)

    print("\nQ38A_PREPROCESSING_BENCHMARK: PASS", flush=True)


if __name__ == "__main__":
    main()
