"""
scripts/smoke_test_vindr_classical_baseline.py

Minimal classical CNN training mechanics smoke test for the VinDr-SpineXR
binary ROI dataset.

Validates: loader → model → forward → loss → backward → optimizer step →
           tiny train loop (5 batches) → tiny val loop (3 batches).

Hard caps enforced:
  - max_train_batches (default 5): train loop exits after this many batches
  - max_val_batches   (default 3): val loop exits after this many batches
  - Epochs: 1

No test evaluation. No checkpoint saving. No QNN work.

Usage:
    python scripts/smoke_test_vindr_classical_baseline.py \
        --root data/processed/vindr_binary_roi_224 \
        --batch-size 8 \
        --max-train-batches 5 \
        --max-val-batches 3 \
        --seed 42
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path setup (run from repo root without package install) ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.cnn import build_model  # reuse existing config-driven CNN

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical CNN baseline smoke test for VinDr-SpineXR binary dataset"
    )
    p.add_argument("--root", default="data/processed/vindr_binary_roi_224",
                   help="Exported dataset root")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-train-batches", type=int, default=5)
    p.add_argument("--max-val-batches", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== VinDr-SpineXR Classical Baseline Smoke Test ===")
    print(f"Root: {args.root}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print()

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=args.root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("Dataset:")
    print(f"  train length: {len(train_ds)}")
    print(f"  val length:   {len(val_ds)}")
    print()

    # ── Model (reuse qcore/models/cnn.py build_model) ─────────────────────────
    # 3-block standard CNN: [1→16→32→64] + AdaptiveAvgPool2d + Linear(64,2)
    # Compatible with input [B, 1, H, W] for any H, W.
    model_config = {
        "conv_channels":  [16, 32, 64],
        "use_batchnorm":  True,
        "dropout":        0.0,
        "pooling":        "adaptive_avg",
        "input_channels": 1,
        "num_classes":    2,
        "block_type":     "standard",
    }
    model = build_model(model_config).to(device)
    n_params = count_params(model)

    print("Model:")
    print("  name: CNN3Block  (build_model from qcore.models.cnn)")
    print(f"  params: {n_params:,}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Batch validation (1 sample batch from train) ──────────────────────────
    train_iter = iter(train_loader)
    x_val_batch, y_val_batch = next(train_iter)
    x_val_batch = x_val_batch.to(device)
    y_val_batch = y_val_batch.to(device)

    B = x_val_batch.shape[0]

    if tuple(x_val_batch.shape) != (B, 1, 224, 224):
        fail(
            f"x shape {tuple(x_val_batch.shape)}, "
            f"expected ({B}, 1, 224, 224)"
        )
    if tuple(y_val_batch.shape) != (B,):
        fail(f"y shape {tuple(y_val_batch.shape)}, expected ({B},)")
    label_set = set(y_val_batch.tolist())
    if not label_set.issubset({0, 1}):
        fail(f"label values {label_set} not ⊆ {{0, 1}}")

    print("Batch validation:")
    print(f"  x shape: {tuple(x_val_batch.shape)}")
    print(f"  y shape: {tuple(y_val_batch.shape)}")
    print(f"  labels valid: PASS")
    print()

    # ── Forward / backward / optimizer step (standalone validation) ───────────
    model.train()
    optimizer.zero_grad()

    logits_val = model(x_val_batch)

    if tuple(logits_val.shape) != (B, 2):
        fail(f"logits shape {tuple(logits_val.shape)}, expected ({B}, 2)")
    if not torch.isfinite(logits_val).all():
        fail("logits contain non-finite values")

    loss_val = criterion(logits_val, y_val_batch)
    if not torch.isfinite(loss_val):
        fail(f"loss is not finite: {loss_val.item()}")

    loss_val.backward()

    gnorm = grad_norm(model)
    if gnorm == 0.0:
        fail("grad norm is 0 — backward pass produced no gradients")

    # Clone parameters before optimizer step
    param_clones = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    optimizer.step()

    # Compute max parameter delta across all parameters
    max_delta = max(
        (p.detach() - param_clones[name]).abs().max().item()
        for name, p in model.named_parameters()
    )
    if max_delta == 0.0:
        fail("max param delta is 0 — optimizer.step() did not update any parameter")

    print("Forward/backward:")
    print(f"  logits shape: {tuple(logits_val.shape)}")
    print(f"  loss finite:  PASS  (loss={loss_val.item():.4f})")
    print(f"  grad norm:    {gnorm:.4f}")
    print(f"  max param delta: {max_delta:.2e}")
    print(f"  param update: PASS")
    print()

    # ── Tiny train loop — 1 epoch, hard cap at max_train_batches ─────────────
    model.train()
    train_loss_sum  = 0.0
    train_correct   = 0
    train_total     = 0
    train_batches   = 0

    # Restart iterator for a clean epoch (the standalone step consumed 1 batch
    # from the previous iterator; start fresh for the reported metrics).
    for x, y in train_loader:
        if train_batches >= args.max_train_batches:
            break  # hard cap enforced

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        preds           = logits.argmax(dim=1)
        train_correct  += (preds == y).sum().item()
        train_total    += y.shape[0]
        train_batches  += 1

    # Paranoia guard — should never trigger given the break above
    if train_batches > args.max_train_batches:
        fail(
            f"Train loop processed {train_batches} batches — "
            f"exceeded hard cap of {args.max_train_batches}"
        )

    mean_train_loss = train_loss_sum / train_batches if train_batches > 0 else float("nan")
    train_acc       = train_correct / train_total    if train_total   > 0 else float("nan")

    print(f"Tiny training (1 epoch, {args.max_train_batches} batches):")
    print(f"  train batches processed: {train_batches}")
    print(f"  train loss: {mean_train_loss:.4f}")
    print(f"  train acc:  {train_acc:.4f}")
    print()

    # ── Tiny val loop — hard cap at max_val_batches, no gradient ─────────────
    model.eval()
    val_loss_sum  = 0.0
    val_correct   = 0
    val_total     = 0
    val_batches   = 0

    with torch.no_grad():
        for x, y in val_loader:
            if val_batches >= args.max_val_batches:
                break  # hard cap enforced

            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = criterion(logits, y)

            val_loss_sum  += loss.item()
            preds          = logits.argmax(dim=1)
            val_correct   += (preds == y).sum().item()
            val_total     += y.shape[0]
            val_batches   += 1

    mean_val_loss = val_loss_sum / val_batches if val_batches > 0 else float("nan")
    val_acc       = val_correct  / val_total   if val_total   > 0 else float("nan")

    print(f"Tiny validation ({args.max_val_batches} batches):")
    print(f"  val batches processed: {val_batches}")
    print(f"  val loss: {mean_val_loss:.4f}")
    print(f"  val acc:  {val_acc:.4f}")
    print()

    print("Smoke test: PASS")


if __name__ == "__main__":
    main()
