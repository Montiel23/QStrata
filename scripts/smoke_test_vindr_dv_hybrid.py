"""
scripts/smoke_test_vindr_dv_hybrid.py

DV Hybrid CNN-QNN smoke test for VinDr-SpineXR binary ROI dataset.
Slice Q18.

Validates end-to-end mechanics:
  VinDr loader → CNN backbone → projection → medical_ansatz DV circuit
    → quantum readout → loss → backward → optimizer step → tiny train/val

Hard limits:
  - max_train_batches (default 3): train loop exits immediately after this many batches
  - max_val_batches   (default 2): val loop exits after this many batches
  - per_batch_timeout_s (120s):   if any single batch exceeds this, abort immediately
  - Epochs: 1
  - No test evaluation, no checkpoint saving

Compatibility note:
  DVHybridCNNQNN uses build_model(cnn_config)[:4] as backbone.
  Layers [:4] = [depthwise_sep_block, depthwise_sep_block, AdaptiveAvgPool2d(1,1), Flatten]
  Since AdaptiveAvgPool2d(1,1) reduces any (H, W) to (1, 1), the backbone output
  is always (B, 128) regardless of spatial input size.
  VinDr 224×224 input is FULLY COMPATIBLE with the existing DVHybridCNNQNN interface
  — no modification to dv_hybrid_cnn_qnn.py is required.

Device: CPU — the DV quantum circuit simulator is CPU-only.

Usage:
    python scripts/smoke_test_vindr_dv_hybrid.py \\
        --root data/processed/vindr_binary_roi_224 \\
        --batch-size 4 \\
        --max-train-batches 3 \\
        --max-val-batches 2 \\
        --seed 42
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN

# ── Constants ──────────────────────────────────────────────────────────────────

PER_BATCH_TIMEOUT_S = 120.0   # abort if any single batch exceeds this

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,      # irrelevant — backbone is [:4], ends at Flatten
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR DV Hybrid smoke test (Slice Q18)"
    )
    p.add_argument("--root", default="data/processed/vindr_binary_roi_224")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train-batches", type=int, default=3)
    p.add_argument("--max-val-batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


def grad_norm_named(model: nn.Module, *param_names: str) -> float:
    """Compute L2 gradient norm across named parameters."""
    total = 0.0
    for name in param_names:
        p = dict(model.named_parameters()).get(name)
        if p is not None and p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # Device: CPU — quantum simulator is CPU-only
    device = torch.device("cpu")

    print("=== VinDr-SpineXR DV Hybrid Smoke Test ===")
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

    # ── Model ─────────────────────────────────────────────────────────────────
    # DVHybridCNNQNN is spatially compatible with 224×224 input:
    # backbone = build_model(cnn_config)[:4]
    # Layer [2] = AdaptiveAvgPool2d(1,1) → always outputs (B, 128) for any H×W
    # No modification to dv_hybrid_cnn_qnn.py required.
    model = DVHybridCNNQNN(
        cnn_config=CNN_CONFIG,
        n_qubits=4,
        depth=1,
        alpha=0.1,
        n_classes=2,
    )
    # No pretrained backbone loaded — smoke test uses random backbone weights.
    n_trainable = model.trainable_param_count()
    n_frozen    = model.frozen_param_count()

    print("Model:")
    print("  name: DVHybridCNNQNN (existing, no modification — 224×224 compatible via AdaptiveAvgPool2d)")
    print(f"  trainable params: {n_trainable}")
    print(f"  frozen params   : {n_frozen}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # ── Batch validation (1 sample batch) ─────────────────────────────────────
    train_iter = iter(train_loader)
    x_val, y_val = next(train_iter)
    # Keep on CPU — quantum circuit is CPU-only
    B = x_val.shape[0]

    if tuple(x_val.shape) != (B, 1, 224, 224):
        fail(f"x shape {tuple(x_val.shape)}, expected ({B}, 1, 224, 224)")
    if tuple(y_val.shape) != (B,):
        fail(f"y shape {tuple(y_val.shape)}, expected ({B},)")
    label_set = set(y_val.tolist())
    if not label_set.issubset({0, 1}):
        fail(f"label values {label_set} not ⊆ {{0, 1}}")

    print("Batch validation:")
    print(f"  x shape: {tuple(x_val.shape)}")
    print(f"  y shape: {tuple(y_val.shape)}")
    print(f"  labels valid: PASS")
    print()

    # ── Forward / backward / optimizer step (standalone validation) ───────────
    print("Running standalone forward/backward (batch_size={}, 4-qubit circuit)...".format(B))
    sys.stdout.flush()

    model.train()
    optimizer.zero_grad()

    t_fwd_start = time.perf_counter()
    logits = model(x_val)
    t_fwd = time.perf_counter() - t_fwd_start

    # Check 120s limit for the forward pass (which contains the quantum loop)
    if t_fwd > PER_BATCH_TIMEOUT_S:
        fail(
            f"Single forward pass exceeded {PER_BATCH_TIMEOUT_S}s limit "
            f"(observed: {t_fwd:.1f}s). Bottleneck likely in quantum circuit loop. "
            f"Batch size={B}, n_qubits=4. Stop — do not continue."
        )

    if tuple(logits.shape) != (B, 2):
        fail(f"logits shape {tuple(logits.shape)}, expected ({B}, 2)")
    if not torch.isfinite(logits).all():
        fail("logits contain non-finite values")

    loss = criterion(logits, y_val)
    if not torch.isfinite(loss):
        fail(f"loss is not finite: {loss.item()}")

    loss.backward()

    # Grad norms
    theta_gn   = model.theta.grad.detach().norm(2).item() if model.theta.grad is not None else 0.0
    proj_gn    = grad_norm_named(model, "proj.weight", "proj.bias")
    readout_gn = grad_norm_named(model, "readout.weight", "readout.bias")

    if theta_gn == 0.0:
        fail("theta grad norm is 0 — quantum parameters not receiving gradient")
    if readout_gn == 0.0:
        fail("readout grad norm is 0 — readout layer not receiving gradient")

    # Probability sum check (from model._prob_sums populated during forward)
    if model._prob_sums:
        prob_sum_mean = float(np.mean(model._prob_sums))
        prob_sum_ok   = abs(prob_sum_mean - 1.0) < 0.01
        prob_sum_str  = f"{prob_sum_mean:.6f} / {'PASS' if prob_sum_ok else 'WARN'}"
    else:
        prob_sum_str = "N/A"
        prob_sum_ok  = True

    # Clone params before optimizer step
    param_clones = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    optimizer.step()

    # Max param delta
    max_delta = max(
        (p.detach() - param_clones[name]).abs().max().item()
        for name, p in model.named_parameters()
        if p.requires_grad and name in param_clones
    )
    if max_delta == 0.0:
        fail("max param delta is 0 — optimizer.step() did not update any parameter")

    print("Forward/backward:")
    print(f"  logits shape     : {tuple(logits.shape)}")
    print(f"  loss finite      : PASS  (loss={loss.item():.4f}, fwd_time={t_fwd:.1f}s)")
    print(f"  backward         : PASS")
    print(f"  theta grad norm  : {theta_gn:.2e}")
    print(f"  proj  grad norm  : {proj_gn:.2e}")
    print(f"  readout grad norm: {readout_gn:.2e}")
    print(f"  prob sum check   : {prob_sum_str}")
    print(f"  max param delta  : {max_delta:.2e}")
    print(f"  param update     : PASS")
    print()

    # ── Tiny train loop — 1 epoch, hard cap at max_train_batches ─────────────
    model.train()
    train_loss_sum = 0.0
    train_correct  = 0
    train_total    = 0
    train_batches  = 0

    for x, y in train_loader:
        if train_batches >= args.max_train_batches:
            break

        t_batch_start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        t_batch = time.perf_counter() - t_batch_start

        # 120-second per-batch hard limit
        if t_batch > PER_BATCH_TIMEOUT_S:
            fail(
                f"Train batch {train_batches + 1} exceeded {PER_BATCH_TIMEOUT_S}s limit "
                f"(observed: {t_batch:.1f}s). "
                f"Suspected bottleneck: quantum circuit loop (per-sample DV simulation). "
                f"n_qubits=4, batch_size={x.shape[0]}. Stop — do not continue."
            )

        train_loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total   += y.shape[0]
        train_batches += 1

        print(
            f"  [train] batch {train_batches}/{args.max_train_batches} | "
            f"loss={loss.item():.4f} | "
            f"time={t_batch:.1f}s",
            flush=True,
        )

    if train_batches > args.max_train_batches:
        fail(f"Train loop ran {train_batches} batches — exceeded hard cap of {args.max_train_batches}")

    mean_train_loss = train_loss_sum / train_batches if train_batches > 0 else float("nan")
    train_acc       = train_correct / train_total    if train_total   > 0 else float("nan")

    print(f"\nTiny training (1 epoch, {args.max_train_batches} batches):")
    print(f"  train batches processed: {train_batches}")
    print(f"  train loss: {mean_train_loss:.4f}")
    print(f"  train acc:  {train_acc:.4f}")
    print()

    # ── Tiny val loop — hard cap at max_val_batches, no gradient ─────────────
    model.eval()
    val_loss_sum = 0.0
    val_correct  = 0
    val_total    = 0
    val_batches  = 0

    with torch.no_grad():
        for x, y in val_loader:
            if val_batches >= args.max_val_batches:
                break

            t_batch_start = time.perf_counter()
            logits = model(x)
            t_batch = time.perf_counter() - t_batch_start

            if t_batch > PER_BATCH_TIMEOUT_S:
                fail(
                    f"Val batch {val_batches + 1} exceeded {PER_BATCH_TIMEOUT_S}s limit "
                    f"(observed: {t_batch:.1f}s). Stop — do not continue."
                )

            loss   = criterion(logits, y)
            preds  = logits.argmax(dim=1)
            val_loss_sum += loss.item()
            val_correct  += (preds == y).sum().item()
            val_total    += y.shape[0]
            val_batches  += 1

            print(
                f"  [val]   batch {val_batches}/{args.max_val_batches} | "
                f"loss={loss.item():.4f} | "
                f"time={t_batch:.1f}s",
                flush=True,
            )

    mean_val_loss = val_loss_sum / val_batches if val_batches > 0 else float("nan")
    val_acc       = val_correct / val_total    if val_total   > 0 else float("nan")

    print(f"\nTiny validation ({args.max_val_batches} batches):")
    print(f"  val batches processed: {val_batches}")
    print(f"  val loss: {mean_val_loss:.4f}")
    print(f"  val acc:  {val_acc:.4f}")
    print()

    print("DV hybrid smoke test: PASS")


if __name__ == "__main__":
    main()
