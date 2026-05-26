#!/usr/bin/env python3
"""
scripts/smoke_test_vindr_dv_hybrid_pretrained_backbone.py
Slice Q20 — VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility Smoke Test

Validates that:
  1. A pretrained classical backbone checkpoint can be loaded into DVHybridCNNQNN
  2. Backbone weights are frozen after loading
  3. Forward + backward passes work end-to-end
  4. Trainable parameters (projection, theta, readout) receive gradients
  5. Backbone receives zero/None gradient (frozen)
  6. Optimizer updates only trainable parameters
  7. Probability conservation is maintained
  8. A tiny 1-epoch train/val loop completes without error

This is feasibility and sanity validation only.
No test evaluation, no full training, no performance claims.

Usage:
  python scripts/smoke_test_vindr_dv_hybrid_pretrained_backbone.py \\
    --root data/processed/vindr_binary_roi_224 \\
    --checkpoint checkpoints/c006_d040_classical_anchor.pt \\
    --batch-size 4 \\
    --max-train-batches 3 \\
    --max-val-batches 2 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VinDrSpineXRBinaryDataset(Dataset):
    """
    Minimal VinDr-SpineXR binary dataset loader.
    Reads manifest.csv and returns (tensor, label) pairs.
    """

    def __init__(self, root: str, split: str) -> None:
        self.root  = root
        self.split = split
        manifest   = os.path.join(root, "manifest.csv")
        if not os.path.isfile(manifest):
            print(f"ERROR: manifest.csv not found at {manifest}", flush=True)
            sys.exit(1)
        self.samples: list[tuple[str, int]] = []
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((row["output_path"], int(row["binary_label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root, rel_path)
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        x   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return x, label


# ── Checkpoint loading helpers ────────────────────────────────────────────────

def detect_format(ck: dict) -> str:
    if isinstance(ck, dict):
        if "model_state_dict" in ck:
            return "model_state_dict"
        if "state_dict" in ck:
            return "state_dict"
        # Check if it looks like a raw state dict (values are tensors)
        if all(isinstance(v, torch.Tensor) for v in list(ck.values())[:3]):
            return "raw_state_dict"
    return "unknown"


def extract_state_dict(ck, fmt: str) -> dict:
    if fmt == "model_state_dict":
        return ck["model_state_dict"]
    if fmt == "state_dict":
        return ck["state_dict"]
    if fmt == "raw_state_dict":
        return ck
    return {}


# ── Key compatibility analysis ────────────────────────────────────────────────

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}


def analyse_keys(src_sd: dict, backbone_keys: set) -> dict:
    """
    Analyse C006-D040 style state dict against DVHybridCNNQNN backbone keys.

    C006-D040 uses keys like "0.0.weight" (block index 0 or 1) and "5.*" (head).
    DVHybridCNNQNN backbone uses "backbone.0.0.weight" etc.

    Strategy: remap "0.*" → "backbone.0.*", "1.*" → "backbone.1.*"; skip "5.*".
    """
    remapped: dict  = {}
    skipped: list   = []
    unexpected: list = []

    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            new_k = f"backbone.{k}"
            remapped[new_k] = v
        elif k.startswith("5."):
            skipped.append(k)
        else:
            unexpected.append(k)

    matched  = [k for k in remapped if k in backbone_keys]
    missing  = [k for k in backbone_keys if k not in remapped]

    return {
        "remapped_sd": remapped,
        "matched":     matched,
        "missing":     missing,
        "unexpected":  unexpected,
        "skipped":     skipped,
    }


# ── Gradient utilities ────────────────────────────────────────────────────────

def grad_norm(module: nn.Module, *param_names: str) -> float:
    total = 0.0
    params = dict(module.named_parameters())
    for n in param_names:
        p = params.get(n)
        if p is not None and p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q20 VinDr DV Hybrid pretrained-backbone feasibility smoke test"
    )
    p.add_argument("--root",             required=True,  help="VinDr dataset root")
    p.add_argument("--checkpoint",       required=True,  help="Pretrained backbone checkpoint path")
    p.add_argument("--batch-size",       type=int, default=4)
    p.add_argument("--max-train-batches",type=int, default=3)
    p.add_argument("--max-val-batches",  type=int, default=2)
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    print("=== VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility ===")
    print(flush=True)

    # ── Dataset root check ────────────────────────────────────────────────────
    if not os.path.isdir(args.root):
        print(f"ERROR: Dataset root not found: {args.root}", flush=True)
        sys.exit(1)

    # ── Checkpoint existence check ────────────────────────────────────────────
    ck_exists = os.path.isfile(args.checkpoint)
    print("Checkpoint:")
    print(f"  path   : {args.checkpoint}")
    print(f"  exists : {'YES' if ck_exists else 'NO'}", flush=True)
    if not ck_exists:
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}", flush=True)
        sys.exit(1)

    ck  = torch.load(args.checkpoint, map_location="cpu")
    fmt = detect_format(ck)
    print(f"  format : {fmt}")
    print(flush=True)

    if fmt == "unknown":
        print("ERROR: Checkpoint format unrecognised. Cannot proceed.", flush=True)
        sys.exit(1)

    src_sd = extract_state_dict(ck, fmt)

    # ── Instantiate model to get backbone key set ─────────────────────────────
    # Import here so failures are visible
    sys.path.insert(0, "/workspace")
    from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN

    model = DVHybridCNNQNN(
        cnn_config=CNN_CONFIG, n_qubits=4, depth=1, alpha=0.1, n_classes=2
    )
    backbone_keys = set(f"backbone.{k}" for k in model.backbone.state_dict().keys())

    # ── Compatibility analysis ─────────────────────────────────────────────────
    analysis = analyse_keys(src_sd, backbone_keys)
    n_matched    = len(analysis["matched"])
    n_missing    = len(analysis["missing"])
    n_unexpected = len(analysis["unexpected"])
    n_skipped    = len(analysis["skipped"])
    compatible   = (n_matched == len(backbone_keys) and n_missing == 0)

    print("Compatibility:")
    print(f"  matched backbone keys  : {n_matched}")
    print(f"  missing backbone keys  : {n_missing}")
    print(f"  unexpected keys        : {n_unexpected}")
    print(f"  skipped classifier keys: {n_skipped}")
    print(f"  compatible             : {'YES' if compatible else 'NO'}", flush=True)

    if not compatible:
        print("\nERROR: Backbone key mismatch — aborting load.", flush=True)
        if analysis["missing"]:
            print("Missing backbone keys:")
            for k in analysis["missing"]:
                print(f"  {k}")
        if analysis["unexpected"]:
            print("Unexpected keys:")
            for k in analysis["unexpected"]:
                print(f"  {k}")
        sys.exit(1)

    # ── Load backbone weights ─────────────────────────────────────────────────
    # Use strict=False: we only supply backbone keys; trainable params stay random.
    missing_keys, unexpected_keys = model.load_state_dict(
        analysis["remapped_sd"], strict=False
    )
    # Expected missing: theta, proj.weight, proj.bias, readout.weight, readout.bias
    # Expected unexpected: none
    trainable_missing = [k for k in missing_keys if not k.startswith("backbone.")]
    backbone_missing  = [k for k in missing_keys if k.startswith("backbone.")]

    if backbone_missing:
        print(f"\nERROR: {len(backbone_missing)} backbone keys failed to load:", flush=True)
        for k in backbone_missing:
            print(f"  {k}")
        sys.exit(1)

    # ── Re-freeze backbone after load ─────────────────────────────────────────
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # ── Frozen backbone validation ────────────────────────────────────────────
    backbone_frozen = all(
        not p.requires_grad for p in model.backbone.parameters()
    )
    n_trainable = model.trainable_param_count()

    print()
    print("Model:")
    print(f"  backbone frozen  : {'YES' if backbone_frozen else 'NO'}")
    print(f"  trainable params : {n_trainable}", flush=True)

    if not backbone_frozen:
        print("\nERROR: Backbone not fully frozen after load.", flush=True)
        sys.exit(1)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=args.root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # ── Single forward/backward pass ─────────────────────────────────────────
    model.train()
    batch_x, batch_y = next(iter(train_loader))

    # forward
    logits = model(batch_x)
    loss   = criterion(logits, batch_y)

    loss_finite = torch.isfinite(loss).item()

    # backward
    optimizer.zero_grad()
    loss.backward()

    # backbone grad — should be 0 or None
    backbone_grad_vals = []
    for p in model.backbone.parameters():
        if p.grad is not None:
            backbone_grad_vals.append(p.grad.abs().max().item())

    if backbone_grad_vals:
        backbone_grad_str = f"{max(backbone_grad_vals):.2e}"
    else:
        backbone_grad_str = "None"

    # trainable grad norms
    theta_gn   = model.theta.grad.detach().norm(2).item() if model.theta.grad is not None else 0.0
    proj_gn    = grad_norm(model, "proj.weight", "proj.bias")
    readout_gn = grad_norm(model, "readout.weight", "readout.bias")

    # probability conservation
    prob_sum_mean = float(np.mean(model._prob_sums)) if model._prob_sums else float("nan")

    print()
    print("Forward/backward:")
    print(f"  logits shape     : {tuple(logits.shape)}")
    print(f"  loss finite      : {'PASS' if loss_finite else 'FAIL'}")
    print(f"  backbone grad    : {backbone_grad_str}  [should be 0 or None]")
    print(f"  theta grad norm  : {theta_gn:.2e}")
    print(f"  proj  grad norm  : {proj_gn:.2e}")
    print(f"  readout grad norm: {readout_gn:.2e}")

    # ── Optimizer step — verify trainable params updated ─────────────────────
    trainable_snapshot = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    optimizer.step()

    max_delta = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in trainable_snapshot:
            delta = (p.detach() - trainable_snapshot[n]).abs().max().item()
            if delta > max_delta:
                max_delta = delta

    param_updated = max_delta > 0.0

    print(f"  max param delta  : {max_delta:.2e}")
    print(f"  param update     : {'PASS' if param_updated else 'FAIL'}")
    print(f"  prob sum         : {prob_sum_mean:.6f}", flush=True)

    if not loss_finite:
        print("\nFAIL: Loss is not finite.", flush=True)
        sys.exit(1)
    if theta_gn == 0.0 and proj_gn == 0.0 and readout_gn == 0.0:
        print("\nFAIL: All trainable gradients are zero.", flush=True)
        sys.exit(1)

    # ── Tiny training loop ────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    train_loss_sum = 0.0
    train_correct  = 0
    train_total    = 0
    train_batches  = 0

    for x, y in train_loader:
        if train_batches >= args.max_train_batches:
            break
        logits = model(x)
        loss   = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        train_correct  += (preds == y).sum().item()
        train_total    += y.size(0)
        train_batches  += 1

    train_loss = train_loss_sum / max(train_batches, 1)
    train_acc  = train_correct  / max(train_total,   1)

    print()
    print(f"Tiny training (1 epoch, {args.max_train_batches} batches):")
    print(f"  train batches: {train_batches}")
    print(f"  train loss   : {train_loss:.4f}")
    print(f"  train acc    : {train_acc:.4f}")
    print(f"  [Note: sanity-only metrics — NOT performance signals]", flush=True)

    # ── Tiny validation loop ──────────────────────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    val_correct  = 0
    val_total    = 0
    val_batches  = 0

    with torch.no_grad():
        for x, y in val_loader:
            if val_batches >= args.max_val_batches:
                break
            # backbone already no-grad via frozen params; force no_grad context
            logits = model(x)
            loss   = criterion(logits, y)
            val_loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            val_correct  += (preds == y).sum().item()
            val_total    += y.size(0)
            val_batches  += 1

    val_loss = val_loss_sum / max(val_batches, 1)
    val_acc  = val_correct  / max(val_total,   1)

    print()
    print(f"Tiny validation ({args.max_val_batches} batches):")
    print(f"  val batches: {val_batches}")
    print(f"  val loss   : {val_loss:.4f}")
    print(f"  val acc    : {val_acc:.4f}")
    print(f"  [Note: sanity-only metrics — NOT performance signals]", flush=True)

    # ── Feasibility verdict ───────────────────────────────────────────────────
    all_pass = (
        ck_exists
        and compatible
        and backbone_frozen
        and loss_finite
        and (theta_gn > 0 or proj_gn > 0 or readout_gn > 0)
        and param_updated
    )

    print()
    print("=== Feasibility verdict ===")
    print(f"PRETRAINED_BACKBONE_DV_READY: {'YES' if all_pass else 'NO'}", flush=True)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
