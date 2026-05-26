"""
scripts/train_vindr_classical_control_tiny_head.py

VinDr-SpineXR Approximate Trainable-Parameter-Matched Classical Control.
Slice Q22.

Replaces the DV quantum head with a tiny classical head of approximately
equivalent trainable parameter count (~574, matching Q21), holding all
else constant. This is a scientific control to determine whether the
compact trainable bottleneck effect (rather than quantum inductive bias)
explains the Q21 improvement over the Q17 classical baseline.

Backbone: C006-D040 pretrained backbone (depthwise_sep [64, 128]),
loaded from checkpoints/c006_d040_classical_anchor.pt.
Backbone is frozen throughout training.

Head: TinyClassicalHead — nn.Linear(128, H) → ReLU → nn.Linear(H, 2)
H is computed dynamically to match Q21 trainable param count (~574).

Device: CUDA mandatory. Classical head runs on GPU.

Usage:
    python scripts/train_vindr_classical_control_tiny_head.py \\
        --root data/processed/vindr_binary_roi_224 \\
        --checkpoint checkpoints/c006_d040_classical_anchor.pt \\
        --batch-size 4 \\
        --epochs 15 \\
        --seed 42
"""

from __future__ import annotations

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
from qcore.models.cnn import build_model

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix as sklearn_cm,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Constants ──────────────────────────────────────────────────────────────────

CHECKPOINT_DIR   = "checkpoints"
CHECKPOINT_NAME  = "vindr_classical_control_tiny_head_best.pt"
EARLY_STOP_PATIENCE = 4
TARGET_PARAMS    = 574  # Q21 trainable param count

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

# ── Seed ───────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR Classical Control Tiny Head (Slice Q22)"
    )
    p.add_argument("--root",       default="data/processed/vindr_binary_roi_224")
    p.add_argument("--checkpoint", default="checkpoints/c006_d040_classical_anchor.pt")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ── Head dim computation ───────────────────────────────────────────────────────

def compute_hidden_dim(in_features: int, target_params: int) -> tuple[int, int]:
    """
    Compute H so that total trainable params ≈ target_params.

    Formula: params = H * (in_features + 3) + 2
    Solving: H = round((target_params - 2) / (in_features + 3))
    """
    H = round((target_params - 2) / (in_features + 3))
    H = max(H, 1)
    actual = H * (in_features + 3) + 2
    return H, actual


# ── Tiny classical head ────────────────────────────────────────────────────────

class TinyClassicalHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, hidden_dim)
        self.act        = nn.ReLU()
        self.readout    = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.projection(x))
        return self.readout(x)


# ── Full model ─────────────────────────────────────────────────────────────────

class VinDrClassicalControl(nn.Module):
    def __init__(self, backbone: nn.Module, head: TinyClassicalHead) -> None:
        super().__init__()
        self.backbone = backbone   # frozen
        self.head     = head       # trainable

    def train(self, mode: bool = True) -> "VinDrClassicalControl":
        """Keep backbone always in eval mode to preserve BatchNorm running stats."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact Q21 forward semantics: with torch.no_grad() around backbone
        with torch.no_grad():
            features = self.backbone(x)   # (B, 128)
        return self.head(features)


# ── Backbone loading ───────────────────────────────────────────────────────────

def build_backbone() -> nn.Module:
    """
    Instantiate the C006-D040 backbone: same Sequential[:4] used in Q21.

    Layers:
      [0] depthwise_sep block (1 → 64 channels)
      [1] depthwise_sep block (64 → 128 channels)
      [2] AdaptiveAvgPool2d(1, 1)
      [3] Flatten
    Output shape: (B, 128)
    """
    full_model = build_model(CNN_CONFIG)
    return nn.Sequential(*list(full_model.children())[:4])


def load_pretrained_backbone(
    model: VinDrClassicalControl, checkpoint_path: str
) -> dict:
    """
    Load C006-D040 backbone weights into VinDrClassicalControl.

    Key remapping (identical to Q21):
      "0.*" → "backbone.0.*"
      "1.*" → "backbone.1.*"
      "5.*" → skipped (classifier head)

    Returns summary dict with matched/skipped/unexpected counts.
    """
    ck = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ck:
        src_sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        src_sd = ck["state_dict"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in list(ck.values())[:3]):
        src_sd = ck
    else:
        print(f"ERROR: Unrecognised checkpoint format at {checkpoint_path}", flush=True)
        sys.exit(1)

    remapped = {}
    skipped  = []
    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            remapped[f"backbone.{k}"] = v
        elif k.startswith("5."):
            skipped.append(k)

    missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
    backbone_keys = set(f"backbone.{k}" for k in model.backbone.state_dict().keys())
    matched = [k for k in remapped if k in backbone_keys]
    missing_backbone = [k for k in missing_keys if k.startswith("backbone.")]

    if missing_backbone:
        print(
            f"ERROR: {len(missing_backbone)} backbone keys failed to load: "
            f"{missing_backbone}",
            flush=True,
        )
        sys.exit(1)

    return {
        "matched":    len(matched),
        "skipped":    len(skipped),
        "unexpected": len(unexpected_keys),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_ml_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    has_both = len(np.unique(y_true)) > 1
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auroc":     roc_auc_score(y_true, y_prob) if has_both else float("nan"),
        "auprc":     average_precision_score(y_true, y_prob) if has_both else float("nan"),
    }


def grad_norm_module(module: nn.Module) -> float:
    """L2 norm of gradients across all parameters in module."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def grad_norm_model(model: VinDrClassicalControl) -> float:
    """L2 norm of all trainable (head) parameter gradients."""
    return grad_norm_module(model.head)


# ── Hard stop helpers ──────────────────────────────────────────────────────────

def check_backbone_grad(model: VinDrClassicalControl, epoch: int) -> None:
    """Hard stop if any backbone parameter received a non-zero gradient."""
    for name, param in model.backbone.named_parameters():
        if param.grad is not None and param.grad.abs().max().item() > 0.0:
            print(
                f"BACKBONE GRADIENT DETECTED — CRITICAL FAILURE\n"
                f"  backbone.{name} received gradient at epoch {epoch}: "
                f"max abs = {param.grad.abs().max().item():.4e}. Aborting.",
                flush=True,
            )
            sys.exit(3)


# ── Latency ────────────────────────────────────────────────────────────────────

def measure_latency_cuda(model: nn.Module, device: torch.device,
                          n_warmup: int = 10, n_samples: int = 100) -> float:
    """
    Time 100 single-sample forward passes on CUDA.
    Returns mean ms per sample.
    """
    model.eval()
    x = torch.randn(1, 1, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            torch.cuda.synchronize()

        timings = []
        for _ in range(n_samples):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return float(np.mean(timings))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    # ── CUDA mandatory ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("CUDA unavailable — stopping", flush=True)
        sys.exit(1)
    device = torch.device("cuda")

    print("=== VinDr Classical Control Tiny Head Training ===")
    print(flush=True)

    # ── Checkpoint / dataset existence check ──────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", flush=True)
        sys.exit(1)
    if not os.path.isdir(args.root):
        print(f"ERROR: Dataset root not found: {args.root}", flush=True)
        sys.exit(1)

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone = build_backbone()

    # ── Head dim computation ──────────────────────────────────────────────────
    # Probe in_features from backbone output on a dummy batch
    with torch.no_grad():
        _dummy = torch.randn(1, 1, 224, 224)
        _feat  = backbone(_dummy)
        in_features = _feat.shape[1]
    del _dummy, _feat

    H, actual_params = compute_hidden_dim(in_features, TARGET_PARAMS)

    print("Head dim computation:")
    print(f"  in_features   : {in_features}")
    print(f"  target_params : {TARGET_PARAMS}")
    print(f"  formula       : H = round(({TARGET_PARAMS} - 2) / ({in_features} + 3))")
    print(f"  computed H    : {H}")
    print(f"  actual trainable params: {actual_params}")
    print(flush=True)

    # ── Assemble model ────────────────────────────────────────────────────────
    head  = TinyClassicalHead(in_features=in_features, hidden_dim=H)
    model = VinDrClassicalControl(backbone=backbone, head=head)

    # ── Load pretrained backbone weights ──────────────────────────────────────
    backbone_load_summary = load_pretrained_backbone(model, args.checkpoint)

    # ── Freeze ALL backbone parameters ────────────────────────────────────────
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # ── Assert frozen ─────────────────────────────────────────────────────────
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            print("BACKBONE FREEZE FAILED", flush=True)
            sys.exit(1)

    # ── Move to CUDA ──────────────────────────────────────────────────────────
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    print("Checkpoint:")
    print(f"  path           : {args.checkpoint}")
    print(f"  backbone loaded: YES")
    print(f"  frozen         : YES")
    print()
    print("Model:")
    print(f"  trainable params: {n_trainable}")
    print(flush=True)

    # ── Probe feature shape on first batch (before training) ─────────────────
    # This prints features.shape to verify in_features matches head input
    _probe_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    _probe_loader = DataLoader(_probe_ds, batch_size=1, shuffle=False, num_workers=0)
    _probe_x, _ = next(iter(_probe_loader))
    _probe_x = _probe_x.to(device)
    with torch.no_grad():
        _feat_shape = model.backbone(_probe_x).shape
    print(f"Feature shape on first batch: {tuple(_feat_shape)}", flush=True)
    del _probe_ds, _probe_loader, _probe_x, _feat_shape

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=args.root, split="val")
    test_ds  = VinDrSpineXRBinaryDataset(root=args.root, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_val_auroc   = float("nan")
    best_val_f1      = float("nan")
    best_epoch       = 0
    patience_counter = 0
    stop_reason      = "max epochs reached"

    train_history: list = []
    val_history:   list = []

    for epoch in range(1, args.epochs + 1):
        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        t_ep_start    = time.perf_counter()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0
        last_head_gn  = 0.0
        last_total_gn = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)

            # NaN/inf loss hard stop
            if not torch.isfinite(loss):
                print("NaN/INF LOSS DETECTED", flush=True)
                sys.exit(3)

            loss.backward()

            # Backbone gradient hard stop
            check_backbone_grad(model, epoch)

            # NaN/inf gradient hard stop
            for name, param in model.head.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        print("NaN/INF GRADIENT DETECTED", flush=True)
                        sys.exit(3)

            last_head_gn  = grad_norm_module(model.head)
            last_total_gn = last_head_gn  # only head params are trainable

            optimizer.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.shape[0]

        t_train = time.perf_counter() - t_ep_start

        mean_train_loss = train_loss / len(train_loader)
        train_acc       = train_correct / train_total

        # ── Val epoch ─────────────────────────────────────────────────────────
        model.eval()
        t_val_start  = time.perf_counter()
        val_loss_sum = 0.0
        val_true, val_pred, val_prob = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits  = model(x)
                loss    = criterion(logits, y)
                val_loss_sum += loss.item()
                probs   = torch.softmax(logits, dim=1)[:, 1]
                preds   = logits.argmax(dim=1)
                val_true.extend(y.cpu().tolist())
                val_pred.extend(preds.cpu().tolist())
                val_prob.extend(probs.cpu().tolist())

        t_val = time.perf_counter() - t_val_start

        mean_val_loss = val_loss_sum / len(val_loader)
        val_metrics   = compute_ml_metrics(val_true, val_pred, val_prob)
        epoch_time    = t_train + t_val

        train_history.append({
            "epoch":    epoch,
            "loss":     mean_train_loss,
            "accuracy": train_acc,
            "time":     epoch_time,
        })
        val_history.append({
            "epoch":       epoch,
            "loss":        mean_val_loss,
            "metrics":     val_metrics,
            "head_gn":     last_head_gn,
            "total_gn":    last_total_gn,
            "time":        epoch_time,
        })

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  train loss: {mean_train_loss:.4f}")
        print(f"  train acc: {train_acc:.4f}")
        print(f"  val loss: {mean_val_loss:.4f}")
        print(f"  val acc: {val_metrics['accuracy']:.4f}")
        print(f"  val f1: {val_metrics['f1']:.4f}")
        print(f"  val auroc: {val_metrics['auroc']:.4f}")
        print(f"  head grad: {last_head_gn:.2e}")
        print(f"  epoch time: {epoch_time:.1f}s")
        sys.stdout.flush()

        # Early stopping
        if mean_val_loss < best_val_loss:
            best_val_loss    = mean_val_loss
            best_val_auroc   = val_metrics["auroc"]
            best_val_f1      = val_metrics["f1"]
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             mean_val_loss,
                "val_auroc":            val_metrics["auroc"],
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                stop_reason = f"early stopping (patience={EARLY_STOP_PATIENCE})"
                break

    print()
    print("Best:")
    print(f"  epoch: {best_epoch}")
    print(f"  val auroc: {best_val_auroc:.4f}")
    print(f"  val f1: {best_val_f1:.4f}")
    print(flush=True)

    # ── Reload best checkpoint ────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Re-freeze backbone after reload
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_loss_sum = 0.0
    test_true, test_pred, test_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y    = x.to(device), y.to(device)
            logits   = model(x)
            loss     = criterion(logits, y)
            test_loss_sum += loss.item()
            probs    = torch.softmax(logits, dim=1)[:, 1]
            preds    = logits.argmax(dim=1)
            test_true.extend(y.cpu().tolist())
            test_pred.extend(preds.cpu().tolist())
            test_prob.extend(probs.cpu().tolist())

    test_loss    = test_loss_sum / len(test_loader)
    test_metrics = compute_ml_metrics(test_true, test_pred, test_prob)
    cm           = sklearn_cm(test_true, test_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Final test:")
    print(f"  loss: {test_loss:.4f}")
    print(f"  acc: {test_metrics['accuracy']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall: {test_metrics['recall']:.4f}")
    print(f"  f1: {test_metrics['f1']:.4f}")
    print(f"  auroc: {test_metrics['auroc']:.4f}")
    print(f"  auprc: {test_metrics['auprc']:.4f}")
    print(f"  confusion: [[{tn}, {fp}], [{fn}, {tp}]]")
    print(flush=True)

    # ── Latency: 100 single-sample passes on CUDA ─────────────────────────────
    lat_mean = measure_latency_cuda(model, device, n_warmup=10, n_samples=100)

    print("Latency:")
    print(f"  ms/sample: {lat_mean:.2f}")
    print(flush=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    # PASS = no NaN/inf, backbone frozen throughout, completed or early stopped,
    #        confusion matrix non-degenerate
    all_one_class = (tn == 0 and tp == 0) or (fp == 0 and fn == 0)
    verdict = "PASS" if not all_one_class else "FAIL (degenerate confusion matrix)"

    print("Verdict:")
    print(f"  CLASSICAL_CONTROL_TINY_HEAD: {verdict}")
    sys.stdout.flush()

    # ── Per-epoch summary ─────────────────────────────────────────────────────
    print()
    print("=== Per-Epoch Summary ===")
    header = (
        "epoch | train_loss | train_acc | val_loss | val_acc | "
        "val_precision | val_recall | val_f1 | val_auroc | val_auprc | "
        "head_grad_norm | total_grad_norm | epoch_time"
    )
    print(header)
    for th, vh in zip(train_history, val_history):
        m = vh["metrics"]
        print(
            f"{th['epoch']:5d} | {th['loss']:.4f} | {th['accuracy']:.4f} | "
            f"{vh['loss']:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auroc']:.4f} | "
            f"{m['auprc']:.4f} | {vh['head_gn']:.4e} | {vh['total_gn']:.4e} | "
            f"{th['time']:.1f}s"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
