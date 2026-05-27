"""
scripts/train_q34a_classical_candidate.py

Q34A Classical NAS Pilot — Trial Execution Script.

Accepts a YAML config and trains a configurable classical MLP head over the
frozen C006-D040 pretrained backbone. Used by run_q34a_classical_nas_pilot.py
as the per-trial command target.

Design decisions (see Q34A report Section 3 for full explanation):
- "backbone_family" in the Q32 search space refers to the trainable HEAD
  architecture, not the frozen backbone (which is always C006-D040). Since the
  frozen backbone outputs 128-dim flat feature vectors (after AdaptiveAvgPool2d
  + Flatten), "standard_cnn" and "depthwise_separable_cnn" are both implemented
  as linear projection heads in this pilot. True convolutional blocks require
  spatial feature maps as input; this pilot limitation is documented in the report.
- BatchNorm is applied as LayerNorm for 1D feature vectors; standard BatchNorm
  is designed for 4D (N, C, H, W) tensors.
- "pooling: adaptive_average" is a no-op — the frozen backbone already applies
  AdaptiveAvgPool2d(1,1) and Flatten before outputting the 128-dim vector.
- No checkpoint is saved — this is a 3–5 epoch pilot only.

CLI:
    python3 scripts/train_q34a_classical_candidate.py --config <yaml_path>

Output lines (machine-readable, parsed by run_q34a_classical_nas_pilot.py):
    Q34A_TRIAL_AUROC: <float>
    Q34A_TRIAL_F1: <float>
    Q34A_TRIAL_ACCURACY: <float>
    Q34A_TRIAL_PARAMS: <int>
    Q34A_TRIAL_LATENCY_MS: <float>
    Q34A_TRIAL_STATUS: completed

Exit code 0 on success, nonzero on failure.
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
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.cnn import build_model

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BACKBONE_CHECKPOINT_DEFAULT = "checkpoints/c006_d040_classical_anchor.pt"
DATASET_ROOT_DEFAULT        = "data/processed/vindr_binary_roi_224"
BACKBONE_OUT_DIM            = 128   # frozen C006-D040 backbone output dimension
MAX_EPOCHS_ALLOWED          = 5     # hard ceiling for Q34A pilot
MAX_TRAINABLE_PARAMS        = 100_000

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q34A Classical NAS Trial")
    p.add_argument("--config", required=True, help="Path to trial YAML config")
    return p.parse_args()


# ── Seeds ──────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Head building ──────────────────────────────────────────────────────────────

def build_nas_head(model_cfg: dict) -> nn.Sequential:
    """
    Build a configurable linear MLP head from model config.

    Architecture:
      depth=shallow  → Linear(128, ch[0]) → Norm → Act → Drop → Linear(ch[0], compress) → Act → Linear(compress, 2)
      depth=medium   → Linear(128, ch[0]) → Norm → Act → Drop → Linear(ch[0], ch[1]) → Norm → Act → Drop
                                          → Linear(ch[1], compress) → Act → Linear(compress, 2)
    """
    backbone_family = model_cfg.get("backbone_family", "standard_cnn")
    channel_width   = model_cfg.get("channel_width",   [32, 64])
    depth           = model_cfg.get("depth",           "shallow")
    compress_dim    = model_cfg.get("compression_dim", 8)
    act_name        = model_cfg.get("activation",      "ReLU")
    norm_name       = model_cfg.get("normalization",   "BatchNorm")
    dropout_rate    = float(model_cfg.get("dropout",   0.0))

    # backbone_family is a label only — both families map to linear layers in this pilot
    _ = backbone_family   # acknowledged; no functional difference in pilot

    def make_act() -> nn.Module:
        return nn.ReLU() if act_name == "ReLU" else nn.GELU()

    def make_norm(dim: int) -> nn.Module:
        # LayerNorm for 1D features (BatchNorm requires 4D tensors)
        return nn.LayerNorm(dim)

    hidden_dims = list(channel_width) if depth == "medium" else [channel_width[0]]

    layers: list[nn.Module] = []
    in_dim = BACKBONE_OUT_DIM

    for hd in hidden_dims:
        layers.append(nn.Linear(in_dim, hd))
        if norm_name in ("BatchNorm", "LayerNorm"):
            layers.append(make_norm(hd))
        layers.append(make_act())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = hd

    # Compression bottleneck
    layers.append(nn.Linear(in_dim, compress_dim))
    layers.append(make_act())

    # Classifier
    layers.append(nn.Linear(compress_dim, 2))

    return nn.Sequential(*layers)


# ── Full model ─────────────────────────────────────────────────────────────────

class NASClassicalModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Sequential) -> None:
        super().__init__()
        self.backbone = backbone   # frozen
        self.head     = head       # trainable

    def train(self, mode: bool = True) -> "NASClassicalModel":
        super().train(mode)
        self.backbone.eval()      # backbone always in eval
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)   # (B, 128)
        return self.head(features)


# ── Backbone loading ───────────────────────────────────────────────────────────

def load_backbone(checkpoint_path: str) -> nn.Module:
    """Build and load the frozen C006-D040 backbone (identical to Q22 pattern)."""
    full_model = build_model(CNN_CONFIG)
    backbone   = nn.Sequential(*list(full_model.children())[:4])

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:
        src_sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        src_sd = ck["state_dict"]
    else:
        print("ERROR: unrecognised checkpoint format", flush=True)
        sys.exit(1)

    remapped = {}
    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            remapped[f"backbone.{k}"] = v

    # Load via a temporary wrapper to reuse load_state_dict pattern
    _tmp = nn.Module()
    _tmp.backbone = backbone
    missing, unexpected = _tmp.load_state_dict(remapped, strict=False)
    backbone_keys = {f"backbone.{k}" for k in backbone.state_dict()}
    missing_backbone = [k for k in missing if k.startswith("backbone.")]
    if missing_backbone:
        print(f"ERROR: {len(missing_backbone)} backbone keys missing: {missing_backbone}", flush=True)
        sys.exit(1)

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


# ── Metrics ────────────────────────────────────────────────────────────────────

def eval_split(model: NASClassicalModel, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    has_both = len(np.unique(y_true)) > 1

    return {
        "loss":     total_loss / len(loader),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


def measure_latency(model: NASClassicalModel, device: torch.device,
                    n_warmup: int = 10, n_samples: int = 50) -> float:
    """Mean ms per single-sample forward pass on device."""
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Validate epoch ceiling ────────────────────────────────────────────────
    epochs_requested = cfg.get("training", {}).get("epochs", 4)
    if epochs_requested > MAX_EPOCHS_ALLOWED:
        print(f"ERROR: epochs={epochs_requested} exceeds Q34A ceiling of {MAX_EPOCHS_ALLOWED}", flush=True)
        sys.exit(1)
    epochs = int(epochs_requested)

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seeds(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        sys.exit(1)
    device = torch.device("cuda")

    print(f"[Q34A] config: {args.config}", flush=True)
    print(f"[Q34A] device: {device}", flush=True)
    print(f"[Q34A] epochs: {epochs}", flush=True)
    print(f"[Q34A] seed:   {seed}", flush=True)

    # ── Data paths ────────────────────────────────────────────────────────────
    dataset_root = cfg.get("dataset", {}).get("root", DATASET_ROOT_DEFAULT)
    if not os.path.isdir(dataset_root):
        dataset_root = DATASET_ROOT_DEFAULT
    ck_path = cfg.get("model", {}).get("checkpoint_path", BACKBONE_CHECKPOINT_DEFAULT)
    if not os.path.isfile(ck_path):
        ck_path = BACKBONE_CHECKPOINT_DEFAULT
    if not os.path.isfile(ck_path):
        print(f"ERROR: backbone checkpoint not found at {ck_path}", flush=True)
        sys.exit(1)
    if not os.path.isdir(dataset_root):
        print(f"ERROR: dataset not found at {dataset_root}", flush=True)
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────────────
    backbone  = load_backbone(ck_path)
    head      = build_nas_head(cfg.get("model", {}))
    model     = NASClassicalModel(backbone, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    print(f"[Q34A] trainable params: {n_trainable}", flush=True)
    print(f"[Q34A] total params:     {n_total}", flush=True)

    if n_trainable > MAX_TRAINABLE_PARAMS:
        print(f"ERROR: trainable_params={n_trainable} exceeds hard limit {MAX_TRAINABLE_PARAMS}", flush=True)
        sys.exit(1)

    # ── Training config ───────────────────────────────────────────────────────
    lr         = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    weight_dec = float(cfg.get("training", {}).get("weight_decay",  0.0))
    batch_size = int(cfg.get("training",   {}).get("batch_size",    4))

    print(f"[Q34A] lr:         {lr}", flush=True)
    print(f"[Q34A] weight_dec: {weight_dec}", flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=dataset_root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=dataset_root, split="val")
    test_ds  = VinDrSpineXRBinaryDataset(root=dataset_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_dec,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    final_train_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss  = 0.0
        correct       = 0
        total         = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if not torch.isfinite(loss):
                print(f"ERROR: NaN/inf loss at epoch {epoch}", flush=True)
                sys.exit(3)
            loss.backward()
            # Check backbone frozen
            for p in model.backbone.parameters():
                if p.grad is not None and p.grad.abs().max().item() > 0:
                    print("ERROR: backbone received gradient — frozen violation", flush=True)
                    sys.exit(3)
            optimizer.step()
            running_loss += loss.item()
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.size(0)

        t_epoch = time.perf_counter() - t0
        final_train_loss = running_loss / len(train_loader)
        val_m = eval_split(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={final_train_loss:.4f} acc={correct/total:.4f} | "
            f"val_loss={val_m['loss']:.4f} auroc={val_m['auroc']:.4f} "
            f"f1={val_m['f1']:.4f} | t={t_epoch:.1f}s",
            flush=True,
        )

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("[Q34A] evaluating on test split ...", flush=True)
    test_m = eval_split(model, test_loader, criterion, device)

    # ── Latency ───────────────────────────────────────────────────────────────
    lat_ms = measure_latency(model, device)

    # ── Machine-readable output ───────────────────────────────────────────────
    print(f"Q34A_TRIAL_AUROC: {test_m['auroc']:.6f}",   flush=True)
    print(f"Q34A_TRIAL_F1: {test_m['f1']:.6f}",         flush=True)
    print(f"Q34A_TRIAL_ACCURACY: {test_m['accuracy']:.6f}", flush=True)
    print(f"Q34A_TRIAL_PARAMS: {n_trainable}",           flush=True)
    print(f"Q34A_TRIAL_LATENCY_MS: {lat_ms:.4f}",       flush=True)
    print(f"Q34A_TRIAL_STATUS: completed",               flush=True)

    # ── Runner-compatible [SUMMARY] block ─────────────────────────────────────
    print("[SUMMARY]",                          flush=True)
    print(f"Trainable params: {n_trainable}",   flush=True)
    print(f"Loss: {final_train_loss:.6f}",      flush=True)

    print(
        f"[Q34A] DONE | auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
        f"params={n_trainable} latency={lat_ms:.2f}ms/sample",
        flush=True,
    )


if __name__ == "__main__":
    main()
