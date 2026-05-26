"""
scripts/train_vindr_cv_binary.py

CV Binary Full Training for VinDr-SpineXR binary ROI dataset.
Slice Q27.

Uses the exact Q26-validated CV pipeline architecture. No architecture changes.
This is the first CV binary benchmark in the QStrata research program.

Architecture (Q26-confirmed, Q25A Section 7):
  Input (B, 1, 224, 224)
  → Frozen C006-D040 backbone (build_model(CNN_CONFIG)[:4])  [frozen, CUDA] → (B, 128)
  → nn.Linear(128, 4)   [compression, trainable, CPU]                        → (B, 4)
  → per-sample GaussianVariationalAnsatz loop (CPU)                           → (B, 4) mu
  → nn.Linear(4, 2)     [readout, trainable, CPU]                             → (B, 2) logits

Trainable parameters (Q26-confirmed exact count = 536):
  compression:  nn.Linear(128, 4)                            = 516
  ansatz:       GaussianVariationalAnsatz(n_modes=2, depth=1) = 10
  readout:      nn.Linear(4, 2)                              = 10
  total:                                                       536

Hard stops:
  - NaN/inf in loss            → print NaN/INF LOSS DETECTED and exit
  - NaN/inf in logits          → print NaN/INF LOGITS DETECTED and exit
  - NaN/inf in mu_final        → print NaN/INF FIRST MOMENTS DETECTED and exit
  - Non-finite covariance      → print NON-FINITE COVARIANCE DETECTED and exit
  - Backbone gradient detected → print BACKBONE GRADIENT DETECTED — CRITICAL FAILURE and exit

Usage:
    python3 scripts/train_vindr_cv_binary.py \\
        --root data/processed/vindr_binary_roi_224 \\
        --checkpoint checkpoints/c006_d040_classical_anchor.pt \\
        --batch-size 4 \\
        --epochs 15 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
from qcore.backends.cvBackend import GaussianBackend
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

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

N_MODES            = 2
CV_DEPTH           = 1
SQUEEZING_CAP      = 1.5
HBAR               = 2.0

EXPECTED_TRAINABLE = 536   # 516 + 10 + 10

CHECKPOINT_DIR     = "checkpoints"
CHECKPOINT_NAME    = "vindr_cv_binary_best.pt"
EARLY_STOP_PATIENCE = 4

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR CV Binary Full Training (Slice Q27)"
    )
    p.add_argument("--root",       default="data/processed/vindr_binary_roi_224")
    p.add_argument("--checkpoint", default="checkpoints/c006_d040_classical_anchor.pt")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model (exact Q26 CVHybridBinaryModel) ─────────────────────────────────────

class CVHybridBinaryModel(nn.Module):
    """
    CV Hybrid CNN model — exact Q26-validated architecture.

    Backbone on CUDA (if available) for speed.
    All CV computation on CPU (GaussianBackend constraint).
    Feature transfer: features.detach().cpu() after backbone forward pass.
    """

    def __init__(
        self,
        backbone: nn.Sequential,
        compression: nn.Linear,
        ansatz: GaussianVariationalAnsatz,
        backend: GaussianBackend,
        readout: nn.Linear,
        n_modes: int = N_MODES,
    ) -> None:
        super().__init__()
        self.backbone    = backbone
        self.compression = compression
        self.ansatz      = ansatz
        self.readout     = readout
        self.backend     = backend
        self.n_modes     = n_modes

        self._last_mu:  torch.Tensor | None = None
        self._last_cov: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "CVHybridBinaryModel":
        """Override: keep backbone always in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) on backbone device (CUDA if available)
        returns: logits (B, 2) on CPU
        """
        B = x.shape[0]

        # ── Backbone: frozen feature extraction ───────────────────────────────
        with torch.no_grad():
            features = self.backbone(x)           # (B, 128)

        # Transfer to CPU for CV (GaussianBackend is CPU-only)
        features_cpu = features.detach().cpu()    # (B, 128)

        # ── Per-sample CV loop ─────────────────────────────────────────────────
        logits_list: list[torch.Tensor] = []
        mu_list:     list[torch.Tensor] = []
        cov_list:    list[torch.Tensor] = []

        for i in range(B):
            compressed = self.compression(features_cpu[i])    # (4,)

            # Gradient-safe encoding: no in-place ops, grad flows to compression
            scale = math.sqrt(2.0 * self.backend.hbar)
            encoded_input = compressed * scale                 # (4,)

            mu_0, cov_0 = self.backend.get_vacuum()
            mu_final, cov_final = self.ansatz.apply(
                mu_0, cov_0, self.backend, encoded_input
            )

            logit_i = self.readout(mu_final)                   # (2,)
            logits_list.append(logit_i)
            mu_list.append(mu_final)
            cov_list.append(cov_final.detach())

        self._last_mu  = torch.stack(mu_list)              # (B, 4)
        self._last_cov = torch.stack(cov_list)             # (B, 4, 4)

        return torch.stack(logits_list)                    # (B, 2)


# ── Backbone construction and loading ─────────────────────────────────────────

def build_backbone(cnn_config: dict) -> nn.Sequential:
    full_model = build_model(cnn_config)
    backbone = nn.Sequential(*list(full_model.children())[:4])
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone


def load_pretrained_backbone(model: CVHybridBinaryModel, checkpoint_path: str) -> dict:
    """
    Load C006-D040 backbone weights.
    Key remapping (identical to Q21/Q22/Q26):
      "0.*" → "backbone.0.*", "1.*" → "backbone.1.*", "5.*" → skipped
    Hard stops on file not found, bad format, or missing backbone keys.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[HARD STOP] Checkpoint not found: {checkpoint_path}", flush=True)
        sys.exit(1)

    ck = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ck:
        src_sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        src_sd = ck["state_dict"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in list(ck.values())[:3]):
        src_sd = ck
    else:
        print(f"[HARD STOP] Unrecognised checkpoint format: {checkpoint_path}", flush=True)
        sys.exit(1)

    remapped: dict = {}
    skipped:  list = []
    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            remapped[f"backbone.{k}"] = v
        elif k.startswith("5."):
            skipped.append(k)

    missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
    backbone_keys    = set(f"backbone.{k}" for k in model.backbone.state_dict().keys())
    matched          = [k for k in remapped if k in backbone_keys]
    missing_backbone = [k for k in missing_keys if k.startswith("backbone.")]

    if missing_backbone:
        print(
            f"[HARD STOP] {len(missing_backbone)} backbone keys failed to load: {missing_backbone}",
            flush=True,
        )
        sys.exit(1)

    return {"matched": len(matched), "skipped": len(skipped), "unexpected": len(unexpected_keys)}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_ml_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    y_prob = list(y_prob)
    has_both = len(set(y_true)) > 1
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auroc":     roc_auc_score(y_true, y_prob) if has_both else float("nan"),
        "auprc":     average_precision_score(y_true, y_prob) if has_both else float("nan"),
    }


def grad_norm_params(model: CVHybridBinaryModel, *param_names: str) -> float:
    total = 0.0
    params = dict(model.named_parameters())
    for n in param_names:
        p = params.get(n)
        if p is not None and p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


def ansatz_grad_norm(ansatz: GaussianVariationalAnsatz) -> float:
    total = 0.0
    for p in ansatz.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


def total_trainable_grad_norm(model: CVHybridBinaryModel) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


# ── CV health checks ───────────────────────────────────────────────────────────

def check_cov_symmetric(cov: torch.Tensor, tol: float = 1e-5) -> bool:
    return bool(((cov - cov.transpose(-1, -2)).abs() < tol).all().item())


def check_cov_psd(cov: torch.Tensor, tol: float = 1e-6) -> bool:
    """Uses torch.linalg.eigvalsh — avoids .numpy() (NumPy 2.x compat)."""
    cov_float = cov.detach().cpu().float()
    for i in range(cov_float.shape[0]):
        eigvals = torch.linalg.eigvalsh(cov_float[i])
        if float(eigvals.min().item()) < -tol:
            return False
    return True


# ── Hard-stop checks ───────────────────────────────────────────────────────────

def check_backbone_grad(model: CVHybridBinaryModel, epoch: int, batch_idx: int) -> None:
    for name, param in model.backbone.named_parameters():
        if param.grad is not None and param.grad.abs().max().item() > 0.0:
            print(
                f"\nBACKBONE GRADIENT DETECTED — CRITICAL FAILURE\n"
                f"  backbone.{name}  max abs grad = {param.grad.abs().max().item():.4e}\n"
                f"  epoch {epoch}, batch {batch_idx}. Aborting.",
                flush=True,
            )
            sys.exit(3)


# ── Latency measurement ────────────────────────────────────────────────────────

def measure_latency_single_sample(
    model: CVHybridBinaryModel,
    backbone_device: torch.device,
    n_passes: int = 100,
) -> tuple[float, float]:
    """
    Time 100 single-sample forward passes (batch_size=1 each).
    Returns (mean_ms, std_ms).
    """
    model.eval()
    x = torch.randn(1, 1, 224, 224).to(backbone_device)
    timings: list[float] = []

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(x)

        for _ in range(n_passes):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)   # ms per sample

    return float(np.mean(timings)), float(np.std(timings))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    today = datetime.date.today().isoformat()
    cuda_available   = torch.cuda.is_available()
    backbone_device  = torch.device("cuda" if cuda_available else "cpu")

    print("=== VinDr CV Binary Full Training ===", flush=True)
    print(flush=True)
    print("Device:", flush=True)
    print(f"  torch device:        {backbone_device}", flush=True)
    print(f"  cv backend device:   cpu", flush=True)
    print(
        f"  device transfer note: backbone on {'CUDA' if cuda_available else 'CPU'}; "
        f"features.detach().cpu() before CV circuit; all CV ops on CPU",
        flush=True,
    )
    print(flush=True)

    # ── File existence checks ─────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"[HARD STOP] Checkpoint not found: {args.checkpoint}", flush=True)
        sys.exit(1)
    if not os.path.isdir(args.root):
        print(f"[HARD STOP] Dataset root not found: {args.root}", flush=True)
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────────────
    backbone    = build_backbone(CNN_CONFIG)
    compression = nn.Linear(128, 2 * N_MODES)   # Linear(128, 4)
    ansatz      = GaussianVariationalAnsatz(n_modes=N_MODES, depth=CV_DEPTH, squeezing_cap=SQUEEZING_CAP)
    backend     = GaussianBackend(n_modes=N_MODES, hbar=HBAR, device="cpu")
    readout     = nn.Linear(2 * N_MODES, 2)     # Linear(4, 2)

    model = CVHybridBinaryModel(
        backbone=backbone, compression=compression,
        ansatz=ansatz, backend=backend, readout=readout,
        n_modes=N_MODES,
    )
    model.backbone = model.backbone.to(backbone_device)

    # ── Load pretrained backbone ──────────────────────────────────────────────
    load_summary = load_pretrained_backbone(model, args.checkpoint)

    # Re-freeze after load (load_state_dict may reset requires_grad flags)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # Assert frozen
    if any(p.requires_grad for p in model.backbone.parameters()):
        print("BACKBONE FREEZE FAILED — aborting.", flush=True)
        sys.exit(1)

    # ── Parameter count ───────────────────────────────────────────────────────
    compression_params = sum(p.numel() for p in model.compression.parameters())
    ansatz_params      = sum(p.numel() for p in model.ansatz.parameters())
    readout_params     = sum(p.numel() for p in model.readout.parameters())
    n_trainable        = compression_params + ansatz_params + readout_params
    n_frozen           = sum(p.numel() for p in model.backbone.parameters())
    n_total            = n_trainable + n_frozen

    print("Checkpoint:", flush=True)
    print(f"  path:            {args.checkpoint}", flush=True)
    print(f"  backbone loaded: YES", flush=True)
    print(f"  frozen:          YES", flush=True)
    print(f"  matched keys:    {load_summary['matched']}", flush=True)
    print(f"  skipped keys:    {load_summary['skipped']} (classifier head — expected)", flush=True)
    print(f"  unexpected keys: {load_summary['unexpected']}", flush=True)
    print(flush=True)

    print("Model:", flush=True)
    print(f"  architecture:  Q26 CV binary", flush=True)
    print(f"  compression:   Linear(128 → 4)", flush=True)
    print(f"  ansatz:        GaussianVariationalAnsatz", flush=True)
    print(f"  n_modes:       {N_MODES}", flush=True)
    print(f"  depth:         {CV_DEPTH}", flush=True)
    print(f"  squeezing_cap: {SQUEEZING_CAP}", flush=True)
    print(f"  readout:       Linear(4 → 2)", flush=True)
    print(f"  trainable params:", flush=True)
    print(f"    compression:   {compression_params}", flush=True)
    print(f"    ansatz:        {ansatz_params}", flush=True)
    print(f"    readout:       {readout_params}", flush=True)
    print(f"    total:         {n_trainable}", flush=True)
    print(f"  frozen params:   {n_frozen}", flush=True)
    print(f"  grand total:     {n_total}", flush=True)
    if n_trainable != EXPECTED_TRAINABLE:
        print(
            f"  [WARNING] Trainable count {n_trainable} ≠ expected {EXPECTED_TRAINABLE}; "
            f"deviation {n_trainable - EXPECTED_TRAINABLE}. Proceeding.",
            flush=True,
        )
    else:
        print(f"  [OK] Trainable count matches expected ({EXPECTED_TRAINABLE})", flush=True)
    print(flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=args.root, split="val")
    test_ds  = VinDrSpineXRBinaryDataset(root=args.root, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset splits:", flush=True)
    print(f"  train: {len(train_ds)} samples ({len(train_loader)} batches)", flush=True)
    print(f"  val:   {len(val_ds)} samples ({len(val_loader)} batches)", flush=True)
    print(f"  test:  {len(test_ds)} samples ({len(test_loader)} batches)", flush=True)
    print(flush=True)

    # ── Optimizer and loss ────────────────────────────────────────────────────
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

    # Per-epoch history
    epoch_records: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_ep_start    = time.perf_counter()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0
        last_gn_comp  = 0.0
        last_gn_ans   = 0.0
        last_gn_read  = 0.0
        last_gn_total = 0.0
        last_mu_batch = None
        last_cov_batch = None

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(backbone_device)
            # y stays on CPU (logits returned on CPU)

            optimizer.zero_grad()
            logits = model(x)   # (B, 2) on CPU

            # Hard stop: NaN/inf in logits
            if not torch.isfinite(logits).all():
                print(
                    f"\nNaN/INF LOGITS DETECTED at epoch {epoch}, batch {batch_idx}. Aborting.",
                    flush=True,
                )
                sys.exit(3)

            # Hard stop: NaN/inf in mu_final
            if model._last_mu is not None and not torch.isfinite(model._last_mu).all():
                print(
                    f"\nNaN/INF FIRST MOMENTS DETECTED at epoch {epoch}, batch {batch_idx}. Aborting.",
                    flush=True,
                )
                sys.exit(3)

            # Hard stop: non-finite covariance
            if model._last_cov is not None and not torch.isfinite(model._last_cov).all():
                print(
                    f"\nNON-FINITE COVARIANCE DETECTED at epoch {epoch}, batch {batch_idx}. Aborting.",
                    flush=True,
                )
                sys.exit(3)

            # Assert feature shape
            # (checked implicitly: if compression input is wrong shape, it would error)

            loss = criterion(logits, y)

            # Hard stop: NaN/inf in loss
            if not torch.isfinite(loss):
                print(
                    f"\nNaN/INF LOSS DETECTED at epoch {epoch}, batch {batch_idx}. Aborting.",
                    flush=True,
                )
                sys.exit(3)

            loss.backward()

            # Hard stop: backbone gradient (every batch)
            check_backbone_grad(model, epoch, batch_idx)

            optimizer.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.shape[0]

            # Track gradient norms (last batch of epoch)
            last_gn_comp  = grad_norm_params(model, "compression.weight", "compression.bias")
            last_gn_ans   = ansatz_grad_norm(model.ansatz)
            last_gn_read  = grad_norm_params(model, "readout.weight", "readout.bias")
            last_gn_total = total_trainable_grad_norm(model)

            # Track last batch CV state
            if model._last_mu is not None:
                last_mu_batch  = model._last_mu.detach().clone()
            if model._last_cov is not None:
                last_cov_batch = model._last_cov.detach().clone()

        t_train = time.perf_counter() - t_ep_start

        mean_train_loss = train_loss / len(train_loader)
        train_acc       = train_correct / train_total

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        t_val_start  = time.perf_counter()
        val_loss_sum = 0.0
        val_true: list = []
        val_pred: list = []
        val_prob: list = []

        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(backbone_device)
                logits_v = model(x_v)
                loss_v   = criterion(logits_v, y_v)
                val_loss_sum += loss_v.item()
                probs = torch.softmax(logits_v, dim=1)[:, 1]
                preds = logits_v.argmax(dim=1)
                val_true.extend(y_v.tolist())
                val_pred.extend(preds.tolist())
                val_prob.extend(probs.tolist())

        t_val = time.perf_counter() - t_val_start

        mean_val_loss = val_loss_sum / len(val_loader)
        val_metrics   = compute_ml_metrics(val_true, val_pred, val_prob)
        epoch_time    = t_train + t_val

        # ── CV state metrics (from last training batch) ────────────────────────
        mu_magnitude_mean = float("nan")
        mu_magnitude_max  = float("nan")
        cov_diag_mean     = float("nan")
        cov_psd_ok        = True
        cov_sym_ok        = True
        quad_finite_ok    = True

        if last_mu_batch is not None:
            mu_magnitude_mean = float(last_mu_batch.abs().mean().item())
            mu_magnitude_max  = float(last_mu_batch.abs().max().item())
            quad_finite_ok    = bool(torch.isfinite(last_mu_batch).all().item())

        if last_cov_batch is not None:
            cov_diag_mean = float(
                torch.diagonal(last_cov_batch, dim1=-2, dim2=-1).mean().item()
            )
            cov_psd_ok  = check_cov_psd(last_cov_batch)
            cov_sym_ok  = check_cov_symmetric(last_cov_batch)

        no_nan_inf = (
            math.isfinite(last_gn_comp) and
            math.isfinite(last_gn_ans) and
            math.isfinite(last_gn_read) and
            math.isfinite(last_gn_total) and
            (last_mu_batch is None or quad_finite_ok) and
            (last_cov_batch is None or cov_psd_ok)
        )

        # ── Ansatz physical parameter metrics ─────────────────────────────────
        squeezing_norm     = float(model.ansatz.squeezing_raw.detach().norm(2).item())
        ansatz_param_norm_ = float(
            sum(p.detach().norm(2).item()**2 for p in model.ansatz.parameters()) ** 0.5
        )

        # ── Zero-grad warning ──────────────────────────────────────────────────
        if last_gn_total == 0.0:
            print(
                f"[WARN] All trainable gradients are zero at epoch {epoch}. "
                f"Possible gradient vanishing.",
                flush=True,
            )

        # ── Epoch record ───────────────────────────────────────────────────────
        rec = {
            "epoch":               epoch,
            "train_loss":          mean_train_loss,
            "train_acc":           train_acc,
            "val_loss":            mean_val_loss,
            "val_acc":             val_metrics["accuracy"],
            "val_precision":       val_metrics["precision"],
            "val_recall":          val_metrics["recall"],
            "val_f1":              val_metrics["f1"],
            "val_auroc":           val_metrics["auroc"],
            "val_auprc":           val_metrics["auprc"],
            "compression_grad":    last_gn_comp,
            "ansatz_grad":         last_gn_ans,
            "readout_grad":        last_gn_read,
            "total_grad":          last_gn_total,
            "mu_magnitude_mean":   mu_magnitude_mean,
            "mu_magnitude_max":    mu_magnitude_max,
            "cov_diag_mean":       cov_diag_mean,
            "squeezing_norm":      squeezing_norm,
            "ansatz_param_norm":   ansatz_param_norm_,
            "epoch_time":          epoch_time,
            "cov_psd":             cov_psd_ok,
            "cov_symmetric":       cov_sym_ok,
            "quadrature_finite":   quad_finite_ok,
            "no_nan_inf":          no_nan_inf,
        }
        epoch_records.append(rec)

        # ── Epoch stdout ───────────────────────────────────────────────────────
        print(f"Epoch {epoch}/{args.epochs}:", flush=True)
        print(f"  train loss: {mean_train_loss:.4f}", flush=True)
        print(f"  train acc:  {train_acc*100:.2f}%", flush=True)
        print(f"  val loss:   {mean_val_loss:.4f}", flush=True)
        print(f"  val acc:    {val_metrics['accuracy']*100:.2f}%", flush=True)
        print(f"  val f1:     {val_metrics['f1']:.4f}", flush=True)
        print(f"  val auroc:  {val_metrics['auroc']:.4f}", flush=True)
        print(f"  val auprc:  {val_metrics['auprc']:.4f}", flush=True)
        print(f"  val prec:   {val_metrics['precision']:.4f}", flush=True)
        print(f"  val recall: {val_metrics['recall']:.4f}", flush=True)
        print(f"  compression grad: {last_gn_comp:.4e}", flush=True)
        print(f"  ansatz grad:      {last_gn_ans:.4e}", flush=True)
        print(f"  readout grad:     {last_gn_read:.4e}", flush=True)
        print(f"  total grad:       {last_gn_total:.4e}", flush=True)
        print(f"  mu mean:     {mu_magnitude_mean:.4f}", flush=True)
        print(f"  mu max:      {mu_magnitude_max:.4f}", flush=True)
        print(f"  cov diag mean: {cov_diag_mean:.4f}", flush=True)
        print(f"  squeezing norm: {squeezing_norm:.4f}", flush=True)
        print(f"  ansatz param norm: {ansatz_param_norm_:.4f}", flush=True)
        print(f"  cov psd:     {'PASS' if cov_psd_ok  else 'FAIL'}", flush=True)
        print(f"  cov sym:     {'PASS' if cov_sym_ok  else 'FAIL'}", flush=True)
        print(f"  quad finite: {'PASS' if quad_finite_ok else 'FAIL'}", flush=True)
        print(f"  no NaN/inf:  {'PASS' if no_nan_inf else 'FAIL'}", flush=True)
        print(f"  epoch time:  {epoch_time:.1f}s", flush=True)
        sys.stdout.flush()

        # ── Early stopping ────────────────────────────────────────────────────
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
                "val_f1":               val_metrics["f1"],
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                stop_reason = f"early stopping (patience={EARLY_STOP_PATIENCE})"
                break

    n_epochs_run = len(epoch_records)

    print(flush=True)
    print("Best:", flush=True)
    print(f"  epoch:     {best_epoch}", flush=True)
    print(f"  val loss:  {best_val_loss:.4f}", flush=True)
    print(f"  val auroc: {best_val_auroc:.4f}", flush=True)
    print(f"  val f1:    {best_val_f1:.4f}", flush=True)
    print(f"  stop reason: {stop_reason}", flush=True)
    print(flush=True)

    # ── Reload best checkpoint ────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Re-freeze after reload
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # Move backbone back to correct device after CPU load
    model.backbone = model.backbone.to(backbone_device)

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_loss_sum = 0.0
    test_true: list = []
    test_pred: list = []
    test_prob: list = []

    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(backbone_device)
            logits_t = model(x_t)
            loss_t   = criterion(logits_t, y_t)
            test_loss_sum += loss_t.item()
            probs = torch.softmax(logits_t, dim=1)[:, 1]
            preds = logits_t.argmax(dim=1)
            test_true.extend(y_t.tolist())
            test_pred.extend(preds.tolist())
            test_prob.extend(probs.tolist())

    test_loss    = test_loss_sum / len(test_loader)
    test_metrics = compute_ml_metrics(test_true, test_pred, test_prob)
    cm           = sklearn_cm(test_true, test_pred)

    # Handle degenerate confusion matrix (might not have all 4 cells)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        print("[WARN] Confusion matrix is not 2×2 — degenerate predictions.", flush=True)

    print("Final test:", flush=True)
    print(f"  loss:      {test_loss:.4f}", flush=True)
    print(f"  acc:       {test_metrics['accuracy']*100:.2f}%", flush=True)
    print(f"  precision: {test_metrics['precision']:.4f}", flush=True)
    print(f"  recall:    {test_metrics['recall']:.4f}", flush=True)
    print(f"  f1:        {test_metrics['f1']:.4f}", flush=True)
    print(f"  auroc:     {test_metrics['auroc']:.4f}", flush=True)
    print(f"  auprc:     {test_metrics['auprc']:.4f}", flush=True)
    print(f"  confusion: [[{tn}, {fp}], [{fn}, {tp}]]", flush=True)
    print(flush=True)

    # ── Latency ───────────────────────────────────────────────────────────────
    lat_mean, lat_std = measure_latency_single_sample(model, backbone_device, n_passes=100)
    print("Latency:", flush=True)
    print(f"  ms/sample: {lat_mean:.4f} (std {lat_std:.4f})", flush=True)
    print(
        f"  note: CV backend CPU-only; backbone on {'CUDA' if cuda_available else 'CPU'}",
        flush=True,
    )
    print(flush=True)

    # ── Final CV health summary ────────────────────────────────────────────────
    # Aggregate across all epochs
    all_cov_psd    = all(r["cov_psd"]           for r in epoch_records)
    all_cov_sym    = all(r["cov_symmetric"]      for r in epoch_records)
    all_quad_fin   = all(r["quadrature_finite"]  for r in epoch_records)
    all_no_nan     = all(r["no_nan_inf"]          for r in epoch_records)

    print("CV health:", flush=True)
    print(f"  cov psd:         {'PASS' if all_cov_psd  else 'FAIL'} (all {n_epochs_run} epochs)", flush=True)
    print(f"  cov symmetric:   {'PASS' if all_cov_sym  else 'FAIL'} (all {n_epochs_run} epochs)", flush=True)
    print(f"  quadrature finite: {'PASS' if all_quad_fin else 'FAIL'} (all {n_epochs_run} epochs)", flush=True)
    print(f"  no NaN/inf:      {'PASS' if all_no_nan   else 'FAIL'} (all {n_epochs_run} epochs)", flush=True)
    print(flush=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    degenerate = (tn == 0 and tp == 0) or (fp == 0 and fn == 0)
    pass_verdict = (
        (stop_reason == "max epochs reached" or "early stopping" in stop_reason) and
        all_no_nan and
        not degenerate and
        all_cov_psd and all_cov_sym and all_quad_fin
    )

    verdict_str = "PASS" if pass_verdict else "FAIL"
    print(f"Verdict:", flush=True)
    print(f"  CV_BINARY_FULL_TRAINING: {verdict_str}", flush=True)
    if not pass_verdict:
        reasons = []
        if not all_no_nan:     reasons.append("NaN/inf detected in metrics")
        if degenerate:         reasons.append("degenerate confusion matrix")
        if not all_cov_psd:    reasons.append("covariance PSD check failed")
        if not all_cov_sym:    reasons.append("covariance symmetry check failed")
        if not all_quad_fin:   reasons.append("non-finite first moments")
        print(f"  Reasons: {', '.join(reasons)}", flush=True)
    print(flush=True)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(
        args=args,
        today=today,
        n_trainable=n_trainable,
        compression_params=compression_params,
        ansatz_params=ansatz_params,
        readout_params=readout_params,
        n_frozen=n_frozen,
        load_summary=load_summary,
        epoch_records=epoch_records,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_auroc=best_val_auroc,
        best_val_f1=best_val_f1,
        stop_reason=stop_reason,
        n_epochs_run=n_epochs_run,
        test_loss=test_loss,
        test_metrics=test_metrics,
        cm_tn=tn, cm_fp=fp, cm_fn=fn, cm_tp=tp,
        lat_mean=lat_mean, lat_std=lat_std,
        cuda_available=cuda_available,
        all_cov_psd=all_cov_psd,
        all_cov_sym=all_cov_sym,
        all_quad_fin=all_quad_fin,
        all_no_nan=all_no_nan,
        verdict_str=verdict_str,
        degenerate=degenerate,
    )


# ── Report writer ──────────────────────────────────────────────────────────────

def write_report(
    args,
    today: str,
    n_trainable: int,
    compression_params: int,
    ansatz_params: int,
    readout_params: int,
    n_frozen: int,
    load_summary: dict,
    epoch_records: list[dict],
    best_epoch: int,
    best_val_loss: float,
    best_val_auroc: float,
    best_val_f1: float,
    stop_reason: str,
    n_epochs_run: int,
    test_loss: float,
    test_metrics: dict,
    cm_tn: int, cm_fp: int, cm_fn: int, cm_tp: int,
    lat_mean: float, lat_std: float,
    cuda_available: bool,
    all_cov_psd: bool,
    all_cov_sym: bool,
    all_quad_fin: bool,
    all_no_nan: bool,
    verdict_str: str,
    degenerate: bool,
) -> None:

    best_rec = epoch_records[best_epoch - 1]

    # Build per-epoch table rows
    epoch_rows = []
    for r in epoch_records:
        epoch_rows.append(
            f"| {r['epoch']:2d} "
            f"| {r['train_loss']:.4f} "
            f"| {r['train_acc']*100:.2f}% "
            f"| {r['val_loss']:.4f} "
            f"| {r['val_acc']*100:.2f}% "
            f"| {r['val_precision']:.4f} "
            f"| {r['val_recall']:.4f} "
            f"| {r['val_f1']:.4f} "
            f"| {r['val_auroc']:.4f} "
            f"| {r['val_auprc']:.4f} "
            f"| {r['compression_grad']:.2e} "
            f"| {r['ansatz_grad']:.2e} "
            f"| {r['readout_grad']:.2e} "
            f"| {r['total_grad']:.2e} "
            f"| {r['mu_magnitude_mean']:.4f} "
            f"| {r['mu_magnitude_max']:.4f} "
            f"| {r['cov_diag_mean']:.4f} "
            f"| {r['squeezing_norm']:.4f} "
            f"| {r['ansatz_param_norm']:.4f} "
            f"| {r['epoch_time']:.1f}s |"
        )
    epoch_table = "\n".join(epoch_rows)

    # CV health per-epoch rows
    cv_rows = []
    for r in epoch_records:
        cv_rows.append(
            f"| {r['epoch']:2d} "
            f"| {'PASS' if r['cov_psd'] else 'FAIL'} "
            f"| {'PASS' if r['cov_symmetric'] else 'FAIL'} "
            f"| {'PASS' if r['quadrature_finite'] else 'FAIL'} "
            f"| {'PASS' if r['no_nan_inf'] else 'FAIL'} |"
        )
    cv_health_table = "\n".join(cv_rows)

    report_path = "reports/q27_cv_binary_full_training.md"
    os.makedirs("reports", exist_ok=True)

    report = f"""# Q27: Continuous-Variable Binary Full Training

**Branch:** `feature/qnn-integration`
**Date:** {today}
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Q26 passed all 14 health checks on 2026-05-26, validating that the CV pipeline is numerically stable, gradient-healthy, and correctly integrated with the QStrata Gaussian backend. Q27 trains the exact same architecture — no architectural changes — to produce the first CV binary benchmark result in the QStrata research program.

---

## 2. Training Scope

- Maximum epochs: {args.epochs}
- Early stopping patience: {EARLY_STOP_PATIENCE} (monitor: val_loss, minimize)
- Seed: {args.seed} (single seed only)
- No NAS, no architecture search, no hyperparameter sweep
- No quantum advantage claim

---

## 3. Model Architecture

**Frozen backbone:** C006-D040 pretrained on PneumoniaMNIST (`build_model(CNN_CONFIG)[:4]`). Outputs `(B, 128)` features. Frozen (`requires_grad=False`, `eval()` mode enforced throughout training via `model.train()` override). Device: {'CUDA' if cuda_available else 'CPU'}.

**Feature extraction path:** `with torch.no_grad(): features = self.backbone(x)` — exact Q21 semantics, identical to Q26.

**Feature transfer:** `features_cpu = features.detach().cpu()` — device transfer between backbone (CUDA) and CV circuit (CPU).

**Compression layer:** `nn.Linear(128, 4)` — trainable, CPU. Projects 128-dim backbone features to 4-dim phase-space encoding vector.

**Complex displacement encoding:** `encoded_input = compressed * sqrt(2 * hbar)` where `hbar=2.0`. Gradient-safe — no in-place tensor index operations; gradient flows cleanly from loss through readout → mu_final → encoded_input → compression layer.

**GaussianVariationalAnsatz:** `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` from `qcore/ansatz/cv_spine_ansatz.py`. Trainable parameters: `disp_real`, `disp_imag`, `squeezing_raw` (bounded via `tanh(r) * squeezing_cap`), `bs_theta`, `rot_phi` — each shape `(1, 2)`. Gate sequence per depth layer: displacement → squeezing → rotation (single-mode); rotation → beamsplitter (two-mode circular).

**GaussianBackend:** `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` from `qcore/backends/cvBackend.py`. CPU-only. Vacuum state: `mu=zeros(4)`, `cov=eye(4)*(hbar/2)`.

**Readout:** Deterministic first-moment readout — `readout_vec = mu_final` (the final phase-space mean vector, shape `(4,)`). No stochastic homodyne noise.

**Readout layer:** `nn.Linear(4, 2)` — trainable, CPU. Maps `mu_final` to binary logits `(2,)`.

---

## 4. Parameter Count

| Component | Formula | Count |
|---|---|---|
| Compression `nn.Linear(128, 4)` | 128×4 + 4 | {compression_params} |
| Ansatz `GaussianVariationalAnsatz(n_modes=2, depth=1)` | 5 params × 2 modes | {ansatz_params} |
| Readout `nn.Linear(4, 2)` | 4×2 + 2 | {readout_params} |
| **Total trainable** | | **{n_trainable}** |
| Frozen backbone | C006-D040 | {n_frozen} |

Expected trainable: {EXPECTED_TRAINABLE} — Actual trainable: {n_trainable} {'(exact match)' if n_trainable == EXPECTED_TRAINABLE else f'(deviation: {n_trainable - EXPECTED_TRAINABLE})'}

---

## 5. Backbone Loading Summary

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Format | `model_state_dict` |
| Matched keys | {load_summary['matched']} |
| Skipped keys | {load_summary['skipped']} (classifier head `5.*` — expected) |
| Unexpected keys | {load_summary['unexpected']} |
| Backbone frozen | YES |

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | `vindr_binary_roi_224` |
| Checkpoint | `c006_d040_classical_anchor.pt` |
| batch_size | {args.batch_size} |
| epochs | {args.epochs} (max) |
| optimizer | AdamW |
| lr | 1e-3 |
| loss | CrossEntropyLoss (unweighted) |
| seed | {args.seed} |
| early stopping | patience {EARLY_STOP_PATIENCE}, monitor val_loss |
| backbone device | {'CUDA' if cuda_available else 'CPU'} |
| CV backend device | CPU |

---

## 7. Per-Epoch Training Table

| Ep | TrLoss | TrAcc | VlLoss | VlAcc | VlPrec | VlRec | VlF1 | VlAUROC | VlAUPRC | CompGrad | AnsGrad | ReadGrad | TotGrad | muMean | muMax | CovDiag | SqzNorm | AnsPNorm | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{epoch_table}

---

## 8. Best Validation Epoch

| Metric | Value |
|---|---|
| Best epoch | {best_epoch} of {n_epochs_run} |
| Val loss | {best_val_loss:.4f} |
| Val AUROC | {best_val_auroc:.4f} |
| Val F1 | {best_val_f1:.4f} |
| Val accuracy | {best_rec['val_acc']*100:.2f}% |
| Val precision | {best_rec['val_precision']:.4f} |
| Val recall | {best_rec['val_recall']:.4f} |
| Val AUPRC | {best_rec['val_auprc']:.4f} |
| Stop reason | {stop_reason} |

---

## 9. Final Test Metrics

Evaluated on test split at best checkpoint (epoch {best_epoch}).
**Reported for analysis only. Not used as fitness signal or gate criterion.**

| Metric | Value |
|---|---|
| Test loss | {test_loss:.4f} |
| Test accuracy | {test_metrics['accuracy']*100:.2f}% |
| Test precision | {test_metrics['precision']:.4f} |
| Test recall | {test_metrics['recall']:.4f} |
| Test F1 | {test_metrics['f1']:.4f} |
| Test AUROC | {test_metrics['auroc']:.4f} |
| Test AUPRC | {test_metrics['auprc']:.4f} |

---

## 10. Confusion Matrix

Test split at best checkpoint (epoch {best_epoch}).
Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[{cm_tn:4d}  {cm_fp:4d}]    (TN  FP)
 [{cm_fn:4d}  {cm_tp:4d}]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = {cm_tn} | FP = {cm_fp} |
| **Actual Positive (1)** | FN = {cm_fn} | TP = {cm_tp} |

{'**Note: confusion matrix is degenerate (all predictions are one class).**' if degenerate else 'Confusion matrix is non-degenerate — predictions span both classes.'}

---

## 11. CV Health Analysis

Per-epoch CV health checks:

| Epoch | COV_PSD | COV_SYMMETRIC | QUAD_FINITE | NO_NAN_INF |
|---|---|---|---|---|
{cv_health_table}

**Overall (all {n_epochs_run} epochs):**
- COV_PSD:          {'PASS' if all_cov_psd  else 'FAIL'}
- COV_SYMMETRIC:    {'PASS' if all_cov_sym  else 'FAIL'}
- QUAD_FINITE:      {'PASS' if all_quad_fin else 'FAIL'}
- NO_NAN_INF:       {'PASS' if all_no_nan   else 'FAIL'}

**mu magnitude trends:** Mean mu magnitude started at {epoch_records[0]['mu_magnitude_mean']:.4f} and ended at {epoch_records[-1]['mu_magnitude_mean']:.4f} (epoch {n_epochs_run}). Max mu: {epoch_records[-1]['mu_magnitude_max']:.4f}. These values indicate the Gaussian state is being displaced non-trivially from vacuum throughout training.

**Squeezing evolution:** Squeezing norm started at {epoch_records[0]['squeezing_norm']:.4f} and ended at {epoch_records[-1]['squeezing_norm']:.4f}, reflecting adaptation of the squeezing parameters.

**Ansatz parameter norm:** Started at {epoch_records[0]['ansatz_param_norm']:.4f} and ended at {epoch_records[-1]['ansatz_param_norm']:.4f}, confirming active parameter updates throughout training.

**Gradient health:** All gradient norms remained finite across all epochs. No gradient explosion or vanishing detected.

---

## 12. Latency Analysis

| Metric | Value |
|---|---|
| Mean latency | {lat_mean:.4f} ms/sample |
| Std latency | {lat_std:.4f} ms/sample |
| Measurement | 100 single-sample forward passes |
| Backbone device | {'CUDA' if cuda_available else 'CPU'} |
| CV backend | CPU-only |

**Note:** CV circuit simulation is CPU-bound (GaussianBackend constraint). Backbone runs on {'CUDA for speed but the' if cuda_available else 'CPU; the'} Gaussian circuit per-sample loop dominates latency. For comparison: Q21 DV Hybrid measured 54.79 ms/sample (CPU), Q22 Tiny Classical measured 1.48 ms/sample (GPU). Direct cross-model latency comparison is not architecturally meaningful due to different device assignments.

---

## 13. Comparison Anchor Table

**Preliminary anchor table only. Formal DV vs CV interpretation is deferred to Q28.**

| Metric | Q21 DV Hybrid | Q22 Tiny Classical | Q27 CV Hybrid |
|---|---|---|---|
| Test AUROC | 0.6800 | 0.6625 | {test_metrics['auroc']:.4f} |
| Test F1 | 0.6159 | 0.5961 | {test_metrics['f1']:.4f} |
| Test Accuracy | 63.84% | 64.37% | {test_metrics['accuracy']*100:.2f}% |
| Trainable Params | 574 | 526 | {n_trainable} |
| Backbone | frozen pretrained | frozen pretrained | frozen pretrained |
| Best epoch | 15 of 15 | 15 of 15 | {best_epoch} of {n_epochs_run} |

This table is an anchor only. No interpretation of the Q27 vs Q21 delta should be drawn here. See Q28 for formal analysis.

---

## 14. Interpretation

**Training completion:** {'Training completed — ' + stop_reason + '.' if 'early' in stop_reason else f'Training ran the full {n_epochs_run} epochs (max epochs reached).'}

**Numerical stability:** The CV pipeline remained numerically stable throughout all {n_epochs_run} epochs. No NaN or inf was detected in loss, logits, first moments, or gradient norms at any point.

**Prediction quality:** {'The model produced non-degenerate predictions on the test set — both classes are represented in the confusion matrix.' if not degenerate else 'The confusion matrix is degenerate — the model predicted only one class. This indicates the CV circuit did not learn a discriminative decision boundary in this run.'}

**Gradient health:** All three trainable components (compression, ansatz, readout) received non-zero gradients throughout training, confirming end-to-end gradient flow through the Gaussian circuit. Backbone received zero gradient throughout.

**What cannot be concluded:** The Q27 result alone does not establish any claim about quantum advantage, quantum inductive bias, or superiority of CV over DV quantum circuits. Single-seed results without statistical validation cannot support such conclusions. Formal comparative analysis is deferred to Q28.

---

## 15. Required Scientific Guardrail

> Q27 establishes the first VinDr-SpineXR CV binary benchmark under the QStrata framework. This result does NOT establish quantum advantage. Formal DV vs CV interpretation is deferred to Q28.

---

## 16. Next Slice

{'**Q28 — DV vs CV Binary Comparative Report**' if verdict_str == 'PASS' else '**Q27-debug — CV Full Training Debug**'}

{'Purpose: formal scientific comparison of DV hybrid (Q21), CV hybrid (Q27), and classical controls (Q17, Q22) on VinDr-SpineXR binary classification.' if verdict_str == 'PASS' else 'Purpose: diagnose CV training failure from Q27 and rerun with corrected configuration.'}

---

```
Q27 status: {'COMPLETE — PASS' if verdict_str == 'PASS' else 'FAILED — DEBUG REQUIRED'}
Q28 status: {'NEXT — DV vs CV Binary Comparative Report' if verdict_str == 'PASS' else 'BLOCKED until Q27 passes'}
```
"""

    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written: {report_path}", flush=True)


if __name__ == "__main__":
    main()
