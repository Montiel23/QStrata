"""
scripts/smoke_test_vindr_cv_binary.py

CV Binary Smoke Test for VinDr-SpineXR binary ROI dataset.
Slice Q26.

Validates the minimal CV pipeline defined in Q25 (design) and Q25A (locked params).

Architecture (from Q25A Section 7):
  Input (4, 1, 224, 224) — batch_size=4
  → Frozen C006-D040 backbone (build_model(CNN_CONFIG)[:4])     [frozen]  → (4, 128)
  → nn.Linear(128, 4)   [compression, trainable, n_modes=2]               → (4, 4)
  → per-sample GaussianVariationalAnsatz loop (CPU)                        → (4, 4) mu_final
  → nn.Linear(4, 2)     [readout, trainable]                               → (4, 2) logits

Health checks performed (from Q25 Section 11):
  Forward:
    FORWARD_PASS        — one batch executes without error
    LOGITS_SHAPE        — logits shape is (4, 2)
    LOGITS_FINITE       — no NaN or inf in logits
    MU_FINITE           — no NaN or inf in first moments mu_final
    COV_FINITE          — no NaN or inf in covariance cov_final

  Gradient:
    LOSS_FINITE         — loss is finite (not NaN or inf)
    GRAD_COMPRESSION    — compression layer received non-zero gradient
    GRAD_ANSATZ         — ansatz parameters received non-zero gradients
    GRAD_READOUT        — readout layer received non-zero gradient
    BACKBONE_FROZEN     — backbone received zero gradient throughout
    GRAD_FINITE         — all gradient norms are finite

  Optimizer:
    PARAMS_UPDATED      — at least one trainable parameter changed after optimizer step

  CV-specific:
    COV_SYMMETRIC       — covariance matrix is symmetric (|cov - cov.T| < 1e-5)
    COV_PSD             — covariance matrix is positive semi-definite
                          (all eigenvalues >= -1e-6, allowing numerical noise)

Hard stops (abort with sys.exit(2) immediately):
  - NaN or inf in loss
  - NaN or inf in any logit
  - NaN or inf in any mu_final element
  - Backbone gradient detected (any backbone param received gradient > 0)

All checks pass → SMOKE TEST RESULT: PASS
Any check fails → SMOKE TEST RESULT: FAIL

Usage:
    python3 scripts/smoke_test_vindr_cv_binary.py \\
        --root data/processed/vindr_binary_roi_224 \\
        --checkpoint checkpoints/c006_d040_classical_anchor.pt \\
        --batch-size 4 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
from qcore.backends.cvBackend import GaussianBackend
from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.cnn import build_model

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

N_MODES        = 2
CV_DEPTH       = 1
SQUEEZING_CAP  = 1.5
HBAR           = 2.0

EXPECTED_TRAINABLE = 536  # 516 (compression) + 10 (ansatz) + 10 (readout)

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR CV Binary Smoke Test (Slice Q26)"
    )
    p.add_argument("--root",       default="data/processed/vindr_binary_roi_224")
    p.add_argument("--checkpoint", default="checkpoints/c006_d040_classical_anchor.pt",
                   help="Pretrained C006-D040 backbone checkpoint path")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model ──────────────────────────────────────────────────────────────────────

class CVHybridBinaryModel(nn.Module):
    """
    CV Hybrid CNN model for VinDr-SpineXR binary classification.

    Architecture (Q25A Section 7):
      backbone    — frozen C006-D040 pretrained on PneumoniaMNIST (CUDA if available)
      compression — nn.Linear(128, 4),  trainable  (CPU, feeds CV circuit)
      ansatz      — GaussianVariationalAnsatz(n_modes=2, depth=1), trainable (CPU)
      readout     — nn.Linear(4, 2),    trainable  (CPU)

    Device split:
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
        self.backbone    = backbone     # frozen; on CUDA if available
        self.compression = compression  # trainable; on CPU
        self.ansatz      = ansatz       # trainable; on CPU
        self.readout     = readout      # trainable; on CPU
        self.backend     = backend      # not nn.Module; CPU-only
        self.n_modes     = n_modes

        # Storage for last batch's Gaussian state (for health checks)
        self._last_mu:  torch.Tensor | None = None
        self._last_cov: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "CVHybridBinaryModel":
        """Override to keep backbone always in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 1, H, W). Device follows backbone device.

        Returns:
            logits: (B, 2) binary classification logits on CPU.
        """
        B = x.shape[0]

        # ── Backbone: frozen feature extraction (CUDA if available) ───────────
        with torch.no_grad():
            features = self.backbone(x)   # (B, 128), requires_grad=False

        # Transfer to CPU for CV computation (GaussianBackend is CPU-only)
        features_cpu = features.detach().cpu()  # (B, 128)

        # ── Per-sample CV loop ─────────────────────────────────────────────────
        logits_list: list[torch.Tensor] = []
        mu_list:     list[torch.Tensor] = []
        cov_list:    list[torch.Tensor] = []

        for i in range(B):
            # Compression: (128,) → (4,), trainable
            compressed = self.compression(features_cpu[i])   # (4,)

            # Gradient-safe encoding: scale to phase-space displacement units
            # encoded_input = compressed * sqrt(2*hbar)
            # Uses multiplication (not in-place index ops) for clean gradient flow
            scale = math.sqrt(2.0 * self.backend.hbar)
            encoded_input = compressed * scale   # (4,), grad flows to compression

            # Initialize vacuum state on CPU
            mu_0, cov_0 = self.backend.get_vacuum()   # mu (4,), cov (4,4)

            # Evolve through Gaussian variational ansatz
            mu_final, cov_final = self.ansatz.apply(
                mu_0, cov_0, self.backend, encoded_input
            )
            # mu_final: (4,), cov_final: (4,4)

            # Deterministic first-moment readout (NOT stochastic homodyne)
            # readout_vec = mu_final directly (Q25 Section 9)
            logit_i = self.readout(mu_final)   # (2,)

            logits_list.append(logit_i)
            mu_list.append(mu_final)
            cov_list.append(cov_final.detach())   # detach cov for storage only

        # Store Gaussian state for health checks (detached copies)
        self._last_mu  = torch.stack(mu_list)              # (B, 4)
        self._last_cov = torch.stack(cov_list)             # (B, 4, 4)

        return torch.stack(logits_list)   # (B, 2)


# ── Backbone loading ───────────────────────────────────────────────────────────

def build_backbone(cnn_config: dict) -> nn.Sequential:
    """
    Build frozen C006-D040 backbone.
    Returns build_model(cnn_config)[:4] — outputs (B, 128) features.
    """
    full_model = build_model(cnn_config)
    backbone = nn.Sequential(*list(full_model.children())[:4])
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone


def load_pretrained_backbone(model: CVHybridBinaryModel, checkpoint_path: str) -> dict:
    """
    Load C006-D040 backbone weights into CVHybridBinaryModel.

    Key remapping (identical to Q21/Q22):
      "0.*" → "backbone.0.*"   (depthwise_sep block 0, 1→64 ch)
      "1.*" → "backbone.1.*"   (depthwise_sep block 1, 64→128 ch)
      "5.*" → skipped          (classifier head, not used)

    Hard stops:
      - Checkpoint file not found
      - Unrecognised checkpoint format
      - Any backbone key fails to load

    Returns:
      dict with matched, skipped, unexpected counts.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[HARD STOP] Checkpoint not found: {checkpoint_path}", flush=True)
        sys.exit(1)

    ck = torch.load(checkpoint_path, map_location="cpu")

    # Detect format (same logic as Q21/Q22)
    if "model_state_dict" in ck:
        src_sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        src_sd = ck["state_dict"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in list(ck.values())[:3]):
        src_sd = ck
    else:
        print(f"[HARD STOP] Unrecognised checkpoint format at {checkpoint_path}", flush=True)
        sys.exit(1)

    # Remap keys
    remapped: dict = {}
    skipped:  list = []
    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            remapped[f"backbone.{k}"] = v
        elif k.startswith("5."):
            skipped.append(k)
        # else: unexpected — ignore silently, count below

    # Load with strict=False (only backbone keys provided)
    missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
    backbone_keys    = set(f"backbone.{k}" for k in model.backbone.state_dict().keys())
    matched          = [k for k in remapped if k in backbone_keys]
    missing_backbone = [k for k in missing_keys if k.startswith("backbone.")]

    if missing_backbone:
        print(
            f"[HARD STOP] {len(missing_backbone)} backbone keys failed to load: "
            f"{missing_backbone}",
            flush=True,
        )
        sys.exit(1)

    return {
        "matched":    len(matched),
        "skipped":    len(skipped),
        "unexpected": len(unexpected_keys),
    }


# ── Health check helpers ───────────────────────────────────────────────────────

def check_tensor_finite(tensor: torch.Tensor, name: str, hard_stop: bool = False) -> bool:
    """Return True if all elements are finite; optionally hard-stop."""
    is_finite = bool(torch.isfinite(tensor).all().item())
    if not is_finite and hard_stop:
        print(f"\n[HARD STOP] {name} contains NaN or inf. Aborting.", flush=True)
        sys.exit(2)
    return is_finite


def check_backbone_frozen(model: CVHybridBinaryModel) -> bool:
    """
    Hard stop if any backbone parameter received a non-zero gradient.
    Returns True only if all backbone grads are None or zero.
    """
    for name, param in model.backbone.named_parameters():
        if param.grad is not None and param.grad.abs().max().item() > 0.0:
            print(
                f"\n[HARD STOP] Backbone parameter 'backbone.{name}' received "
                f"gradient: max abs = {param.grad.abs().max().item():.4e}. "
                f"Backbone must remain frozen. Aborting.",
                flush=True,
            )
            sys.exit(2)
    return True


def grad_norm(model: CVHybridBinaryModel, *param_names: str) -> float:
    """Compute L2 gradient norm over named parameters."""
    total = 0.0
    params = dict(model.named_parameters())
    for n in param_names:
        p = params.get(n)
        if p is not None and p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


def ansatz_grad_norm(ansatz: GaussianVariationalAnsatz) -> float:
    """Compute combined L2 gradient norm over all ansatz parameters."""
    total = 0.0
    for p in ansatz.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


def check_cov_symmetric(cov: torch.Tensor, tol: float = 1e-5) -> bool:
    """Check that covariance matrix is symmetric."""
    return bool(((cov - cov.transpose(-1, -2)).abs() < tol).all().item())


def check_cov_psd(cov: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check that covariance matrix is positive semi-definite.
    All eigenvalues must be >= -tol (allow small numerical noise).
    Uses torch.linalg.eigvalsh (avoids numpy, compatible with NumPy 2.x env).
    """
    cov_float = cov.detach().cpu().float()   # (B, 4, 4)
    # Per-sample check
    for i in range(cov_float.shape[0]):
        eigvals = torch.linalg.eigvalsh(cov_float[i])   # ascending order
        if float(eigvals.min().item()) < -tol:
            return False
    return True


# ── Main smoke test ────────────────────────────────────────────────────────────

def run_smoke_test() -> None:
    args = parse_args()
    set_seeds(args.seed)

    print("=" * 70, flush=True)
    print("Q26 — CV Binary Smoke Test", flush=True)
    print(f"Date:       {datetime.date.today().isoformat()}", flush=True)
    print(f"Branch:     feature/qnn-integration", flush=True)
    print(f"Seed:       {args.seed}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Root:       {args.root}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print("=" * 70, flush=True)

    # ── Device setup ──────────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    backbone_device = torch.device("cuda" if cuda_available else "cpu")
    cpu_device      = torch.device("cpu")

    print(f"\n[DEVICE] CUDA available: {cuda_available}", flush=True)
    print(f"[DEVICE] Backbone device: {backbone_device}", flush=True)
    print(f"[DEVICE] CV backend device: cpu (GaussianBackend constraint)", flush=True)

    # ── Build model components ────────────────────────────────────────────────
    print("\n[BUILD] Constructing model components...", flush=True)

    backbone    = build_backbone(CNN_CONFIG)
    compression = nn.Linear(128, 2 * N_MODES)   # 128→4, 516 trainable params
    ansatz      = GaussianVariationalAnsatz(n_modes=N_MODES, depth=CV_DEPTH, squeezing_cap=SQUEEZING_CAP)
    backend     = GaussianBackend(n_modes=N_MODES, hbar=HBAR, device="cpu")
    readout     = nn.Linear(2 * N_MODES, 2)      # 4→2, 10 trainable params

    model = CVHybridBinaryModel(
        backbone=backbone,
        compression=compression,
        ansatz=ansatz,
        backend=backend,
        readout=readout,
        n_modes=N_MODES,
    )

    # Move backbone to GPU; CV components stay on CPU
    model.backbone = model.backbone.to(backbone_device)
    # compression, ansatz, readout remain on CPU

    # ── Parameter count ───────────────────────────────────────────────────────
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable

    compression_params = sum(p.numel() for p in model.compression.parameters())
    ansatz_params      = sum(p.numel() for p in model.ansatz.parameters())
    readout_params     = sum(p.numel() for p in model.readout.parameters())

    print(f"\n[PARAMS] Trainable parameter breakdown:", flush=True)
    print(f"  compression (Linear 128→4):  {compression_params}", flush=True)
    print(f"  ansatz (GaussianVariational): {ansatz_params}", flush=True)
    print(f"  readout (Linear 4→2):         {readout_params}", flush=True)
    print(f"  Total trainable:              {n_trainable}", flush=True)
    print(f"  Total frozen (backbone):      {n_frozen}", flush=True)
    print(f"  Grand total:                  {n_total}", flush=True)
    print(f"  Expected trainable:           {EXPECTED_TRAINABLE}", flush=True)

    if n_trainable != EXPECTED_TRAINABLE:
        print(
            f"  [WARNING] Trainable count {n_trainable} ≠ expected {EXPECTED_TRAINABLE}. "
            f"Deviation: {n_trainable - EXPECTED_TRAINABLE}. "
            f"Proceeding — exact count logged above.",
            flush=True,
        )
    else:
        print(f"  [OK] Trainable count matches expected ({EXPECTED_TRAINABLE}).", flush=True)

    # ── Verify backbone frozen at construction ────────────────────────────────
    print("\n[FREEZE] Verifying backbone is frozen...", flush=True)
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            print(
                f"[HARD STOP] Backbone parameter 'backbone.{name}' has requires_grad=True "
                f"at construction. Aborting.",
                flush=True,
            )
            sys.exit(2)
    print("[FREEZE] All backbone parameters: requires_grad=False — OK", flush=True)

    # ── Load pretrained backbone ───────────────────────────────────────────────
    print(f"\n[BACKBONE] Loading pretrained weights from {args.checkpoint}...", flush=True)
    load_summary = load_pretrained_backbone(model, args.checkpoint)
    print(f"[BACKBONE] Matched keys:    {load_summary['matched']}", flush=True)
    print(f"[BACKBONE] Skipped keys:    {load_summary['skipped']} (classifier head — expected)", flush=True)
    print(f"[BACKBONE] Unexpected keys: {load_summary['unexpected']}", flush=True)

    if load_summary["matched"] != 28:
        print(
            f"[WARNING] Expected 28 matched backbone keys, got {load_summary['matched']}.",
            flush=True,
        )

    # ── Load data — one batch only ─────────────────────────────────────────────
    print(f"\n[DATA] Loading one batch from {args.root}...", flush=True)
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    loader   = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch.to(backbone_device)
    y_batch = y_batch.to(cpu_device)   # labels on CPU for loss computation

    print(f"[DATA] x_batch shape: {tuple(x_batch.shape)}", flush=True)
    print(f"[DATA] y_batch shape: {tuple(y_batch.shape)}, values: {y_batch.tolist()}", flush=True)

    # ── Set training mode (backbone stays eval) ───────────────────────────────
    model.train()
    # Confirm backbone is in eval mode after .train() override
    assert not model.backbone.training, "[HARD STOP] Backbone entered train mode after model.train()"

    # ── Optimizer setup ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Snapshot trainable params before update ───────────────────────────────
    params_before: dict[str, torch.Tensor] = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    # ══════════════════════════════════════════════════════════════════════════
    # FORWARD PASS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[FORWARD] Running forward pass (batch_size=4)...", flush=True)

    try:
        logits = model(x_batch)   # (4, 2)
    except Exception as exc:
        print(f"\n[HARD STOP] Forward pass raised exception: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(2)

    print(f"[FORWARD] logits shape:  {tuple(logits.shape)}", flush=True)
    print(f"[FORWARD] logits values: {logits.detach().tolist()}", flush=True)
    print(f"[FORWARD] mu_final (last batch): {model._last_mu.tolist()}", flush=True)

    # ── Hard stop: NaN/inf in logits ──────────────────────────────────────────
    check_tensor_finite(logits,          "logits",    hard_stop=True)
    check_tensor_finite(model._last_mu,  "mu_final",  hard_stop=True)
    check_tensor_finite(model._last_cov, "cov_final", hard_stop=True)

    # ── Compute loss ──────────────────────────────────────────────────────────
    # Move logits to CPU for loss (labels are on CPU)
    loss = criterion(logits.cpu(), y_batch)
    print(f"[FORWARD] loss value:    {loss.item():.6f}", flush=True)

    # ── Hard stop: NaN/inf in loss ────────────────────────────────────────────
    check_tensor_finite(loss, "loss", hard_stop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BACKWARD PASS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[BACKWARD] Running backward pass...", flush=True)
    optimizer.zero_grad()
    loss.backward()

    # ── Hard stop: backbone gradient check ────────────────────────────────────
    check_backbone_frozen(model)
    print("[BACKWARD] Backbone gradient check: PASS (zero gradient confirmed)", flush=True)

    # ── Gradient norms ────────────────────────────────────────────────────────
    gn_compression = grad_norm(model, "compression.weight", "compression.bias")
    gn_ansatz      = ansatz_grad_norm(model.ansatz)
    gn_readout     = grad_norm(model, "readout.weight", "readout.bias")

    print(f"[BACKWARD] Gradient norms:", flush=True)
    print(f"  compression: {gn_compression:.4e}", flush=True)
    print(f"  ansatz:      {gn_ansatz:.4e}", flush=True)
    print(f"  readout:     {gn_readout:.4e}", flush=True)

    # ── Hard stop: NaN/inf in gradient norms ──────────────────────────────────
    for gn_name, gn_val in [
        ("compression", gn_compression),
        ("ansatz", gn_ansatz),
        ("readout", gn_readout),
    ]:
        if not math.isfinite(gn_val):
            print(
                f"\n[HARD STOP] Gradient norm '{gn_name}' is not finite: {gn_val}. Aborting.",
                flush=True,
            )
            sys.exit(2)

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMIZER STEP
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[OPTIMIZER] Running AdamW step (lr=1e-3)...", flush=True)
    optimizer.step()
    print("[OPTIMIZER] Optimizer step complete.", flush=True)

    # ── Verify parameters updated ─────────────────────────────────────────────
    any_updated = False
    for n, p in model.named_parameters():
        if p.requires_grad:
            if not torch.allclose(p.detach(), params_before[n], atol=0.0):
                any_updated = True
                break

    print(f"[OPTIMIZER] Parameters updated after step: {any_updated}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HEALTH CHECKS — evaluate all, collect results
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60, flush=True)
    print("[CHECKS] Evaluating all health checks...", flush=True)

    checks: dict[str, bool] = {}

    # Forward checks
    checks["FORWARD_PASS"]  = True   # reached here — no exception
    checks["LOGITS_SHAPE"]  = (tuple(logits.shape) == (args.batch_size, 2))
    checks["LOGITS_FINITE"] = bool(torch.isfinite(logits).all().item())
    checks["MU_FINITE"]     = bool(torch.isfinite(model._last_mu).all().item())
    checks["COV_FINITE"]    = bool(torch.isfinite(model._last_cov).all().item())

    # Loss check
    checks["LOSS_FINITE"]   = math.isfinite(loss.item())

    # Gradient checks
    checks["GRAD_COMPRESSION"] = (gn_compression > 0.0)
    checks["GRAD_ANSATZ"]      = (gn_ansatz > 0.0)
    checks["GRAD_READOUT"]     = (gn_readout > 0.0)
    checks["BACKBONE_FROZEN"]  = True   # reached here — hard stop would have fired
    checks["GRAD_FINITE"]      = (
        math.isfinite(gn_compression) and
        math.isfinite(gn_ansatz) and
        math.isfinite(gn_readout)
    )

    # Optimizer check
    checks["PARAMS_UPDATED"] = any_updated

    # CV-specific checks
    cov_last = model._last_cov.float()   # (B, 4, 4)
    checks["COV_SYMMETRIC"] = check_cov_symmetric(cov_last, tol=1e-5)
    checks["COV_PSD"]       = check_cov_psd(cov_last, tol=1e-6)

    # ── Print check table ──────────────────────────────────────────────────────
    print(f"\n{'Check':<22} {'Result'}", flush=True)
    print("─" * 35, flush=True)
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name:<20} {status}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    all_pass = all(checks.values())
    failed   = [k for k, v in checks.items() if not v]

    print("\n" + "=" * 70, flush=True)
    if all_pass:
        print("SMOKE TEST RESULT: PASS", flush=True)
        print("All 14 health checks passed.", flush=True)
        print("Q26 is complete. Q27 (CV Binary Full Training) is unblocked.", flush=True)
    else:
        print("SMOKE TEST RESULT: FAIL", flush=True)
        print(f"Failed checks ({len(failed)}): {', '.join(failed)}", flush=True)
        print("Q27 is BLOCKED until Q26 passes.", flush=True)
    print("=" * 70, flush=True)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n[SUMMARY]", flush=True)
    print(f"  Trainable params:    {n_trainable}", flush=True)
    print(f"  Frozen params:       {n_frozen}", flush=True)
    print(f"  Loss:                {loss.item():.6f}", flush=True)
    print(f"  Logits:              {logits.detach().tolist()}", flush=True)
    print(f"  mu_final:            {model._last_mu.tolist()}", flush=True)
    print(f"  Grad (compression):  {gn_compression:.4e}", flush=True)
    print(f"  Grad (ansatz):       {gn_ansatz:.4e}", flush=True)
    print(f"  Grad (readout):      {gn_readout:.4e}", flush=True)
    print(f"  Params updated:      {any_updated}", flush=True)

    # Exit code: 0 = PASS, 1 = FAIL
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    run_smoke_test()
