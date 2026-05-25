"""
qcore/nas/evaluator.py

NAS v1 candidate evaluator — Slice 25.

Implements the stable benchmark protocol v1 as a callable function:
  - Fixed random seed applied before any data loading or model instantiation
  - Epoch-level validation accuracy tracking
  - Best-validation checkpoint saved in memory (no file I/O)
  - Test evaluation at best-validation checkpoint
  - Returns a populated EvaluatorResult dataclass

Public API
----------
    EvaluatorResult — dataclass holding all per-candidate metrics
    evaluate_candidate(candidate_id, config_path, seed=42) -> EvaluatorResult
"""

from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Repo root on path — allows import from any working directory ──────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter
from qcore.models.cnn import build_model


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluatorResult:
    """
    Structured result returned by evaluate_candidate().

    Accuracy fields are stored as percentages (0–100 scale), not fractions.
    test_acc_analysis_only is for post-hoc analysis only — it must never be
    used as a fitness signal or in Pareto dominance calculations.
    """
    candidate_id:           str        # Candidate identifier, e.g. "C001"
    block_type:             str        # Block type from config, e.g. "standard"
    conv_channels:          list       # Channel configuration, e.g. [32, 64]
    params:                 int        # Total trainable parameter count
    best_val_acc:           float      # Best validation accuracy across all epochs (%)
    best_epoch:             int        # Epoch (1-based) at which best val acc occurred
    final_train_acc:        float      # Training accuracy at the last epoch (%)
    test_acc_analysis_only: float      # Test accuracy at best-val checkpoint (%) — analysis only
    mean_epoch_time:        float      # Mean wall-clock time per training epoch (seconds)
    latency_ms:             float      # Mean inference latency (ms/batch)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_seed(seed: int) -> None:
    """Apply fixed seed to Python, NumPy, and PyTorch. Must be called before
    any data loading or model instantiation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _compute_class_weights(
    dataset_adapter, num_classes: int, device: str
) -> torch.Tensor:
    labels   = [dataset_adapter[i][1].item() for i in range(len(dataset_adapter))]
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts   = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels_t == c).sum().float()
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_candidate(
    candidate_id: str,
    config_path: str,
    seed: int = 42,
) -> EvaluatorResult:
    """
    Train and evaluate a single NAS candidate using the stable benchmark
    protocol v1.

    Parameters
    ----------
    candidate_id : str
        Human-readable identifier (e.g. "C001"). Used in log output only.
    config_path : str
        Path to the candidate's YAML config file.
    seed : int
        Random seed. Must be 42 for all v1 evaluations. Applied to Python
        random, NumPy, and PyTorch before any data loading or model
        instantiation.

    Returns
    -------
    EvaluatorResult
        Fully populated result object. Accuracy fields are in percent (0–100).

    Raises
    ------
    RuntimeError
        On any failure during config loading, dataset loading, model building,
        training, or evaluation. No partial result is returned.
    """
    # ── 1. Seed — must precede ALL data and model operations ─────────────────
    _apply_seed(seed)

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            f"evaluate_candidate({candidate_id!r}): CUDA device not available. "
            "GPU is required for NAS v1 evaluation."
        )

    print(f"\n{'='*60}")
    print(f"Evaluating : {candidate_id}  (seed={seed})")
    print(f"Config     : {config_path}")
    print(f"Device     : {device}  —  {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # ── 3. Load config ────────────────────────────────────────────────────────
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        raise RuntimeError(
            f"evaluate_candidate({candidate_id!r}): failed to load config "
            f"'{config_path}': {exc}"
        ) from exc

    epochs      = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]
    lr          = config["training"]["lr"]
    use_cw      = config["training"]["class_weights"]
    num_classes = config["dataset"]["num_classes"]
    block_type  = config["model"].get("block_type", "standard")
    conv_channels = list(config["model"]["conv_channels"])

    print(f"\nTraining settings (YAML, no overrides):")
    print(f"  block_type={block_type}, conv_channels={conv_channels}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, class_weights={use_cw}")

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    try:
        train_ds = get_dataset(config["dataset"]["name"], "train")
        val_ds   = get_dataset(config["dataset"]["name"], "val")
        test_ds  = get_dataset(config["dataset"]["name"], "test")
    except Exception as exc:
        raise RuntimeError(
            f"evaluate_candidate({candidate_id!r}): dataset loading failed: {exc}"
        ) from exc

    train_adapted = TorchDatasetAdapter(train_ds)
    val_adapted   = TorchDatasetAdapter(val_ds)
    test_adapted  = TorchDatasetAdapter(test_ds)

    print(f"\nSplit sizes: train={len(train_adapted)}, "
          f"val={len(val_adapted)}, test={len(test_adapted)}")

    # Seeded generator for reproducible train-loader shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_adapted, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader  = DataLoader(val_adapted,  batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_adapted, batch_size=batch_size, shuffle=False)

    # ── 5. Class weights ──────────────────────────────────────────────────────
    cw = _compute_class_weights(train_adapted, num_classes, device) if use_cw else None
    if cw is not None:
        print(f"Class weights: {cw.cpu().tolist()}")

    # ── 6. Model ──────────────────────────────────────────────────────────────
    model_cfg = {**config["model"], **config["dataset"]}
    try:
        model = build_model(model_cfg).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"evaluate_candidate({candidate_id!r}): model build failed: {exc}"
        ) from exc

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    # ── 7. Training with best-checkpoint tracking ─────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(weight=cw)

    epoch_times     = []
    final_train_acc = 0.0
    best_val_acc    = -1.0
    best_epoch      = -1
    best_weights    = None   # in-memory; no file I/O

    print(f"\nTraining ({epochs} epochs):")
    print(
        f"  {'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  "
        f"{'train_acc':>9}  {'val_acc':>8}  {'time(s)':>7}  {'best?':>5}"
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        run_loss, n_cor, n_tot = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            n_cor    += (logits.argmax(1) == yb).sum().item()
            n_tot    += xb.size(0)

        train_loss = run_loss / n_tot
        train_acc  = n_cor   / n_tot

        # Validate
        model.eval()
        v_loss, v_cor, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                v_loss += loss_fn(logits, yb).item() * xb.size(0)
                v_cor  += (logits.argmax(1) == yb).sum().item()
                v_tot  += xb.size(0)

        val_loss = v_loss / v_tot
        val_acc  = v_cor  / v_tot

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"  {epoch:>6d}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
            f"{train_acc:>9.4f}  {val_acc:>8.4f}  {epoch_time:>7.2f}  "
            f"{'  ✓' if is_best else ''}"
        )

        final_train_acc = train_acc

    if best_weights is None:
        raise RuntimeError(
            f"evaluate_candidate({candidate_id!r}): no best-validation checkpoint "
            "was saved — the training loop may have produced no valid epochs."
        )

    # ── 8. Restore best-validation weights ───────────────────────────────────
    model.load_state_dict(best_weights)
    print(
        f"\nRestored best weights from epoch {best_epoch} "
        f"(best val acc = {best_val_acc * 100:.2f}%)"
    )

    # ── 9. Test evaluation at best-validation checkpoint ─────────────────────
    model.eval()
    t_cor, t_tot    = 0, 0
    batch_latencies = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            t_start = time.time()
            logits  = model(xb)
            batch_latencies.append((time.time() - t_start) * 1000)
            t_cor += (logits.argmax(1).cpu() == yb).sum().item()
            t_tot += xb.size(0)

    test_acc        = t_cor / t_tot
    mean_latency_ms = sum(batch_latencies) / len(batch_latencies)
    mean_epoch_time = sum(epoch_times)     / len(epoch_times)

    print(f"\n{'─'*60}")
    print(f"RESULT — {candidate_id}")
    print(f"  Parameters         : {n_params:,}")
    print(f"  Best val accuracy  : {best_val_acc * 100:.2f}%  (epoch {best_epoch})")
    print(f"  Final train acc    : {final_train_acc * 100:.2f}%")
    print(f"  Test acc*          : {test_acc * 100:.2f}%  ({t_cor}/{t_tot})")
    print(f"  Mean epoch time    : {mean_epoch_time:.2f}s")
    print(f"  Mean latency       : {mean_latency_ms:.3f} ms/batch")
    print(f"  * evaluated at best-val checkpoint (epoch {best_epoch})")
    print(f"{'─'*60}")

    # ── 10. Return structured result (accuracy as %, not fraction) ────────────
    return EvaluatorResult(
        candidate_id=           candidate_id,
        block_type=             block_type,
        conv_channels=          conv_channels,
        params=                 n_params,
        best_val_acc=           best_val_acc    * 100.0,
        best_epoch=             best_epoch,
        final_train_acc=        final_train_acc * 100.0,
        test_acc_analysis_only= test_acc        * 100.0,
        mean_epoch_time=        mean_epoch_time,
        latency_ms=             mean_latency_ms,
    )
