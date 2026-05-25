"""
scripts/benchmark_nas_v0_candidates_stable.py

NAS v0 stable benchmark — Slice 20.

Implements the stable benchmark protocol (v1):
  - Fixed random seed (42) before data loading and model instantiation
  - Epoch-level validation tracking
  - Best-checkpoint selection (in-memory, no file I/O)
  - Test evaluation using best-validation weights

Usage:
    python scripts/benchmark_nas_v0_candidates_stable.py \
        --config experiments/configs/binary_baseline.yaml \
        --candidate_id C001
"""

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Seed — must be set before any data loading or model instantiation ─────────
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter
from qcore.models.cnn import build_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset_adapter, num_classes: int, device: str) -> torch.Tensor:
    labels = [dataset_adapter[i][1].item() for i in range(len(dataset_adapter))]
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels_t == c).sum().float()
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ── Core benchmark ────────────────────────────────────────────────────────────

def run_stable_benchmark(candidate_id: str, config_path: str, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Stable Benchmark: {candidate_id}")
    print(f"Config : {config_path}")
    print(f"Seed   : {SEED}")
    print(f"{'='*60}")

    config = load_config(config_path)

    # ── Hyperparameters from config — no overrides ────────────────────────────
    epochs      = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]
    lr          = config["training"]["lr"]
    use_cw      = config["training"]["class_weights"]
    num_classes = config["dataset"]["num_classes"]

    print(f"\nTraining settings (from YAML):")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"class_weights={use_cw}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = get_dataset(config["dataset"]["name"], "train")
    val_ds   = get_dataset(config["dataset"]["name"], "val")
    test_ds  = get_dataset(config["dataset"]["name"], "test")

    train_adapted = TorchDatasetAdapter(train_ds)
    val_adapted   = TorchDatasetAdapter(val_ds)
    test_adapted  = TorchDatasetAdapter(test_ds)

    print(f"\nSplit sizes: train={len(train_adapted)}, "
          f"val={len(val_adapted)}, test={len(test_adapted)}")

    # Use a generator seeded to 42 for the training loader so shuffle is
    # reproducible; val/test loaders are not shuffled.
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_adapted, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader  = DataLoader(val_adapted,   batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_adapted,  batch_size=batch_size, shuffle=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    if use_cw:
        cw = compute_class_weights(train_adapted, num_classes, device)
        print(f"Class weights: {cw.cpu().tolist()}")
    else:
        cw = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {}
    for k, v in config["model"].items():
        model_cfg[k] = v
    for k, v in config["dataset"].items():
        model_cfg[k] = v

    model    = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    # ── Training with best-checkpoint tracking ────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss(weight=cw)

    epoch_times     = []
    final_train_acc = 0.0

    # Best-checkpoint state
    best_val_acc    = -1.0
    best_epoch      = -1
    best_weights    = None          # in-memory copy — no file I/O

    print(f"\nTraining ({epochs} epochs):")
    print(f"  {'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  "
          f"{'train_acc':>9}  {'val_acc':>8}  {'time(s)':>7}  {'best?':>5}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
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
        train_acc  = n_cor / n_tot

        # ── Validate ──────────────────────────────────────────────────────────
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
        val_acc  = v_cor / v_tot

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ── Best-checkpoint bookkeeping (in-memory, no file I/O) ─────────────
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

    # ── Restore best-validation weights ──────────────────────────────────────
    assert best_weights is not None, "No best weights saved — training may have failed."
    model.load_state_dict(best_weights)
    print(f"\nRestored best weights from epoch {best_epoch} "
          f"(best val acc = {best_val_acc * 100:.2f}%)")

    # ── Test evaluation at best-validation checkpoint ─────────────────────────
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
    mean_epoch_time = sum(epoch_times) / len(epoch_times)

    # ── Results block ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"RESULTS — {candidate_id}  ({config_path})")
    print(f"{'─'*60}")
    print(f"  Seed                     : {SEED}")
    print(f"  Parameters               : {n_params:,}")
    print(f"  Best val accuracy        : {best_val_acc * 100:.2f}%  (epoch {best_epoch})")
    print(f"  Final train accuracy     : {final_train_acc * 100:.2f}%")
    print(f"  Test accuracy*           : {test_acc * 100:.2f}%  ({t_cor}/{t_tot})")
    print(f"  Mean epoch time          : {mean_epoch_time:.2f}s")
    print(f"  Mean latency             : {mean_latency_ms:.3f} ms/batch")
    print(f"  * evaluated at best-val checkpoint (epoch {best_epoch})")
    print(f"{'─'*60}")

    return {
        "candidate_id":       candidate_id,
        "config_path":        config_path,
        "params":             n_params,
        "best_val_acc":       best_val_acc,
        "best_epoch":         best_epoch,
        "final_train_acc":    final_train_acc,
        "test_acc":           test_acc,
        "mean_epoch_time":    mean_epoch_time,
        "mean_latency_ms":    mean_latency_ms,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NAS v0 stable benchmark (seed=42, best-checkpoint)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the candidate YAML config file.",
    )
    parser.add_argument(
        "--candidate_id",
        required=True,
        help="Human-readable candidate identifier (e.g. C001).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", f"Expected CUDA, got: {device}. Check GPU runtime."
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

    run_stable_benchmark(args.candidate_id, args.config, device)


if __name__ == "__main__":
    main()
