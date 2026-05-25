"""
smoke_test_binary_torch_pipeline.py — Binary torch pipeline smoke test.

Validates: registry → TorchDatasetAdapter → DataLoader → tiny CNN forward pass.
No training. No optimizer. No loss. No checkpoint.
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter


def main():
    # ── Step 1: Load pneumoniamnist train split ───────────────────────────────
    print("Loading pneumoniamnist train split ...")
    ds = get_dataset("pneumoniamnist", "train")

    # ── Step 2: Wrap + DataLoader ─────────────────────────────────────────────
    adapted = TorchDatasetAdapter(ds)
    loader  = DataLoader(adapted, batch_size=16, shuffle=True)

    # ── Step 3: Fetch one batch ───────────────────────────────────────────────
    x, y = next(iter(loader))
    print(f"x shape : {x.shape}")
    print(f"x dtype : {x.dtype}")
    print(f"y shape : {y.shape}")
    print(f"y dtype : {y.dtype}")
    print(f"x.min() : {x.min().item():.6f}")
    print(f"x.max() : {x.max().item():.6f}")

    # ── Step 4: Assertions ────────────────────────────────────────────────────
    assert x.shape == (16, 1, 28, 28), \
        f"[FAIL] x.shape expected (16,1,28,28), got {x.shape}"
    assert x.dtype == torch.float32, \
        f"[FAIL] x.dtype expected torch.float32, got {x.dtype}"
    assert y.shape == (16,), \
        f"[FAIL] y.shape expected (16,), got {y.shape}"
    assert y.dtype == torch.long, \
        f"[FAIL] y.dtype expected torch.long, got {y.dtype}"
    assert x.min() >= 0.0 and x.max() <= 1.0, \
        f"[FAIL] x values outside [0,1]: min={x.min():.6f} max={x.max():.6f}"

    # ── Step 5: Tiny inline CNN ───────────────────────────────────────────────
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    model.eval()

    # ── Step 6: Forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x)

    # ── Step 7: Assertions on logits ─────────────────────────────────────────
    assert logits.shape == (16, 2), \
        f"[FAIL] logits.shape expected (16,2), got {logits.shape}"
    assert not torch.isnan(logits).any(), \
        "[FAIL] NaN values found in logits"

    # ── Step 8 ────────────────────────────────────────────────────────────────
    print()
    print("=== BINARY TORCH PIPELINE SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
