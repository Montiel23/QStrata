"""
smoke_test_gpu_torch.py — GPU torch smoke test.

Validates: CUDA available → tensor on GPU → tiny CNN forward pass on GPU.
No training. No optimizer. No loss. No qcore imports.
"""

import torch
import torch.nn as nn


def main():
    # ── Step 1: torch version ─────────────────────────────────────────────────
    print(f"torch version : {torch.__version__}")

    # ── Step 2: CUDA availability ─────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA not available — check runtime config"
    print("torch.cuda.is_available() : True")

    # ── Step 3: GPU device name ───────────────────────────────────────────────
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU device name : {device_name}")

    # ── Step 4: Allocate tensor on GPU ────────────────────────────────────────
    x = torch.randn(16, 1, 28, 28).cuda()
    assert x.device.type == "cuda", "Tensor not on GPU"
    print(f"x.shape  : {x.shape}")
    print(f"x.dtype  : {x.dtype}")
    print(f"x.device : {x.device}")

    # ── Step 5: Define tiny CNN on GPU ────────────────────────────────────────
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2),
    ).cuda()

    # ── Step 6: Forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x)

    # ── Step 7: Assertions on logits ─────────────────────────────────────────
    assert logits.shape == (16, 2), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaNs detected in logits"
    print(f"logits.shape  : {logits.shape}")
    print(f"logits.dtype  : {logits.dtype}")
    print(f"logits.device : {logits.device}")

    # ── Step 8 ────────────────────────────────────────────────────────────────
    print()
    print("=== GPU TORCH SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
