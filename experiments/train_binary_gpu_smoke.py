"""
train_binary_gpu_smoke.py — Binary PneumoniaMNIST GPU training script.

Validates: CUDA available → load data → tiny CNN → 5-epoch train loop →
test evaluation → inference latency → GO/NO-GO summary.
No plots. No checkpoint saving. No external logging.
"""

import sys
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.registry import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter


def mean(values):
    return sum(values) / len(values)


def main():
    # ── Section 1: Device check ───────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    assert DEVICE == "cuda", f"Expected CUDA, got: {DEVICE}. Check GPU runtime."
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Section 2: Config ─────────────────────────────────────────────────────
    EPOCHS     = 5
    BATCH_SIZE = 64
    LR         = 1e-3

    # ── Section 3: Load datasets ──────────────────────────────────────────────
    print("Loading datasets ...")
    train_ds = get_dataset("pneumoniamnist", "train")
    val_ds   = get_dataset("pneumoniamnist", "val")
    test_ds  = get_dataset("pneumoniamnist", "test")

    train_loader = DataLoader(TorchDatasetAdapter(train_ds), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TorchDatasetAdapter(val_ds),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TorchDatasetAdapter(test_ds),  batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    # ── Section 4: Define tiny CNN ────────────────────────────────────────────
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2),
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")

    # ── Section 5: Train loop ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()
    epoch_times = []

    final_train_acc = 0.0
    final_val_acc   = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train pass
        model.train()
        running_loss = 0.0
        n_correct    = 0
        n_total      = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds         = logits.argmax(dim=1)
            n_correct    += (preds == yb).sum().item()
            n_total      += xb.size(0)

        train_loss = running_loss / n_total
        train_acc  = n_correct / n_total

        # Val pass
        model.eval()
        val_running_loss = 0.0
        val_n_correct    = 0
        val_n_total      = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss   = loss_fn(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
                preds             = logits.argmax(dim=1)
                val_n_correct    += (preds == yb).sum().item()
                val_n_total      += xb.size(0)

        val_loss = val_running_loss / val_n_total
        val_acc  = val_n_correct / val_n_total

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"{epoch_time:.2f}s"
        )

        final_train_acc = train_acc
        final_val_acc   = val_acc

    # ── Section 6: Test evaluation + inference latency ────────────────────────
    model.eval()
    batch_latencies = []
    all_preds       = []
    all_labels      = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            t0 = time.time()
            logits = model(xb)
            batch_latencies.append((time.time() - t0) * 1000)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.tolist())

    test_correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    test_accuracy = test_correct / len(all_labels)
    mean_latency_ms = mean(batch_latencies)

    print(f"Test accuracy          : {test_accuracy * 100:.2f}%  ({test_correct}/{len(all_labels)})")
    print(f"Mean inference latency : {mean_latency_ms:.3f} ms per batch")

    # ── Section 7: Summary block ──────────────────────────────────────────────
    verdict = "GO" if test_accuracy >= 0.70 else "NO-GO"
    mean_epoch_time = mean(epoch_times)

    print()
    print("=== GPU TRAINING SUMMARY ===")
    print(f"Device             : {DEVICE}")
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Mean epoch time    : {mean_epoch_time:.2f}s")
    print(f"Final train accuracy: {final_train_acc * 100:.2f}%")
    print(f"Final val accuracy  : {final_val_acc * 100:.2f}%")
    print(f"Test accuracy       : {test_accuracy * 100:.2f}%")
    print(f"Mean inference latency: {mean_latency_ms:.2f}ms per batch")
    print(f"{verdict}")


if __name__ == "__main__":
    main()
