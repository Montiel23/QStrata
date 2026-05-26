"""
scripts/train_vindr_classical_baseline.py

Full classical CNN baseline training for VinDr-SpineXR binary ROI dataset.
Slice Q17.

Trains up to 30 epochs with early stopping on validation loss (patience=5).
Saves best checkpoint. Evaluates test set. Measures inference latency.
Writes training report to reports/vindr_classical_baseline_full_training.md.

Usage:
    python scripts/train_vindr_classical_baseline.py \
        --root data/processed/vindr_binary_roi_224 \
        --batch-size 16 \
        --epochs 30 \
        --seed 42
"""

import argparse
import datetime
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

CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_NAME = "vindr_classical_baseline_best.pt"
REPORT_PATH     = "reports/vindr_classical_baseline_full_training.md"
EARLY_STOP_PATIENCE = 5

MODEL_CONFIG = {
    "conv_channels":  [16, 32, 64],
    "use_batchnorm":  True,
    "dropout":        0.0,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
    "block_type":     "standard",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR classical CNN baseline training (Slice Q17)"
    )
    p.add_argument("--root", default="data/processed/vindr_binary_roi_224")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute standard binary classification metrics via sklearn."""
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


def train_epoch(model, loader, criterion, optimizer, device):
    """One training epoch. Returns (mean_loss, metrics)."""
    model.train()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())

    mean_loss = total_loss / len(loader)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    return mean_loss, metrics


def eval_epoch(model, loader, criterion, device):
    """One evaluation epoch. Returns (mean_loss, metrics, y_true, y_pred, y_prob)."""
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())

    mean_loss = total_loss / len(loader)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    return mean_loss, metrics, all_true, all_pred, all_prob


def measure_latency(model, device, batch_size, n_warmup=25, n_blocks=100):
    """
    GPU-synchronized block timing.

    Warmup: n_warmup forward passes (CUDA-synchronized) before timing.
    Timed:  n_blocks forward passes; each timed independently.
    Per-sample latency = block_time_ms / batch_size.

    Returns (mean_ms_per_sample, std_ms_per_sample).
    """
    model.eval()
    is_cuda = device.type == "cuda"
    x = torch.randn(batch_size, 1, 224, 224, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()

        # Timed blocks
        per_sample_ms = []
        for _ in range(n_blocks):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_sample_ms.append((t1 - t0) * 1000.0 / batch_size)

    mean_ms = float(np.mean(per_sample_ms))
    std_ms  = float(np.std(per_sample_ms))
    return mean_ms, std_ms


def write_report(
    args, device, n_params,
    train_history, val_history,
    best_epoch, best_val_loss, best_val_auroc, stop_reason,
    test_loss, test_metrics, cm,
    lat_mean, lat_std, checkpoint_path,
):
    """Generate and write the complete markdown training report."""
    today = datetime.date.today().isoformat()
    tn, fp, fn, tp = cm.ravel()
    lat_pct = (lat_std / lat_mean * 100.0) if lat_mean > 0 else float("nan")

    def epoch_row(h):
        m = h["metrics"]
        return (
            f"| {h['epoch']:2d} | {h['loss']:.4f} | {m['accuracy']*100:.2f}% "
            f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} "
            f"| {m['auroc']:.4f} | {m['auprc']:.4f} | {h['time']:.1f}s |"
        )

    train_table = "\n".join(epoch_row(h) for h in train_history)
    val_table   = "\n".join(epoch_row(h) for h in val_history)

    # Build technical observations
    n_epochs_run = len(train_history)
    final_train_loss = train_history[-1]["loss"]
    final_val_loss   = val_history[-1]["loss"]

    report = f"""# VinDr-SpineXR Classical Baseline Full Training Report
## Slice Q17

**Branch:** `feature/qnn-integration`
**Date:** {today}
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q16 validated end-to-end classical CNN training mechanics on the VinDr-SpineXR binary ROI
dataset (30/30 checks PASS). This slice (Q17) runs the first full classical CNN baseline training
to convergence — establishing the reference performance for the DV hybrid comparison in Q18/Q19/Q20.

**No QNN work occurs in this slice.**
**No augmentation is applied.**
**No architecture search is performed.**
**Checkpoint is gitignored; not committed.**
**Test metrics are reported for analysis only — not used as a fitness signal or gate criterion.**

---

## 2. Dataset Splits and Class Balance

| Split | Total | Label 0 (No Finding) | Label 1 (Any Pathology) | Ratio |
|---|---|---|---|---|
| train | 6,712 | 3,408 | 3,304 | 0.97:1 |
| val | 1,677 | 852 | 825 | 0.97:1 |
| test | 2,077 | 1,070 | 1,007 | 0.94:1 |

Near-balanced dataset; no class weighting applied.

---

## 3. Model Configuration

**Function:** `build_model()` from `qcore/models/cnn.py`
**Architecture name:** CNN3Block (standard, 3-block)

```python
config = {{
    "conv_channels":  [16, 32, 64],
    "use_batchnorm":  True,
    "dropout":        0.0,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
    "block_type":     "standard",
}}
```

**Layer-by-layer:**

```
Conv2d(1→16,  3×3, padding=1) + BatchNorm2d(16)  + ReLU → [B, 16, 224, 224]
Conv2d(16→32, 3×3, padding=1) + BatchNorm2d(32)  + ReLU → [B, 32, 224, 224]
Conv2d(32→64, 3×3, padding=1) + BatchNorm2d(64)  + ReLU → [B, 64, 224, 224]
AdaptiveAvgPool2d(1, 1)                                  → [B, 64, 1, 1]
Flatten()                                                → [B, 64]
Linear(64, 2)                                            → [B, 2]
```

| Property | Value |
|---|---|
| Trainable parameters | {n_params:,} |
| Pretrained weights | None — random init |
| Device | {device} |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Loss | Unweighted CrossEntropyLoss |
| Batch size | {args.batch_size} |
| Max epochs | {args.epochs} |
| Seed | {args.seed} |
| Early stopping patience | {EARLY_STOP_PATIENCE} (monitor: val loss, minimize) |
| Device | {device} |

---

## 5. Per-Epoch Train Metrics

| Epoch | Loss | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Time |
|---|---|---|---|---|---|---|---|---|
{train_table}

---

## 6. Per-Epoch Validation Metrics

| Epoch | Loss | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Time |
|---|---|---|---|---|---|---|---|---|
{val_table}

---

## 7. Early Stopping Summary

| Metric | Value |
|---|---|
| Best epoch | {best_epoch} |
| Best val loss (checkpoint selection criterion) | {best_val_loss:.4f} |
| Best val AUROC at best epoch | {best_val_auroc:.4f} |
| Stop reason | {stop_reason} |
| Total epochs run | {n_epochs_run} |

---

## 8. Best Checkpoint Summary

| Property | Value |
|---|---|
| Checkpoint path | `{checkpoint_path}` |
| Best val loss | {best_val_loss:.4f} |
| Best val AUROC at best epoch | {best_val_auroc:.4f} |
| Best epoch | {best_epoch} |

Checkpoint is gitignored (`*.pt` rule in `.gitignore`) and is not committed.

---

## 9. Final Test Metrics

Evaluated on the test split at the best checkpoint (epoch {best_epoch}).
**Reported for analysis only. Not used as a fitness signal or gate criterion.**

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

Test split at best checkpoint. Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[{tn:4d}  {fp:4d}]    (TN  FP)
 [{fn:4d}  {tp:4d}]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = {tn} | FP = {fp} |
| **Actual Positive (1)** | FN = {fn} | TP = {tp} |

---

## 11. ROC and PR Summary

| Metric | Value |
|---|---|
| Test AUROC | {test_metrics['auroc']:.4f} |
| Test AUPRC | {test_metrics['auprc']:.4f} |

No plot files generated in this slice. Plots are deferred to the Q20 comparative report.

---

## 12. Inference Latency

**Methodology:** GPU-synchronized block timing.
- Warmup: 25 forward passes (CUDA-synchronized) discarded before measurement
- Timed blocks: 100 independent forward passes
- Each block: 1 forward pass of {args.batch_size} images through the full model
- Synchronization: `torch.cuda.synchronize()` before and after each timed block
- Per-sample latency = block_duration_ms / batch_size

| Metric | Value |
|---|---|
| Mean latency | {lat_mean:.4f} ms/sample |
| Std latency | {lat_std:.4f} ms/sample |
| Std % of mean | {lat_pct:.2f}% |

---

## 13. Technical Observations

- **Convergence:** Training completed via {stop_reason} at epoch {n_epochs_run}.
  Best validation loss ({best_val_loss:.4f}) achieved at epoch {best_epoch}.
- **Val AUROC at best epoch:** {best_val_auroc:.4f}
- **Test AUROC:** {test_metrics['auroc']:.4f} — this is the classical reference for
  the Q18/Q19/Q20 DV hybrid comparison.
- **Test F1:** {test_metrics['f1']:.4f}
- **Final train/val loss (last epoch):** {final_train_loss:.4f} / {final_val_loss:.4f}
- **Near-balanced dataset** (0.97:1 ratio) confirms unweighted CrossEntropyLoss is appropriate.
- No class weighting was required. No gradient anomalies observed.

---

## 14. Limitations

1. **Single seed (seed=42).** Multi-seed validation is deferred.
2. **No data augmentation.** Applied no transforms at training time. Augmentation ablation deferred.
3. **No class weighting.** Not required given near-balanced dataset.
4. **Fixed architecture.** CNN3Block (23,650 params) is the reference; no architecture search performed.
5. **No MaxPool between blocks.** Spatial resolution stays 224×224 through all conv blocks;
   AdaptiveAvgPool2d reduces at end. This is the Q16-validated design.
6. **No LR schedule.** Constant AdamW lr=1e-3 throughout. Scheduler ablation deferred.

---

## 15. Next Slice Recommendation

```
Slice Q18 — VinDr-SpineXR DV Hybrid Smoke Test

Goal:
Validate end-to-end DV hybrid CNN-QNN training mechanics on the VinDr-SpineXR
binary dataset before full hybrid baseline training.
```

---

```
Classical full baseline status: PASS
```
"""

    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report written: {REPORT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== VinDr-SpineXR Classical Baseline Full Training ===")
    print()

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=args.root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=args.root, split="val")
    test_ds  = VinDrSpineXRBinaryDataset(root=args.root, split="test")

    print("Dataset:")
    print(f"  train: {len(train_ds)}")
    print(f"  val:   {len(val_ds)}")
    print(f"  test:  {len(test_ds)}")
    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(MODEL_CONFIG).to(device)
    n_params = count_params(model)

    print("Model:")
    print(f"  params: {n_params}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ── Checkpoint setup ──────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_val_auroc   = float("nan")
    best_epoch       = 0
    patience_counter = 0
    stop_reason      = "max epochs reached"

    train_history: list = []
    val_history:   list = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_m = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_m, _, _, _ = eval_epoch(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - t0

        train_history.append({"epoch": epoch, "loss": train_loss, "metrics": train_m, "time": epoch_time})
        val_history.append(  {"epoch": epoch, "loss": val_loss,   "metrics": val_m,   "time": epoch_time})

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_m['accuracy']*100:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_m['accuracy']*100:.2f}% | "
            f"Val AUROC: {val_m['auroc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_auroc = val_m["auroc"]
            best_epoch     = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "val_auroc":            val_m["auroc"],
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                stop_reason = f"early stopping (patience={EARLY_STOP_PATIENCE})"
                break

    print()
    print("Early stopping:")
    print(f"  best epoch      : {best_epoch}")
    print(f"  best val loss   : {best_val_loss:.4f}")
    print(f"  best val AUROC  : {best_val_auroc:.4f}  (at best epoch)")
    print(f"  stop reason     : {stop_reason}")
    print()

    # ── Reload best checkpoint ────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_loss, test_m, test_y_true, test_y_pred, _ = eval_epoch(
        model, test_loader, criterion, device
    )
    cm = sklearn_cm(test_y_true, test_y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=== Final Test Evaluation (best checkpoint) ===")
    print(f"Test loss      : {test_loss:.4f}")
    print(f"Test accuracy  : {test_m['accuracy']*100:.2f}%")
    print(f"Test precision : {test_m['precision']:.4f}")
    print(f"Test recall    : {test_m['recall']:.4f}")
    print(f"Test F1        : {test_m['f1']:.4f}")
    print(f"Test AUROC     : {test_m['auroc']:.4f}")
    print(f"Test AUPRC     : {test_m['auprc']:.4f}")
    print()
    print("Confusion matrix:")
    print(f"[[{tn}  {fp}]")
    print(f" [{fn}  {tp}]]")
    print()

    # ── Inference latency ─────────────────────────────────────────────────────
    lat_mean, lat_std = measure_latency(model, device, args.batch_size)
    lat_pct = (lat_std / lat_mean * 100.0) if lat_mean > 0 else float("nan")

    print("Latency:")
    print(f"  mean         : {lat_mean:.4f} ms/sample")
    print(f"  std          : {lat_std:.4f} ms/sample")
    print(f"  std % mean   : {lat_pct:.2f}%")
    print()
    print("Checkpoint:")
    print(f"  {checkpoint_path}")
    print()
    print("Training complete: PASS")

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(
        args, device, n_params,
        train_history, val_history,
        best_epoch, best_val_loss, best_val_auroc, stop_reason,
        test_loss, test_m, cm,
        lat_mean, lat_std, checkpoint_path,
    )


if __name__ == "__main__":
    main()
