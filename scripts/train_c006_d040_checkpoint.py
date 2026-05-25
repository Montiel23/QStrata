#!/usr/bin/env python3
"""
scripts/train_c006_d040_checkpoint.py

Slice Q6 — C006-D040 Classical Anchor Checkpoint Training

Trains C006-D040 (depthwise_sep, conv_channels=[64,128]) on PneumoniaMNIST
using the existing repo dataset interface, model builder, and stable benchmark
protocol v1.

Reuse note:
    evaluate_candidate() in qcore/nas/evaluator.py implements the same
    training loop and in-memory best-val checkpoint tracking, but does not
    write a checkpoint to disk and does not perform hybrid backbone
    compatibility verification. This script adds those two steps using the
    same primitives (build_model, get_dataset, TorchDatasetAdapter) to avoid
    duplicating training-loop logic beyond what is strictly required.

Produces:
    checkpoints/c006_d040_classical_anchor.pt   — best-val model checkpoint
    reports/c006_d040_checkpoint_training.md    — training report

Do not modify any existing source files.
"""

from __future__ import annotations

import copy
import datetime
import os
import random
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qcore.data.registry      import get_dataset
from qcore.data.torch_adapter import TorchDatasetAdapter
from qcore.models.cnn         import build_model
from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN

# ── Hard-coded paths and constants ────────────────────────────────────────────
SEED        = 42
CONFIG_PATH = os.path.join(_REPO_ROOT, "experiments", "configs",
                           "binary_baseline_depthwise_sep_wide.yaml")
CKPT_PATH   = os.path.join(_REPO_ROOT, "checkpoints",
                            "c006_d040_classical_anchor.pt")
REPORT_PATH = os.path.join(_REPO_ROOT, "reports",
                            "c006_d040_checkpoint_training.md")

# DVHybridCNNQNN flat config (identical to runner used in Q4/Q6)
HYBRID_CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}


# ── Seeding — stable benchmark protocol v1 ────────────────────────────────────
def _apply_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
print("=== C006-D040 Classical Anchor Checkpoint Training ===")
print(f"Config  : {CONFIG_PATH}")
print(f"Seed    : {SEED}")
print()

_apply_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {device}"
      + (f"  — {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

epochs      = config["training"]["epochs"]
batch_size  = config["training"]["batch_size"]
lr          = config["training"]["lr"]
use_cw      = config["training"]["class_weights"]
num_classes = config["dataset"]["num_classes"]
dataset_name = config["dataset"]["name"]

print(f"Epochs     : {epochs}")
print(f"Batch size : {batch_size}")
print(f"LR         : {lr}")
print(f"Dataset    : {dataset_name}")
print(f"Num classes: {num_classes}")
print()

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading PneumoniaMNIST splits ...", flush=True)
train_ds = TorchDatasetAdapter(get_dataset(dataset_name, "train"))
val_ds   = TorchDatasetAdapter(get_dataset(dataset_name, "val"))
test_ds  = TorchDatasetAdapter(get_dataset(dataset_name, "test"))
print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          generator=g, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

# ── Class weights (balanced — same formula as evaluate_candidate) ─────────────
if use_cw:
    labels_list = [train_ds[i][1].item() for i in range(len(train_ds))]
    labels_t    = torch.tensor(labels_list, dtype=torch.long)
    counts      = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels_t == c).sum().float()
    cw = (counts.sum() / (num_classes * counts)).to(device)
    print(f"Class weights: {cw.cpu().tolist()}")
else:
    cw = None

# ── Build model ───────────────────────────────────────────────────────────────
model_cfg = {**config["model"], **config["dataset"]}
model     = build_model(model_cfg).to(device)
n_params  = sum(p.numel() for p in model.parameters())
print(f"\nModel      : C006-D040 (depthwise_sep, [64,128])")
print(f"Parameters : {n_params:,}")

# ── Optimizer and loss ────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn   = nn.CrossEntropyLoss(weight=cw)

# ── Training loop with best-val checkpoint tracking ───────────────────────────
print(f"\nTraining ({epochs} epochs):\n")

epoch_records: list[dict] = []
best_val_acc  = -1.0
best_epoch    = -1
best_weights  = None    # in-memory; saved to disk after loop

for epoch in range(1, epochs + 1):
    t0 = time.time()

    # ── Train ─────────────────────────────────────────────────────────────────
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
    train_acc  = n_cor   / n_tot * 100.0

    # ── Validate ──────────────────────────────────────────────────────────────
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
    val_acc  = v_cor  / v_tot * 100.0
    epoch_time = time.time() - t0

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_weights = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch:2d}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.2f}%"
        + ("  ★" if is_best else ""),
        flush=True,
    )

    epoch_records.append({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "epoch_time": epoch_time,
    })

# ── Restore and save best-val checkpoint ──────────────────────────────────────
print(f"\nRestoring best weights (epoch {best_epoch}, val acc {best_val_acc:.2f}%) ...",
      flush=True)
model.load_state_dict(best_weights)
model.eval()

os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epoch":            best_epoch,
        "best_val_acc":     best_val_acc,
        "config":           config,
    },
    CKPT_PATH,
)
print(f"Checkpoint saved: {CKPT_PATH}")

# ── Checkpoint reload verification ───────────────────────────────────────────
ckpt_reload_pass = False
reload_val_acc   = None
try:
    ckpt   = torch.load(CKPT_PATH, map_location=device)
    model2 = build_model(model_cfg).to(device)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()

    # Verify reload val acc matches saved val acc
    v_cor2, v_tot2 = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb   = xb.to(device), yb.to(device)
            logits2   = model2(xb)
            v_cor2   += (logits2.argmax(1) == yb).sum().item()
            v_tot2   += xb.size(0)
    reload_val_acc   = v_cor2 / v_tot2 * 100.0
    val_acc_match    = abs(reload_val_acc - best_val_acc) < 0.01
    ckpt_reload_pass = True
    print(f"Checkpoint reload : PASS (reloaded val acc: {reload_val_acc:.2f}%)")
except Exception as e:
    print(f"Checkpoint reload : FAIL ({e})")
    reload_val_acc = None
    val_acc_match  = False

# ── Test evaluation at best-val checkpoint (analysis only) ────────────────────
print("Evaluating on test split (analysis only) ...", flush=True)
t_cor, t_tot = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb    = xb.to(device)
        logits = model(xb)
        t_cor += (logits.argmax(1).cpu() == yb).sum().item()
        t_tot += xb.size(0)
test_acc = t_cor / t_tot * 100.0
print(f"Test acc (analysis only) : {test_acc:.2f}%")

# ── Hybrid backbone compatibility verification (state dict only) ──────────────
backbone_load_pass = False
backbone_load_note = ""
try:
    hybrid = DVHybridCNNQNN(
        cnn_config=HYBRID_CNN_CONFIG,
        n_qubits=4, depth=1, alpha=0.1, n_classes=2,
    )

    # Extract backbone keys: all keys from the classical model that exist in
    # hybrid.backbone.state_dict(). Layers 2 (AdaptiveAvgPool2d) and 3
    # (Flatten) have no learnable parameters, so backbone keys are only 0.* and 1.*
    full_state     = model.state_dict()
    backbone_keys  = set(hybrid.backbone.state_dict().keys())
    backbone_state = {k: v for k, v in full_state.items() if k in backbone_keys}

    missing = backbone_keys - set(backbone_state.keys())
    extra   = set(backbone_state.keys()) - backbone_keys

    if missing or extra:
        raise ValueError(
            f"Key mismatch — missing: {missing}, extra: {extra}"
        )

    hybrid.backbone.load_state_dict(backbone_state, strict=True)
    backbone_load_pass = True
    backbone_load_note = (
        f"{len(backbone_keys)} keys loaded, "
        f"{sum(p.numel() for p in hybrid.backbone.parameters())} params"
    )
    print(f"Hybrid backbone load : PASS ({backbone_load_note})")
except Exception as e:
    backbone_load_note = str(e)
    print(f"Hybrid backbone load : FAIL ({e})")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=== C006-D040 Classical Checkpoint Training Complete ===")
print(f"Best val acc    : {best_val_acc:.2f}%  (epoch {best_epoch})")
print(f"Test acc        : {test_acc:.2f}%  [analysis only]")
print(f"Checkpoint path : {CKPT_PATH}")
print(f"Checkpoint reload        : {'PASS' if ckpt_reload_pass else 'FAIL'}")
print(f"Hybrid backbone load     : {'PASS' if backbone_load_pass else 'FAIL'}")

# ── Markdown report ───────────────────────────────────────────────────────────
run_date = datetime.date.today().isoformat()

epoch_table_rows = "\n".join(
    f"| {r['epoch']:2d} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | "
    f"{r['train_acc']:.2f}% | {r['val_acc']:.2f}% | {r['epoch_time']:.1f}s |"
    for r in epoch_records
)

reload_match_str  = "PASS" if (ckpt_reload_pass and val_acc_match) else "FAIL"
reload_match_note = (f"reloaded={reload_val_acc:.2f}%, saved={best_val_acc:.2f}%"
                     if reload_val_acc is not None else "N/A")

report = f"""# C006-D040 Classical Anchor Checkpoint Training

- **Status:** Complete
- **Date:** {run_date}
- **Branch:** feature/qnn-integration
- **Slice:** Q6

---

## 1. Title

C006-D040 Classical Anchor Checkpoint Training — Slice Q6

Training the C006-D040 depthwise-separable CNN architecture on PneumoniaMNIST
to produce a pretrained backbone for `DVHybridCNNQNN`.

---

## 2. Training Configuration

| Parameter | Value |
|---|---|
| Config path | `{CONFIG_PATH}` |
| Dataset | PneumoniaMNIST binary (`{dataset_name}`) |
| Split sizes | Train: {len(train_ds):,} / Val: {len(val_ds):,} / Test: {len(test_ds):,} |
| Seed | {SEED} |
| Epochs | {epochs} |
| Batch size | {batch_size} |
| Optimizer | Adam |
| Learning rate | {lr} |
| Loss function | CrossEntropyLoss (balanced class weights) |
| Class weights | {cw.cpu().tolist() if cw is not None else 'None'} |
| Device | {device} |
| Architecture | depthwise_sep, conv_channels=[64, 128], params={n_params:,} |

---

## 3. Per-Epoch Results

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Time |
|---|---|---|---|---|---|
{epoch_table_rows}

---

## 4. Best Epoch Summary

| Item | Value |
|---|---|
| Best epoch | {best_epoch} |
| Best val accuracy | {best_val_acc:.2f}% |
| Checkpoint path | `checkpoints/c006_d040_classical_anchor.pt` |

Checkpoint contents: `model_state_dict`, `epoch`, `best_val_acc`, `config`.

---

## 5. Test Accuracy

**Test accuracy: {test_acc:.2f}%** — analysis only; not used as fitness signal or gate criterion.

Evaluated at the best-val checkpoint (epoch {best_epoch}).

---

## 6. Checkpoint Verification

| Check | Result | Notes |
|---|---|---|
| Checkpoint file exists | {'PASS' if os.path.exists(CKPT_PATH) else 'FAIL'} | `{CKPT_PATH}` |
| Checkpoint reloads without error | {'PASS' if ckpt_reload_pass else 'FAIL'} | `build_model()` + `load_state_dict()` |
| Reloaded model val acc matches saved val acc | {reload_match_str} | {reload_match_note} |
| Backbone state dict loads into `DVHybridCNNQNN` | {'PASS' if backbone_load_pass else 'FAIL'} | {backbone_load_note} |

---

## 7. Notes

- **Architecture:** C006-D040 uses depthwise-separable convolution blocks (`build_block("depthwise_sep", ...)`). The model[:4] backbone slice (blocks 0–1 + AdaptiveAvgPool2d + Flatten) maps to `DVHybridCNNQNN.backbone`.
- **Backbone compatibility:** {len(set(hybrid.backbone.state_dict().keys()) if backbone_load_pass else set()) if backbone_load_pass else 0} backbone keys were transferred. Layers 2 (AdaptiveAvgPool2d) and 3 (Flatten) carry no learnable parameters — only layers 0 and 1 (the depthwise-sep blocks) contribute to the backbone state dict.
- **Val acc reference:** The classical anchor `v1_classical_anchor` achieved best_val_acc = 91.79% in the Slice 29–30 stability validation (4 seeds, epochs varied). The result here uses seed=42 and epochs={epochs}.
- **Next step:** `DVHybridCNNQNN.__init__` should be updated in Q7 to load `checkpoints/c006_d040_classical_anchor.pt` during backbone construction, replacing the current random initialisation.
"""

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write(report)

print(f"\nReport written to: {REPORT_PATH}")
print("=== DONE ===")
