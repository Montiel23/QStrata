#!/usr/bin/env python3
"""
scripts/run_dv_hybrid_multiseed_stability.py

Slice Q9 — DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation

Runs DVHybridCNNQNN on PneumoniaMNIST across four seeds (42, 7, 123, 999)
with 10 epochs per seed. Collects per-seed and aggregate metrics, applies
the hard-coded stability gate, and emits exactly one of the two approved
verdict strings:
  "Stable enough for VinDr-SpineXR binary planning"
  "Not stable enough; stop and investigate"

Architecture: unchanged from Q8 (same DVHybridCNNQNN, same CNN_CONFIG).
Backbone: frozen pretrained C006-D040 loaded from checkpoints/c006_d040_classical_anchor.pt.

If a seed's backbone load or freeze verification fails, that seed is recorded
as FAILED and the runner continues with remaining seeds. Training exceptions
are similarly recorded per-seed.

Does NOT modify any existing source file.
Does NOT overwrite any existing report.
"""

from __future__ import annotations

import copy
import datetime
import os
import random
import sys
import time
import traceback

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Hard-coded configuration ──────────────────────────────────────────────────
SEEDS      = [42, 7, 123, 999]
EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-3
N_QUBITS   = 4
DEPTH      = 1
ALPHA      = 0.1
N_CLASSES  = 2
DEVICE     = "cpu"   # Quantum simulator is CPU-only

CKPT_PATH = os.path.join(_REPO_ROOT, "checkpoints", "c006_d040_classical_anchor.pt")

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

REPORT_PATH = os.path.join(
    _REPO_ROOT, "reports", "dv_hybrid_multiseed_stability.md"
)

# Reference values from Q8
Q8_BEST_VAL_ACC = 92.18
Q8_TEST_ACC     = 87.98

# Guard: must not collide with any existing report
_EXISTING_REPORTS = [
    os.path.join(_REPO_ROOT, "reports", "dv_hybrid_pneumoniamnist_baseline.md"),
    os.path.join(_REPO_ROOT, "reports", "dv_hybrid_pneumoniamnist_pretrained_baseline.md"),
    os.path.join(_REPO_ROOT, "reports", "dv_hybrid_pneumoniamnist_full_baseline.md"),
]
for _er in _EXISTING_REPORTS:
    assert os.path.realpath(REPORT_PATH) != os.path.realpath(_er), \
        f"FATAL: Q9 report path collides with existing report: {_er}"


# ── Seeding — stable benchmark protocol v1 ────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Gradient norm helper ──────────────────────────────────────────────────────
def param_grad_norm(params: list) -> float:
    """L2 norm of gradients. Returns 0.0 if all gradients are None."""
    total_sq = sum(
        p.grad.detach().norm().item() ** 2
        for p in params if p.grad is not None
    )
    return float(total_sq ** 0.5)


# ── Imports ───────────────────────────────────────────────────────────────────
from qcore.data.registry            import get_dataset
from qcore.data.torch_adapter       import TorchDatasetAdapter
from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

# ── Data loading (once — datasets shared across seeds) ────────────────────────
print("=== Slice Q9 — DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation ===")
print(f"Seeds      : {SEEDS}")
print(f"Epochs/seed: {EPOCHS}")
print(f"Checkpoint : {CKPT_PATH}")
print(f"Report     : {REPORT_PATH}")
print()

print("Loading PneumoniaMNIST splits ...", flush=True)
train_ds = TorchDatasetAdapter(get_dataset("pneumoniamnist", "train"))
val_ds   = TorchDatasetAdapter(get_dataset("pneumoniamnist", "val"))
test_ds  = TorchDatasetAdapter(get_dataset("pneumoniamnist", "test"))
print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")

# ── Class weights ─────────────────────────────────────────────────────────────
train_labels = torch.tensor(
    [int(train_ds[i][1]) for i in range(len(train_ds))], dtype=torch.long
)
counts        = torch.bincount(train_labels, minlength=N_CLASSES).float()
weights       = 1.0 / (counts + 1e-6)
weights       = weights / weights.sum()
class_weights = weights.to(DEVICE)
print(f"  Class counts  : {counts.long().tolist()}")
print(f"  Class weights : [{class_weights[0].item():.6f}, {class_weights[1].item():.6f}]")
print()


# ─────────────────────────────────────────────────────────────────────────────
def run_seed(seed: int) -> dict:
    """
    Run one seed: set seed, instantiate model, load backbone, verify,
    train EPOCHS epochs, eval on test.
    Returns a result dict with training_status = COMPLETED or FAILED.
    Failures are recorded and returned; the caller continues to the next seed.
    """
    result: dict = {
        "seed":                              seed,
        "best_val_acc":                      None,
        "best_epoch":                        None,
        "test_acc_analysis_only":            None,
        "train_acc_at_best_epoch":           None,
        "val_loss_at_best_epoch":            None,
        "test_precision":                    None,
        "test_recall":                       None,
        "test_f1":                           None,
        "test_auroc":                        None,
        "test_auprc":                        None,
        "confusion_matrix":                  None,
        "theta_grad_norm_epoch1":            None,
        "projection_grad_norm_epoch1":       None,
        "readout_grad_norm_epoch1":          None,
        "projection_grad_active_all_epochs": False,
        "probability_sum_valid_all_epochs":  False,
        "majority_class_collapse":           True,
        "mean_epoch_time":                   None,
        "training_status":                   "FAILED",
        "failure_detail":                    None,
        "epoch_records":                     [],
    }

    print(f"─── Seed {seed} ────────────────────────────────────────────────────────", flush=True)

    # Apply seed
    set_seed(seed)

    # Instantiate model
    try:
        model = DVHybridCNNQNN(
            cnn_config=CNN_CONFIG,
            n_qubits=N_QUBITS,
            depth=DEPTH,
            alpha=ALPHA,
            n_classes=N_CLASSES,
        )
        model = model.to(DEVICE)
    except Exception as exc:
        result["failure_detail"] = (
            f"Model instantiation failed: {exc}\n{traceback.format_exc()}"
        )
        print(f"  Model instantiation FAILED: {exc}", flush=True)
        return result

    # Load pretrained backbone
    try:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        full_classical_state = ckpt["model_state_dict"]

        backbone_expected_keys = set(model.backbone.state_dict().keys())
        backbone_state = {
            k: v for k, v in full_classical_state.items()
            if k in backbone_expected_keys
        }

        missing = backbone_expected_keys - set(backbone_state.keys())
        extra   = set(backbone_state.keys()) - backbone_expected_keys
        if missing or extra:
            raise ValueError(
                f"State dict key mismatch — missing: {missing}, extra: {extra}"
            )

        model.backbone.load_state_dict(backbone_state, strict=True)
        n_bb_keys   = len(backbone_expected_keys)
        n_bb_params = sum(p.numel() for p in model.backbone.parameters())
        print(
            f"  Backbone load : PASS  "
            f"[{n_bb_keys} keys, {n_bb_params:,} params — "
            f"checkpoint epoch {ckpt['epoch']}, val acc {ckpt['best_val_acc']:.2f}%]",
            flush=True,
        )
    except Exception as exc:
        result["failure_detail"] = (
            f"Backbone load failed (seed {seed}): {exc}\n{traceback.format_exc()}"
        )
        print(f"  Backbone load : FAIL  [{exc}]", flush=True)
        return result

    # Verify backbone frozen
    trainable_names = [
        name for name, p in model.backbone.named_parameters() if p.requires_grad
    ]
    if trainable_names:
        result["failure_detail"] = (
            f"Backbone not fully frozen (seed {seed}): {trainable_names}"
        )
        print(f"  Backbone frozen : FAIL  — trainable: {trainable_names}", flush=True)
        return result

    n_bb_param_count = sum(1 for _ in model.backbone.parameters())
    print(
        f"  Backbone frozen : PASS  "
        f"[all {n_bb_param_count} backbone params have requires_grad=False]",
        flush=True,
    )
    model.backbone.eval()

    # DataLoaders — created after set_seed so shuffle is reproducible per seed
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    epoch_records:    list[dict] = []
    best_val_acc      = -1.0
    best_epoch_num    = 1
    best_model_state  = None

    ep1_theta_gn = ep1_proj_gn = ep1_readout_gn = None

    # Training loop
    current_epoch = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            current_epoch = epoch
            t0 = time.time()
            model.train()

            train_loss_acc  = 0.0
            train_correct   = 0
            train_total     = 0
            theta_gnorms:   list[float] = []
            proj_gnorms:    list[float] = []
            readout_gnorms: list[float] = []
            all_prob_sums:  list[float] = []

            for bx, by in train_loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                optimizer.zero_grad()
                logits = model(bx)
                loss   = criterion(logits, by)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss_acc += loss.item() * bx.size(0)
                    preds           = logits.argmax(dim=1)
                    train_correct  += (preds == by).sum().item()
                    train_total    += bx.size(0)

                theta_gnorms.append(    param_grad_norm([model.theta]))
                proj_gnorms.append(     param_grad_norm(list(model.proj.parameters())))
                readout_gnorms.append(  param_grad_norm(list(model.readout.parameters())))
                all_prob_sums.extend(model._prob_sums)

            # Validation
            model.eval()
            val_loss_acc = 0.0
            val_correct  = 0
            val_total    = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(DEVICE)
                    by = by.to(DEVICE)
                    logits = model(bx)
                    val_loss_acc += criterion(logits, by).item() * bx.size(0)
                    val_correct  += (logits.argmax(1) == by).sum().item()
                    val_total    += bx.size(0)

            train_loss = train_loss_acc / train_total
            val_loss   = val_loss_acc   / val_total
            train_acc  = train_correct  / train_total * 100.0
            val_acc    = val_correct    / val_total   * 100.0
            epoch_time = time.time() - t0

            theta_gn   = float(np.mean(theta_gnorms))   if theta_gnorms   else 0.0
            proj_gn    = float(np.mean(proj_gnorms))     if proj_gnorms    else 0.0
            readout_gn = float(np.mean(readout_gnorms))  if readout_gnorms else 0.0
            prob_mean  = float(np.mean(all_prob_sums))   if all_prob_sums  else 0.0
            prob_std   = float(np.std(all_prob_sums))    if all_prob_sums  else 0.0

            if epoch == 1:
                ep1_theta_gn   = theta_gn
                ep1_proj_gn    = proj_gn
                ep1_readout_gn = readout_gn

            print(
                f"Epoch {epoch:2d}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}% | "
                f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2f}% | "
                f"θ grad: {theta_gn:.2e} | Proj grad: {proj_gn:.2e} | "
                f"Epoch: {epoch_time:.1f}s",
                flush=True,
            )

            epoch_records.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "theta_gn":   theta_gn,
                "proj_gn":    proj_gn,
                "readout_gn": readout_gn,
                "prob_mean":  prob_mean,
                "prob_std":   prob_std,
                "epoch_time": epoch_time,
            })

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_epoch_num   = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                print(
                    f"  ★ New best val acc: {best_val_acc:.2f}% (epoch {best_epoch_num})",
                    flush=True,
                )

    except Exception as exc:
        last = epoch_records[-1] if epoch_records else {}
        result["failure_detail"] = (
            f"Training crashed at epoch {current_epoch} (seed {seed}): {exc}\n"
            f"Last train_acc={last.get('train_acc', 'N/A'):.4f}  "  # type: ignore[arg-type]
            f"val_acc={last.get('val_acc', 'N/A'):.4f}  "
            f"theta_gn={last.get('theta_gn', 'N/A'):.2e}  "
            f"proj_gn={last.get('proj_gn', 'N/A'):.2e}\n"
            f"{traceback.format_exc()}"
        ) if last else (
            f"Training crashed at epoch {current_epoch} (seed {seed}): {exc}\n"
            f"{traceback.format_exc()}"
        )
        result["epoch_records"]  = epoch_records
        result["training_status"] = "FAILED"
        print(f"  Training FAILED at epoch {current_epoch}: {exc}", flush=True)
        return result

    # Reload best checkpoint before test evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    print(
        f"Best checkpoint reloaded before test eval : YES  "
        f"[epoch {best_epoch_num}, val acc {best_val_acc:.2f}%]",
        flush=True,
    )

    # Test evaluation (analysis only — not a fitness gate)
    all_preds:  list[int]   = []
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx     = bx.to(DEVICE)
            logits = model(bx)
            soft   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(by.numpy().tolist())
            all_probs.extend(soft.tolist())

    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np  = np.array(all_probs)

    test_acc  = float((all_preds_np == all_labels_np).mean() * 100.0)
    precision = float(precision_score(all_labels_np, all_preds_np, zero_division=0))
    recall    = float(recall_score(   all_labels_np, all_preds_np, zero_division=0))
    f1        = float(f1_score(       all_labels_np, all_preds_np, zero_division=0))
    cm        = confusion_matrix(all_labels_np, all_preds_np)
    auroc     = float(roc_auc_score(all_labels_np, all_probs_np))
    auprc     = float(average_precision_score(all_labels_np, all_probs_np))

    tn_v, fp_v = int(cm[0, 0]), int(cm[0, 1])
    fn_v, tp_v = int(cm[1, 0]), int(cm[1, 1])

    # Derived stability checks
    proj_grad_active_all = all(r["proj_gn"] > 0 for r in epoch_records)
    prob_sums_valid      = all(abs(r["prob_mean"] - 1.0) < 1e-4 for r in epoch_records)
    majority_collapse    = not (tn_v > 0 and tp_v > 0)

    mean_epoch_time = float(np.mean([r["epoch_time"] for r in epoch_records]))

    # Populate result
    result.update({
        "best_val_acc":                      best_val_acc,
        "best_epoch":                        best_epoch_num,
        "test_acc_analysis_only":            test_acc,
        "train_acc_at_best_epoch":           epoch_records[best_epoch_num - 1]["train_acc"],
        "val_loss_at_best_epoch":            epoch_records[best_epoch_num - 1]["val_loss"],
        "test_precision":                    precision,
        "test_recall":                       recall,
        "test_f1":                           f1,
        "test_auroc":                        auroc,
        "test_auprc":                        auprc,
        "confusion_matrix":                  {"tn": tn_v, "fp": fp_v, "fn": fn_v, "tp": tp_v},
        "theta_grad_norm_epoch1":            ep1_theta_gn,
        "projection_grad_norm_epoch1":       ep1_proj_gn,
        "readout_grad_norm_epoch1":          ep1_readout_gn,
        "projection_grad_active_all_epochs": proj_grad_active_all,
        "probability_sum_valid_all_epochs":  prob_sums_valid,
        "majority_class_collapse":           majority_collapse,
        "mean_epoch_time":                   mean_epoch_time,
        "training_status":                   "COMPLETED",
        "failure_detail":                    None,
        "epoch_records":                     epoch_records,
    })

    # Per-seed stdout summary (exact format from spec)
    proj_ep1_active = ep1_proj_gn is not None and ep1_proj_gn > 0
    proj_ep1_label  = "ACTIVE" if proj_ep1_active else "ZERO"
    proj_ep1_str    = f"{ep1_proj_gn:.2e}  [{proj_ep1_label}]" if ep1_proj_gn is not None else f"N/A  [ZERO]"

    print()
    print(f"=== Seed {seed} — Complete ===")
    print(f"Best val acc    : {best_val_acc:.2f}%  (epoch {best_epoch_num})")
    print(f"Test acc        : {test_acc:.2f}%  [analysis only]")
    print(f"F1              : {f1:.4f}")
    print(f"AUROC           : {auroc:.4f}")
    print(f"Proj grad ep1   : {proj_ep1_str}")
    print(f"Proj grad all   : {'ACTIVE ALL EPOCHS' if proj_grad_active_all else 'DEGRADED'}")
    print(f"Prob sums valid : {'YES' if prob_sums_valid else 'NO'}")
    print(f"Collapse absent : {'YES' if not majority_collapse else 'NO'}")
    print(f"Training status : COMPLETED")
    print(f"Mean epoch time : {mean_epoch_time:.1f}s")
    print()

    return result


# ── Run all four seeds ────────────────────────────────────────────────────────
all_results: list[dict] = []
for _seed in SEEDS:
    _r = run_seed(_seed)
    all_results.append(_r)

# ── Aggregate statistics ──────────────────────────────────────────────────────
completed = [r for r in all_results if r["training_status"] == "COMPLETED"]
n_completed = len(completed)
n_seeds     = len(SEEDS)


def _agg(values: list[float]) -> tuple[float, float, float, float]:
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a))


if completed:
    mean_val,   std_val,   min_val,   max_val   = _agg([r["best_val_acc"]          for r in completed])
    mean_test,  std_test,  _,         _          = _agg([r["test_acc_analysis_only"] for r in completed])
    mean_f1,    std_f1,    _,         _          = _agg([r["test_f1"]               for r in completed])
    mean_auroc, std_auroc, _,         _          = _agg([r["test_auroc"]            for r in completed])
    mean_ep_time = float(np.mean([r["mean_epoch_time"] for r in completed]))
else:
    mean_val = std_val = min_val = max_val = 0.0
    mean_test = std_test = mean_f1 = std_f1 = mean_auroc = std_auroc = mean_ep_time = 0.0

# ── Hard-coded stability gate ─────────────────────────────────────────────────
# Gate 1: std(best_val_acc) <= 1.0%
gate1_val  = std_val
gate1_pass = (gate1_val <= 1.0) and (n_completed == n_seeds)

# Gate 2: No seed >2.5% below mean val acc
if completed:
    gaps    = [mean_val - r["best_val_acc"] for r in completed]
    max_gap = float(max(gaps))
else:
    max_gap = float("inf")
gate2_pass = (max_gap <= 2.5) and (n_completed == n_seeds)

# Gate 3: std(test_acc) <= 2.0%
gate3_val  = std_test
gate3_pass = (gate3_val <= 2.0) and (n_completed == n_seeds)

# Gate 4: All runs completed without failure
gate4_pass = n_completed == n_seeds

# Gate 5: proj grad norm > 0 at epoch 1 for all seeds
n_proj_ep1_active = sum(
    1 for r in completed
    if r["projection_grad_norm_epoch1"] is not None and r["projection_grad_norm_epoch1"] > 0
)
gate5_pass = (n_proj_ep1_active == n_seeds) and (n_completed == n_seeds)

# Gate 6: proj grads active all epochs for all seeds
n_proj_all_active = sum(1 for r in completed if r["projection_grad_active_all_epochs"])
gate6_pass = (n_proj_all_active == n_seeds) and (n_completed == n_seeds)

# Gate 7: prob sums valid for all seeds
n_prob_valid = sum(1 for r in completed if r["probability_sum_valid_all_epochs"])
gate7_pass   = (n_prob_valid == n_seeds) and (n_completed == n_seeds)

# Gate 8: majority-class collapse absent for all seeds
n_collapse_absent = sum(1 for r in completed if not r["majority_class_collapse"])
gate8_pass        = (n_collapse_absent == n_seeds) and (n_completed == n_seeds)

overall_pass = all([
    gate1_pass, gate2_pass, gate3_pass, gate4_pass,
    gate5_pass, gate6_pass, gate7_pass, gate8_pass,
])

verdict_str = (
    "Stable enough for VinDr-SpineXR binary planning"
    if overall_pass else
    "Not stable enough; stop and investigate"
)


# ── Aggregate and gate stdout ─────────────────────────────────────────────────
def pf(b: bool) -> str:
    return "PASS" if b else "FAIL"


print("=== Q9 Aggregate Results ===")
print(f"Seeds           : {', '.join(str(s) for s in SEEDS)}")
print()
print(f"Mean best_val_acc : {mean_val:.2f}%  (std: {std_val:.2f}%)")
print(f"Min  best_val_acc : {min_val:.2f}%")
print(f"Max  best_val_acc : {max_val:.2f}%")
print(f"Mean test_acc     : {mean_test:.2f}%  (std: {std_test:.2f}%)  [analysis only]")
print(f"Mean F1           : {mean_f1:.4f}  (std: {std_f1:.4f})")
print(f"Mean AUROC        : {mean_auroc:.4f}  (std: {std_auroc:.4f})")
print(f"Mean epoch time   : {mean_ep_time:.1f}s")
print()
print("=== Stability Gate ===")
print(f"std(best_val_acc) <= 1.0%                         : {pf(gate1_pass)}  ({std_val:.2f}%)")
print(f"No seed >2.5% below mean val acc                  : {pf(gate2_pass)}  (max gap: {max_gap:.2f}%)")
print(f"std(test_acc) <= 2.0%                             : {pf(gate3_pass)}  ({std_test:.2f}%)")
print(f"All runs completed without failure                 : {pf(gate4_pass)}")
print(f"Proj grad > 0 at epoch 1 for all seeds            : {pf(gate5_pass)}")
print(f"Proj grads active all epochs for all seeds        : {pf(gate6_pass)}")
print(f"Prob sums valid for all seeds                     : {pf(gate7_pass)}")
print(f"Majority-class collapse absent for all seeds      : {pf(gate8_pass)}")
print()
print(f"=== VERDICT: {verdict_str} ===")
print()


# ── Build markdown report ─────────────────────────────────────────────────────
run_date = datetime.date.today().isoformat()


def _str_or_na(val: object, fmt: str) -> str:
    if val is None:
        return "N/A"
    return format(val, fmt)  # type: ignore[arg-type]


# Per-seed results table rows
per_seed_rows: list[str] = []
for r in all_results:
    s    = r["seed"]
    bva  = f"{r['best_val_acc']:.2f}%"             if r["best_val_acc"]             is not None else "N/A"
    be   = str(r["best_epoch"])                      if r["best_epoch"]               is not None else "N/A"
    ta   = f"{r['test_acc_analysis_only']:.2f}%"    if r["test_acc_analysis_only"]   is not None else "N/A"
    tac  = f"{r['train_acc_at_best_epoch']:.2f}%"   if r["train_acc_at_best_epoch"]  is not None else "N/A"
    vl   = f"{r['val_loss_at_best_epoch']:.4f}"     if r["val_loss_at_best_epoch"]   is not None else "N/A"
    pre  = f"{r['test_precision']:.4f}"              if r["test_precision"]           is not None else "N/A"
    rec  = f"{r['test_recall']:.4f}"                 if r["test_recall"]              is not None else "N/A"
    f1v  = f"{r['test_f1']:.4f}"                    if r["test_f1"]                  is not None else "N/A"
    aur  = f"{r['test_auroc']:.4f}"                 if r["test_auroc"]               is not None else "N/A"
    aupr = f"{r['test_auprc']:.4f}"                 if r["test_auprc"]               is not None else "N/A"
    cm_d = r["confusion_matrix"]
    cm_s = (
        f"TN={cm_d['tn']}, FP={cm_d['fp']}, FN={cm_d['fn']}, TP={cm_d['tp']}"
        if cm_d else "N/A"
    )
    pgn1 = (
        f"{r['projection_grad_norm_epoch1']:.2e}"
        if r["projection_grad_norm_epoch1"] is not None else "N/A"
    )
    pga  = "YES" if r["projection_grad_active_all_epochs"] else "NO"
    pbv  = "YES" if r["probability_sum_valid_all_epochs"]  else "NO"
    col  = "NO"  if r["majority_class_collapse"]           else "YES"   # YES = collapse ABSENT
    met  = f"{r['mean_epoch_time']:.1f}s"                 if r["mean_epoch_time"] is not None else "N/A"
    sts  = r["training_status"]
    per_seed_rows.append(
        f"| {s} | {bva} | {be} | {ta} | {tac} | {vl} "
        f"| {pre} | {rec} | {f1v} | {aur} | {aupr} "
        f"| {cm_s} | {pgn1} | {pga} | {pbv} | {col} | {met} | {sts} |"
    )

per_seed_table = "\n".join(per_seed_rows)

# Stability gate table rows
gate_table_rows = "\n".join([
    f"| std(best_val_acc) | ≤ 1.0% | {std_val:.2f}% | {pf(gate1_pass)} |",
    f"| Max seed gap below mean val acc | ≤ 2.5% | {max_gap:.2f}% | {pf(gate2_pass)} |",
    f"| std(test_acc) [analysis] | ≤ 2.0% | {std_test:.2f}% | {pf(gate3_pass)} |",
    f"| All runs completed | 0 failures | {n_seeds - n_completed} failures | {pf(gate4_pass)} |",
    f"| Proj grad > 0 epoch 1 all seeds | All seeds | {n_proj_ep1_active}/{n_seeds} seeds | {pf(gate5_pass)} |",
    f"| Proj grads active all epochs all seeds | All seeds | {n_proj_all_active}/{n_seeds} seeds | {pf(gate6_pass)} |",
    f"| Prob sums valid all seeds | All seeds | {n_prob_valid}/{n_seeds} seeds | {pf(gate7_pass)} |",
    f"| Collapse absent all seeds | All seeds | {n_collapse_absent}/{n_seeds} seeds | {pf(gate8_pass)} |",
])

# Interpretation and next step paragraphs
if overall_pass:
    interpretation = (
        f"All eight stability gate criteria passed. The DV hybrid model achieves "
        f"mean best val acc {mean_val:.2f}% ± {std_val:.2f}% across four seeds "
        f"(range {min_val:.2f}–{max_val:.2f}%), within the ≤1.0% std threshold. "
        f"Gradient flow through the quantum projection layer remained active across all "
        f"epochs for all seeds, and quantum probability sums remained valid (≈1.0) "
        f"throughout all runs. No majority-class collapse was observed in any seed. "
        f"The DV hybrid baseline is considered stable and suitable for progression to "
        f"the next experimental stage."
    )
    next_step_para = (
        "The DV hybrid baseline is stable across seeds. The recommended next step (Q10) "
        "is to begin VinDr-SpineXR binary classification planning: identify the target "
        "binary task (e.g., normal vs. abnormal), prepare the data pipeline (resizing, "
        "normalisation, class balancing), confirm that `DVHybridCNNQNN` is compatible "
        "with the new dataset's image dimensions, and run a 3-epoch sanity check "
        "analogous to Slice Q7 on PneumoniaMNIST."
    )
else:
    failing_items: list[str] = []
    if not gate1_pass: failing_items.append(f"std(val_acc) = {std_val:.2f}% > 1.0%")
    if not gate2_pass: failing_items.append(f"max seed gap = {max_gap:.2f}% > 2.5%")
    if not gate3_pass: failing_items.append(f"std(test_acc) = {std_test:.2f}% > 2.0%")
    if not gate4_pass: failing_items.append(f"{n_seeds - n_completed} seed(s) failed to complete")
    if not gate5_pass: failing_items.append(f"proj grad = 0 at epoch 1 for ≥1 seed")
    if not gate6_pass: failing_items.append(f"proj grad degraded in ≥1 epoch for ≥1 seed")
    if not gate7_pass: failing_items.append(f"prob sums deviated from 1.0 for ≥1 seed")
    if not gate8_pass: failing_items.append(f"majority-class collapse present in ≥1 seed")
    failing_str = "; ".join(failing_items)
    interpretation = (
        f"One or more stability gate criteria failed: {failing_str}. "
        f"The DV hybrid baseline cannot be considered stable at this stage. "
        f"Investigation of the failing criteria is required before proceeding "
        f"to VinDr-SpineXR binary planning."
    )
    next_step_para = (
        f"Stop and investigate the failing gate criteria before proceeding. "
        f"Failing items: {failing_str}. "
        f"Depending on the failure mode, consider: (a) inspecting seed-sensitive "
        f"initialisation in the quantum theta parameter for seeds with high val acc "
        f"variance; (b) reviewing gradient flow through the projection layer for any "
        f"seed where proj_gn = 0; (c) inspecting confusion matrices for seeds "
        f"exhibiting majority-class collapse."
    )

# Failure forensics appendix
failed_results = [r for r in all_results if r["training_status"] == "FAILED"]
if failed_results:
    forensics_lines = ["\n---\n\n## Appendix A — Failure Forensics\n"]
    for r in failed_results:
        forensics_lines.append(f"### Seed {r['seed']}\n")
        forensics_lines.append(f"```\n{r['failure_detail']}\n```\n")
    forensics_section = "\n".join(forensics_lines)
else:
    forensics_section = ""

# ── Write report ──────────────────────────────────────────────────────────────
report = f"""# DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation

- **Status:** {'Complete' if n_completed == n_seeds else f'Partial — {n_seeds - n_completed} seed(s) FAILED; see Appendix A'}
- **Date:** {run_date}
- **Branch:** feature/qnn-integration
- **Slice:** Q9

---

## 1. Title

DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation — Slice Q9

Multi-seed (4 seeds × 10 epochs) stability screening of `DVHybridCNNQNN` on PneumoniaMNIST.
Q8 validated the architecture at seed=42 over 30 epochs (best val acc 92.18%); this slice
checks whether the result is stable across seeds before committing to more expensive
experiments or new datasets.

---

## 2. Context

Q8 full baseline (seed=42, 30 epochs) achieved 92.18% best val acc, with projection
gradients active throughout and no majority-class collapse. The result was meaningful but
based on a single seed. Q9 answers: was Q8 stable, or was it seed luck?

Four seeds (42, 7, 123, 999) are run for 10 epochs each. Q8 showed the model plateaus in
the 90–92% range by epoch 10 before further improvement requires 20+ epochs; 10 epochs is
therefore sufficient for stability screening before committing to expensive full reruns per
seed.

Architecture is unchanged from Q8. All training conditions are identical (same checkpoint,
same class weights, same batch size, same LR) except for the per-seed random initialisation
of the trainable hybrid parameters.

---

## 3. Architecture Summary

Architecture unchanged from Q8 (`DVHybridCNNQNN`).

| Component | Type | Frozen / Trainable | Parameters | Source |
|---|---|---|---|---|
| CNN backbone (`model[:4]`) | 2× depthwise-sep + AdaptiveAvgPool2d + Flatten | **Frozen** | 9,612 | Pretrained C006-D040 (Q6) |
| Projection layer | `nn.Linear(128, 4)`, no activation | **Trainable** | 516 | Random init per seed |
| Quantum theta | `nn.Parameter` shape `(1, 2, 4, 3)` | **Trainable** | 24 | Random init per seed |
| Readout layer | `nn.Linear(16, 2)` | **Trainable** | 34 | Random init per seed |
| **Total trainable** | | | **574** | |
| **Total frozen** | | | **9,612** | |

---

## 4. Run Configuration

| Parameter | Value |
|---|---|
| Seeds | {', '.join(str(s) for s in SEEDS)} |
| Epochs per seed | {EPOCHS} |
| Batch size | {BATCH_SIZE} |
| Optimizer | Adam |
| Learning rate | {LR} |
| Loss | `nn.CrossEntropyLoss(weight=balanced)` |
| Class weights | [{class_weights[0].item():.6f}, {class_weights[1].item():.6f}] |
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Device | {DEVICE} (quantum simulator CPU-only) |
| Test accuracy role | **Analysis only — not a fitness gate** |

---

## 5. Per-Seed Results

| Seed | Best Val Acc | Best Ep | Test Acc* | Train Acc @ Best | Val Loss @ Best | Precision | Recall | F1 | AUROC | AUPRC | Confusion Matrix | Proj GN Ep1 | Proj All Active | Prob Valid | Collapse Absent | Mean Ep Time | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{per_seed_table}

*Test accuracy is analysis-only — not a fitness gate.

---

## 6. Aggregate Statistics

| Metric | Value |
|---|---|
| mean_best_val_acc | {mean_val:.2f}% |
| std_best_val_acc | {std_val:.2f}% |
| min_best_val_acc | {min_val:.2f}% |
| max_best_val_acc | {max_val:.2f}% |
| mean_test_acc (analysis only) | {mean_test:.2f}% |
| std_test_acc (analysis only) | {std_test:.2f}% |
| mean_f1 | {mean_f1:.4f} |
| std_f1 | {std_f1:.4f} |
| mean_auroc | {mean_auroc:.4f} |
| std_auroc | {std_auroc:.4f} |
| mean_epoch_time | {mean_ep_time:.1f}s |

---

## 7. Stability Gate

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
{gate_table_rows}

---

## 8. Comparison Against Q8 Seed=42 Full Baseline

| Metric | Q8 (seed=42, 30ep) | Q9 mean (4 seeds, 10ep) |
|---|---|---|
| best_val_acc | {Q8_BEST_VAL_ACC:.2f}% | {mean_val:.2f}% ± {std_val:.2f}% |
| test_acc (analysis) | {Q8_TEST_ACC:.2f}% | {mean_test:.2f}% ± {std_test:.2f}% |
| mean_f1 | — | {mean_f1:.4f} ± {std_f1:.4f} |
| mean_auroc | — | {mean_auroc:.4f} ± {std_auroc:.4f} |

---

## 9. Interpretation

{interpretation}

---

## 10. Explicit Verdict

VERDICT: {verdict_str}

---

## 11. Recommended Next Step

{next_step_para}
{forensics_section}"""

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Final guard before write — use realpath (works even if REPORT_PATH does not yet exist)
for _er in _EXISTING_REPORTS:
    assert os.path.realpath(REPORT_PATH) != os.path.realpath(_er), \
        f"FATAL: Q9 report path collides with existing report: {_er}"

with open(REPORT_PATH, "w") as f:
    f.write(report)

print(f"Report written to: {REPORT_PATH}")
print()
status_label = "COMPLETE" if n_completed == n_seeds else f"PARTIAL — {n_seeds - n_completed} seed(s) FAILED"
print(f"=== OVERALL: {status_label} — Q9 multi-seed stability runner finished ===")
