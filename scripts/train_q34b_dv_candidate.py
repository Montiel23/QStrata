"""
scripts/train_q34b_dv_candidate.py

Q34B DV NAS Pilot — Trial Execution Script.

Accepts a YAML config and trains a DV quantum hybrid model over the frozen
C006-D040 pretrained backbone. Used by run_q34b_dv_nas_pilot.py as the
per-trial command target.

Architecture (identical to Q21):
    Input (B, 1, 224, 224)
    → Frozen C006-D040 backbone         [frozen]    → (B, 128)
    → nn.Linear(128, n_qubits)          [trainable] → (B, n_qubits)
    → medical_ansatz (per-sample loop)  [trainable] → (B, 2^n_qubits)
    → nn.Linear(2^n_qubits, 2)          [trainable] → (B, 2)

Uses qcore.models.dv_hybrid_cnn_qnn.DVHybridCNNQNN directly. Configurable
dimensions are n_qubits (qubit_count) and depth (ansatz_depth). All other
Q34B search dimensions map to their fixed medical_ansatz values (documented
below under "Q34B Pilot Substitutions").

Q34B PILOT SUBSTITUTIONS (documented, not errors):
  rotation_family:
    Q33A search space: RY | RX+RY | RX+RY+RZ
    Pilot implementation: medical_ansatz always applies RX+RY+RZ variational
    layer at every depth. "RY" and "RX+RY" config values are substituted to
    RX+RY+RZ. Implementing per-rotation-family ansatz factories is deferred
    to a full NAS execution.

  entanglement_topology:
    Q33A search space: linear | circular
    Pilot implementation: medical_ansatz uses circular ring+cross entanglement
    (CNOT(q, q+1 mod n) + CNOT(q, q+2 mod n) for n>2). "linear" config values
    are substituted to circular. Separate linear topology is deferred.

  reuploading_frequency:
    Q33A search space: none | every_layer
    Pilot implementation: medical_ansatz always applies angle encoding (RY+RZ)
    at the start of every depth layer — equivalent to "every_layer". "none"
    config values are substituted to "every_layer". Single-pass encoding without
    reuploading is deferred.

  compression_dim:
    Q33A search space: 2 | 4 (independent bottleneck before quantum encoding)
    Pilot implementation: proj layer is always nn.Linear(128, n_qubits); there
    is no independent compression bottleneck between the backbone output and the
    quantum circuit input. compression_dim from the config is logged but does not
    affect architecture. A two-stage proj(128→compress_dim→n_qubits) is deferred.

  measurement_strategy:
    Q33A search space: pauli_z_per_qubit | multi_qubit_expectation_avg | concatenated
    Pilot implementation: full probability vector (2^n_qubits,) → nn.Linear readout.
    "pauli_z_per_qubit" is the only pilot option. Other strategies are deferred.

  classical_projection_layer:
    Pilot implementation: always nn.Linear (no activation). "linear" is the only
    pilot option.

Device:
    Quantum circuit simulation is CPU-only in the QStrata DV backend (matrix
    multiplication over complex state vectors). The backbone and all training
    also run on CPU. CUDA is checked for environment sanity only.

CLI:
    python3 scripts/train_q34b_dv_candidate.py --config <yaml_path>

Output lines (machine-readable, parsed by run_q34b_dv_nas_pilot.py):
    Q34B_TRIAL_AUROC: <float>
    Q34B_TRIAL_F1: <float>
    Q34B_TRIAL_ACCURACY: <float>
    Q34B_TRIAL_PARAMS: <int>
    Q34B_TRIAL_LATENCY_MS: <float>
    Q34B_TRIAL_GRADIENT_HEALTH: PASS|FAIL
    Q34B_TRIAL_NAN_INF: PASS|FAIL
    Q34B_TRIAL_STATUS: completed

Exit 0 on success, nonzero on failure.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BACKBONE_CHECKPOINT_DEFAULT = "checkpoints/c006_d040_classical_anchor.pt"
DATASET_ROOT_DEFAULT        = "data/processed/vindr_binary_roi_224"
MAX_EPOCHS_ALLOWED          = 5      # hard ceiling for Q34B pilot
MAX_TRAINABLE_PARAMS        = 100_000
GRAD_NORM_MAX               = 1.0e6  # exploding gradient hard stop
PROB_SUM_DEV_MAX            = 0.01   # 1% max deviation from 1.0

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q34B DV NAS Trial")
    p.add_argument("--config", required=True, help="Path to trial YAML config")
    return p.parse_args()


# ── Seeds ──────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Backbone loading ───────────────────────────────────────────────────────────

def load_pretrained_backbone(model: DVHybridCNNQNN, checkpoint_path: str) -> None:
    """
    Load C006-D040 backbone weights into DVHybridCNNQNN.
    Identical key remapping to Q21: "0.*" → "backbone.0.*", "1.*" → "backbone.1.*".
    Hard stop if any backbone key is missing.
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ck:
        src_sd = ck["model_state_dict"]
    elif "state_dict" in ck:
        src_sd = ck["state_dict"]
    else:
        print("ERROR: unrecognised checkpoint format", flush=True)
        sys.exit(1)

    remapped: dict = {}
    for k, v in src_sd.items():
        if k.startswith("0.") or k.startswith("1."):
            remapped[f"backbone.{k}"] = v

    missing, _ = model.load_state_dict(remapped, strict=False)
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        print(
            f"ERROR: {len(backbone_missing)} backbone keys missing: {backbone_missing}",
            flush=True,
        )
        sys.exit(1)

    # Re-freeze backbone after load
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    if any(p.requires_grad for p in model.backbone.parameters()):
        print("ERROR: backbone not fully frozen after load", flush=True)
        sys.exit(1)


# ── Gradient helpers ───────────────────────────────────────────────────────────

def grad_norm_param(p: nn.Parameter | None) -> float:
    if p is None or p.grad is None:
        return 0.0
    return float(p.grad.detach().norm(2).item())


def compute_grad_norms(model: DVHybridCNNQNN) -> dict[str, float]:
    theta_gn   = grad_norm_param(model.theta)
    proj_gn    = (
        grad_norm_param(model.proj.weight) ** 2
        + grad_norm_param(model.proj.bias) ** 2
    ) ** 0.5
    readout_gn = (
        grad_norm_param(model.readout.weight) ** 2
        + grad_norm_param(model.readout.bias) ** 2
    ) ** 0.5
    total_gn   = (theta_gn**2 + proj_gn**2 + readout_gn**2) ** 0.5
    return {
        "theta":   theta_gn,
        "proj":    proj_gn,
        "readout": readout_gn,
        "total":   total_gn,
    }


def check_backbone_grad(model: DVHybridCNNQNN, epoch: int) -> None:
    """Hard stop if backbone received any gradient."""
    for name, param in model.backbone.named_parameters():
        if param.grad is not None and param.grad.abs().max().item() > 0.0:
            print(
                f"ERROR: backbone parameter 'backbone.{name}' received gradient "
                f"at epoch {epoch}. Backbone must be frozen. Aborting.",
                flush=True,
            )
            sys.exit(3)


def check_grad_norms(grads: dict[str, float], epoch: int) -> None:
    """Hard stop on NaN, inf, or exploding gradients."""
    for name, val in grads.items():
        if not np.isfinite(val):
            print(
                f"ERROR: gradient '{name}' is not finite at epoch {epoch}: {val}. Aborting.",
                flush=True,
            )
            sys.exit(3)
        if val > GRAD_NORM_MAX:
            print(
                f"ERROR: gradient '{name}' exploded at epoch {epoch}: "
                f"{val:.4e} > {GRAD_NORM_MAX:.0e}. Aborting.",
                flush=True,
            )
            sys.exit(3)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_split(
    model: DVHybridCNNQNN,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    all_prob_sums: list[float] = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            if model._prob_sums:
                all_prob_sums.extend(model._prob_sums)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)
    has_both   = len(np.unique(y_true_arr)) > 1

    # Probability conservation check
    if all_prob_sums:
        max_dev = max(abs(s - 1.0) for s in all_prob_sums)
        if max_dev > PROB_SUM_DEV_MAX:
            print(
                f"ERROR: probability sum deviation {max_dev:.4e} > {PROB_SUM_DEV_MAX:.0%}. "
                f"Invalid quantum circuit output. Aborting.",
                flush=True,
            )
            sys.exit(3)

    return {
        "loss":     total_loss / len(loader),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1":       float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true_arr, y_prob_arr)) if has_both else float("nan"),
    }


def measure_latency(model: DVHybridCNNQNN,
                    n_warmup: int = 5, n_samples: int = 20) -> float:
    """
    Mean ms per single-sample forward pass (CPU timing).
    Uses smaller warmup and sample count than Q34A because DV circuit is slow.
    """
    model.eval()
    x = torch.randn(1, 1, 224, 224)  # CPU tensor (DV circuit is CPU-only)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        timings: list[float] = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Epoch ceiling ─────────────────────────────────────────────────────────
    epochs_requested = cfg.get("training", {}).get("epochs", 4)
    if epochs_requested > MAX_EPOCHS_ALLOWED:
        print(
            f"ERROR: epochs={epochs_requested} exceeds Q34B ceiling of {MAX_EPOCHS_ALLOWED}",
            flush=True,
        )
        sys.exit(1)
    epochs = int(epochs_requested)

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seeds(seed)

    # ── CUDA sanity check (DV runs on CPU; CUDA checked for environment) ──────
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — DV pilot runs on CPU regardless", flush=True)
    device = torch.device("cpu")   # DV quantum circuit is CPU-only

    # ── Read searchable dimensions ────────────────────────────────────────────
    model_cfg    = cfg.get("model", {})
    qubit_count  = int(model_cfg.get("qubit_count", 4))
    ansatz_depth = int(model_cfg.get("ansatz_depth", 1))

    # Dimensions acknowledged but substituted in this pilot (see module docstring)
    rotation_family          = model_cfg.get("rotation_family",          "RX+RY+RZ")
    entanglement_topology    = model_cfg.get("entanglement_topology",    "circular")
    encoding_strategy        = model_cfg.get("encoding_strategy",        "angle_encoding")
    reuploading_frequency    = model_cfg.get("reuploading_frequency",    "every_layer")
    measurement_strategy     = model_cfg.get("measurement_strategy",     "pauli_z_per_qubit")
    compression_dim          = model_cfg.get("compression_dim",          qubit_count)
    classical_proj_layer     = model_cfg.get("classical_projection_layer", "linear")

    print(f"[Q34B] config:                    {args.config}", flush=True)
    print(f"[Q34B] device:                    {device} (DV circuit is CPU-only)", flush=True)
    print(f"[Q34B] epochs:                    {epochs}", flush=True)
    print(f"[Q34B] seed:                      {seed}", flush=True)
    print(f"[Q34B] qubit_count:               {qubit_count}", flush=True)
    print(f"[Q34B] ansatz_depth:              {ansatz_depth}", flush=True)
    print(f"[Q34B] rotation_family:           {rotation_family} → SUBSTITUTED to RX+RY+RZ (medical_ansatz fixed)", flush=True)
    print(f"[Q34B] entanglement_topology:     {entanglement_topology} → SUBSTITUTED to circular (medical_ansatz fixed)", flush=True)
    print(f"[Q34B] encoding_strategy:         {encoding_strategy} (angle_encoding, reuploading at every layer)", flush=True)
    print(f"[Q34B] reuploading_frequency:     {reuploading_frequency} → SUBSTITUTED to every_layer (medical_ansatz fixed)", flush=True)
    print(f"[Q34B] measurement_strategy:      {measurement_strategy} (full prob vector → linear readout)", flush=True)
    print(f"[Q34B] compression_dim:           {compression_dim} → SUBSTITUTED to n_qubits={qubit_count} (no independent bottleneck)", flush=True)
    print(f"[Q34B] classical_projection_layer:{classical_proj_layer}", flush=True)

    # ── Data paths ────────────────────────────────────────────────────────────
    dataset_root = cfg.get("dataset", {}).get("root", DATASET_ROOT_DEFAULT)
    if not os.path.isdir(dataset_root):
        dataset_root = DATASET_ROOT_DEFAULT
    ck_path = model_cfg.get("checkpoint_path", BACKBONE_CHECKPOINT_DEFAULT)
    if not os.path.isfile(ck_path):
        ck_path = BACKBONE_CHECKPOINT_DEFAULT
    if not os.path.isfile(ck_path):
        print(f"ERROR: backbone checkpoint not found at {ck_path}", flush=True)
        sys.exit(1)
    if not os.path.isdir(dataset_root):
        print(f"ERROR: dataset not found at {dataset_root}", flush=True)
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────────────
    model = DVHybridCNNQNN(
        cnn_config=CNN_CONFIG,
        n_qubits=qubit_count,
        depth=ansatz_depth,
        alpha=0.1,
        n_classes=2,
    )

    load_pretrained_backbone(model, ck_path)
    model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    print(f"[Q34B] trainable params:          {n_trainable}", flush=True)
    print(f"[Q34B] total params:              {n_total}", flush=True)
    print(f"[Q34B]   proj(128→{qubit_count}):       {model.proj.weight.numel() + model.proj.bias.numel()}", flush=True)
    print(f"[Q34B]   theta shape:             {tuple(model.theta.shape)} = {model.theta.numel()}", flush=True)
    print(f"[Q34B]   readout(2^{qubit_count}→2):     {model.readout.weight.numel() + model.readout.bias.numel()}", flush=True)

    if n_trainable > MAX_TRAINABLE_PARAMS:
        print(
            f"ERROR: trainable_params={n_trainable} exceeds hard limit {MAX_TRAINABLE_PARAMS}",
            flush=True,
        )
        sys.exit(1)

    # ── Training config ───────────────────────────────────────────────────────
    lr         = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    weight_dec = float(cfg.get("training", {}).get("weight_decay",  0.0))
    batch_size = int(cfg.get("training",   {}).get("batch_size",    4))

    print(f"[Q34B] lr:                        {lr}", flush=True)
    print(f"[Q34B] weight_decay:              {weight_dec}", flush=True)
    print(f"[Q34B] batch_size:                {batch_size}", flush=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VinDrSpineXRBinaryDataset(root=dataset_root, split="train")
    val_ds   = VinDrSpineXRBinaryDataset(root=dataset_root, split="val")
    test_ds  = VinDrSpineXRBinaryDataset(root=dataset_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_dec,
    )

    # ── Gradient health tracking ──────────────────────────────────────────────
    # GRADIENT_HEALTH: PASS if at least one of (theta, proj, readout) was non-zero
    # in the final epoch and no gradients were NaN/inf/exploding.
    # FAIL if all were zero (barren plateau) or any were NaN/inf/exploding.
    gradient_health_pass = True
    nan_inf_pass         = True
    final_grad_norms: dict[str, float] = {}

    # ── Training loop ─────────────────────────────────────────────────────────
    final_train_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()   # backbone always in eval
        t0           = time.perf_counter()
        running_loss = 0.0
        correct      = 0
        total        = 0
        last_grads: dict[str, float] = {}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)

            if not torch.isfinite(loss):
                print(
                    f"ERROR: NaN/inf loss at epoch {epoch}. "
                    f"Q34B_TRIAL_NAN_INF: FAIL",
                    flush=True,
                )
                nan_inf_pass = False
                sys.exit(3)

            loss.backward()

            # Backbone frozen check
            check_backbone_grad(model, epoch)

            last_grads = compute_grad_norms(model)
            check_grad_norms(last_grads, epoch)

            optimizer.step()
            running_loss += loss.item()
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.size(0)

        t_epoch          = time.perf_counter() - t0
        final_train_loss = running_loss / len(train_loader)
        final_grad_norms = last_grads

        val_m = eval_split(model, val_loader, criterion)

        # Barren plateau warning: zero gradient norm
        if last_grads.get("total", 0.0) == 0.0:
            print(
                f"[Q34B] WARNING: all trainable gradients are zero at epoch {epoch} "
                f"— possible barren plateau",
                flush=True,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={final_train_loss:.4f} acc={correct/total:.4f} | "
            f"val_loss={val_m['loss']:.4f} auroc={val_m['auroc']:.4f} "
            f"f1={val_m['f1']:.4f} | "
            f"θ={last_grads.get('theta', 0.0):.2e} "
            f"proj={last_grads.get('proj', 0.0):.2e} "
            f"readout={last_grads.get('readout', 0.0):.2e} | "
            f"t={t_epoch:.1f}s",
            flush=True,
        )

    # ── Gradient health verdict ───────────────────────────────────────────────
    theta_gn   = final_grad_norms.get("theta",   0.0)
    proj_gn    = final_grad_norms.get("proj",    0.0)
    readout_gn = final_grad_norms.get("readout", 0.0)

    # PASS: at least one component non-zero (not complete barren plateau)
    #       AND no component exceeds GRAD_NORM_MAX (no explosion)
    any_nonzero  = (theta_gn > 0.0 or proj_gn > 0.0 or readout_gn > 0.0)
    all_finite   = (np.isfinite(theta_gn) and np.isfinite(proj_gn) and np.isfinite(readout_gn))
    no_explosion = (theta_gn <= GRAD_NORM_MAX and proj_gn <= GRAD_NORM_MAX and readout_gn <= GRAD_NORM_MAX)

    if any_nonzero and all_finite and no_explosion:
        gradient_health_pass = True
    else:
        gradient_health_pass = False
        print(
            f"[Q34B] GRADIENT HEALTH FAIL: any_nonzero={any_nonzero} "
            f"all_finite={all_finite} no_explosion={no_explosion}",
            flush=True,
        )

    grad_health_str = "PASS" if gradient_health_pass else "FAIL"
    nan_inf_str     = "PASS" if nan_inf_pass else "FAIL"

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("[Q34B] evaluating on test split ...", flush=True)
    test_m = eval_split(model, test_loader, criterion)

    # ── Latency ───────────────────────────────────────────────────────────────
    print("[Q34B] measuring latency (CPU, DV circuit) ...", flush=True)
    lat_ms = measure_latency(model)

    # ── Machine-readable output ───────────────────────────────────────────────
    print(f"Q34B_TRIAL_AUROC: {test_m['auroc']:.6f}",     flush=True)
    print(f"Q34B_TRIAL_F1: {test_m['f1']:.6f}",           flush=True)
    print(f"Q34B_TRIAL_ACCURACY: {test_m['accuracy']:.6f}", flush=True)
    print(f"Q34B_TRIAL_PARAMS: {n_trainable}",             flush=True)
    print(f"Q34B_TRIAL_LATENCY_MS: {lat_ms:.4f}",         flush=True)
    print(f"Q34B_TRIAL_GRADIENT_HEALTH: {grad_health_str}", flush=True)
    print(f"Q34B_TRIAL_NAN_INF: {nan_inf_str}",            flush=True)
    print(f"Q34B_TRIAL_STATUS: completed",                 flush=True)

    # ── Runner-compatible [SUMMARY] block ─────────────────────────────────────
    print("[SUMMARY]",                           flush=True)
    print(f"Trainable params: {n_trainable}",    flush=True)
    print(f"Loss: {final_train_loss:.6f}",       flush=True)

    print(
        f"[Q34B] DONE | auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
        f"params={n_trainable} lat={lat_ms:.2f}ms/sample "
        f"grad={grad_health_str} nan_inf={nan_inf_str}",
        flush=True,
    )


if __name__ == "__main__":
    main()
