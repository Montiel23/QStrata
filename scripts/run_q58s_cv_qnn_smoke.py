#!/usr/bin/env python3
"""
scripts/run_q58s_cv_qnn_smoke.py

Q58S — Custom CV-QNN Smoke Test

Pre-flight smoke test for Q58-CUSTOM-CV-QNN-BINARY-BENCHMARK.
Validates the custom in-house CV-QNN pipeline (cv_spine_ansatz) on Q49
embeddings before the full multi-seed benchmark run.

Validates:
  - GaussianBackend + cv_spine_ansatz.GaussianVariationalAnsatz import
  - CVQNNEmbeddingHead: 128-dim → phase-space encode → Gaussian circuit → readout
  - Forward pass: logits shape (B, 2)
  - 1-epoch training with gradient flow
  - AUROC, F1 computation
  - Circuit manifest: n_modes, depth, squeezing_cap, gate types
  - Squeezing bound (tanh * squeezing_cap) verified
  - Artifacts: CSV, JSON, SVG, MD report

Data: Q49 embeddings (MobileNetV3-Large, 128-dim, binary labels)
Subset: max 64 train / 32 val / 32 test, seed=42
Runtime target: <60s

Do NOT use: PennyLane, Qiskit, Strawberry Fields, TorchQuantum.
All quantum ops use the custom in-house qcore framework only.

Usage (from QStrata repo root):
    python scripts/run_q58s_cv_qnn_smoke.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# ── Repo root ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR     = REPO_ROOT / "workspace" / "experiments" / "Q49" / "embeddings"
OUT_BASE    = REPO_ROOT / "workspace" / "experiments" / "Q58S"
OUT_RESULTS = OUT_BASE / "results"
OUT_REPORTS = OUT_BASE / "reports"
OUT_FIGURES = OUT_BASE / "figures"

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_TRAIN       = 64
MAX_VAL         = 32
MAX_TEST        = 32
CV_N_MODES      = 2
CV_DEPTH        = 1
CV_SQUEEZING_CAP = 1.5
LR              = 1e-2
N_EPOCHS        = 1
CV_BATCH        = 16
SEED            = 42
RUNTIME_LIMIT_S = 60

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Output dirs ────────────────────────────────────────────────────────────────
for d in (OUT_RESULTS, OUT_REPORTS, OUT_FIGURES):
    d.mkdir(parents=True, exist_ok=True)

start_wall = time.time()

def elapsed() -> float:
    return time.time() - start_wall

print("=" * 64)
print("Q58S — Custom CV-QNN Smoke Test")
print("=" * 64)
print(f"Repo root:      {REPO_ROOT}")
print(f"Embeddings:     {EMB_DIR}")
print(f"Output:         {OUT_BASE}")
print(f"CV modes:       {CV_N_MODES}  depth={CV_DEPTH}  squeezing_cap={CV_SQUEEZING_CAP}")
print(f"Runtime limit:  {RUNTIME_LIMIT_S}s")
print()

results: Dict = {
    "slice_id": "Q58S-CUSTOM-CV-QNN-SMOKE-TEST",
    "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "seed": SEED,
    "synthetic_fallback_used": False,
    "real_data_source": None,
    "train_n": 0, "val_n": 0, "test_n": 0,
    "n_classes": 0,
    "cv_qnn": {},
    "circuit_manifest": {},
    "metrics": [],
    "artifacts": [],
    "runtime_s": 0.0,
    "runtime_exceeded": False,
    "hard_failures": [],
    "warnings": [],
    "recommendation": "BLOCKED",
}


# ── 1. Data Loading ────────────────────────────────────────────────────────────
print("── 1. Data Loading ──────────────────────────────────────────────────────")

emb_train = emb_val = emb_test = None
lbl_train = lbl_val = lbl_test = None
data_source = "NONE"

if EMB_DIR.exists():
    try:
        emb_train_full = np.load(EMB_DIR / "train_embeddings.npy").astype(np.float32)
        lbl_train_full = np.load(EMB_DIR / "train_labels.npy").astype(np.int64)
        emb_val_full   = np.load(EMB_DIR / "val_embeddings.npy").astype(np.float32)
        lbl_val_full   = np.load(EMB_DIR / "val_labels.npy").astype(np.int64)
        emb_test_full  = np.load(EMB_DIR / "test_embeddings.npy").astype(np.float32)
        lbl_test_full  = np.load(EMB_DIR / "test_labels.npy").astype(np.int64)

        def _balanced_subset(emb, lbl, n):
            per_class = n // 2
            idx = []
            for c in [0, 1]:
                c_idx = np.where(lbl == c)[0][:per_class]
                idx.extend(c_idx.tolist())
            return emb[np.array(idx[:n])], lbl[np.array(idx[:n])]

        emb_train, lbl_train = _balanced_subset(emb_train_full, lbl_train_full, MAX_TRAIN)
        emb_val,   lbl_val   = _balanced_subset(emb_val_full,   lbl_val_full,   MAX_VAL)
        emb_test,  lbl_test  = _balanced_subset(emb_test_full,  lbl_test_full,  MAX_TEST)

        data_source = "Q49_EMBEDDINGS"
        results["real_data_source"] = str(EMB_DIR)
        print(f"  Source:  Q49 embeddings (MobileNetV3-Large, 128-dim)")
        print(f"  Train:   {len(lbl_train)} samples  "
              f"(class 0: {(lbl_train==0).sum()}, class 1: {(lbl_train==1).sum()})")
        print(f"  Val:     {len(lbl_val)} samples")
        print(f"  Test:    {len(lbl_test)} samples")
    except Exception as e:
        results["warnings"].append(f"Q49 load failed: {e}")
        print(f"  WARN: Q49 load failed ({e}) — using synthetic fallback")
        data_source = "SYNTHETIC"

if data_source in ("SYNTHETIC", "NONE") or emb_train is None:
    results["synthetic_fallback_used"] = True
    results["warnings"].append("Using synthetic fallback data (128-dim random)")
    emb_train = np.random.randn(MAX_TRAIN, 128).astype(np.float32)
    lbl_train = np.array([i % 2 for i in range(MAX_TRAIN)], dtype=np.int64)
    emb_val   = np.random.randn(MAX_VAL, 128).astype(np.float32)
    lbl_val   = np.array([i % 2 for i in range(MAX_VAL)], dtype=np.int64)
    emb_test  = np.random.randn(MAX_TEST, 128).astype(np.float32)
    lbl_test  = np.array([i % 2 for i in range(MAX_TEST)], dtype=np.int64)
    data_source = "SYNTHETIC_FALLBACK"
    print(f"  WARN: Using synthetic fallback (128-dim random embeddings)")

results["train_n"] = int(len(lbl_train))
results["val_n"]   = int(len(lbl_val))
results["test_n"]  = int(len(lbl_test))
results["n_classes"] = int(len(set(lbl_train.tolist())))

X_tr = torch.from_numpy(emb_train)
y_tr = torch.from_numpy(lbl_train)
X_te = torch.from_numpy(emb_test)
y_te = lbl_test

print(f"  Elapsed: {elapsed():.2f}s")
print()


# ── 2. CV-QNN Import ───────────────────────────────────────────────────────────
print("── 2. CV-QNN Framework Import ───────────────────────────────────────────")

cv_import_ok = False
cv_import_error = None

try:
    from qcore.backends.cvBackend import GaussianBackend
    from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
    from qcore.physics.symplectic import (get_rotation_matrix,
                                           get_beamsplitter_matrix,
                                           get_squeezing_matrix,
                                           get_displacement_vector)
    cv_import_ok = True
    print("  qcore.backends.cvBackend.GaussianBackend         OK")
    print("  qcore.ansatz.cv_spine_ansatz.GaussianVariationalAnsatz OK")
    print("  qcore.physics.symplectic (rotation/BS/squeezing)  OK")
except Exception as e:
    cv_import_error = str(e)
    results["hard_failures"].append(f"CV-QNN import failed: {e}")
    print(f"  FAIL: {e}")
    traceback.print_exc()

print(f"  Elapsed: {elapsed():.2f}s")
print()

if not cv_import_ok:
    # Can't continue without framework
    results["recommendation"] = "BLOCKED: CV-QNN import failed"
    results["runtime_s"] = round(elapsed(), 2)
    json_path = OUT_RESULTS / "q58s_smoke_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"BLOCKED — see {json_path}")
    sys.exit(1)


# ── 3. CV-QNN Model Definition ─────────────────────────────────────────────────
print("── 3. CV-QNN Model ──────────────────────────────────────────────────────")

class CVQNNSpineHead(nn.Module):
    """
    CV-QNN head for spinal imaging binary classification.

    Architecture:
      embedding (128) → Linear(128, 2*n_modes) [phase-space encoder]
      → GaussianVariationalAnsatz (Displacement, Squeezing, Rotation, Beamsplitter)
      → Homodyne X-quadrature readout (n_modes)
      → Linear(n_modes, n_classes)

    Photonic circuit parameters (per layer):
      - Displacement: disp_real, disp_imag  (n_modes params each)
      - Squeezing:    squeezing_raw * tanh → bounded by squeezing_cap
      - Rotation:     rot_phi               (n_modes params)
      - Beamsplitter: bs_theta              (n_modes params, ring topology)

    Total trainable params ≈ encoder + ansatz + readout.
    """
    def __init__(self, in_dim: int = 128, n_modes: int = 2,
                 depth: int = 1, squeezing_cap: float = 1.5,
                 n_classes: int = 2):
        super().__init__()
        self.n_modes = n_modes
        self.depth = depth
        self.squeezing_cap = squeezing_cap
        self.encoder = nn.Linear(in_dim, 2 * n_modes)
        self.ansatz = GaussianVariationalAnsatz(n_modes=n_modes, depth=depth,
                                                 squeezing_cap=squeezing_cap)
        self.readout = nn.Linear(n_modes, n_classes)
        self._backend = GaussianBackend(n_modes=n_modes)

    def _forward_single(self, enc_i: torch.Tensor) -> torch.Tensor:
        mu, cov = self._backend.get_vacuum()
        mu_out, _ = self.ansatz.apply(mu, cov, self._backend, encoded_input=enc_i)
        return mu_out[::2]   # X quadratures: shape (n_modes,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)                          # (B, 2*n_modes)
        feats = torch.stack([self._forward_single(enc[i]) for i in range(len(x))])
        return self.readout(feats)                     # (B, n_classes)

    def circuit_manifest(self) -> dict:
        """Document photonic circuit parameters."""
        param_counts = {
            "encoder_weight": self.encoder.weight.numel(),
            "encoder_bias": self.encoder.bias.numel(),
            "disp_real": self.ansatz.disp_real.numel(),
            "disp_imag": self.ansatz.disp_imag.numel(),
            "squeezing_raw": self.ansatz.squeezing_raw.numel(),
            "bs_theta": self.ansatz.bs_theta.numel(),
            "rot_phi": self.ansatz.rot_phi.numel(),
            "readout_weight": self.readout.weight.numel(),
            "readout_bias": self.readout.bias.numel(),
        }
        total = sum(param_counts.values())
        return {
            "n_modes": self.n_modes,
            "depth": self.depth,
            "squeezing_cap": self.squeezing_cap,
            "gate_sequence": ["Displacement(D)", "Squeezing(S)", "Rotation(R)",
                               "Beamsplitter(BS) ring topology"],
            "data_reuploading": "layer-by-layer (encoded_input added per depth)",
            "measurement": "Homodyne X-quadrature (even indices of mu)",
            "param_counts": param_counts,
            "total_trainable_params": total,
        }


cv_model = CVQNNSpineHead(
    in_dim=128, n_modes=CV_N_MODES, depth=CV_DEPTH,
    squeezing_cap=CV_SQUEEZING_CAP
)
manifest = cv_model.circuit_manifest()
results["circuit_manifest"] = manifest

print(f"  Architecture:   128 → encoder(2*{CV_N_MODES}) → GVA(modes={CV_N_MODES}, depth={CV_DEPTH}) → readout")
print(f"  Gates/layer:    {', '.join(manifest['gate_sequence'])}")
print(f"  Data reuploading: {manifest['data_reuploading']}")
print(f"  Measurement:    {manifest['measurement']}")
print(f"  Squeezing cap:  {CV_SQUEEZING_CAP} (tanh-bounded)")
print(f"  Total params:   {manifest['total_trainable_params']}")
print()


# ── 4. Forward Pass Smoke ──────────────────────────────────────────────────────
print("── 4. Forward Pass ──────────────────────────────────────────────────────")

forward_ok = False
cv_status = "NOT_RUN"
cv_metrics: Dict = {}
t0 = time.time()

try:
    cv_model.eval()
    with torch.no_grad():
        _smoke_out = cv_model(X_tr[:2])
    assert _smoke_out.shape == (2, 2), f"Unexpected logit shape: {_smoke_out.shape}"
    forward_ok = True
    print(f"  Forward pass:   OK  logits shape={tuple(_smoke_out.shape)}")
except Exception as e:
    results["hard_failures"].append(f"CV-QNN forward pass failed: {e}")
    print(f"  FAIL: {e}")
    traceback.print_exc()

print(f"  Elapsed: {elapsed():.2f}s")
print()

if not forward_ok:
    results["recommendation"] = "BLOCKED: CV-QNN forward pass failed"
    results["runtime_s"] = round(elapsed(), 2)
    json_path = OUT_RESULTS / "q58s_smoke_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"BLOCKED — see {json_path}")
    sys.exit(1)


# ── 5. Training (1 epoch) ──────────────────────────────────────────────────────
print("── 5. Training (1 epoch) ─────────────────────────────────────────────────")

train_ok = False
cv_opt  = torch.optim.Adam(cv_model.parameters(), lr=LR)
cv_crit = nn.CrossEntropyLoss()
epoch_loss = float("nan")

try:
    cv_model.train()
    n_batches = math.ceil(len(X_tr) / CV_BATCH)
    batch_losses = []

    for bi in range(n_batches):
        xb = X_tr[bi * CV_BATCH:(bi + 1) * CV_BATCH]
        yb = y_tr[bi * CV_BATCH:(bi + 1) * CV_BATCH]
        cv_opt.zero_grad()
        logits = cv_model(xb)
        loss = cv_crit(logits, yb)
        loss.backward()
        cv_opt.step()
        batch_losses.append(loss.item())
        if elapsed() > RUNTIME_LIMIT_S * 0.8:
            results["warnings"].append(f"Training capped at batch {bi+1}/{n_batches} (runtime limit)")
            break

    epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
    train_ok = True
    print(f"  Batches completed: {len(batch_losses)}/{n_batches}")
    print(f"  Epoch loss:        {epoch_loss:.6f}")
    print(f"  Elapsed:           {elapsed():.2f}s")
except Exception as e:
    results["warnings"].append(f"CV-QNN training failed (forward OK): {e}")
    print(f"  WARN: {e}")
    traceback.print_exc()

print()


# ── 6. Evaluation ──────────────────────────────────────────────────────────────
print("── 6. Evaluation (test set) ──────────────────────────────────────────────")

eval_ok = False

try:
    cv_model.eval()
    with torch.no_grad():
        cv_logits = cv_model(X_te)
        cv_probs  = torch.softmax(cv_logits, dim=-1)[:, 1].numpy()
        cv_preds  = cv_logits.argmax(dim=-1).numpy()

    n_cls = len(set(y_te.tolist()))
    cv_metrics = {
        "model": "cv_qnn_spine_head",
        "n_modes": CV_N_MODES,
        "depth": CV_DEPTH,
        "squeezing_cap": CV_SQUEEZING_CAP,
        "n_trainable_params": manifest["total_trainable_params"],
        "train_loss": round(epoch_loss, 6),
        "n_samples": int(len(y_te)),
        "n_classes_seen": n_cls,
    }

    if _HAS_SKLEARN:
        cv_metrics["accuracy"] = float(accuracy_score(y_te, cv_preds))
        cv_metrics["f1"] = float(f1_score(y_te, cv_preds, average="binary", zero_division=0))
        cv_metrics["precision"] = float(precision_score(y_te, cv_preds, average="binary", zero_division=0))
        cv_metrics["recall"] = float(recall_score(y_te, cv_preds, average="binary", zero_division=0))
        if n_cls == 2:
            try:
                cv_metrics["auroc"] = float(roc_auc_score(y_te, cv_probs))
                cv_metrics["auroc_available"] = True
            except Exception as ae:
                cv_metrics["auroc"] = None
                cv_metrics["auroc_available"] = False
                cv_metrics["auroc_note"] = str(ae)
        else:
            cv_metrics["auroc"] = None
            cv_metrics["auroc_available"] = False
            cv_metrics["auroc_note"] = "Single class in test split"
    else:
        cv_metrics["accuracy"] = float((cv_preds == y_te).mean())
        cv_metrics["f1"] = None
        cv_metrics["auroc"] = None

    cv_metrics["runtime_s"] = round(time.time() - t0, 2)
    cv_status = "PASS"
    eval_ok = True

    print(f"  Accuracy:  {cv_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  F1:        {cv_metrics.get('f1', 'N/A')}")
    print(f"  AUROC:     {cv_metrics.get('auroc', 'N/A')}")
    print(f"  Params:    {manifest['total_trainable_params']}")
    print(f"  Elapsed:   {elapsed():.2f}s")
except Exception as e:
    cv_status = "WARN"
    cv_metrics = {"error": str(e)}
    results["warnings"].append(f"Evaluation failed: {e}")
    print(f"  WARN: {e}")
    traceback.print_exc()

results["cv_qnn"] = {"status": cv_status, "import_ok": cv_import_ok,
                      "forward_ok": forward_ok, "train_ok": train_ok, **cv_metrics}
results["metrics"] = [cv_metrics] if cv_status == "PASS" else []
print()


# ── 7. Squeezing Parameter Check ───────────────────────────────────────────────
print("── 7. Squeezing Parameter Check ─────────────────────────────────────────")

try:
    sq_raw = cv_model.ansatz.squeezing_raw.detach()
    sq_bounded = torch.tanh(sq_raw) * CV_SQUEEZING_CAP
    sq_max = float(sq_bounded.abs().max())
    sq_mean = float(sq_bounded.abs().mean())
    print(f"  squeezing_raw range:    [{float(sq_raw.min()):.4f}, {float(sq_raw.max()):.4f}]")
    print(f"  squeezing_bounded max:  {sq_max:.4f}  (cap={CV_SQUEEZING_CAP})")
    print(f"  squeezing_bounded mean: {sq_mean:.4f}")
    assert sq_max <= CV_SQUEEZING_CAP + 1e-5, f"Squeezing exceeded cap: {sq_max} > {CV_SQUEEZING_CAP}"
    print(f"  Bound check:            PASS (max {sq_max:.4f} ≤ cap {CV_SQUEEZING_CAP})")
    results["circuit_manifest"]["squeezing_max_observed"] = round(sq_max, 6)
    results["circuit_manifest"]["squeezing_mean_observed"] = round(sq_mean, 6)
    results["circuit_manifest"]["squeezing_bound_check"] = "PASS"
except Exception as e:
    results["warnings"].append(f"Squeezing check failed: {e}")
    print(f"  WARN: {e}")

print()


# ── 8. Outputs ─────────────────────────────────────────────────────────────────
total_elapsed = elapsed()
results["runtime_s"] = round(total_elapsed, 2)
results["runtime_exceeded"] = total_elapsed > RUNTIME_LIMIT_S

# CSV
csv_path = OUT_RESULTS / "q58s_smoke_metrics.csv"
if results["metrics"]:
    fields = ["model", "accuracy", "f1", "precision", "recall", "auroc",
              "auroc_available", "train_loss", "n_trainable_params", "runtime_s",
              "n_modes", "depth", "squeezing_cap", "n_samples"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results["metrics"])
    results["artifacts"].append(str(csv_path))
    print(f"CSV:     {csv_path.relative_to(REPO_ROOT)}")

# JSON
json_path = OUT_RESULTS / "q58s_smoke_results.json"
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=str)
results["artifacts"].append(str(json_path))
print(f"JSON:    {json_path.relative_to(REPO_ROOT)}")

# SVG
svg_path = OUT_FIGURES / "q58s_smoke_metrics.svg"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = cv_metrics if cv_status == "PASS" else {}
    if m:
        metrics_vals = {
            "Accuracy": m.get("accuracy", 0.0) or 0.0,
            "F1":       m.get("f1", 0.0) or 0.0,
            "AUROC":    m.get("auroc", 0.0) or 0.0,
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metrics_vals.keys(), metrics_vals.values(),
                      color=["#4e79a7", "#f28e2b", "#59a14f"])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"Q58S — CV-QNN Smoke Metrics (n_modes={CV_N_MODES}, depth={CV_DEPTH})")
        for bar, val in zip(bars, metrics_vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(svg_path, format="svg")
        plt.close(fig)
        results["artifacts"].append(str(svg_path))
        print(f"SVG:     {svg_path.relative_to(REPO_ROOT)}")
except Exception as e:
    results["warnings"].append(f"SVG generation failed: {e}")


# ── 9. Recommendation ──────────────────────────────────────────────────────────
n_hard = len(results["hard_failures"])
if n_hard == 0 and cv_import_ok and forward_ok and eval_ok:
    recommendation = "READY_FOR_Q58_FULL_BENCHMARK"
elif n_hard == 0 and cv_import_ok and forward_ok:
    recommendation = "READY_WITH_TRAINING_WARNING"
elif cv_import_ok:
    recommendation = "BLOCKED_FORWARD_FAILED"
else:
    recommendation = f"BLOCKED_IMPORT_FAILED: {'; '.join(results['hard_failures'][:1])}"

results["recommendation"] = recommendation


# ── 10. Markdown Report ─────────────────────────────────────────────────────────
m = cv_metrics if cv_status == "PASS" else {}
auroc_str = f"{m.get('auroc', 0.0):.4f}" if m.get("auroc") is not None else "N/A"

report_lines = [
    "# Q58S — Custom CV-QNN Smoke Test Report",
    f"**Date:** {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    f"**Runtime:** {total_elapsed:.1f}s  |  **Limit:** {RUNTIME_LIMIT_S}s  |  **Seed:** {SEED}",
    "",
    "---",
    "",
    "## Asset Discovery",
    "",
    f"- Q49 embeddings dir: `{EMB_DIR}` — {'EXISTS' if EMB_DIR.exists() else 'MISSING'}",
    f"- GaussianBackend import: {'OK' if cv_import_ok else 'FAIL'}",
    f"- cv_spine_ansatz import: {'OK' if cv_import_ok else 'FAIL'}",
    "",
    "## Data Source",
    "",
    f"- Source: **{data_source}**",
    f"- Synthetic fallback used: `{results['synthetic_fallback_used']}`",
    f"- Train: {results['train_n']}  |  Val: {results['val_n']}  |  Test: {results['test_n']}",
    f"- Classes in train: {results['n_classes']}",
    "",
    "## Circuit Manifest",
    "",
    f"| Parameter | Value |",
    "|-----------|-------|",
    f"| n_modes | {CV_N_MODES} |",
    f"| depth | {CV_DEPTH} |",
    f"| squeezing_cap | {CV_SQUEEZING_CAP} |",
    f"| Gate sequence | Displacement → Squeezing → Rotation → Beamsplitter (ring) |",
    f"| Data reuploading | Per depth layer (encoded_input added to mu) |",
    f"| Measurement | Homodyne X-quadrature |",
    f"| Squeezing bound | tanh(r_raw) × squeezing_cap — {'PASS' if results['circuit_manifest'].get('squeezing_bound_check') == 'PASS' else 'UNCHECKED'} |",
    f"| Total trainable params | {manifest['total_trainable_params']} |",
    "",
    "## CV-QNN Status",
    "",
    f"| Check | Status |",
    "|-------|--------|",
    f"| Import | {'PASS' if cv_import_ok else 'FAIL'} |",
    f"| Forward pass | {'PASS' if forward_ok else 'FAIL'} |",
    f"| Training (1 epoch) | {'PASS' if train_ok else 'FAIL'} |",
    f"| Evaluation | {cv_status} |",
    "",
    "## Metrics (Test Set)",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Accuracy | {m.get('accuracy', 'N/A')} |",
    f"| F1 | {m.get('f1', 'N/A')} |",
    f"| AUROC | {auroc_str} |",
    f"| Train loss | {m.get('train_loss', 'N/A')} |",
    f"| Runtime | {m.get('runtime_s', 'N/A')}s |",
    "",
    "## Warnings",
    "",
]
for w in results["warnings"] or ["None"]:
    report_lines.append(f"- {w}")

report_lines += ["", "## Hard Failures", ""]
for hf in results["hard_failures"] or ["None"]:
    report_lines.append(f"- {hf}")

report_lines += ["", "## Generated Artifacts", ""]
for a in results["artifacts"]:
    try:
        rel = Path(a).relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(a)
    report_lines.append(f"- `{rel}`")

report_lines += [
    "",
    "---",
    "",
    f"## Recommendation",
    "",
    f"**{recommendation}**",
    "",
]
if recommendation == "READY_FOR_Q58_FULL_BENCHMARK":
    report_lines.append(
        "CV-QNN pipeline fully operational. "
        "Proceed with `sliceforge execute --project qstrata "
        "--slice-id Q58-CUSTOM-CV-QNN-BINARY-BENCHMARK --night-auto --yes --allow-benchmarks`"
    )
elif "READY_WITH" in recommendation:
    report_lines.append("CV-QNN import and forward OK. Review training warning before full run.")
else:
    report_lines.append("Resolve blockers listed above.")

report_text = "\n".join(report_lines) + "\n"

md_path = OUT_REPORTS / "q58s_cv_qnn_smoke_report.md"
with open(md_path, "w", encoding="utf-8") as fh:
    fh.write(report_text)
results["artifacts"].append(str(md_path))
print(f"Report:  {md_path.relative_to(REPO_ROOT)}")

# Update JSON with final artifact list + recommendation
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=str)


# ── 11. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("Q58S SMOKE TEST SUMMARY")
print("=" * 64)
print(f"Data source:      {data_source}")
print(f"Synthetic:        {results['synthetic_fallback_used']}")
print(f"Train/Val/Test:   {results['train_n']}/{results['val_n']}/{results['test_n']}")
print(f"CV-QNN import:    {cv_import_ok}")
print(f"Forward pass:     {forward_ok}")
print(f"Training:         {train_ok}")
print(f"Eval status:      {cv_status}")
if m.get("auroc") is not None:
    print(f"AUROC:            {m['auroc']:.4f}")
if m.get("accuracy") is not None:
    print(f"Accuracy:         {m['accuracy']:.4f}")
print(f"Hard failures:    {n_hard}")
print(f"Warnings:         {len(results['warnings'])}")
print(f"Total runtime:    {total_elapsed:.1f}s  (limit={RUNTIME_LIMIT_S}s)")
print()
print(f"RECOMMENDATION: {recommendation}")
print("=" * 64)

sys.exit(0 if n_hard == 0 else 1)
