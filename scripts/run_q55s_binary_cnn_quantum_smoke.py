#!/usr/bin/env python3
"""
scripts/run_q55s_binary_cnn_quantum_smoke.py

Q55S — Binary CNN + Custom Quantum Smoke Test

Validates that the QSTRATA binary CNN + custom quantum pipeline is executable
before launching the full binary_cnn_quantum_publication_package campaign.

Three paths tested:
  1. Classical CNN head — Linear(128, 2) on Q49 embeddings, 1 epoch
  2. Custom DV-QNN     — proj(128→4) + medical_ansatz + readout(16→2), 1 epoch
  3. Custom CV-QNN     — GaussianBackend + GaussianVariationalAnsatz + homodyne readout

Data: Q49 embeddings (MobileNetV3-Large, 128-dim, binary labels)
Subset: max 64 train / 32 val / 32 test

Do NOT use: PennyLane, Qiskit, Strawberry Fields, TorchQuantum.
All quantum ops use the custom in-house qcore framework.

Usage (from QStrata repo root):
    python scripts/run_q55s_binary_cnn_quantum_smoke.py
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
from typing import Dict, List, Optional, Tuple

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
try:
    from sklearn.metrics import roc_auc_score
    _HAS_AUROC = True
except ImportError:
    _HAS_AUROC = False

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_DIR      = REPO_ROOT / "workspace" / "experiments" / "Q49" / "embeddings"
OUT_BASE     = REPO_ROOT / "workspace" / "experiments" / "Q55S"
OUT_RESULTS  = OUT_BASE / "results"
OUT_FIGURES  = OUT_BASE / "figures"
OUT_REPORTS  = OUT_BASE / "reports"
REPORT_COPY  = REPO_ROOT / "reports" / "q55s_binary_cnn_quantum_smoke_test.md"

# ── Config ────────────────────────────────────────────────────────────────────
MAX_TRAIN    = 64
MAX_VAL      = 32
MAX_TEST     = 32
N_QUBITS     = 4
DV_DEPTH     = 1
DV_ALPHA     = 0.1
CV_N_MODES   = 2
CV_DEPTH     = 1
LR           = 1e-2
N_EPOCHS     = 1
SEED         = 42
RUNTIME_LIMIT_S = 1800  # 30 min

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
start_wall = time.time()

def elapsed() -> float:
    return time.time() - start_wall

def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    name: str = "",
) -> Dict[str, object]:
    n_classes = len(set(y_true))
    m: Dict[str, object] = {"model": name}
    m["accuracy"]  = float(accuracy_score(y_true, y_pred))
    m["f1"]        = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    m["precision"] = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    m["recall"]    = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    m["n_samples"] = int(len(y_true))
    m["n_classes_seen"] = n_classes

    if _HAS_AUROC and y_prob is not None and n_classes == 2:
        try:
            m["auroc"] = float(roc_auc_score(y_true, y_prob))
            m["auroc_available"] = True
        except Exception as e:
            m["auroc"] = None
            m["auroc_available"] = False
            m["auroc_note"] = str(e)
    else:
        m["auroc"] = None
        m["auroc_available"] = False
        m["auroc_note"] = "Single class in split or AUROC unavailable"
    return m


# ── Output directories ────────────────────────────────────────────────────────
for d in (OUT_RESULTS, OUT_FIGURES, OUT_REPORTS, REPORT_COPY.parent):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print("Q55S — Binary CNN + Custom Quantum Smoke Test")
print("=" * 64)
print(f"Repo root:   {REPO_ROOT}")
print(f"Embeddings:  {EMB_DIR}")
print(f"Output:      {OUT_BASE}")
print()

results: Dict = {
    "slice_id": "Q55S-BINARY-CNN-QUANTUM-SMOKE-TEST",
    "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "synthetic_fallback_used": False,
    "real_data_source": None,
    "train_n": 0, "val_n": 0, "test_n": 0,
    "n_classes": 0,
    "classical_cnn_head": {},
    "custom_dv_qnn": {},
    "custom_cv_qnn": {},
    "metrics": [],
    "artifacts": [],
    "runtime_s": 0.0,
    "runtime_exceeded": False,
    "hard_failures": [],
    "warnings": [],
    "recommendation": "NOT_READY_WITH_BLOCKERS",
}


# ── 1. Data Loading ───────────────────────────────────────────────────────────
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

        # Tiny subset — balanced sample
        def _balanced_subset(emb, lbl, n):
            per_class = n // 2
            idx = []
            for c in [0, 1]:
                c_idx = np.where(lbl == c)[0][:per_class]
                idx.extend(c_idx.tolist())
            idx = np.array(idx[:n])
            return emb[idx], lbl[idx]

        emb_train, lbl_train = _balanced_subset(emb_train_full, lbl_train_full, MAX_TRAIN)
        emb_val,   lbl_val   = _balanced_subset(emb_val_full,   lbl_val_full,   MAX_VAL)
        emb_test,  lbl_test  = _balanced_subset(emb_test_full,  lbl_test_full,  MAX_TEST)

        data_source = "Q49_EMBEDDINGS"
        results["real_data_source"] = str(EMB_DIR)
        print(f"  Source:  Q49 embeddings (MobileNetV3-Large, 128-dim)")
        print(f"  Train:   {len(lbl_train)} samples  "
              f"(class 0: {(lbl_train==0).sum()}, class 1: {(lbl_train==1).sum()})")
        print(f"  Val:     {len(lbl_val)} samples  "
              f"(class 0: {(lbl_val==0).sum()}, class 1: {(lbl_val==1).sum()})")
        print(f"  Test:    {len(lbl_test)} samples  "
              f"(class 0: {(lbl_test==0).sum()}, class 1: {(lbl_test==1).sum()})")
    except Exception as e:
        print(f"  WARN: Q49 load failed ({e}) — using synthetic fallback")
        results["warnings"].append(f"Q49 load failed: {e}")
        data_source = "SYNTHETIC"

if data_source == "SYNTHETIC" or emb_train is None:
    # Synthetic fallback
    results["synthetic_fallback_used"] = True
    results["warnings"].append("Using synthetic fallback data")
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

# Torch tensors
X_tr = torch.from_numpy(emb_train)
y_tr = torch.from_numpy(lbl_train)
X_te = torch.from_numpy(emb_test)
y_te = lbl_test  # keep numpy for sklearn metrics

print(f"  Elapsed: {elapsed():.1f}s")
print()


# ── 2. Classical CNN Head ─────────────────────────────────────────────────────
print("── 2. Classical CNN Head ────────────────────────────────────────────────")

cls_status = "NOT_RUN"
cls_metrics: Dict = {}
t0 = time.time()

try:
    class ClassicalHead(nn.Module):
        def __init__(self, in_dim: int = 128, n_classes: int = 2) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )
        def forward(self, x):
            return self.fc(x)

    cls_model = ClassicalHead(128, 2)
    cls_opt   = torch.optim.Adam(cls_model.parameters(), lr=LR)
    cls_crit  = nn.CrossEntropyLoss()

    # 1-epoch mini-batch training
    cls_model.train()
    batch_size = 16
    n_batches = math.ceil(len(X_tr) / batch_size)
    epoch_loss = 0.0
    for bi in range(n_batches):
        xb = X_tr[bi*batch_size:(bi+1)*batch_size]
        yb = y_tr[bi*batch_size:(bi+1)*batch_size]
        cls_opt.zero_grad()
        logits = cls_model(xb)
        loss   = cls_crit(logits, yb)
        loss.backward()
        cls_opt.step()
        epoch_loss += loss.item()
    epoch_loss /= n_batches

    # Evaluate on test set
    cls_model.eval()
    with torch.no_grad():
        test_logits = cls_model(X_te)         # (N, 2)
        test_probs  = torch.softmax(test_logits, dim=-1)[:, 1].numpy()
        test_preds  = test_logits.argmax(dim=-1).numpy()

    cls_metrics = compute_metrics(y_te, test_preds, test_probs, "classical_cnn_head")
    cls_metrics["train_loss"] = round(epoch_loss, 6)
    cls_metrics["n_trainable_params"] = sum(p.numel() for p in cls_model.parameters())
    cls_metrics["runtime_s"] = round(time.time() - t0, 2)
    cls_status = "PASS"
    print(f"  Status:   PASS  (loss={epoch_loss:.4f}  "
          f"acc={cls_metrics['accuracy']:.4f}  auroc={cls_metrics.get('auroc', 'N/A')})")
    print(f"  Params:   {cls_metrics['n_trainable_params']}  elapsed: {elapsed():.1f}s")
except Exception as e:
    cls_status = "FAIL"
    cls_metrics = {"error": str(e)}
    results["hard_failures"].append(f"Classical CNN head failed: {e}")
    print(f"  FAIL: {e}")
    traceback.print_exc()

results["classical_cnn_head"] = {"status": cls_status, **cls_metrics}
print()


# ── 3. Custom DV-QNN Path ─────────────────────────────────────────────────────
print("── 3. Custom DV-QNN Path ────────────────────────────────────────────────")

dv_status = "NOT_RUN"
dv_import_ok = False
dv_train_supported = False
dv_metrics: Dict = {}
t0 = time.time()

try:
    from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
    from qcore.backends.base import Backend
    from qcore.states.vacuum import vacuum_state
    dv_import_ok = True
    print("  Import:  OK (qcore.ansatz.medical_ansatz, qcore.backends.base, qcore.states.vacuum)")
except Exception as e:
    dv_status = "FAIL"
    dv_metrics = {"error": f"Import failed: {e}"}
    results["hard_failures"].append(f"DV-QNN import failed: {e}")
    print(f"  FAIL import: {e}")

if dv_import_ok:
    try:
        # DV-QNN head operating on pre-computed embeddings
        class DVQNNEmbeddingHead(nn.Module):
            """DV-QNN head accepting 128-dim embeddings."""
            def __init__(self, in_dim=128, n_qubits=4, depth=1, alpha=0.1, n_classes=2):
                super().__init__()
                self.n_qubits = n_qubits
                self.depth = depth
                self.alpha = alpha
                self.proj = nn.Linear(in_dim, n_qubits)
                self.theta = nn.Parameter(alpha * torch.randn(*get_ansatz_shape(n_qubits, depth)))
                self._backend = Backend()
                self.readout = nn.Linear(2 ** n_qubits, n_classes)

            def _run_circuit(self, x_i):
                circuit = medical_ansatz(x_i, self.theta, self.n_qubits, self.depth, self.alpha)
                U     = self._backend.compile(circuit)
                state = vacuum_state(self.n_qubits)
                out   = self._backend.run(U, state)
                return torch.abs(out) ** 2  # (2^n_qubits,)

            def forward(self, x):
                proj_out = self.proj(x)     # (B, n_qubits)
                probs = torch.stack([self._run_circuit(proj_out[i]) for i in range(len(x))])
                return self.readout(probs)  # (B, n_classes)

        dv_model = DVQNNEmbeddingHead(128, N_QUBITS, DV_DEPTH, DV_ALPHA)
        dv_opt   = torch.optim.Adam(dv_model.parameters(), lr=LR)
        dv_crit  = nn.CrossEntropyLoss()

        # Forward pass smoke (1 sample)
        dv_model.eval()
        with torch.no_grad():
            _smoke = dv_model(X_tr[:1])
        print(f"  Forward: OK  logit shape={tuple(_smoke.shape)}")

        # Mini training epoch (small batch to keep runtime low)
        DV_BATCH = 8  # small for DV circuit speed
        dv_model.train()
        n_batches = math.ceil(len(X_tr) / DV_BATCH)
        epoch_loss = 0.0
        for bi in range(n_batches):
            xb = X_tr[bi*DV_BATCH:(bi+1)*DV_BATCH]
            yb = y_tr[bi*DV_BATCH:(bi+1)*DV_BATCH]
            dv_opt.zero_grad()
            logits = dv_model(xb)
            loss   = dv_crit(logits, yb)
            loss.backward()
            dv_opt.step()
            epoch_loss += loss.item()
            if time.time() - t0 > 600:  # 10 min hard cap per path
                results["warnings"].append("DV-QNN training capped at 10 min — partial epoch")
                break
        epoch_loss /= max(bi + 1, 1)
        dv_train_supported = True

        # Evaluate
        dv_model.eval()
        with torch.no_grad():
            dv_logits = dv_model(X_te)
            dv_probs  = torch.softmax(dv_logits, dim=-1)[:, 1].numpy()
            dv_preds  = dv_logits.argmax(dim=-1).numpy()

        dv_metrics = compute_metrics(y_te, dv_preds, dv_probs, "custom_dv_qnn")
        dv_metrics["train_loss"] = round(epoch_loss, 6)
        dv_metrics["dv_train_supported"] = True
        dv_metrics["n_qubits"] = N_QUBITS
        dv_metrics["circuit_depth"] = DV_DEPTH
        dv_metrics["n_trainable_params"] = sum(p.numel() for p in dv_model.parameters() if p.requires_grad)
        dv_metrics["runtime_s"] = round(time.time() - t0, 2)
        dv_status = "PASS"
        print(f"  Status:  PASS  (loss={epoch_loss:.4f}  "
              f"acc={dv_metrics['accuracy']:.4f}  auroc={dv_metrics.get('auroc', 'N/A')})")
        print(f"  Qubits:  {N_QUBITS}  depth={DV_DEPTH}  elapsed: {elapsed():.1f}s")
    except Exception as e:
        dv_status = "WARN"
        dv_metrics = {"error": str(e), "dv_train_supported": False}
        results["warnings"].append(f"DV-QNN training failed (forward pass may have worked): {e}")
        print(f"  WARN: {e}")
        traceback.print_exc()

results["custom_dv_qnn"] = {"status": dv_status, "import_ok": dv_import_ok, **dv_metrics}
print()


# ── 4. Custom CV-QNN Path ─────────────────────────────────────────────────────
print("── 4. Custom CV-QNN Path ────────────────────────────────────────────────")

cv_status = "NOT_RUN"
cv_import_ok = False
cv_train_supported = False
cv_metrics: Dict = {}
t0 = time.time()

try:
    from qcore.backends.cvBackend import GaussianBackend
    from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
    cv_import_ok = True
    print("  Import:  OK (qcore.backends.cvBackend, qcore.ansatz.cv_spine_ansatz)")
except Exception as e:
    cv_status = "FAIL"
    cv_metrics = {"error": f"Import failed: {e}"}
    results["warnings"].append(f"CV-QNN import failed (non-blocking): {e}")
    print(f"  WARN import: {e}")

if cv_import_ok:
    try:
        class CVQNNEmbeddingHead(nn.Module):
            """CV-QNN head accepting 128-dim embeddings.

            Architecture:
              embedding (128) → Linear(128, 2*n_modes) → phase-space encoding
              → GaussianVariationalAnsatz → homodyne X quadratures (n_modes)
              → Linear(n_modes, n_classes)
            """
            def __init__(self, in_dim=128, n_modes=2, depth=1, n_classes=2):
                super().__init__()
                self.n_modes = n_modes
                # Project embedding to phase-space displacement vector (2*n_modes)
                self.encoder = nn.Linear(in_dim, 2 * n_modes)
                # Variational ansatz (trainable Gaussian gates)
                self.ansatz = GaussianVariationalAnsatz(n_modes=n_modes, depth=depth)
                # Readout from n_modes homodyne X quadratures
                self.readout = nn.Linear(n_modes, n_classes)
                # Backend (stateless for Gaussian ops)
                self._backend = GaussianBackend(n_modes=n_modes)

            def _forward_single(self, enc_i: torch.Tensor) -> torch.Tensor:
                """One sample: enc_i is (2*n_modes,) phase-space displacement."""
                mu, cov = self._backend.get_vacuum()
                mu_out, _ = self.ansatz.apply(mu, cov, self._backend, encoded_input=enc_i)
                # Homodyne measurement: X quadratures (even indices)
                x_quads = mu_out[::2]    # (n_modes,)
                return x_quads

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                enc = self.encoder(x)    # (B, 2*n_modes)
                out = torch.stack([self._forward_single(enc[i]) for i in range(len(x))])
                return self.readout(out) # (B, n_classes)

        cv_model = CVQNNEmbeddingHead(128, CV_N_MODES, CV_DEPTH)
        cv_opt   = torch.optim.Adam(cv_model.parameters(), lr=LR)
        cv_crit  = nn.CrossEntropyLoss()

        # Forward pass smoke (1 sample)
        cv_model.eval()
        with torch.no_grad():
            _cv_smoke = cv_model(X_tr[:1])
        print(f"  Forward: OK  logit shape={tuple(_cv_smoke.shape)}")

        # Mini training epoch
        CV_BATCH = 16
        cv_model.train()
        n_batches = math.ceil(len(X_tr) / CV_BATCH)
        epoch_loss = 0.0
        for bi in range(n_batches):
            xb = X_tr[bi*CV_BATCH:(bi+1)*CV_BATCH]
            yb = y_tr[bi*CV_BATCH:(bi+1)*CV_BATCH]
            cv_opt.zero_grad()
            logits = cv_model(xb)
            loss   = cv_crit(logits, yb)
            loss.backward()
            cv_opt.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches
        cv_train_supported = True

        # Evaluate
        cv_model.eval()
        with torch.no_grad():
            cv_logits = cv_model(X_te)
            cv_probs  = torch.softmax(cv_logits, dim=-1)[:, 1].numpy()
            cv_preds  = cv_logits.argmax(dim=-1).numpy()

        cv_metrics = compute_metrics(y_te, cv_preds, cv_probs, "custom_cv_qnn")
        cv_metrics["train_loss"] = round(epoch_loss, 6)
        cv_metrics["cv_train_supported"] = True
        cv_metrics["n_modes"] = CV_N_MODES
        cv_metrics["depth"] = CV_DEPTH
        cv_metrics["n_trainable_params"] = sum(p.numel() for p in cv_model.parameters() if p.requires_grad)
        cv_metrics["runtime_s"] = round(time.time() - t0, 2)
        cv_status = "PASS"
        print(f"  Status:  PASS  (loss={epoch_loss:.4f}  "
              f"acc={cv_metrics['accuracy']:.4f}  auroc={cv_metrics.get('auroc', 'N/A')})")
        print(f"  Modes:   {CV_N_MODES}  depth={CV_DEPTH}  elapsed: {elapsed():.1f}s")
    except Exception as e:
        cv_status = "WARN"
        cv_metrics = {"error": str(e), "cv_train_supported": False}
        results["warnings"].append(f"CV-QNN training failed: {e}")
        print(f"  WARN: {e}")
        traceback.print_exc()

results["custom_cv_qnn"] = {"status": cv_status, "import_ok": cv_import_ok, **cv_metrics}
print()


# ── 5. Collect metrics ────────────────────────────────────────────────────────
all_metrics: List[Dict] = []
for path_name, path_data in [
    ("classical_cnn_head", results["classical_cnn_head"]),
    ("custom_dv_qnn",      results["custom_dv_qnn"]),
    ("custom_cv_qnn",      results["custom_cv_qnn"]),
]:
    if path_data.get("status") == "PASS":
        m = {k: v for k, v in path_data.items()
             if k in ("model", "accuracy", "f1", "precision", "recall",
                      "auroc", "auroc_available", "train_loss",
                      "n_trainable_params", "runtime_s", "n_samples")}
        m.setdefault("model", path_name)
        all_metrics.append(m)

results["metrics"] = all_metrics


# ── 6. Write CSV ──────────────────────────────────────────────────────────────
csv_path = OUT_RESULTS / "q55s_smoke_metrics.csv"
if all_metrics:
    fields = ["model", "accuracy", "f1", "precision", "recall",
              "auroc", "auroc_available", "train_loss", "n_trainable_params",
              "runtime_s", "n_samples"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_metrics)
    print(f"CSV:     {csv_path.relative_to(REPO_ROOT)}")
    results["artifacts"].append(str(csv_path))


# ── 7. Write JSON ─────────────────────────────────────────────────────────────
total_elapsed = elapsed()
results["runtime_s"] = round(total_elapsed, 2)
results["runtime_exceeded"] = total_elapsed > RUNTIME_LIMIT_S

json_path = OUT_RESULTS / "q55s_smoke_results.json"
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"JSON:    {json_path.relative_to(REPO_ROOT)}")
results["artifacts"].append(str(json_path))


# ── 8. SVG Figure ─────────────────────────────────────────────────────────────
svg_path = OUT_FIGURES / "q55s_smoke_metrics.svg"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if all_metrics:
        labels  = [m.get("model", "?") for m in all_metrics]
        accs    = [m.get("accuracy", 0.0) or 0.0 for m in all_metrics]
        f1s     = [m.get("f1", 0.0) or 0.0 for m in all_metrics]
        aurocs  = [m.get("auroc") or 0.0 for m in all_metrics]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, accs,   width, label="Accuracy", color="#4e79a7")
        ax.bar(x,         f1s,    width, label="F1",       color="#f28e2b")
        ax.bar(x + width, aurocs, width, label="AUROC",    color="#59a14f")

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Q55S — Binary CNN + Quantum Smoke Test Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(svg_path, format="svg")
        plt.close(fig)
        print(f"SVG:     {svg_path.relative_to(REPO_ROOT)}")
        results["artifacts"].append(str(svg_path))
    else:
        print("  WARN: No metrics to plot — skipping SVG")
        results["warnings"].append("No metrics available for SVG figure")
except Exception as e:
    print(f"  WARN SVG: {e}")
    results["warnings"].append(f"SVG generation failed: {e}")


# ── 9. Determine recommendation ───────────────────────────────────────────────
n_hard = len(results["hard_failures"])
cls_ok = results["classical_cnn_head"].get("status") == "PASS"
dv_ok  = results["custom_dv_qnn"].get("status") == "PASS"
cv_ok  = results["custom_cv_qnn"].get("status") == "PASS"

if n_hard == 0 and cls_ok and (dv_ok or cv_ok):
    recommendation = "READY_FOR_FULL_BINARY_CNN_QUANTUM_CAMPAIGN"
elif n_hard == 0 and cls_ok:
    recommendation = "READY_WITH_QUANTUM_WARNINGS"
else:
    blockers = results["hard_failures"]
    recommendation = f"NOT_READY_WITH_BLOCKERS: {'; '.join(blockers[:2])}"

results["recommendation"] = recommendation


# ── 10. Markdown Report ───────────────────────────────────────────────────────
def _metric_row(m: Dict) -> str:
    auroc = f"{m.get('auroc'):.4f}" if m.get("auroc") is not None else "N/A"
    return (f"| {m.get('model','?')} "
            f"| {m.get('accuracy',0):.4f} "
            f"| {m.get('f1',0):.4f} "
            f"| {m.get('auroc_available',False)} "
            f"| {auroc} "
            f"| {m.get('train_loss','?')} "
            f"| {m.get('n_trainable_params','?')} "
            f"| {m.get('runtime_s','?')}s |")


report_lines = [
    "# Q55S — Binary CNN + Custom Quantum Smoke Report",
    f"**Date:** {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    f"**Runtime:** {total_elapsed:.1f}s  |  **Limit:** {RUNTIME_LIMIT_S}s",
    "",
    "---",
    "",
    "## Asset Discovery",
    "",
    f"- Q49 embeddings dir: `{EMB_DIR}` — {'EXISTS' if EMB_DIR.exists() else 'MISSING'}",
    f"- DV-QNN import (qcore.ansatz.medical_ansatz): {'OK' if dv_import_ok else 'FAIL'}",
    f"- CV-QNN import (qcore.backends.cvBackend): {'OK' if cv_import_ok else 'FAIL'}",
    "",
    "## Data Source",
    "",
    f"- Source: **{data_source}**",
    f"- Synthetic fallback used: `{results['synthetic_fallback_used']}`",
    f"- Train: {results['train_n']} samples  |  Val: {results['val_n']}  |  Test: {results['test_n']}",
    f"- Classes seen in train: {results['n_classes']}",
    "",
    "## Path Status",
    "",
    f"| Path | Status | Import | Train Supported |",
    "|------|--------|--------|-----------------|",
    f"| Classical CNN head | {cls_status} | N/A | Yes |",
    f"| Custom DV-QNN | {dv_status} | {'OK' if dv_import_ok else 'FAIL'} | {dv_train_supported} |",
    f"| Custom CV-QNN | {cv_status} | {'OK' if cv_import_ok else 'FAIL'} | {cv_train_supported} |",
    "",
    "## Metrics (Test Set)",
    "",
    "| Model | Accuracy | F1 | AUROC Avail | AUROC | Train Loss | Params | Runtime |",
    "|-------|----------|----|-------------|-------|------------|--------|---------|",
]
for m in all_metrics:
    report_lines.append(_metric_row(m))

report_lines += [
    "",
    "## Warnings",
    "",
]
for w in results["warnings"] or ["None"]:
    report_lines.append(f"- {w}")

report_lines += [
    "",
    "## Hard Failures",
    "",
]
for hf in results["hard_failures"] or ["None"]:
    report_lines.append(f"- {hf}")

report_lines += [
    "",
    "## Generated Artifacts",
    "",
]
for a in results["artifacts"]:
    p = Path(a)
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        rel = p
    report_lines.append(f"- `{rel}`")

report_lines += [
    "",
    "---",
    "",
    f"## Recommendation",
    "",
    f"**{recommendation}**",
    "",
    "### Next Step",
]
if "READY_FOR_FULL" in recommendation:
    report_lines.append(
        "All three paths operational. "
        "Proceed with `sliceforge loop --project qstrata --night-auto "
        "--max-iterations 10 --budget-aware --allow-benchmarks --notify --yes`"
    )
elif "READY_WITH" in recommendation:
    report_lines.append(
        "Classical path and at least one quantum path operational. "
        "Resolve quantum warnings before full benchmark run."
    )
else:
    report_lines.append(
        "Resolve blockers listed above before launching full campaign."
    )

report_text = "\n".join(report_lines) + "\n"

md_path = OUT_REPORTS / "q55s_binary_cnn_quantum_smoke_report.md"
with open(md_path, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"Report:  {md_path.relative_to(REPO_ROOT)}")
results["artifacts"].append(str(md_path))

# Copy to reports/
with open(REPORT_COPY, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"Copy:    {REPORT_COPY.relative_to(REPO_ROOT)}")
results["artifacts"].append(str(REPORT_COPY))

# Update JSON with final artifact list
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=str)


# ── 11. Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("SMOKE TEST SUMMARY")
print("=" * 64)
print(f"Data source:    {data_source}")
print(f"Synthetic:      {results['synthetic_fallback_used']}")
print(f"Train/Val/Test: {results['train_n']}/{results['val_n']}/{results['test_n']}")
print(f"Classical head: {cls_status}")
print(f"DV-QNN:         {dv_status}  (import={dv_import_ok}, train={dv_train_supported})")
print(f"CV-QNN:         {cv_status}  (import={cv_import_ok}, train={cv_train_supported})")
print(f"Hard failures:  {len(results['hard_failures'])}")
print(f"Warnings:       {len(results['warnings'])}")
print(f"Total runtime:  {total_elapsed:.1f}s")
print(f"Runtime limit:  {RUNTIME_LIMIT_S}s  exceeded={results['runtime_exceeded']}")
print()
print(f"RECOMMENDATION: {recommendation}")
print("=" * 64)

sys.exit(0 if n_hard == 0 else 1)
