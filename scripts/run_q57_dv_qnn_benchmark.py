"""
scripts/run_q57_dv_qnn_benchmark.py

Q57 — Custom DV-QNN Binary Benchmark.

Evaluates the custom in-house DV-QNN (Discrete Variable Quantum Neural Network)
framework for binary classification on VinDr-SpineXR and BUU-LSPINE.

Architecture:
  MobileNetV3-Large (frozen, ImageNet pretrained)
    → frozen linear projection 960→128 (random init, seed=42)
    → DV-QNN head:
        proj     Linear(128→4)              [trainable, 516 params]
        theta    (depth=1, 2, n_qubits=4, 3) [trainable,  24 params]
        ansatz   medical_ansatz DV circuit
        readout  Linear(16→2)               [trainable,  34 params]
  Trainable DV-QNN head params: 574

Custom in-house quantum framework only — no PennyLane, Qiskit, TorchQuantum.

Datasets:
  VinDr-SpineXR  — Q49 pre-computed 128-dim embeddings
  BUU-LSPINE     — synthetic procedural spine ROIs (same generator as Q56)

Decision rule:
  Same datasets, splits, and seeds as Q56 [42, 7, 123].
  Baseline established when mean AUROC reported across 3 seeds per dataset.

Usage:
  python scripts/run_q57_dv_qnn_benchmark.py --dry-run
  python scripts/run_q57_dv_qnn_benchmark.py --smoke
  python scripts/run_q57_dv_qnn_benchmark.py --full

Runtime contract:
  --smoke : 1 seed × 2 datasets × 4 epochs  (~5–15 min; DV circuit is CPU-bound)
  --full  : 3 seeds × 2 datasets × 4 epochs  (~15–45 min depending on hardware)
  --dry-run: Validates imports and paths; no training

Device note:
  DV quantum circuit simulator is CPU-only. Backbone used for BUU feature
  extraction runs on GPU if available; QNN head training runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────

SLICE_ID = "Q57"

SMOKE_SEEDS = [42]
FULL_SEEDS  = [42, 7, 123]

# DV-QNN circuit hyperparameters
N_QUBITS  = 4
DV_DEPTH  = 1
DV_ALPHA  = 0.1

EPOCHS       = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 8   # small batch for DV circuit throughput

CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

FEAT_DIM_RAW     = 960   # MobileNetV3-Large avgpool output
BACKBONE_OUT_DIM = 128   # frozen projection output

MOBILENET_LARGE_BACKBONE_PARAMS = 5_483_032
PROJECTION_PARAMS               = FEAT_DIM_RAW * BACKBONE_OUT_DIM + BACKBONE_OUT_DIM  # 123,008

# DV-QNN head param counts (n_qubits=4, depth=1)
QNN_PROJ_PARAMS    = BACKBONE_OUT_DIM * N_QUBITS + N_QUBITS   # 128*4+4    = 516
QNN_THETA_PARAMS   = DV_DEPTH * 2 * N_QUBITS * 3              # 1*2*4*3   = 24
QNN_READOUT_PARAMS = (2 ** N_QUBITS) * 2 + 2                  # 16*2+2    = 34
QNN_HEAD_TRAINABLE = QNN_PROJ_PARAMS + QNN_THETA_PARAMS + QNN_READOUT_PARAMS  # 574

# Baseline reference values
Q46_WINNER_AUROC  = 0.987344   # MobileNetV3-Large full pipeline (Q46 winner)
Q46_WINNER_F1     = 0.933100
Q56_VINDR_AUROC   = 0.973061   # Q56 classical CNN baseline (3-seed mean, val set)
Q56_BUU_AUROC     = 1.000000   # Q56 classical CNN baseline on synthetic BUU

# BUU-LSPINE synthetic dataset config (mirrors Q56)
BUU_TRAIN_N_PER_CLASS = 500
BUU_VAL_N_PER_CLASS   = 100
BUU_TEST_N_PER_CLASS  = 100

# Paths
Q49_EMBEDDING_DIR = "workspace/experiments/Q49/embeddings"
RESULTS_DIR       = "workspace/experiments/Q57/results"
LEADERBOARD_PATH  = "workspace/projects/qstrata/leaderboards/q57-dv-qnn.csv"
VALIDATION_REPORT = "workspace/experiments/Q57/validation_report.md"

LEADERBOARD_FIELDS = [
    "rank", "dataset", "seed", "auroc", "f1", "accuracy",
    "n_qubits", "circuit_depth", "alpha",
    "params_backbone", "params_projection", "params_qnn_head", "params_total",
    "runtime_s",
    "delta_auroc_vs_q46winner",
    "delta_auroc_vs_q56baseline",
]

SUMMARY_FIELDS = [
    "rank", "dataset", "row_type",
    "mean_auroc", "std_auroc", "ci95_lo_auroc", "ci95_hi_auroc",
    "mean_f1",   "std_f1",   "ci95_lo_f1",   "ci95_hi_f1",
    "mean_accuracy",
    "n_qubits", "circuit_depth", "params_total", "decision",
]


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# ── DV-QNN Embedding Head ─────────────────────────────────────────────────────

def build_dv_qnn_head(seed: int, in_dim: int = BACKBONE_OUT_DIM):
    """
    Build a fresh DVQNNEmbeddingHead for one (dataset, seed) run.

    Architecture:
      in_dim → proj Linear(in_dim, n_qubits) → medical_ansatz → readout Linear(2^n, 2)

    Operates directly on pre-computed embeddings (no CNN backbone needed).
    Device: CPU — quantum simulator is CPU-only.
    """
    import torch
    import torch.nn as nn
    from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
    from qcore.backends.base import Backend
    from qcore.states.vacuum import vacuum_state

    class _DVQNNEmbeddingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_qubits = N_QUBITS
            self.depth    = DV_DEPTH
            self.alpha    = DV_ALPHA

            # Trainable: projection + theta + readout (574 params total)
            self.proj    = nn.Linear(in_dim, N_QUBITS)
            self.theta   = nn.Parameter(DV_ALPHA * torch.randn(*get_ansatz_shape(N_QUBITS, DV_DEPTH)))
            self._backend = Backend()
            self.readout = nn.Linear(2 ** N_QUBITS, 2)

        def _run_circuit(self, x_i: "torch.Tensor") -> "torch.Tensor":
            circuit = medical_ansatz(x_i, self.theta, self.n_qubits, self.depth, self.alpha)
            U       = self._backend.compile(circuit)
            state   = vacuum_state(self.n_qubits)
            out     = self._backend.run(U, state)
            return torch.abs(out) ** 2  # (2^n_qubits,)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            proj_out   = self.proj(x)   # (B, n_qubits)
            probs      = torch.stack([self._run_circuit(proj_out[i]) for i in range(len(x))])
            return self.readout(probs)  # (B, 2)

    set_seeds(seed)
    return _DVQNNEmbeddingHead()


# ── CI95 ──────────────────────────────────────────────────────────────────────

def ci95(values: List[float]) -> Tuple[float, float, float, float]:
    """Return (mean, std, lo, hi) using t-distribution CI95."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if n == 1:
        return values[0], 0.0, values[0], values[0]
    arr  = np.array(values, dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1))
    se   = std / math.sqrt(n)
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return mean, std, lo, hi


# ── Synthetic BUU-LSPINE ROI (mirrors Q56) ───────────────────────────────────

def synth_roi(seed: int, label: int, sz: int = 224) -> np.ndarray:
    """Procedural spine ROI image. Returns uint8 [sz, sz]."""
    rng = np.random.RandomState(seed)
    img = np.zeros((sz, sz), dtype=np.float32)
    cy = sz // 2 + rng.randint(-15, 15)
    cx = sz // 2 + rng.randint(-15, 15)
    ry = rng.randint(45, 68)
    rx = rng.randint(55, 80)
    Y, X = np.ogrid[:sz, :sz]
    d = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
    img += np.clip(35 + 25 * np.exp(-d * 0.3), 0, 80).astype(np.float32)
    cm = (d <= 1.05) & (d >= 0.82)
    nc = int(cm.sum())
    if nc:
        img[cm] = np.clip(195 + rng.randint(-12, 18, size=(nc,)), 0, 255)
    tm = d < 0.82
    img[tm] = np.clip((90 + rng.normal(0, 12, (sz, sz)).astype(np.float32))[tm], 60, 155)
    if label == 1:
        for _ in range(rng.randint(1, 4)):
            a = rng.uniform(0, 6.283)
            ex = int(cx + rx * float(np.cos(a)))
            ey = int(cy + ry * float(np.sin(a)))
            sr = rng.randint(6, 18)
            sm = (Y - ey) ** 2 + (X - ex) ** 2 < sr ** 2
            ns = int(sm.sum())
            if ns:
                img[sm] = np.clip(210 + rng.randint(-15, 25, size=(ns,)), 0, 255)
        by0 = cy + ry + rng.randint(2, 8)
        bm  = (Y >= by0) & (Y <= by0 + rng.randint(4, 10)) & (d < 2.5)
        img[bm] = np.clip(img[bm] * 0.55, 0, 255)
    img += rng.normal(0, 6, (sz, sz)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def build_buu_split(n_per_class: int, seed_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic BUU-LSPINE images → (imgs uint8 [N,224,224], labels int [N])."""
    imgs, labels = [], []
    for cls in range(2):
        for i in range(n_per_class):
            img = synth_roi(seed=seed_offset + cls * n_per_class + i, label=cls)
            imgs.append(img)
            labels.append(cls)
    idx = np.arange(len(imgs))
    rng = np.random.RandomState(seed_offset)
    rng.shuffle(idx)
    return np.stack([imgs[i] for i in idx]), np.array([labels[i] for i in idx])


# ── BUU-LSPINE feature extraction (mirrors Q56) ───────────────────────────────

def extract_buu_features(
    imgs_uint8: np.ndarray,
    backbone,
    projection,
    device,
) -> np.ndarray:
    """
    imgs_uint8: [N, 224, 224] uint8
    Returns float32 numpy [N, 128].
    backbone+projection run on device (GPU if available).
    """
    import torch
    import torchvision.transforms as T
    from run_q38a_preprocessing_benchmark import _clahe_torch

    transform = T.Compose([
        T.Lambda(lambda x: _clahe_torch(x, CLAHE_CLIP, CLAHE_TILE)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_emb = []
    backbone.eval()
    projection.eval()
    with torch.no_grad():
        for i in range(0, len(imgs_uint8), 32):
            batch = imgs_uint8[i:i + 32]
            tensors = []
            for img in batch:
                t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
                tensors.append(transform(t))
            x = torch.stack(tensors).to(device)
            emb = projection(backbone(x))
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ── Embedding dataset ─────────────────────────────────────────────────────────

def make_embedding_loaders(
    train_emb: np.ndarray, train_lbl: np.ndarray,
    val_emb: np.ndarray,   val_lbl: np.ndarray,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _ds(emb, lbl):
        return TensorDataset(
            torch.from_numpy(emb.astype(np.float32)),
            torch.from_numpy(lbl.astype(np.int64)),
        )

    train_loader = DataLoader(_ds(train_emb, train_lbl),
                              batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(_ds(val_emb,   val_lbl),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_head(model, loader, criterion, device) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    has_both = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


# ── Train one (dataset, seed) run ─────────────────────────────────────────────

def run_one(
    dataset_id: str,
    seed: int,
    q49_dir: str,
    buu_features_cache: dict,
) -> dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # DV-QNN head runs on CPU (quantum simulator constraint)
    device = torch.device("cpu")

    set_seeds(seed)

    if dataset_id == "vindr_spinexr":
        train_emb = np.load(os.path.join(q49_dir, "train_embeddings.npy"))
        train_lbl = np.load(os.path.join(q49_dir, "train_labels.npy"))
        val_emb   = np.load(os.path.join(q49_dir, "val_embeddings.npy"))
        val_lbl   = np.load(os.path.join(q49_dir, "val_labels.npy"))
        q56_ref   = Q56_VINDR_AUROC
    else:
        train_emb = buu_features_cache["train_emb"]
        train_lbl = buu_features_cache["train_lbl"]
        val_emb   = buu_features_cache["val_emb"]
        val_lbl   = buu_features_cache["val_lbl"]
        q56_ref   = Q56_BUU_AUROC

    train_loader, val_loader = make_embedding_loaders(
        train_emb, train_lbl, val_emb, val_lbl
    )

    # Fresh DV-QNN head per seed
    head      = build_dv_qnn_head(seed=seed).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.time()
    for epoch in range(EPOCHS):
        head.train()
        epoch_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            epoch_batches += 1
        print(f"      epoch {epoch + 1}/{EPOCHS}  batches={epoch_batches}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    runtime_s = time.time() - t0
    metrics   = eval_head(head, val_loader, criterion, device)

    p_total = (
        MOBILENET_LARGE_BACKBONE_PARAMS
        + PROJECTION_PARAMS
        + QNN_HEAD_TRAINABLE
    )

    return {
        "dataset":                  dataset_id,
        "seed":                     seed,
        "auroc":                    round(metrics["auroc"],    6),
        "f1":                       round(metrics["f1"],       6),
        "accuracy":                 round(metrics["accuracy"], 6),
        "n_qubits":                 N_QUBITS,
        "circuit_depth":            DV_DEPTH,
        "alpha":                    DV_ALPHA,
        "params_backbone":          MOBILENET_LARGE_BACKBONE_PARAMS,
        "params_projection":        PROJECTION_PARAMS,
        "params_qnn_head":          QNN_HEAD_TRAINABLE,
        "params_total":             p_total,
        "runtime_s":                round(runtime_s, 2),
        "delta_auroc_vs_q46winner":  round(metrics["auroc"] - Q46_WINNER_AUROC, 6),
        "delta_auroc_vs_q56baseline": round(metrics["auroc"] - q56_ref, 6),
    }


# ── MobileNetV3-Large backbone (for BUU feature extraction) ───────────────────

def build_mobilenet_extractor(seed: int, device):
    import torch
    import torch.nn as nn
    from torchvision import models

    full_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    backbone   = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    set_seeds(seed)
    projection = nn.Linear(FEAT_DIM_RAW, BACKBONE_OUT_DIM)
    for p in projection.parameters():
        p.requires_grad = False
    projection.eval()

    return backbone.to(device), projection.to(device)


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_per_seed_csv(rows: List[dict], path: str) -> None:
    ranked = sorted(rows, key=lambda r: r["auroc"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(ranked)


def write_summary_csv(all_rows: List[dict], path: str) -> None:
    from collections import defaultdict
    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for r in all_rows:
        by_dataset[r["dataset"]].append(r)

    summary_rows = []
    for ds_id, rows in by_dataset.items():
        aurocs     = [r["auroc"]    for r in rows]
        f1s        = [r["f1"]       for r in rows]
        accuracies = [r["accuracy"] for r in rows]
        ma, sa, la, ha = ci95(aurocs)
        mf, sf, lf, hf = ci95(f1s)
        decision = "WINNER" if ma > 0.7 else "BELOW_THRESHOLD"
        summary_rows.append({
            "rank":           "",
            "dataset":        ds_id,
            "row_type":       "summary",
            "mean_auroc":     round(ma, 6),
            "std_auroc":      round(sa, 6),
            "ci95_lo_auroc":  round(la, 6),
            "ci95_hi_auroc":  round(ha, 6),
            "mean_f1":        round(mf, 6),
            "std_f1":         round(sf, 6),
            "ci95_lo_f1":     round(lf, 6),
            "ci95_hi_f1":     round(hf, 6),
            "mean_accuracy":  round(float(np.mean(accuracies)), 6),
            "n_qubits":       N_QUBITS,
            "circuit_depth":  DV_DEPTH,
            "params_total":   rows[0]["params_total"],
            "decision":       decision,
        })

    summary_rows.sort(key=lambda r: r["mean_auroc"], reverse=True)
    for i, r in enumerate(summary_rows, 1):
        r["rank"] = i

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


# ── Dry-run validation ────────────────────────────────────────────────────────

def run_dry_run(project_root: str) -> bool:
    print("=" * 70)
    print("Q57 Custom DV-QNN Binary Benchmark — DRY-RUN VALIDATION")
    print("=" * 70)

    all_ok = True

    checks = [
        ("Q49 embeddings dir",   os.path.join(project_root, Q49_EMBEDDING_DIR)),
        ("train_embeddings.npy", os.path.join(project_root, Q49_EMBEDDING_DIR, "train_embeddings.npy")),
        ("val_embeddings.npy",   os.path.join(project_root, Q49_EMBEDDING_DIR, "val_embeddings.npy")),
        ("test_embeddings.npy",  os.path.join(project_root, Q49_EMBEDDING_DIR, "test_embeddings.npy")),
    ]
    print("\n[1] File dependencies:")
    for name, path in checks:
        ok = os.path.exists(path)
        sym = "+" if ok else "x"
        print(f"  [{sym}] {name:<30} {path}")
        if not ok:
            all_ok = False

    print("\n[2] Quantum framework imports:")
    qcore_imports = [
        ("qcore.ansatz.medical_ansatz", "medical_ansatz, get_ansatz_shape"),
        ("qcore.backends.base",         "Backend"),
        ("qcore.states.vacuum",         "vacuum_state"),
    ]
    torch_missing = False
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_missing = True
    for module, symbols in qcore_imports:
        try:
            __import__(module, fromlist=symbols.split(", "))
            print(f"  [+] {module}")
        except ImportError as e:
            err_str = str(e)
            if torch_missing and "torch" in err_str:
                # qcore modules depend on torch; expect this outside docker
                print(f"  [~] {module}  — requires docker-qstrata-gpu-1 (torch absent)")
            else:
                print(f"  [x] {module}  — {e}")
                all_ok = False

    print("\n[3] General imports:")
    import_failures = []
    for pkg in ["torch", "torchvision", "sklearn", "numpy"]:
        try:
            __import__(pkg)
            print(f"  [+] {pkg}")
        except ImportError:
            print(f"  [~] {pkg}  (requires docker-qstrata-gpu-1)")
            import_failures.append(pkg)

    docker_only = {"torch", "torchvision", "sklearn"}
    if import_failures and not (set(import_failures) - docker_only):
        print("  -> torch/torchvision/sklearn require docker-qstrata-gpu-1")
    elif import_failures:
        all_ok = False

    print("\n[4] Output paths (created if missing):")
    for rel in [RESULTS_DIR, os.path.dirname(LEADERBOARD_PATH), os.path.dirname(VALIDATION_REPORT)]:
        abs_p = os.path.join(project_root, rel)
        os.makedirs(abs_p, exist_ok=True)
        print(f"  [+] {abs_p}")

    print("\n[5] DV-QNN architecture:")
    total_p = MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + QNN_HEAD_TRAINABLE
    print(f"  N_QUBITS               = {N_QUBITS}")
    print(f"  DV_DEPTH               = {DV_DEPTH}")
    print(f"  DV_ALPHA               = {DV_ALPHA}")
    print(f"  QNN proj params        = {QNN_PROJ_PARAMS:,}  (Linear {BACKBONE_OUT_DIM}->{N_QUBITS})")
    print(f"  QNN theta params       = {QNN_THETA_PARAMS:,}  (ansatz depth×2×qubits×3)")
    print(f"  QNN readout params     = {QNN_READOUT_PARAMS:,}  (Linear {2**N_QUBITS}->2)")
    print(f"  QNN head trainable     = {QNN_HEAD_TRAINABLE:,}")
    print(f"  Backbone (frozen)      = {MOBILENET_LARGE_BACKBONE_PARAMS:,}")
    print(f"  Projection (frozen)    = {PROJECTION_PARAMS:,}")
    print(f"  Total params           = {total_p:,}")
    print(f"  FULL_SEEDS             = {FULL_SEEDS}")
    print(f"  EPOCHS                 = {EPOCHS}")
    print(f"  BATCH_SIZE             = {BATCH_SIZE}")
    print(f"  Q46_WINNER_AUROC ref   = {Q46_WINNER_AUROC}")
    print(f"  Q56_VINDR_AUROC ref    = {Q56_VINDR_AUROC}")

    if all_ok and import_failures:
        status = "READY_WITH_WARNINGS"
    elif all_ok:
        status = "READY"
    else:
        status = "BLOCKED"
    print(f"\n  Status: {status}")
    print("=" * 70)
    return all_ok


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(seeds: List[int], project_root: str) -> None:
    import torch

    # Feature extraction uses GPU if available; QNN head always on CPU
    feat_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnn_device  = torch.device("cpu")
    print(f"  Feature extraction device : {feat_device}")
    print(f"  QNN training device       : {qnn_device}  (quantum simulator CPU-only)")

    q49_dir = os.path.join(project_root, Q49_EMBEDDING_DIR)

    # Build MobileNetV3-Large for BUU-LSPINE feature extraction
    print("  Loading MobileNetV3-Large backbone for BUU feature extraction...")
    backbone, projection = build_mobilenet_extractor(seed=42, device=feat_device)

    # Pre-extract BUU-LSPINE features once (same for all seeds, mirrors Q56)
    print("  Generating BUU-LSPINE synthetic data and extracting features...")
    train_imgs, train_lbl = build_buu_split(BUU_TRAIN_N_PER_CLASS, seed_offset=1000)
    val_imgs,   val_lbl   = build_buu_split(BUU_VAL_N_PER_CLASS,   seed_offset=2000)

    train_emb_buu = extract_buu_features(train_imgs, backbone, projection, feat_device)
    val_emb_buu   = extract_buu_features(val_imgs,   backbone, projection, feat_device)

    buu_features_cache = {
        "train_emb": train_emb_buu, "train_lbl": train_lbl,
        "val_emb":   val_emb_buu,   "val_lbl":   val_lbl,
    }
    print(f"  BUU-LSPINE features: train={train_emb_buu.shape}, val={val_emb_buu.shape}")

    all_rows:   List[dict] = []
    vindr_rows: List[dict] = []
    buu_rows:   List[dict] = []

    datasets = ["vindr_spinexr", "buu_lspine"]

    for dataset_id in datasets:
        for seed in seeds:
            label = "VinDr-SpineXR" if dataset_id == "vindr_spinexr" else "BUU-LSPINE (synthetic)"
            print(f"\n  -> {label}  seed={seed}")
            row = run_one(
                dataset_id=dataset_id,
                seed=seed,
                q49_dir=q49_dir,
                buu_features_cache=buu_features_cache,
            )
            all_rows.append(row)
            if dataset_id == "vindr_spinexr":
                vindr_rows.append(row)
            else:
                buu_rows.append(row)
            print(f"    AUROC={row['auroc']:.4f}  F1={row['f1']:.4f}  "
                  f"dQ46={row['delta_auroc_vs_q46winner']:+.4f}  "
                  f"dQ56={row['delta_auroc_vs_q56baseline']:+.4f}  "
                  f"runtime={row['runtime_s']:.1f}s")

    # Write per-seed CSVs
    results_abs = os.path.join(project_root, RESULTS_DIR)
    os.makedirs(results_abs, exist_ok=True)
    write_per_seed_csv(all_rows,   os.path.join(results_abs, "metrics.csv"))
    write_per_seed_csv(vindr_rows, os.path.join(results_abs, "vindr_metrics.csv"))
    write_per_seed_csv(buu_rows,   os.path.join(results_abs, "buu_lspine_metrics.csv"))

    # Write leaderboard summary CSV
    lb_abs = os.path.join(project_root, LEADERBOARD_PATH)
    write_summary_csv(all_rows, lb_abs)

    # Compute per-dataset summaries for report
    v_aurocs = [r["auroc"] for r in vindr_rows]
    b_aurocs = [r["auroc"] for r in buu_rows]
    v_f1s    = [r["f1"]    for r in vindr_rows]
    b_f1s    = [r["f1"]    for r in buu_rows]

    v_ma, v_sa, v_la, v_ha = ci95(v_aurocs)
    b_ma, b_sa, b_la, b_ha = ci95(b_aurocs)
    v_mf, _,    v_lf, v_hf = ci95(v_f1s)
    b_mf, _,    b_lf, b_hf = ci95(b_f1s)

    # Write validation report
    report_abs = os.path.join(project_root, VALIDATION_REPORT)
    os.makedirs(os.path.dirname(report_abs), exist_ok=True)
    _write_validation_report(
        report_abs, seeds, vindr_rows, buu_rows,
        v_ma, v_sa, v_la, v_ha, v_mf, v_lf, v_hf,
        b_ma, b_sa, b_la, b_ha, b_mf, b_lf, b_hf,
    )

    print(f"\n  Results written:")
    print(f"    {os.path.join(results_abs, 'metrics.csv')}")
    print(f"    {os.path.join(results_abs, 'vindr_metrics.csv')}")
    print(f"    {os.path.join(results_abs, 'buu_lspine_metrics.csv')}")
    print(f"    {lb_abs}")
    print(f"    {report_abs}")


# ── Validation report writer ──────────────────────────────────────────────────

def _write_validation_report(
    path: str,
    seeds: List[int],
    vindr_rows: List[dict],
    buu_rows: List[dict],
    v_ma, v_sa, v_la, v_ha,
    v_mf, v_lf, v_hf,
    b_ma, b_sa, b_la, b_ha,
    b_mf, b_lf, b_hf,
) -> None:
    n_seeds   = len(seeds)
    total_p   = MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + QNN_HEAD_TRAINABLE
    n_buu_tr  = BUU_TRAIN_N_PER_CLASS * 2

    lines = [
        "# Q57 Custom DV-QNN Binary Benchmark — Validation Report",
        "",
        f"**Slice ID:** Q57-CUSTOM-DV-QNN-BINARY-BENCHMARK  ",
        f"**Date:** 2026-06-03  ",
        f"**Seeds:** {seeds}  ",
        f"**Epochs:** {EPOCHS}  ",
        f"**LR:** {LR}  |  **Weight Decay:** {WEIGHT_DECAY}  ",
        "",
        "## Architecture",
        "",
        "| Component | Details | Params |",
        "|-----------|---------|--------|",
        f"| Backbone  | MobileNetV3-Large (ImageNet, frozen) | {MOBILENET_LARGE_BACKBONE_PARAMS:,} |",
        f"| Projection | Linear({FEAT_DIM_RAW}→{BACKBONE_OUT_DIM}), frozen random | {PROJECTION_PARAMS:,} |",
        f"| QNN proj   | Linear({BACKBONE_OUT_DIM}→{N_QUBITS}), trainable | {QNN_PROJ_PARAMS:,} |",
        f"| QNN theta  | Variational angles (depth={DV_DEPTH}, n_qubits={N_QUBITS}) | {QNN_THETA_PARAMS:,} |",
        f"| QNN readout| Linear({2**N_QUBITS}→2), trainable | {QNN_READOUT_PARAMS:,} |",
        f"| **QNN head total** | Trainable DV-QNN parameters | **{QNN_HEAD_TRAINABLE:,}** |",
        f"| **Grand total** | All components | **{total_p:,}** |",
        "",
        "## DV-QNN Circuit Details",
        "",
        f"- **Framework:** Custom in-house qcore DV simulator (no PennyLane/Qiskit/TorchQuantum)",
        f"- **Ansatz:** `medical_ansatz` — H init + data reuploading (atan) + variational + ring+cross CNOT",
        f"- **Qubits:** {N_QUBITS}  |  **Depth:** {DV_DEPTH}  |  **Alpha:** {DV_ALPHA}",
        f"- **State space:** {2**N_QUBITS}-dimensional Hilbert space",
        f"- **Circuit ops per sample:** 4H + 8 encoding + 12 variational + 8 CNOT = 32 gates",
        f"- **Gradient flow:** data reuploading via torch.atan preserves autograd end-to-end",
        "",
        "## Datasets",
        "",
        "| Dataset | Source | Train N | Val N | Notes |",
        "|---------|--------|---------|-------|-------|",
        f"| VinDr-SpineXR | Real DICOM ROIs | 6,712 | 1,677 | Q49 pre-computed 128-dim embeddings |",
        f"| BUU-LSPINE | Synthetic procedural | {n_buu_tr} | {BUU_VAL_N_PER_CLASS * 2} | Same generator as Q56 |",
        "",
        "## Results",
        "",
        "### VinDr-SpineXR",
        "",
        "| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 |",
        "|------|-------|-----|----------|-------------|----------|",
    ]
    for r in vindr_rows:
        lines.append(
            f"| {r['seed']} | {r['auroc']:.4f} | {r['f1']:.4f} | "
            f"{r['accuracy']:.4f} | {r['runtime_s']:.1f} | "
            f"{r['delta_auroc_vs_q56baseline']:+.4f} |"
        )

    lines += [
        f"| **Mean** | **{v_ma:.4f}** | **{v_mf:.4f}** | | | |",
        f"| CI95 | [{v_la:.4f}, {v_ha:.4f}] | [{v_lf:.4f}, {v_hf:.4f}] | | | |",
        "",
        f"**AUROC:** {v_ma:.4f} ± {v_sa:.4f} (95% CI: [{v_la:.4f}, {v_ha:.4f}])  ",
        f"**F1:**    {v_mf:.4f} (95% CI: [{v_lf:.4f}, {v_hf:.4f}])  ",
        f"**Δ AUROC vs Q46 winner:** {v_ma - Q46_WINNER_AUROC:+.4f}  ",
        f"**Δ AUROC vs Q56 CNN baseline:** {v_ma - Q56_VINDR_AUROC:+.4f}  ",
        "",
        "### BUU-LSPINE (Synthetic)",
        "",
        "| Seed | AUROC | F1 | Accuracy | Runtime (s) | Δ vs Q56 |",
        "|------|-------|-----|----------|-------------|----------|",
    ]
    for r in buu_rows:
        lines.append(
            f"| {r['seed']} | {r['auroc']:.4f} | {r['f1']:.4f} | "
            f"{r['accuracy']:.4f} | {r['runtime_s']:.1f} | "
            f"{r['delta_auroc_vs_q56baseline']:+.4f} |"
        )

    lines += [
        f"| **Mean** | **{b_ma:.4f}** | **{b_mf:.4f}** | | | |",
        f"| CI95 | [{b_la:.4f}, {b_ha:.4f}] | [{b_lf:.4f}, {b_hf:.4f}] | | | |",
        "",
        f"**AUROC:** {b_ma:.4f} ± {b_sa:.4f} (95% CI: [{b_la:.4f}, {b_ha:.4f}])  ",
        f"**F1:**    {b_mf:.4f} (95% CI: [{b_lf:.4f}, {b_hf:.4f}])  ",
        f"**Δ AUROC vs Q56 CNN baseline:** {b_ma - Q56_BUU_AUROC:+.4f}  ",
        "",
        "## Q56 CNN Baseline Comparison",
        "",
        "| Dataset | Q56 CNN AUROC | Q57 DV-QNN AUROC | Δ AUROC | Q57 Params (trainable) |",
        "|---------|--------------|------------------|---------|------------------------|",
        f"| VinDr-SpineXR | {Q56_VINDR_AUROC:.4f} | {v_ma:.4f} | {v_ma - Q56_VINDR_AUROC:+.4f} | {QNN_HEAD_TRAINABLE:,} |",
        f"| BUU-LSPINE | {Q56_BUU_AUROC:.4f} | {b_ma:.4f} | {b_ma - Q56_BUU_AUROC:+.4f} | {QNN_HEAD_TRAINABLE:,} |",
        "",
        "## Validation Checklist",
        "",
        "- [x] Custom DV-QNN head integrated with MobileNetV3-Large feature extractor",
        "- [x] Trained on both datasets (VinDr-SpineXR + BUU-LSPINE) across all seeds",
        "- [x] Leaderboard CSV written with AUROC/F1/CI95/circuit_depth/params/runtime",
        "- [x] Validation report documents DV-QNN results vs CNN baseline",
        "- [x] Only custom in-house quantum framework used (qcore — no PennyLane/Qiskit)",
        "",
        "## Notes",
        "",
        f"- VinDr-SpineXR uses Q49 pre-computed 128-dim embeddings (MobileNetV3-Large + "
        f"frozen random projection 960→{BACKBONE_OUT_DIM}). QNN head trained fresh per seed.",
        f"- BUU-LSPINE uses synthetic procedural spine ROI images (same generator as Q56). "
        f"Features extracted via MobileNetV3-Large + frozen seeded random projection.",
        f"- Trainable parameters per run: DV-QNN head only ({QNN_HEAD_TRAINABLE} params). "
        f"Backbone and projection are fully frozen.",
        f"- DV-QNN performs per-sample circuit simulation (CPU-only). "
        f"Training is slower than classical head but gradient flows end-to-end.",
        f"- Reference baselines: Q46 winner AUROC = {Q46_WINNER_AUROC:.4f} | "
        f"Q56 CNN VinDr AUROC = {Q56_VINDR_AUROC:.4f} | Q56 CNN BUU AUROC = {Q56_BUU_AUROC:.4f}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q57 Custom DV-QNN Binary Benchmark"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke",   action="store_true", help="1 seed (seed=42)")
    mode.add_argument("--full",    action="store_true", help="3 seeds [42,7,123]")
    mode.add_argument("--dry-run", action="store_true", help="Validate deps, no training")
    parser.add_argument("--project-root", type=str, default=_PROJECT_ROOT)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    if args.dry_run or (not args.smoke and not args.full):
        ok = run_dry_run(project_root)
        sys.exit(0 if ok else 1)

    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    phase = "SMOKE" if args.smoke else "FULL"

    print("=" * 70)
    print(f"Q57 Custom DV-QNN Binary Benchmark — Phase: {phase}")
    print("=" * 70)
    print(f"  Seeds: {seeds}  |  Epochs: {EPOCHS}  |  LR: {LR}  |  batch_size: {BATCH_SIZE}")
    print(f"  DV-QNN: n_qubits={N_QUBITS}  depth={DV_DEPTH}  alpha={DV_ALPHA}")
    print(f"  Trainable QNN head params: {QNN_HEAD_TRAINABLE}")
    print()

    try:
        import torch
        import torchvision
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Verify qcore quantum framework (no external quantum libs)
    try:
        from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
        from qcore.backends.base import Backend
        from qcore.states.vacuum import vacuum_state
    except ImportError as e:
        print(f"ERROR: qcore quantum framework unavailable: {e}")
        sys.exit(1)

    t_wall = time.time()
    run_benchmark(seeds=seeds, project_root=project_root)
    elapsed = time.time() - t_wall
    print(f"\n  Done.  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
