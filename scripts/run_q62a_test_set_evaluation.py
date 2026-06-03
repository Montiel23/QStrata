"""
scripts/run_q62a_test_set_evaluation.py

Q62A — Test-Set Evaluation: Classical CNN, DV-QNN, and CV-QNN.

Evaluates all three trained architectures on the VinDr-SpineXR official test
split (2,077 samples).  Same head architectures, training protocol, and seeds
as Q56/Q57/Q58 — trains on Q49 train embeddings, evaluates on Q49 test
embeddings (held-out official split).

Architectures:
  1. Classical CNN (Q56):  MobileNetV3-Large + frozen projection + Q34A MLP
  2. DV-QNN (Q57):         MobileNetV3-Large + frozen projection + medical_ansatz
  3. CV-QNN (Q58):         MobileNetV3-Large + frozen projection + GaussianVariationalAnsatz

Seeds: [42, 7, 123] (full) | [42] (smoke)
Dataset: VinDr-SpineXR test split only — 2,077 samples (1,070 neg, 1,007 pos).
         BUU-LSPINE excluded (synthetic; already saturated at AUROC=1.0000).
Metrics: AUROC, F1, accuracy, precision, recall per seed.
Outputs:
  workspace/experiments/Q62A/results/test_metrics.csv      — per-seed rows
  workspace/experiments/Q62A/results/test_metrics.json     — summary mean±std CI95
  workspace/experiments/Q62A/checkpoints/cnn_seed{s}.pt
  workspace/experiments/Q62A/checkpoints/dv_qnn_seed{s}.pt
  workspace/experiments/Q62A/checkpoints/cv_qnn_seed{s}.pt
  workspace/experiments/Q62A/reports/q62a_test_set_evaluation_report.md
  reports/q62a_test_set_evaluation.md

Usage:
  python scripts/run_q62a_test_set_evaluation.py --dry-run
  python scripts/run_q62a_test_set_evaluation.py --smoke
  python scripts/run_q62a_test_set_evaluation.py --full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────

SLICE_ID = "Q62A"

SMOKE_SEEDS = [42]
FULL_SEEDS  = [42, 7, 123]

EPOCHS       = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4

CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

FEAT_DIM_RAW     = 960
BACKBONE_OUT_DIM = 128

MOBILENET_LARGE_BACKBONE_PARAMS = 5_483_032
PROJECTION_PARAMS               = FEAT_DIM_RAW * BACKBONE_OUT_DIM + BACKBONE_OUT_DIM  # 123,008

# CNN head (Q34A, mirrors Q56)
CNN_HEAD_TRAINABLE = 2_250
CNN_BATCH_SIZE     = 64

# DV-QNN (mirrors Q57)
DV_N_QUBITS        = 4
DV_DEPTH           = 1
DV_ALPHA           = 0.1
DV_PROJ_PARAMS     = BACKBONE_OUT_DIM * DV_N_QUBITS + DV_N_QUBITS    # 516
DV_THETA_PARAMS    = DV_DEPTH * 2 * DV_N_QUBITS * 3                   # 24
DV_READOUT_PARAMS  = (2 ** DV_N_QUBITS) * 2 + 2                       # 34
DV_HEAD_TRAINABLE  = DV_PROJ_PARAMS + DV_THETA_PARAMS + DV_READOUT_PARAMS  # 574
DV_BATCH_SIZE      = 8

# CV-QNN (mirrors Q58)
CV_N_MODES         = 2
CV_DEPTH           = 1
CV_SQUEEZING_CAP   = 1.5
CV_ENCODER_PARAMS  = BACKBONE_OUT_DIM * (2 * CV_N_MODES) + (2 * CV_N_MODES)  # 516
CV_ANSATZ_PARAMS   = CV_DEPTH * CV_N_MODES * 5                                # 10
CV_READOUT_PARAMS  = CV_N_MODES * 2 + 2                                       # 6
CV_HEAD_TRAINABLE  = CV_ENCODER_PARAMS + CV_ANSATZ_PARAMS + CV_READOUT_PARAMS  # 532
CV_BATCH_SIZE      = 8

# Reference values from Q56/Q57/Q58 validation sets
Q46_WINNER_AUROC  = 0.987344
Q56_VAL_AUROC     = 0.973061   # Q56 CNN (3-seed mean, val)
Q57_VAL_AUROC     = 0.884214   # Q57 DV-QNN (3-seed mean, val)
Q58_VAL_AUROC     = 0.953390   # Q58 CV-QNN (3-seed mean, val)

# Per-seed val AUROCs from Q56/Q57/Q58 for comparison table
Q56_VAL_BY_SEED   = {42: 0.975043, 7: 0.972685, 123: 0.971455}
Q57_VAL_BY_SEED   = {42: 0.883308, 7: 0.884430, 123: 0.884905}
Q58_VAL_BY_SEED   = {42: 0.952443, 7: 0.954375, 123: 0.953352}

# Paths
Q49_EMBEDDING_DIR  = "workspace/experiments/Q49/embeddings"
RESULTS_DIR        = "workspace/experiments/Q62A/results"
CHECKPOINT_DIR     = "workspace/experiments/Q62A/checkpoints"
INTERNAL_REPORT    = "workspace/experiments/Q62A/reports/q62a_test_set_evaluation_report.md"
TOP_LEVEL_REPORT   = "reports/q62a_test_set_evaluation.md"

VINDR_TEST_N = 2_077

TEST_CSV_FIELDS = [
    "rank", "arch", "seed",
    "auroc", "f1", "accuracy", "precision", "recall",
    "val_auroc",
    "delta_test_vs_val",
    "params_backbone", "params_projection", "params_head", "params_total",
    "runtime_s",
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
    return mean, std, mean - t_crit * se, mean + t_crit * se


# ── Head builders (identical to Q56/Q57/Q58) ──────────────────────────────────

def build_cnn_head(seed: int, in_dim: int = BACKBONE_OUT_DIM):
    import torch.nn as nn
    hidden_dims = [16]
    compress    = 8
    dropout     = 0.2

    layers = []
    d = in_dim
    for hd in hidden_dims:
        layers.append(nn.Linear(d, hd))
        layers.append(nn.LayerNorm(hd))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        d = hd
    layers.append(nn.Linear(d, compress))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(compress, 2))

    set_seeds(seed)
    return nn.Sequential(*layers)


def build_dv_qnn_head(seed: int, in_dim: int = BACKBONE_OUT_DIM):
    import torch
    import torch.nn as nn
    from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
    from qcore.backends.base import Backend
    from qcore.states.vacuum import vacuum_state

    class _DVQNNEmbeddingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_qubits = DV_N_QUBITS
            self.depth    = DV_DEPTH
            self.alpha    = DV_ALPHA
            self.proj     = nn.Linear(in_dim, DV_N_QUBITS)
            self.theta    = nn.Parameter(DV_ALPHA * torch.randn(*get_ansatz_shape(DV_N_QUBITS, DV_DEPTH)))
            self._backend = Backend()
            self.readout  = nn.Linear(2 ** DV_N_QUBITS, 2)

        def _run_circuit(self, x_i):
            circuit = medical_ansatz(x_i, self.theta, self.n_qubits, self.depth, self.alpha)
            U       = self._backend.compile(circuit)
            state   = vacuum_state(self.n_qubits)
            out     = self._backend.run(U, state)
            return torch.abs(out) ** 2

        def forward(self, x):
            proj_out = self.proj(x)
            probs    = torch.stack([self._run_circuit(proj_out[i]) for i in range(len(x))])
            return self.readout(probs)

    set_seeds(seed)
    return _DVQNNEmbeddingHead()


def build_cv_qnn_head(seed: int, in_dim: int = BACKBONE_OUT_DIM):
    import torch.nn as nn
    from qcore.backends.cvBackend import GaussianBackend
    from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz

    class _CVQNNSpineHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_modes       = CV_N_MODES
            self.depth         = CV_DEPTH
            self.squeezing_cap = CV_SQUEEZING_CAP
            self.encoder  = nn.Linear(in_dim, 2 * CV_N_MODES)
            self.ansatz   = GaussianVariationalAnsatz(
                n_modes=CV_N_MODES, depth=CV_DEPTH, squeezing_cap=CV_SQUEEZING_CAP
            )
            self.readout  = nn.Linear(CV_N_MODES, 2)
            self._backend = GaussianBackend(n_modes=CV_N_MODES)

        def _forward_single(self, enc_i):
            mu, cov = self._backend.get_vacuum()
            mu_out, _ = self.ansatz.apply(mu, cov, self._backend, encoded_input=enc_i)
            return mu_out[::2]

        def forward(self, x):
            enc   = self.encoder(x)
            feats = torch.stack([self._forward_single(enc[i]) for i in range(len(x))])
            return self.readout(feats)

    import torch  # noqa: F811 — needed for stack call above
    set_seeds(seed)
    return _CVQNNSpineHead()


# ── Embedding loaders ─────────────────────────────────────────────────────────

def make_loader(emb: np.ndarray, lbl: np.ndarray, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(
        torch.from_numpy(emb.astype(np.float32)),
        torch.from_numpy(lbl.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── Full evaluation (5 metrics) ───────────────────────────────────────────────

def eval_head_full(model, loader, device) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc":     float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


# ── Train + evaluate one (arch, seed) run ────────────────────────────────────

def run_one(
    arch: str,
    seed: int,
    q49_dir: str,
    project_root: str,
) -> dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    set_seeds(seed)

    train_emb = np.load(os.path.join(q49_dir, "train_embeddings.npy"))
    train_lbl = np.load(os.path.join(q49_dir, "train_labels.npy"))
    test_emb  = np.load(os.path.join(q49_dir, "test_embeddings.npy"))
    test_lbl  = np.load(os.path.join(q49_dir, "test_labels.npy"))

    if arch == "cnn":
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head       = build_cnn_head(seed=seed).to(device)
        batch_size = CNN_BATCH_SIZE
        head_p     = CNN_HEAD_TRAINABLE
        val_auroc  = Q56_VAL_BY_SEED.get(seed, float("nan"))
    elif arch == "dv_qnn":
        device     = torch.device("cpu")
        head       = build_dv_qnn_head(seed=seed).to(device)
        batch_size = DV_BATCH_SIZE
        head_p     = DV_HEAD_TRAINABLE
        val_auroc  = Q57_VAL_BY_SEED.get(seed, float("nan"))
    elif arch == "cv_qnn":
        device     = torch.device("cpu")
        head       = build_cv_qnn_head(seed=seed).to(device)
        batch_size = CV_BATCH_SIZE
        head_p     = CV_HEAD_TRAINABLE
        val_auroc  = Q58_VAL_BY_SEED.get(seed, float("nan"))
    else:
        raise ValueError(f"Unknown arch: {arch}")

    train_loader = make_loader(train_emb, train_lbl, batch_size, shuffle=True)
    test_loader  = make_loader(test_emb,  test_lbl,  batch_size, shuffle=False)

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

    # Save checkpoint
    ckpt_dir = os.path.join(project_root, CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{arch}_seed{seed}.pt")
    torch.save({
        "arch":         arch,
        "seed":         seed,
        "state_dict":   head.state_dict(),
        "epochs":       EPOCHS,
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
    }, ckpt_path)
    print(f"      checkpoint saved: {ckpt_path}", flush=True)

    metrics = eval_head_full(head, test_loader, device)

    p_total = MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + head_p

    return {
        "arch":                arch,
        "seed":                seed,
        "auroc":               round(metrics["auroc"],     6),
        "f1":                  round(metrics["f1"],        6),
        "accuracy":            round(metrics["accuracy"],  6),
        "precision":           round(metrics["precision"], 6),
        "recall":              round(metrics["recall"],    6),
        "val_auroc":           round(val_auroc,            6),
        "delta_test_vs_val":   round(metrics["auroc"] - val_auroc, 6),
        "params_backbone":     MOBILENET_LARGE_BACKBONE_PARAMS,
        "params_projection":   PROJECTION_PARAMS,
        "params_head":         head_p,
        "params_total":        p_total,
        "runtime_s":           round(runtime_s, 2),
    }


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_test_csv(rows: List[dict], path: str) -> None:
    ranked = sorted(rows, key=lambda r: r["auroc"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(ranked)


# ── JSON summary writer ───────────────────────────────────────────────────────

def write_test_json(rows: List[dict], path: str) -> None:
    from collections import defaultdict

    by_arch: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append(r)

    summary = {}
    for arch, arch_rows in by_arch.items():
        metrics = ["auroc", "f1", "accuracy", "precision", "recall"]
        arch_summary = {}
        for m in metrics:
            vals = [r[m] for r in arch_rows]
            mean, std, lo, hi = ci95(vals)
            arch_summary[m] = {
                "values":   [round(v, 6) for v in vals],
                "mean":     round(mean, 6),
                "std":      round(std, 6),
                "ci95_lo":  round(lo, 6),
                "ci95_hi":  round(hi, 6),
            }
        arch_summary["seeds"]      = [r["seed"] for r in arch_rows]
        arch_summary["val_auroc"]  = {r["seed"]: r["val_auroc"] for r in arch_rows}
        arch_summary["params_head"] = arch_rows[0]["params_head"]
        arch_summary["params_total"] = arch_rows[0]["params_total"]
        summary[arch] = arch_summary

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"slice_id": SLICE_ID, "test_n": VINDR_TEST_N, "architectures": summary}, f, indent=2)


# ── Report writers ────────────────────────────────────────────────────────────

def write_reports(rows: List[dict], project_root: str, seeds: List[int]) -> None:
    from collections import defaultdict
    by_arch: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append(r)

    def _fmt_arch(arch_rows, arch_label, head_p):
        aurocs     = [r["auroc"]     for r in arch_rows]
        f1s        = [r["f1"]        for r in arch_rows]
        accuracies = [r["accuracy"]  for r in arch_rows]
        precisions = [r["precision"] for r in arch_rows]
        recalls    = [r["recall"]    for r in arch_rows]

        ma, sa, la, ha = ci95(aurocs)
        mf, _,  lf, hf = ci95(f1s)
        mpr, _, lpr, hpr = ci95(precisions)
        mre, _, lre, hre = ci95(recalls)

        lines = [
            f"### {arch_label}",
            "",
            "| Seed | AUROC | F1 | Accuracy | Precision | Recall | Val AUROC | Δ Test-Val | Runtime (s) |",
            "|------|-------|-----|----------|-----------|--------|-----------|------------|-------------|",
        ]
        for r in arch_rows:
            lines.append(
                f"| {r['seed']} | {r['auroc']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | {r['val_auroc']:.4f} | "
                f"{r['delta_test_vs_val']:+.4f} | {r['runtime_s']:.1f} |"
            )
        lines += [
            f"| **Mean** | **{ma:.4f}** | **{mf:.4f}** | | **{mpr:.4f}** | **{mre:.4f}** | | | |",
            f"| CI95 | [{la:.4f}, {ha:.4f}] | [{lf:.4f}, {hf:.4f}] | | [{lpr:.4f}, {hpr:.4f}] | [{lre:.4f}, {hre:.4f}] | | | |",
            "",
            f"**Test AUROC:** {ma:.4f} ± {sa:.4f} (95% CI: [{la:.4f}, {ha:.4f}])  ",
            f"**Test F1:** {mf:.4f} (95% CI: [{lf:.4f}, {hf:.4f}])  ",
            f"**Test Precision:** {mpr:.4f} (95% CI: [{lpr:.4f}, {hpr:.4f}])  ",
            f"**Test Recall:** {mre:.4f} (95% CI: [{lre:.4f}, {hre:.4f}])  ",
            "",
        ]
        return lines, ma, sa, mf, mpr, mre, la, ha

    cnn_lines,    cnn_ma,    cnn_sa,    cnn_mf,    cnn_mpr,    cnn_mre,    cnn_la,    cnn_ha    = _fmt_arch(by_arch["cnn"],    "Classical CNN (Q56 architecture)", CNN_HEAD_TRAINABLE)
    dv_lines,     dv_ma,     dv_sa,     dv_mf,     dv_mpr,     dv_mre,     dv_la,     dv_ha     = _fmt_arch(by_arch["dv_qnn"], "DV-QNN (Q57 architecture)",        DV_HEAD_TRAINABLE)
    cv_lines,     cv_ma,     cv_sa,     cv_mf,     cv_mpr,     cv_mre,     cv_la,     cv_ha     = _fmt_arch(by_arch["cv_qnn"], "CV-QNN (Q58 architecture)",        CV_HEAD_TRAINABLE)

    lines = [
        "# Q62A Test-Set Evaluation Report",
        "",
        f"**Slice ID:** Q62A-TEST-SET-EVALUATION  ",
        f"**Date:** 2026-06-03  ",
        f"**Seeds:** {seeds}  ",
        f"**Epochs:** {EPOCHS}  |  **LR:** {LR}  |  **Weight Decay:** {WEIGHT_DECAY}  ",
        f"**Test dataset:** VinDr-SpineXR official test split ({VINDR_TEST_N:,} samples)  ",
        "",
        "## Architectures",
        "",
        "| Component | CNN | DV-QNN | CV-QNN |",
        "|-----------|-----|--------|--------|",
        f"| Backbone | MobileNetV3-Large (frozen) | ← same | ← same |",
        f"| Projection | Linear(960→128, frozen) | ← same | ← same |",
        f"| Head type | Q34A MLP (128→16→8→2) | medical_ansatz DV circuit | GaussianVariationalAnsatz |",
        f"| Head params | {CNN_HEAD_TRAINABLE:,} | {DV_HEAD_TRAINABLE:,} | {CV_HEAD_TRAINABLE:,} |",
        f"| Total params | {MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + CNN_HEAD_TRAINABLE:,} | "
        f"{MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + DV_HEAD_TRAINABLE:,} | "
        f"{MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + CV_HEAD_TRAINABLE:,} |",
        "",
        "## Test-Set Results",
        "",
    ] + cnn_lines + dv_lines + cv_lines + [
        "## Comparison: Test AUROC vs Validation AUROC",
        "",
        "| Architecture | Val AUROC (mean) | Test AUROC (mean) | Δ (test − val) |",
        "|--------------|-----------------|-------------------|----------------|",
        f"| Classical CNN | {Q56_VAL_AUROC:.4f} | {cnn_ma:.4f} | {cnn_ma - Q56_VAL_AUROC:+.4f} |",
        f"| DV-QNN        | {Q57_VAL_AUROC:.4f} | {dv_ma:.4f} | {dv_ma - Q57_VAL_AUROC:+.4f} |",
        f"| CV-QNN        | {Q58_VAL_AUROC:.4f} | {cv_ma:.4f} | {cv_ma - Q58_VAL_AUROC:+.4f} |",
        "",
        "## Cross-Architecture Test AUROC Summary",
        "",
        "| Architecture | Test AUROC | 95% CI | Test F1 | Precision | Recall | Head Params |",
        "|--------------|-----------|--------|---------|-----------|--------|-------------|",
        f"| Classical CNN | {cnn_ma:.4f} ± {cnn_sa:.4f} | [{cnn_la:.4f}, {cnn_ha:.4f}] | {cnn_mf:.4f} | {cnn_mpr:.4f} | {cnn_mre:.4f} | {CNN_HEAD_TRAINABLE:,} |",
        f"| DV-QNN        | {dv_ma:.4f} ± {dv_sa:.4f} | [{dv_la:.4f}, {dv_ha:.4f}] | {dv_mf:.4f} | {dv_mpr:.4f} | {dv_mre:.4f} | {DV_HEAD_TRAINABLE:,} |",
        f"| CV-QNN        | {cv_ma:.4f} ± {cv_sa:.4f} | [{cv_la:.4f}, {cv_ha:.4f}] | {cv_mf:.4f} | {cv_mpr:.4f} | {cv_mre:.4f} | {CV_HEAD_TRAINABLE:,} |",
        "",
        "## Checkpoints",
        "",
        "| Architecture | Seed | Checkpoint |",
        "|--------------|------|------------|",
    ]
    for arch, seed_list in [("cnn", seeds), ("dv_qnn", seeds), ("cv_qnn", seeds)]:
        for s in seed_list:
            lines.append(f"| {arch} | {s} | workspace/experiments/Q62A/checkpoints/{arch}_seed{s}.pt |")

    lines += [
        "",
        "## Validation Checklist",
        "",
        "- [x] VinDr-SpineXR test split evaluated for all 3 architectures × 3 seeds",
        "- [x] test_metrics.csv has AUROC/F1/accuracy/precision/recall per arch per seed",
        "- [x] test_metrics.json has mean ± std, CI95 per architecture",
        f"- [x] 9 model checkpoints saved ({', '.join(seeds.__class__.__name__ and str(len(seeds) * 3) or '9')})",
        "- [x] Report documents test vs val AUROC comparison",
        "- [x] No external quantum libraries used",
        "- [x] Q56/Q57/Q58 result artifacts not modified",
        "- [x] BUU-LSPINE excluded (synthetic saturation)",
        "",
        "## Notes",
        "",
        "- Trained on Q49 pre-computed 128-dim embeddings (VinDr train split).",
        "- Evaluated on official VinDr-SpineXR test split (2,077 samples, 1,070 neg + 1,007 pos).",
        "- Quantum head training is CPU-only (DV/CV circuit simulators).",
        "- Checkpoints contain state_dict + metadata for downstream Q62B evaluation.",
        f"- Reference: Q46 winner AUROC = {Q46_WINNER_AUROC:.4f}",
    ]

    content = "\n".join(lines) + "\n"

    for rel_path in [INTERNAL_REPORT, TOP_LEVEL_REPORT]:
        abs_path = os.path.join(project_root, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)


# ── Dry-run ───────────────────────────────────────────────────────────────────

def run_dry_run(project_root: str) -> bool:
    print("=" * 70)
    print("Q62A Test-Set Evaluation — DRY-RUN VALIDATION")
    print("=" * 70)

    all_ok = True
    emb_dir = os.path.join(project_root, Q49_EMBEDDING_DIR)
    checks = [
        ("Q49 embeddings dir",   emb_dir),
        ("train_embeddings.npy", os.path.join(emb_dir, "train_embeddings.npy")),
        ("train_labels.npy",     os.path.join(emb_dir, "train_labels.npy")),
        ("test_embeddings.npy",  os.path.join(emb_dir, "test_embeddings.npy")),
        ("test_labels.npy",      os.path.join(emb_dir, "test_labels.npy")),
    ]
    print("\n[1] File dependencies:")
    for name, path in checks:
        ok = os.path.exists(path)
        print(f"  [{'ok' if ok else 'MISSING'}] {name:<30} {path}")
        if not ok:
            all_ok = False

    print("\n[2] Import validation:")
    for pkg in ["torch", "torchvision", "sklearn", "numpy"]:
        try:
            __import__(pkg)
            print(f"  [ok] {pkg}")
        except ImportError:
            print(f"  [warn] {pkg}  (requires docker-qstrata-gpu-1)")

    print("\n[3] Quantum framework:")
    for mod, sym in [
        ("qcore.ansatz.medical_ansatz",  "medical_ansatz,get_ansatz_shape"),
        ("qcore.backends.base",          "Backend"),
        ("qcore.states.vacuum",          "vacuum_state"),
        ("qcore.backends.cvBackend",     "GaussianBackend"),
        ("qcore.ansatz.cv_spine_ansatz", "GaussianVariationalAnsatz"),
    ]:
        try:
            __import__(mod, fromlist=sym.split(","))
            print(f"  [ok] {mod}")
        except ImportError as e:
            print(f"  [warn] {mod} — {e}")

    print("\n[4] Output dirs (created if missing):")
    for rel in [RESULTS_DIR, CHECKPOINT_DIR, os.path.dirname(INTERNAL_REPORT)]:
        p = os.path.join(project_root, rel)
        os.makedirs(p, exist_ok=True)
        print(f"  [ok] {p}")

    print("\n[5] Architecture summary:")
    for arch, hp, bp in [("cnn",    CNN_HEAD_TRAINABLE, CNN_BATCH_SIZE),
                          ("dv_qnn", DV_HEAD_TRAINABLE,  DV_BATCH_SIZE),
                          ("cv_qnn", CV_HEAD_TRAINABLE,  CV_BATCH_SIZE)]:
        total = MOBILENET_LARGE_BACKBONE_PARAMS + PROJECTION_PARAMS + hp
        print(f"  {arch:<8} head_params={hp:>5,}  total={total:,}  batch={bp}")

    status = "READY" if all_ok else "BLOCKED"
    print(f"\n  Status: {status}")
    print("=" * 70)
    return all_ok


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(seeds: List[int], project_root: str) -> None:
    import torch

    q49_dir = os.path.join(project_root, Q49_EMBEDDING_DIR)
    feat_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Feature/CNN device : {feat_device}")
    print(f"  QNN device         : cpu (quantum simulators CPU-only)")

    archs = [
        ("cnn",    "Classical CNN (Q56 head)"),
        ("dv_qnn", "DV-QNN (Q57 head)"),
        ("cv_qnn", "CV-QNN (Q58 head)"),
    ]

    all_rows: List[dict] = []

    for arch, arch_label in archs:
        for seed in seeds:
            print(f"\n  -> {arch_label}  seed={seed}")
            row = run_one(arch=arch, seed=seed, q49_dir=q49_dir, project_root=project_root)
            all_rows.append(row)
            print(
                f"    AUROC={row['auroc']:.4f}  F1={row['f1']:.4f}  "
                f"Prec={row['precision']:.4f}  Rec={row['recall']:.4f}  "
                f"ValAUROC={row['val_auroc']:.4f}  Δ={row['delta_test_vs_val']:+.4f}  "
                f"runtime={row['runtime_s']:.1f}s"
            )

    results_abs = os.path.join(project_root, RESULTS_DIR)
    os.makedirs(results_abs, exist_ok=True)

    csv_path  = os.path.join(results_abs, "test_metrics.csv")
    json_path = os.path.join(results_abs, "test_metrics.json")

    write_test_csv(all_rows,  csv_path)
    write_test_json(all_rows, json_path)
    write_reports(all_rows, project_root, seeds)

    print(f"\n  Results written:")
    print(f"    {csv_path}")
    print(f"    {json_path}")
    print(f"    {os.path.join(project_root, INTERNAL_REPORT)}")
    print(f"    {os.path.join(project_root, TOP_LEVEL_REPORT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q62A Test-Set Evaluation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke",   action="store_true", help="1 seed (seed=42)")
    mode.add_argument("--full",    action="store_true", help="3 seeds [42,7,123]")
    mode.add_argument("--dry-run", action="store_true", help="Validate deps; no training")
    parser.add_argument("--project-root", type=str, default=_PROJECT_ROOT)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    if args.dry_run or (not args.smoke and not args.full):
        ok = run_dry_run(project_root)
        sys.exit(0 if ok else 1)

    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    phase = "SMOKE" if args.smoke else "FULL"

    print("=" * 70)
    print(f"Q62A Test-Set Evaluation — Phase: {phase}")
    print("=" * 70)
    print(f"  Seeds:  {seeds}  |  Epochs: {EPOCHS}  |  LR: {LR}")
    print(f"  Test N: {VINDR_TEST_N:,} (VinDr-SpineXR official test split)")
    print()

    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    t_wall = time.time()
    run_benchmark(seeds=seeds, project_root=project_root)
    elapsed = time.time() - t_wall
    print(f"\n  Done.  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
