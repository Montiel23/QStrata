"""
scripts/run_q62b_probability_export.py

Q62B — Per-Sample Probability Export: CNN, DV-QNN, CV-QNN on VinDr-SpineXR.

Loads Q62A checkpoints (no re-training) and exports per-sample prediction
probabilities for both test and val splits.  Outputs feed directly into Q62C
(ROC curves, PR curves, confusion matrices).

Architectures (same as Q62A / Q56 / Q57 / Q58):
  1. Classical CNN   — MobileNetV3-Large + frozen projection + Q34A MLP
  2. DV-QNN          — MobileNetV3-Large + frozen projection + medical_ansatz
  3. CV-QNN          — MobileNetV3-Large + frozen projection + GaussianVariationalAnsatz

Seeds:   [42, 7, 123] (full) | [42] (smoke)
Dataset: VinDr-SpineXR test split (2,077 samples) + val split (1,677 samples)

Outputs:
  workspace/experiments/Q62B/probabilities/cnn_vindr_test_probs.csv
  workspace/experiments/Q62B/probabilities/cnn_vindr_val_probs.csv
  workspace/experiments/Q62B/probabilities/dv_qnn_vindr_test_probs.csv
  workspace/experiments/Q62B/probabilities/dv_qnn_vindr_val_probs.csv
  workspace/experiments/Q62B/probabilities/cv_qnn_vindr_test_probs.csv
  workspace/experiments/Q62B/probabilities/cv_qnn_vindr_val_probs.csv
  workspace/experiments/Q62B/reports/q62b_probability_export_report.md

Usage:
  python scripts/run_q62b_probability_export.py --dry-run
  python scripts/run_q62b_probability_export.py --smoke
  python scripts/run_q62b_probability_export.py --full
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np

_SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Constants (mirror Q62A) ────────────────────────────────────────────────────

SLICE_ID = "Q62B"

SMOKE_SEEDS = [42]
FULL_SEEDS  = [42, 7, 123]

FEAT_DIM_RAW     = 960
BACKBONE_OUT_DIM = 128

# CNN head (Q34A, mirrors Q56 / Q62A)
CNN_BATCH_SIZE = 64

# DV-QNN (mirrors Q57 / Q62A)
DV_N_QUBITS   = 4
DV_DEPTH       = 1
DV_ALPHA       = 0.1
DV_BATCH_SIZE  = 8

# CV-QNN (mirrors Q58 / Q62A)
CV_N_MODES         = 2
CV_DEPTH           = 1
CV_SQUEEZING_CAP   = 1.5
CV_BATCH_SIZE      = 8

# Paths
Q49_EMBEDDING_DIR = "workspace/experiments/Q49/embeddings"
Q62A_CKPT_DIR     = "workspace/experiments/Q62A/checkpoints"
PROB_OUT_DIR      = "workspace/experiments/Q62B/probabilities"
REPORT_PATH       = "workspace/experiments/Q62B/reports/q62b_probability_export_report.md"

VINDR_TEST_N = 2_077
VINDR_VAL_N  = 1_677

PROB_CSV_FIELDS = ["sample_id", "y_true", "y_score", "y_pred", "seed"]


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    import random
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


# ── Head builders (identical to Q62A) ─────────────────────────────────────────

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
    import torch
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

    set_seeds(seed)
    return _CVQNNSpineHead()


# ── DataLoader helper ──────────────────────────────────────────────────────────

def make_loader(emb: np.ndarray, lbl: np.ndarray, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(
        torch.from_numpy(emb.astype(np.float32)),
        torch.from_numpy(lbl.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


# ── Inference → probability list ──────────────────────────────────────────────

def infer_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_score, y_pred) arrays."""
    import torch

    model.eval()
    y_true_list, y_score_list, y_pred_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            y_true_list.extend(y.cpu().tolist())
            y_score_list.extend(probs.cpu().tolist())
            y_pred_list.extend(preds.cpu().tolist())

    return (
        np.array(y_true_list,  dtype=np.int32),
        np.array(y_score_list, dtype=np.float64),
        np.array(y_pred_list,  dtype=np.int32),
    )


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def load_checkpoint(arch: str, seed: int, project_root: str):
    import torch

    ckpt_path = os.path.join(project_root, Q62A_CKPT_DIR, f"{arch}_seed{seed}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if arch == "cnn":
        model = build_cnn_head(seed=seed)
    elif arch == "dv_qnn":
        model = build_dv_qnn_head(seed=seed)
    elif arch == "cv_qnn":
        model = build_cv_qnn_head(seed=seed)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── CSV writer ─────────────────────────────────────────────────────────────────

def write_prob_csv(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROB_CSV_FIELDS)
        writer.writeheader()
        for i in range(len(y_true)):
            writer.writerow({
                "sample_id": i,
                "y_true":    int(y_true[i]),
                "y_score":   f"{float(y_score[i]):.8f}",
                "y_pred":    int(y_pred[i]),
                "seed":      seed,
            })


# ── Per-arch export (all seeds → one CSV with seed column) ────────────────────

def export_arch(
    arch: str,
    seeds: List[int],
    test_emb: np.ndarray,
    test_lbl: np.ndarray,
    val_emb: np.ndarray,
    val_lbl: np.ndarray,
    project_root: str,
    batch_size: int,
) -> Dict:
    import torch

    device = torch.device("cpu")  # checkpoints are CPU; QNN backends are CPU-only

    # Accumulate rows across seeds (one row per sample × seed)
    test_rows: List[dict] = []
    val_rows:  List[dict] = []

    for seed in seeds:
        print(f"    seed={seed}  loading checkpoint …", flush=True)
        model = load_checkpoint(arch, seed, project_root)
        model = model.to(device)

        test_loader = make_loader(test_emb, test_lbl, batch_size)
        val_loader  = make_loader(val_emb,  val_lbl,  batch_size)

        t0 = time.time()
        t_true, t_score, t_pred = infer_probs(model, test_loader, device)
        v_true, v_score, v_pred = infer_probs(model, val_loader,  device)
        elapsed = time.time() - t0

        print(
            f"    seed={seed}  test={len(t_true)}  val={len(v_true)}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        for i in range(len(t_true)):
            test_rows.append({
                "sample_id": i,
                "y_true":    int(t_true[i]),
                "y_score":   float(t_score[i]),
                "y_pred":    int(t_pred[i]),
                "seed":      seed,
            })
        for i in range(len(v_true)):
            val_rows.append({
                "sample_id": i,
                "y_true":    int(v_true[i]),
                "y_score":   float(v_score[i]),
                "y_pred":    int(v_pred[i]),
                "seed":      seed,
            })

    prob_dir = os.path.join(project_root, PROB_OUT_DIR)
    os.makedirs(prob_dir, exist_ok=True)

    test_csv = os.path.join(prob_dir, f"{arch}_vindr_test_probs.csv")
    val_csv  = os.path.join(prob_dir, f"{arch}_vindr_val_probs.csv")

    _write_rows(test_rows, test_csv)
    _write_rows(val_rows,  val_csv)

    print(f"    -> {test_csv}  ({len(test_rows)} rows)", flush=True)
    print(f"    -> {val_csv}   ({len(val_rows)} rows)",  flush=True)

    return {
        "arch":         arch,
        "seeds":        seeds,
        "test_csv":     test_csv,
        "val_csv":      val_csv,
        "test_rows":    len(test_rows),
        "val_rows":     len(val_rows),
        "test_samples": len(test_lbl),
        "val_samples":  len(val_lbl),
    }


def _write_rows(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROB_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "sample_id": r["sample_id"],
                "y_true":    r["y_true"],
                "y_score":   f"{r['y_score']:.8f}",
                "y_pred":    r["y_pred"],
                "seed":      r["seed"],
            })


# ── Report writer ──────────────────────────────────────────────────────────────

def write_report(
    export_results: List[dict],
    seeds: List[int],
    project_root: str,
    elapsed_total: float,
) -> None:
    lines = [
        "# Q62B Probability Export Report",
        "",
        f"**Slice ID:** Q62B-PROBABILITY-EXPORT  ",
        f"**Date:** 2026-06-03  ",
        f"**Seeds:** {seeds}  ",
        f"**Checkpoints from:** Q62A (no re-training)  ",
        f"**Total runtime:** {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)  ",
        "",
        "## Exported Files",
        "",
        "| Architecture | Split | Rows | Samples × Seeds | CSV |",
        "|-------------|-------|------|-----------------|-----|",
    ]
    for r in export_results:
        arch = r["arch"]
        ns   = r["test_samples"]
        nv   = r["val_samples"]
        nsds = len(r["seeds"])
        lines.append(
            f"| {arch} | test | {r['test_rows']} | {ns} × {nsds} | `{os.path.basename(r['test_csv'])}` |"
        )
        lines.append(
            f"| {arch} | val  | {r['val_rows']} | {nv} × {nsds} | `{os.path.basename(r['val_csv'])}` |"
        )

    lines += [
        "",
        "## CSV Schema",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| sample_id | int | 0-based index into the split |",
        "| y_true | int | Ground-truth label (0=normal, 1=abnormal) |",
        "| y_score | float | Softmax probability of positive class ∈ [0,1] |",
        "| y_pred | int | Argmax prediction (0 or 1) |",
        "| seed | int | Training seed (42, 7, or 123) |",
        "",
        "## Validation Checklist",
        "",
        "- [x] CNN probabilities exported (test + val)",
        "- [x] DV-QNN probabilities exported (test + val)",
        "- [x] CV-QNN probabilities exported (test + val)",
        "- [x] y_score values are softmax probabilities ∈ [0,1]",
        "- [x] CSV schema: sample_id, y_true, y_score, y_pred, seed",
        "- [x] Q62A checkpoints used (no re-training)",
        "- [x] No external quantum libraries used",
        "",
        "## Notes",
        "",
        "- Inference only — model weights are frozen checkpoints from Q62A.",
        "- Each CSV contains rows for all seeds (seed column distinguishes them).",
        "- Q62C will use these files to generate ROC/PR curves and confusion matrices.",
    ]

    content = "\n".join(lines) + "\n"
    abs_path = os.path.join(project_root, REPORT_PATH)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Report: {abs_path}", flush=True)


# ── Dry-run ────────────────────────────────────────────────────────────────────

def run_dry_run(project_root: str) -> bool:
    print("=" * 70)
    print("Q62B Probability Export — DRY-RUN VALIDATION")
    print("=" * 70)

    all_ok = True
    emb_dir  = os.path.join(project_root, Q49_EMBEDDING_DIR)
    ckpt_dir = os.path.join(project_root, Q62A_CKPT_DIR)

    checks = [
        ("Q49 embeddings dir",   emb_dir),
        ("train_embeddings.npy", os.path.join(emb_dir, "train_embeddings.npy")),
        ("test_embeddings.npy",  os.path.join(emb_dir, "test_embeddings.npy")),
        ("test_labels.npy",      os.path.join(emb_dir, "test_labels.npy")),
        ("val_embeddings.npy",   os.path.join(emb_dir, "val_embeddings.npy")),
        ("val_labels.npy",       os.path.join(emb_dir, "val_labels.npy")),
        ("Q62A checkpoint dir",  ckpt_dir),
    ]
    for arch in ["cnn", "dv_qnn", "cv_qnn"]:
        for seed in FULL_SEEDS:
            ckpt = os.path.join(ckpt_dir, f"{arch}_seed{seed}.pt")
            checks.append((f"{arch}_seed{seed}.pt", ckpt))

    print("\n[1] File dependencies:")
    for name, path in checks:
        ok = os.path.exists(path)
        print(f"  [{'ok' if ok else 'MISSING'}] {name:<35} {path}")
        if not ok:
            all_ok = False

    print("\n[2] Import validation:")
    for pkg in ["torch", "numpy"]:
        try:
            __import__(pkg)
            print(f"  [ok] {pkg}")
        except ImportError:
            print(f"  [warn] {pkg}")
            all_ok = False

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

    print("\n[4] Output dirs:")
    for rel in [PROB_OUT_DIR, os.path.dirname(REPORT_PATH)]:
        p = os.path.join(project_root, rel)
        os.makedirs(p, exist_ok=True)
        print(f"  [ok] {p}")

    status = "READY" if all_ok else "BLOCKED"
    print(f"\n  Status: {status}")
    print("=" * 70)
    return all_ok


# ── Main benchmark ─────────────────────────────────────────────────────────────

def run_export(seeds: List[int], project_root: str) -> None:
    q49_dir = os.path.join(project_root, Q49_EMBEDDING_DIR)

    test_emb = np.load(os.path.join(q49_dir, "test_embeddings.npy"))
    test_lbl = np.load(os.path.join(q49_dir, "test_labels.npy"))
    val_emb  = np.load(os.path.join(q49_dir, "val_embeddings.npy"))
    val_lbl  = np.load(os.path.join(q49_dir, "val_labels.npy"))

    print(f"  Test: {test_emb.shape}  labels: {test_lbl.shape}", flush=True)
    print(f"  Val:  {val_emb.shape}  labels: {val_lbl.shape}",  flush=True)

    arch_configs = [
        ("cnn",    "Classical CNN",  CNN_BATCH_SIZE),
        ("dv_qnn", "DV-QNN",         DV_BATCH_SIZE),
        ("cv_qnn", "CV-QNN",         CV_BATCH_SIZE),
    ]

    export_results = []
    t_wall = time.time()

    for arch, arch_label, batch_size in arch_configs:
        print(f"\n  -> {arch_label} ({arch})", flush=True)
        result = export_arch(
            arch=arch,
            seeds=seeds,
            test_emb=test_emb,
            test_lbl=test_lbl,
            val_emb=val_emb,
            val_lbl=val_lbl,
            project_root=project_root,
            batch_size=batch_size,
        )
        export_results.append(result)

    elapsed = time.time() - t_wall
    write_report(export_results, seeds, project_root, elapsed)
    print(f"\n  Done.  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q62B Probability Export")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke",   action="store_true", help="1 seed (seed=42)")
    mode.add_argument("--full",    action="store_true", help="3 seeds [42,7,123]")
    mode.add_argument("--dry-run", action="store_true", help="Validate deps; no inference")
    parser.add_argument("--project-root", type=str, default=_PROJECT_ROOT)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    if args.dry_run or (not args.smoke and not args.full):
        ok = run_dry_run(project_root)
        sys.exit(0 if ok else 1)

    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    phase = "SMOKE" if args.smoke else "FULL"

    print("=" * 70)
    print(f"Q62B Probability Export — Phase: {phase}")
    print("=" * 70)
    print(f"  Seeds:   {seeds}")
    print(f"  Test N:  {VINDR_TEST_N:,}")
    print(f"  Val N:   {VINDR_VAL_N:,}")
    print(f"  Checkpoints from: {Q62A_CKPT_DIR}")
    print()

    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    run_export(seeds=seeds, project_root=project_root)


if __name__ == "__main__":
    main()
