"""
scripts/train_q34c_cv_candidate.py

Q34C CV NAS Pilot — per-trial training script.

Reads a YAML config and trains a CV quantum hybrid model (1 trial).
Used by run_q34c_cv_nas_pilot.py as the per-trial command target.

Architecture:
    Input (B, 1, 224, 224)
    -> Frozen C006-D040 backbone [frozen, CUDA if avail] -> (B, 128)
    -> nn.Linear(128, compression_dim) [trainable, CPU]  -> (B, compression_dim)
    -> CappedGaussianAnsatz per-sample loop [CPU]         -> (B, 2*n_modes) mu
    -> nn.Linear(2*n_modes, 2) [readout, CPU]             -> (B, 2) logits

Pilot substitutions (per Q34C-Preflight readiness check):
    beam_splitter_topology:    always circular (hardcoded in GaussianVariationalAnsatz)
    readout_strategy:          always first_moments (mu_final)
    encoding_strategy:         always displacement_encoding
    covariance_parameterization: always symplectic (apply_symplectic throughout)
    displacement_cap:          searched value applied via tanh(disp_raw) x cap (inline)

Output lines (machine-readable, parsed by run_q34c_cv_nas_pilot.py):
    Q34C_TRIAL_AUROC: <float>
    Q34C_TRIAL_F1: <float>
    Q34C_TRIAL_ACCURACY: <float>
    Q34C_TRIAL_PARAMS: <int>
    Q34C_TRIAL_LATENCY_MS: <float>
    Q34C_TRIAL_GRADIENT_HEALTH: PASS|FAIL
    Q34C_TRIAL_NAN_INF: PASS|FAIL
    Q34C_TRIAL_STABILITY_TAXONOMY: valid|covariance_not_psd|unstable_covariance|nan_state|unstable_training
    Q34C_TRIAL_STATUS: completed

Exit 0 on success, nonzero on failure.
"""
from __future__ import annotations

import argparse
import math
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

from qcore.ansatz.cv_spine_ansatz import GaussianVariationalAnsatz
from qcore.backends.cvBackend import GaussianBackend
from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
from qcore.models.cnn import build_model
from qcore.physics.symplectic import (
    get_beamsplitter_matrix,
    get_displacement_vector,
    get_rotation_matrix,
    get_squeezing_matrix,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Constants ──────────────────────────────────────────────────────────────────

CNN_CONFIG = {
    "block_type": "depthwise_sep", "conv_channels": [64, 128], "dropout": 0.3,
    "use_batchnorm": True, "pooling": "adaptive_avg",
    "input_channels": 1, "num_classes": 2,
}
MAX_EPOCHS_ALLOWED   = 5
MAX_TRAINABLE_PARAMS = 100_000
GRAD_NORM_MAX        = 1e6
HBAR                 = 2.0
COV_UNSTABLE_THRESH  = 1000.0   # Q33B §9: covariance diagonal > 1000× vacuum


# ── CappedGaussianAnsatz: displacement_cap via tanh ───────────────────────────

class CappedGaussianAnsatz(GaussianVariationalAnsatz):
    """Adds bounded displacement (tanh(disp_raw) × cap) without modifying qcore."""

    def __init__(self, n_modes, depth, squeezing_cap=1.5, displacement_cap=2.0):
        super().__init__(n_modes, depth, squeezing_cap)
        self.displacement_cap = float(displacement_cap)

    def apply(self, mu, cov, backend, encoded_input=None):
        for d in range(self.depth):
            if encoded_input is not None:
                mu = mu + encoded_input
            for i in range(self.n_modes):
                dr    = torch.tanh(self.disp_real[d, i]) * self.displacement_cap
                di    = torch.tanh(self.disp_imag[d, i]) * self.displacement_cap
                D     = get_displacement_vector(self.n_modes, i,
                                                torch.complex(dr, di), hbar=backend.hbar)
                mu    = mu + D.to(mu.device)
                sq    = torch.tanh(self.squeezing_raw[d, i]) * self.squeezing_cap
                mu, cov = backend.apply_symplectic(mu, cov,
                                                   get_squeezing_matrix(self.n_modes, i, sq))
                mu, cov = backend.apply_symplectic(mu, cov,
                                                   get_rotation_matrix(self.n_modes, i,
                                                                       self.rot_phi[d, i]))
            for i in range(self.n_modes):  # circular beam-splitter entanglement
                m1, m2 = i, (i + 1) % self.n_modes
                mu, cov = backend.apply_symplectic(mu, cov,
                                                   get_rotation_matrix(self.n_modes, m1,
                                                                       self.rot_phi[d, i]))
                mu, cov = backend.apply_symplectic(mu, cov,
                                                   get_beamsplitter_matrix(self.n_modes, m1, m2,
                                                                           self.bs_theta[d, i]))
        return mu, cov


# ── CVCandidateModel ──────────────────────────────────────────────────────────

class CVCandidateModel(nn.Module):
    def __init__(self, backbone, compression, ansatz, backend, readout, n_modes):
        super().__init__()
        self.backbone    = backbone
        self.compression = compression
        self.ansatz      = ansatz
        self.backend     = backend
        self.readout     = readout
        self.n_modes     = n_modes
        self._last_mu = self._last_cov = None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        feats_cpu = feats.detach().cpu()
        scale = math.sqrt(2.0 * self.backend.hbar)
        logits, mus, covs = [], [], []
        for i in range(feats_cpu.shape[0]):
            mu0, cov0   = self.backend.get_vacuum()
            mu_f, cov_f = self.ansatz.apply(mu0, cov0, self.backend,
                                            self.compression(feats_cpu[i]) * scale)
            logits.append(self.readout(mu_f))
            mus.append(mu_f)
            covs.append(cov_f.detach())
        self._last_mu  = torch.stack(mus)
        self._last_cov = torch.stack(covs)
        return torch.stack(logits)


# ── Backbone ──────────────────────────────────────────────────────────────────

def load_backbone(model, ck_path):
    ck  = torch.load(ck_path, map_location="cpu", weights_only=False)
    src = ck.get("model_state_dict") or ck.get("state_dict") or ck
    remapped = {f"backbone.{k}": v for k, v in src.items()
                if k.startswith("0.") or k.startswith("1.")}
    missing, _ = model.load_state_dict(remapped, strict=False)
    if any(k.startswith("backbone.") for k in missing):
        print("ERROR: backbone keys missing after load", flush=True); sys.exit(1)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()


# ── Stability ─────────────────────────────────────────────────────────────────

def check_psd(cov_batch, tol=1e-6):
    """Returns True if all batched covariance matrices are positive semi-definite."""
    for i in range(cov_batch.shape[0]):
        if float(torch.linalg.eigvalsh(cov_batch[i].float()).min()) < -tol:
            return False
    return True


def classify_stability(nan_inf_pass, cov_psd_pass, cov_unstable, grad_health_pass):
    if not nan_inf_pass:     return "nan_state"
    if not cov_psd_pass:     return "covariance_not_psd"
    if cov_unstable:         return "unstable_covariance"
    if not grad_health_pass: return "unstable_training"
    return "valid"


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_split(model, loader, criterion, device=None):
    model.eval()
    y_true, y_pred, y_prob, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            if device is not None:
                x = x.to(device)
            logits = model(x)
            total_loss += criterion(logits, y.to(logits.device)).item()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_prob.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
    has_both = len(set(y_true)) > 1
    return {
        "loss":     total_loss / max(len(loader), 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Q34C CV NAS Trial")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = int(cfg.get("training", {}).get("epochs", 2))
    if epochs > MAX_EPOCHS_ALLOWED:
        print(f"ERROR: epochs={epochs} > ceiling {MAX_EPOCHS_ALLOWED}", flush=True); sys.exit(1)

    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc        = cfg.get("model", {})
    n_modes   = int(mc.get("n_modes", 2))
    cv_depth  = int(mc.get("cv_depth", 1))
    sq_cap    = float(mc.get("squeezing_cap", 1.5))
    disp_cap  = float(mc.get("displacement_cap", 2.0))
    comp_dim  = int(mc.get("compression_dim", 2 * n_modes))
    lr        = float(cfg.get("training", {}).get("learning_rate", 5e-4))
    wd        = float(cfg.get("training", {}).get("weight_decay", 0.0))
    batch_sz  = int(cfg.get("training", {}).get("batch_size", 4))
    ds_root   = cfg.get("dataset", {}).get("root", "data/processed/vindr_binary_roi_224")
    ck_path   = mc.get("checkpoint_path", "checkpoints/c006_d040_classical_anchor.pt")

    print(f"[Q34C] n_modes={n_modes} cv_depth={cv_depth} sq_cap={sq_cap} "
          f"disp_cap={disp_cap} comp_dim={comp_dim} lr={lr} wd={wd} epochs={epochs}", flush=True)
    print(f"[Q34C] substitutions: topology=circular readout=first_moments "
          f"encoding=displacement_encoding cov_param=symplectic", flush=True)

    backbone    = nn.Sequential(*list(build_model(CNN_CONFIG).children())[:4]).to(device)
    for p_ in backbone.parameters(): p_.requires_grad = False
    backbone.eval()
    ansatz  = CappedGaussianAnsatz(n_modes, cv_depth, sq_cap, disp_cap)
    backend = GaussianBackend(n_modes, hbar=HBAR, device="cpu")
    model   = CVCandidateModel(backbone, nn.Linear(128, comp_dim),
                               ansatz, backend, nn.Linear(2 * n_modes, 2), n_modes)
    load_backbone(model, ck_path)

    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[Q34C] trainable_params={n_trainable}", flush=True)
    if n_trainable > MAX_TRAINABLE_PARAMS:
        print(f"ERROR: params={n_trainable} > {MAX_TRAINABLE_PARAMS}", flush=True); sys.exit(1)

    train_ds = VinDrSpineXRBinaryDataset(root=ds_root, split="train")
    test_ds  = VinDrSpineXRBinaryDataset(root=ds_root, split="test")
    train_ld = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad], lr=lr, weight_decay=wd)

    nan_inf_pass = cov_psd_pass = grad_health_pass = True
    cov_unstable = False
    final_train_loss = float("nan")
    gn_last = 0.0

    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.perf_counter(); running = 0.0
        for x, y in train_ld:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y.to(logits.device))
            if not torch.isfinite(loss):
                print("ERROR: non-finite loss", flush=True); nan_inf_pass = False; sys.exit(3)
            loss.backward()

            for n_, prm in model.backbone.named_parameters():
                if prm.grad is not None and prm.grad.abs().max() > 0:
                    print(f"ERROR: backbone gradient on backbone.{n_}", flush=True); sys.exit(3)

            gn_last = float(sum(
                float(p_.grad.detach().norm(2) ** 2)
                for p_ in model.parameters() if p_.requires_grad and p_.grad is not None
            ) ** 0.5)
            if not math.isfinite(gn_last) or gn_last > GRAD_NORM_MAX:
                grad_health_pass = False
            optimizer.step()
            running += loss.item()

            if model._last_mu is not None and not torch.isfinite(model._last_mu).all():
                nan_inf_pass = False
            if model._last_cov is not None:
                if not torch.isfinite(model._last_cov).all():
                    nan_inf_pass = False
                elif not check_psd(model._last_cov):
                    cov_psd_pass = False
                elif float(model._last_cov.diagonal(dim1=-2, dim2=-1).abs().max()) > COV_UNSTABLE_THRESH:
                    cov_unstable = True

        final_train_loss = running / len(train_ld)
        print(f"Epoch {epoch}/{epochs} | loss={final_train_loss:.4f} "
              f"gn={gn_last:.2e} t={time.perf_counter()-t0:.1f}s", flush=True)

    grad_health_pass = grad_health_pass and gn_last > 0.0

    test_m = eval_split(model, test_ld, criterion, device=device)

    model.eval()
    x_lat = torch.randn(1, 1, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(2): model(x_lat)
        t0 = time.perf_counter()
        for _ in range(5): model(x_lat)
    lat_ms = (time.perf_counter() - t0) / 5 * 1000.0

    stability = classify_stability(nan_inf_pass, cov_psd_pass, cov_unstable, grad_health_pass)
    grad_str  = "PASS" if grad_health_pass else "FAIL"
    nan_str   = "PASS" if nan_inf_pass else "FAIL"

    print(f"Q34C_TRIAL_AUROC: {test_m['auroc']:.6f}", flush=True)
    print(f"Q34C_TRIAL_F1: {test_m['f1']:.6f}", flush=True)
    print(f"Q34C_TRIAL_ACCURACY: {test_m['accuracy']:.6f}", flush=True)
    print(f"Q34C_TRIAL_PARAMS: {n_trainable}", flush=True)
    print(f"Q34C_TRIAL_LATENCY_MS: {lat_ms:.4f}", flush=True)
    print(f"Q34C_TRIAL_GRADIENT_HEALTH: {grad_str}", flush=True)
    print(f"Q34C_TRIAL_NAN_INF: {nan_str}", flush=True)
    print(f"Q34C_TRIAL_STABILITY_TAXONOMY: {stability}", flush=True)
    print(f"Q34C_TRIAL_STATUS: completed", flush=True)
    print("[SUMMARY]", flush=True)
    print(f"Trainable params: {n_trainable}", flush=True)
    print(f"Loss: {final_train_loss:.6f}", flush=True)
    print(f"[Q34C] DONE | auroc={test_m['auroc']:.4f} f1={test_m['f1']:.4f} "
          f"params={n_trainable} lat={lat_ms:.2f}ms stability={stability}", flush=True)


if __name__ == "__main__":
    main()
