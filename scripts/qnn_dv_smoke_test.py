#!/usr/bin/env python3
"""
scripts/qnn_dv_smoke_test.py

Slice Q2 — DV Minimal Train-Step Smoke Validation

Validates that the existing DV quantum stack runs one complete train step
(forward → loss → backward → optimizer.step() → parameter update) using a
single synthetic sample. No dataset loading. No epoch loop. No training.

Reuses TwoDQClassifier from experiments/models/basic_qmodel.py.
Individual DV stack components (ansatz, backend, vacuum_state,
measure_probability) are validated independently before the integrated step.

Do not modify any existing source files.
"""

from __future__ import annotations

import os
import sys

# ── Repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

# ── Configuration ─────────────────────────────────────────────────────────────
N_QUBITS  = 4
DEPTH     = 1
ALPHA     = 0.1
N_CLASSES = 2
LR        = 1e-2

# ── Result accumulator ────────────────────────────────────────────────────────
results: dict[int, tuple[str, bool, str]] = {}


def record(n: int, label: str, passed: bool, detail: str = "") -> None:
    results[n] = (label, passed, detail)
    status = "PASS" if passed else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"{n:2d}. {label:<36}: {status}{suffix}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
print("=== Q2 DV Minimal Train-Step Smoke Validation ===")
print()

# ── 1. medical_ansatz instantiation ──────────────────────────────────────────
try:
    from qcore.ansatz.medical_ansatz import medical_ansatz, get_ansatz_shape
    # Trivial smoke call: build a circuit with zero weights and zero input
    _shape   = get_ansatz_shape(N_QUBITS, DEPTH)
    _theta_t = torch.zeros(*_shape)
    _x_t     = torch.zeros(N_QUBITS)
    _circ    = medical_ansatz(_x_t, _theta_t, N_QUBITS, DEPTH, ALPHA)
    assert len(_circ) > 0, "circuit has no ops"
    record(1, "medical_ansatz instantiation", True,
           f"circuit ops={len(_circ)}")
except Exception as e:
    record(1, "medical_ansatz instantiation", False, str(e))
    print("\n=== OVERALL: FAILED — cannot import/run medical_ansatz ===")
    sys.exit(1)

# ── 2. Backend initialisation ─────────────────────────────────────────────────
try:
    from qcore.backends.base import Backend
    _backend = Backend()
    record(2, "Backend initialisation", True)
except Exception as e:
    record(2, "Backend initialisation", False, str(e))
    print("\n=== OVERALL: FAILED — cannot initialise Backend ===")
    sys.exit(1)

# ── 3. vacuum_state validity ──────────────────────────────────────────────────
try:
    from qcore.states.vacuum import vacuum_state
    _vac = vacuum_state(N_QUBITS)
    expected_dim = 2 ** N_QUBITS
    assert _vac.shape == (expected_dim,), \
        f"shape={tuple(_vac.shape)}, expected ({expected_dim},)"
    vac_norm = (torch.abs(_vac) ** 2).sum().item()
    assert abs(vac_norm - 1.0) < 1e-5, \
        f"norm={vac_norm:.8f}, expected 1.0"
    record(3, "vacuum_state validity", True,
           f"shape=({expected_dim},) norm={vac_norm:.6f}")
except Exception as e:
    record(3, "vacuum_state validity", False, str(e))
    print("\n=== OVERALL: FAILED — vacuum_state invalid ===")
    sys.exit(1)

# ── 4. measure_probability validity ──────────────────────────────────────────
try:
    from qcore.measurement.probability import measure_probability
    # Run the trivial circuit built in step 1 through the backend
    _U_t   = _backend.compile(_circ)
    _s_t   = vacuum_state(N_QUBITS)
    _out_t = _backend.run(_U_t, _s_t)
    # Per-wire marginal probability P(|1⟩) on wire 0
    _p0 = measure_probability(_out_t, 0, N_QUBITS)
    _p0_val = float(_p0.real)
    assert 0.0 <= _p0_val <= 1.0 + 1e-6, \
        f"P(wire=0)={_p0_val:.6f} out of [0,1]"
    # Full probability distribution via |psi|^2
    _full_p = (torch.abs(_out_t) ** 2)
    _psum   = _full_p.sum().item()
    assert abs(_psum - 1.0) < 1e-4, \
        f"prob_sum={_psum:.6f}, expected 1.0"
    record(4, "measure_probability validity", True,
           f"P(wire0={_p0_val:.4f}) full_sum={_psum:.6f}")
except Exception as e:
    record(4, "measure_probability validity", False, str(e))
    print("\n=== OVERALL: FAILED — measure_probability invalid ===")
    sys.exit(1)

# ── Train-step setup ──────────────────────────────────────────────────────────
try:
    from experiments.models.basic_qmodel import TwoDQClassifier
    model     = TwoDQClassifier(n_qubits=N_QUBITS, depth=DEPTH,
                                alpha=ALPHA, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # CrossEntropyLoss — matches train_medmnist.py pattern
    criterion = nn.CrossEntropyLoss()
    # Synthetic single-sample input: shape (n_qubits,) — matches TwoDQClassifier.forward(x)
    x_dummy = torch.randn(N_QUBITS)
    # Binary label: shape (1,) for CrossEntropyLoss(logits=(1,C), target=(1,))
    y_dummy = torch.tensor([1], dtype=torch.long)
    # Snapshot theta before the optimizer step for update comparison
    theta_before = model.theta.data.clone()
except Exception as e:
    for k in range(5, 14):
        record(k, f"Validation {k}", False, f"setup error: {e}")
    print(f"\n=== OVERALL: FAILED — model setup error: {e} ===")
    sys.exit(1)

# ── 5. Forward pass ───────────────────────────────────────────────────────────
logits = out = None
try:
    optimizer.zero_grad()
    logits, out = model(x_dummy)
    record(5, "Forward pass completes", True,
           f"logits.shape={tuple(logits.shape)}")
except Exception as e:
    record(5, "Forward pass completes", False, str(e))
    for k in range(6, 14):
        record(k, f"Validation {k}", False, "forward failed")
    print(f"\n=== OVERALL: FAILED — forward pass error ===")
    sys.exit(1)

# ── 6. Output shape valid ─────────────────────────────────────────────────────
logits_2d = None
try:
    # model returns logits.view(-1) → shape (N_CLASSES,)
    # CrossEntropyLoss expects (N, C); unsqueeze to (1, N_CLASSES)
    logits_2d = logits.unsqueeze(0)
    assert logits_2d.shape == (1, N_CLASSES), \
        f"expected (1,{N_CLASSES}), got {tuple(logits_2d.shape)}"
    record(6, "Output shape valid", True,
           f"logits.unsqueeze(0).shape={tuple(logits_2d.shape)}")
except Exception as e:
    record(6, "Output shape valid", False, str(e))
    for k in range(7, 14):
        record(k, f"Validation {k}", False, "shape invalid")
    print(f"\n=== OVERALL: FAILED — output shape invalid ===")
    sys.exit(1)

# ── 7. Loss computes ──────────────────────────────────────────────────────────
loss = None
try:
    loss = criterion(logits_2d, y_dummy)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    record(7, "Loss computes", True, f"loss={loss.item():.6f}")
except Exception as e:
    record(7, "Loss computes", False, str(e))
    for k in range(8, 14):
        record(k, f"Validation {k}", False, "loss failed")
    print(f"\n=== OVERALL: FAILED — loss computation error ===")
    sys.exit(1)

# ── 8. loss.backward() ───────────────────────────────────────────────────────
try:
    loss.backward()
    record(8, "loss.backward() completes", True)
except Exception as e:
    record(8, "loss.backward() completes", False, str(e))
    for k in range(9, 14):
        record(k, f"Validation {k}", False, "backward failed")
    print(f"\n=== OVERALL: FAILED — backward pass error ===")
    sys.exit(1)

# ── 9. Nonzero theta gradient ─────────────────────────────────────────────────
try:
    assert model.theta.grad is not None, "theta.grad is None after backward()"
    grad_norm = model.theta.grad.norm().item()
    assert grad_norm > 0.0, f"theta.grad.norm()={grad_norm:.6e} — expected > 0"
    record(9, "Nonzero theta gradient", True,
           f"theta.grad.norm()={grad_norm:.6e}")
except Exception as e:
    record(9, "Nonzero theta gradient", False, str(e))

# ── 10. optimizer.step() ─────────────────────────────────────────────────────
try:
    optimizer.step()
    record(10, "optimizer.step() completes", True)
except Exception as e:
    record(10, "optimizer.step() completes", False, str(e))
    for k in range(11, 14):
        record(k, f"Validation {k}", False, "optimizer step failed")
    print(f"\n=== OVERALL: FAILED — optimizer step error ===")
    sys.exit(1)

# ── 11. Parameter update confirmed ───────────────────────────────────────────
try:
    theta_after = model.theta.data
    assert not torch.equal(theta_before, theta_after), \
        "theta unchanged after optimizer.step() — no parameter update"
    max_delta = (theta_after - theta_before).abs().max().item()
    record(11, "Parameter update confirmed", True,
           f"max|Δtheta|={max_delta:.6e}")
except Exception as e:
    record(11, "Parameter update confirmed", False, str(e))

# ── 12. Outputs finite (no NaN / inf) ────────────────────────────────────────
try:
    logits_det = logits.detach()
    assert torch.isfinite(logits_det).all().item(), \
        f"non-finite logits: {logits_det.tolist()}"
    record(12, "Outputs finite (no NaN / inf)", True,
           f"logits={[round(v, 4) for v in logits_det.tolist()]}")
except Exception as e:
    record(12, "Outputs finite (no NaN / inf)", False, str(e))

# ── 13. Measurements physically valid ────────────────────────────────────────
try:
    # out is the full quantum state vector; |psi_i|^2 must be a probability dist
    probs = (torch.abs(out) ** 2).detach()
    prob_sum  = probs.sum().item()
    all_nonneg = bool((probs >= 0).all().item())
    assert all_nonneg, "negative probabilities detected in state vector"
    assert abs(prob_sum - 1.0) < 1e-4, \
        f"prob_sum={prob_sum:.6f}, expected 1.0"
    record(13, "Measurements physically valid", True,
           f"prob_sum={prob_sum:.6f} all_nonneg={all_nonneg}")
except Exception as e:
    record(13, "Measurements physically valid", False, str(e))

# ── Overall verdict ───────────────────────────────────────────────────────────
print()
all_pass = all(v[1] for v in results.values())
if all_pass:
    print("=== OVERALL: ALL PASS — DV train-step interface validated ===")
else:
    failed_items = [k for k, v in results.items() if not v[1]]
    print(f"=== OVERALL: FAILED — see items {failed_items} above ===")
    sys.exit(1)
