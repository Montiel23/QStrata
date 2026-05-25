#!/usr/bin/env python3
"""
scripts/qnn_dv_gradient_restoration_smoke_test.py

Slice Q5 — DV Gradient Restoration Smoke Validation

Validates that replacing np.arctan with torch.atan in medical_ansatz restores
full differentiable gradient flow from loss → readout → quantum circuit →
projection layer.

Uses DVHybridCNNQNN with a synthetic dummy batch (no dataset, no training).
Runs one forward pass → loss → backward → optimizer.step() and confirms:
  - theta grad norm > 0       (was > 0 in Q4, must remain > 0)
  - projection grad norm > 0  (was 0 in Q4 — the critical restored criterion)
  - readout grad norm > 0     (was > 0 in Q4, must remain > 0)
  - max |Δproj| > 0           (projection params must update after step)
  - max |Δtheta| > 0          (theta params must update after step)
  - prob sum ≈ 1.0            (unitary preservation)

13 validation conditions from Slice Q5 specification.

Do not modify any existing source files.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

# ── Configuration ─────────────────────────────────────────────────────────────
N_QUBITS   = 4
DEPTH      = 1
ALPHA      = 0.1
N_CLASSES  = 2
BATCH_SIZE = 2     # small synthetic batch — enough to exercise the loop
LR         = 1e-2

CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

# ── Result accumulator ────────────────────────────────────────────────────────
results: dict[int, tuple[str, bool, str]] = {}


def record(n: int, label: str, passed: bool, detail: str = "") -> None:
    results[n] = (label, passed, detail)


# ─────────────────────────────────────────────────────────────────────────────
print("=== Q5 Gradient Restoration Validation ===")
print()

# ── Import model ──────────────────────────────────────────────────────────────
try:
    from qcore.models.dv_hybrid_cnn_qnn import DVHybridCNNQNN
    model = DVHybridCNNQNN(
        cnn_config=CNN_CONFIG,
        n_qubits=N_QUBITS,
        depth=DEPTH,
        alpha=ALPHA,
        n_classes=N_CLASSES,
    )
    model.train()   # backbone.eval() maintained by override
except Exception as e:
    print(f"FATAL: cannot instantiate DVHybridCNNQNN: {e}")
    for k in range(1, 14):
        record(k, f"Validation {k}", False, f"model instantiation failed: {e}")
    sys.exit(1)

# ── Synthetic dummy batch ─────────────────────────────────────────────────────
torch.manual_seed(42)
x_dummy = torch.randn(BATCH_SIZE, 1, 28, 28)    # (B, 1, 28, 28) — correct input shape
y_dummy = torch.randint(0, N_CLASSES, (BATCH_SIZE,), dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
)

# Snapshot parameters before step
proj_w_before  = model.proj.weight.data.clone()
proj_b_before  = model.proj.bias.data.clone()
theta_before   = model.theta.data.clone()

# ── Validation 1 — Forward pass completes ─────────────────────────────────────
logits = None
try:
    optimizer.zero_grad()
    logits = model(x_dummy)                         # (B, 2)
    record(1, "Forward pass completes", True,
           f"logits.shape={tuple(logits.shape)}")
except Exception as e:
    record(1, "Forward pass completes", False, str(e))
    for k in range(2, 14):
        record(k, f"Validation {k}", False, "forward failed")
    for n, (lbl, ok, det) in results.items():
        print(f"{n:2d}. {lbl:<40}: {'PASS' if ok else 'FAIL'}"
              + (f"  [{det}]" if det else ""))
    print(f"\n=== OVERALL: FAILED ===")
    sys.exit(1)

# ── Validation 2 — Loss computes ──────────────────────────────────────────────
loss = None
try:
    loss = criterion(logits, y_dummy)
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    record(2, "Loss computes", True, f"loss={loss.item():.6f}")
except Exception as e:
    record(2, "Loss computes", False, str(e))
    for k in range(3, 14):
        record(k, f"Validation {k}", False, "loss failed")
    for n, (lbl, ok, det) in results.items():
        print(f"{n:2d}. {lbl:<40}: {'PASS' if ok else 'FAIL'}"
              + (f"  [{det}]" if det else ""))
    print(f"\n=== OVERALL: FAILED ===")
    sys.exit(1)

# ── Validation 3 — loss.backward() completes ─────────────────────────────────
try:
    loss.backward()
    record(3, "loss.backward() completes", True)
except Exception as e:
    record(3, "loss.backward() completes", False, str(e))
    for k in range(4, 14):
        record(k, f"Validation {k}", False, "backward failed")
    for n, (lbl, ok, det) in results.items():
        print(f"{n:2d}. {lbl:<40}: {'PASS' if ok else 'FAIL'}"
              + (f"  [{det}]" if det else ""))
    print(f"\n=== OVERALL: FAILED ===")
    sys.exit(1)

# ── Collect gradient norms ─────────────────────────────────────────────────────
def grad_norm_list(params) -> float:
    """L2 norm of gradients across a list of parameters. Returns 0 if all None."""
    total_sq = sum(
        p.grad.detach().norm().item() ** 2
        for p in params if p.grad is not None
    )
    return float(total_sq ** 0.5)

theta_gn   = grad_norm_list([model.theta])
proj_gn    = grad_norm_list(list(model.proj.parameters()))
readout_gn = grad_norm_list(list(model.readout.parameters()))
prob_mean  = (sum(model._prob_sums) / len(model._prob_sums)) if model._prob_sums else 0.0

# ── Validation 4 — optimizer.step() completes ────────────────────────────────
try:
    optimizer.step()
    record(4, "optimizer.step() completes", True)
except Exception as e:
    record(4, "optimizer.step() completes", False, str(e))
    for k in range(5, 14):
        record(k, f"Validation {k}", False, "optimizer step failed")
    for n, (lbl, ok, det) in results.items():
        print(f"{n:2d}. {lbl:<40}: {'PASS' if ok else 'FAIL'}"
              + (f"  [{det}]" if det else ""))
    print(f"\n=== OVERALL: FAILED ===")
    sys.exit(1)

# ── Parameter deltas ──────────────────────────────────────────────────────────
max_delta_proj = max(
    (model.proj.weight.data - proj_w_before).abs().max().item(),
    (model.proj.bias.data   - proj_b_before).abs().max().item(),
)
max_delta_theta = (model.theta.data - theta_before).abs().max().item()

# ── Print required stdout metrics ─────────────────────────────────────────────
print(f"Theta grad norm      : {theta_gn:.2e}")
print(f"Proj  grad norm      : {proj_gn:.2e}")
print(f"Readout grad norm    : {readout_gn:.2e}")
print()
print(f"Max |Δproj|          : {max_delta_proj:.2e}")
print(f"Max |Δtheta|         : {max_delta_theta:.2e}")
print()
print(f"Prob sum mean        : {prob_mean:.6f}")
print()

# ── Validation 5 — theta grad norm > 0 ───────────────────────────────────────
record(5, "theta grad norm > 0",
       theta_gn > 0.0,
       f"theta_gn={theta_gn:.2e}")

# ── Validation 6 — projection grad norm > 0 (critical gate) ──────────────────
record(6, "projection grad norm > 0  [CRITICAL]",
       proj_gn > 0.0,
       f"proj_gn={proj_gn:.2e}")

# ── Validation 7 — readout grad norm > 0 ─────────────────────────────────────
record(7, "readout grad norm > 0",
       readout_gn > 0.0,
       f"readout_gn={readout_gn:.2e}")

# ── Validation 8 — projection params changed ─────────────────────────────────
record(8, "projection params changed after step",
       max_delta_proj > 0.0,
       f"max|Δproj|={max_delta_proj:.2e}")

# ── Validation 9 — theta params changed ──────────────────────────────────────
record(9, "theta params changed after step",
       max_delta_theta > 0.0,
       f"max|Δtheta|={max_delta_theta:.2e}")

# ── Validation 10 — probability sums ≈ 1.0 ───────────────────────────────────
record(10, "prob sums ≈ 1.0",
       abs(prob_mean - 1.0) < 1e-4,
       f"prob_mean={prob_mean:.6f}")

# ── Validation 11 — all outputs finite ───────────────────────────────────────
logits_det = logits.detach()
record(11, "logits finite",
       bool(torch.isfinite(logits_det).all().item()),
       f"logits_range=[{logits_det.min().item():.4f}, {logits_det.max().item():.4f}]")

# ── Validation 12 — no NaN ───────────────────────────────────────────────────
has_nan = bool(torch.isnan(logits_det).any().item())
record(12, "no NaN in outputs",
       not has_nan,
       f"has_nan={has_nan}")

# ── Validation 13 — no inf ───────────────────────────────────────────────────
has_inf = bool(torch.isinf(logits_det).any().item())
record(13, "no inf in outputs",
       not has_inf,
       f"has_inf={has_inf}")

# ── Print all results ─────────────────────────────────────────────────────────
for n, (lbl, ok, det) in sorted(results.items()):
    status = "PASS" if ok else "FAIL"
    suffix = f"  [{det}]" if det else ""
    print(f"{n:2d}. {lbl:<44}: {status}{suffix}")

print()
all_pass     = all(v[1] for v in results.values())
proj_ok      = results.get(6, (None, False, None))[1]
verdict_text = ("Projection gradient restored"
                if proj_ok else
                "Projection gradient still broken")
print(f"=== Gradient path verdict: {verdict_text} ===")
print()

if all_pass:
    print("=== OVERALL: ALL PASS — Q5 gradient restoration validated ===")
else:
    failed = [k for k, v in results.items() if not v[1]]
    print(f"=== OVERALL: FAILED — see items {failed} above ===")
    sys.exit(1)
