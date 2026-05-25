"""
scripts/smoke_depthwise_config.py — Smoke test for build_model() block_type extension.

Checks:
  1. standard block (binary_baseline.yaml, block_type omitted)
  2. depthwise_sep block (binary_baseline_depthwise_sep.yaml)
  3. parameter counts differ between the two

No training. No dataset loading. No optimizer.
"""

import os
import sys

import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.models.cnn import build_model

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")
INPUT = torch.randn(4, 1, 28, 28)

passed = []
failed = []


def report(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {label}{suffix}")
    (passed if ok else failed).append(label)


# ── Check 1: standard baseline (block_type omitted from config) ───────────────
print("\nCheck 1 — standard baseline (block_type omitted)")
try:
    with open(os.path.join(CONFIGS_DIR, "binary_baseline.yaml")) as f:
        cfg = yaml.safe_load(f)

    model_cfg = {}
    for k, v in cfg["model"].items():
        model_cfg[k] = v
    for k, v in cfg["dataset"].items():
        model_cfg[k] = v

    model_std = build_model(model_cfg)
    model_std.eval()

    with torch.no_grad():
        logits_std = model_std(INPUT)

    params_std = sum(p.numel() for p in model_std.parameters())
    nan_std    = torch.isnan(logits_std).any().item()

    report("logits shape (4, 2)",  tuple(logits_std.shape) == (4, 2),
           str(tuple(logits_std.shape)))
    report("no NaNs in logits",    not nan_std)
    report("param count > 0",      params_std > 0, str(params_std))
    print(f"  logits shape  : {tuple(logits_std.shape)}")
    print(f"  param count   : {params_std}")

except Exception as exc:
    report("Check 1 exception", False, str(exc))
    params_std = None
    logits_std = None

# ── Check 2: depthwise_sep ────────────────────────────────────────────────────
print("\nCheck 2 — depthwise_sep block")
try:
    with open(os.path.join(CONFIGS_DIR, "binary_baseline_depthwise_sep.yaml")) as f:
        cfg_dw = yaml.safe_load(f)

    model_cfg_dw = {}
    for k, v in cfg_dw["model"].items():
        model_cfg_dw[k] = v
    for k, v in cfg_dw["dataset"].items():
        model_cfg_dw[k] = v

    model_dw = build_model(model_cfg_dw)
    model_dw.eval()

    with torch.no_grad():
        logits_dw = model_dw(INPUT)

    params_dw = sum(p.numel() for p in model_dw.parameters())
    nan_dw    = torch.isnan(logits_dw).any().item()

    report("logits shape (4, 2)",  tuple(logits_dw.shape) == (4, 2),
           str(tuple(logits_dw.shape)))
    report("no NaNs in logits",    not nan_dw)
    report("param count > 0",      params_dw > 0, str(params_dw))
    print(f"  logits shape  : {tuple(logits_dw.shape)}")
    print(f"  param count   : {params_dw}")

except Exception as exc:
    report("Check 2 exception", False, str(exc))
    params_dw = None

# ── Check 3: parameter counts differ ─────────────────────────────────────────
print("\nCheck 3 — parameter counts differ")
if params_std is not None and params_dw is not None:
    counts_differ = params_std != params_dw
    report(
        "standard params != depthwise_sep params",
        counts_differ,
        f"standard={params_std}  depthwise_sep={params_dw}",
    )
else:
    report("parameter count comparison (skipped — prior check failed)", False)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failed:
    print(f"CHECKS FAILED — see above  ({len(failed)} failed, {len(passed)} passed)")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED  ({len(passed)} checks)")
