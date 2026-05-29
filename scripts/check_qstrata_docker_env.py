"""
Q38D — QStrata Docker environment self-check.

Runs inside the target container (CPU or GPU) and reports the
execution environment against the expected spec.  Exits 0 on pass,
1 on any check failure.

Usage (from host):
    docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/check_qstrata_docker_env.py
    docker exec qstrata-eda        python3 /workspace/scripts/check_qstrata_docker_env.py
"""

import importlib.util
import os
import platform
import sys

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[34mINFO\033[0m"

failures: list[str] = []
warnings: list[str] = []


def _check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    tag = PASS if ok else (WARN if warn_only else FAIL)
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    if not ok:
        if warn_only:
            warnings.append(label)
        else:
            failures.append(label)


def section(title: str) -> None:
    print(f"\n── {title} {'─' * (60 - len(title))}")


# ── Python ────────────────────────────────────────────────────────────────────

section("Python runtime")
py_ver = sys.version_info
_check("Python >= 3.10", py_ver >= (3, 10), f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}")
_check("Running inside container (/workspace exists)", os.path.isdir("/workspace"))

# ── PYTHONPATH ────────────────────────────────────────────────────────────────

section("PYTHONPATH")
pythonpath = os.environ.get("PYTHONPATH", "")
ws_in_path = "/workspace" in pythonpath or "/workspace" in sys.path
_check("PYTHONPATH includes /workspace", ws_in_path, pythonpath or "<not set>", warn_only=True)

# ── PyTorch ───────────────────────────────────────────────────────────────────

section("PyTorch")
try:
    import torch  # noqa: E402  (import after path check intentional)
    torch_ver = torch.__version__
    _check("torch importable", True, torch_ver)
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        _check("CUDA available", True, f"{gpu_name} / CUDA {cuda_ver}")
        _check("CUDA version >= 12.1", cuda_ver >= "12.1", cuda_ver)
        _check("torch build targets cu121", "+cu121" in torch_ver, torch_ver)
    else:
        _check("CPU-only build (no CUDA)", True, torch_ver)
        _check("torch wheel is CPU variant", "cpu" in torch_ver or "+cu" not in torch_ver, torch_ver)
except ImportError as exc:
    _check("torch importable", False, str(exc))

# ── NumPy ─────────────────────────────────────────────────────────────────────

section("NumPy")
try:
    import numpy as np
    np_ver = np.__version__
    _check("numpy importable", True, np_ver)
    major = int(np_ver.split(".")[0])
    # GPU image was built with "numpy<2" intent; 2.x works but triggers ABI warning.
    # CPU image uses system pip default (currently 2.x).
    # Both are functional; flag 2.x as a warning so rebuilds can pin correctly.
    if major >= 2:
        _check(
            "numpy < 2 (ABI-safe for torch 2.2.x+cu121)",
            False,
            f"{np_ver} — ABI warning at import; functional but Dockerfile pin drifted",
            warn_only=True,
        )
    else:
        _check("numpy < 2 (ABI pin satisfied)", True, np_ver)
except ImportError as exc:
    _check("numpy importable", False, str(exc))

# ── Key project dependencies ──────────────────────────────────────────────────

section("Project dependencies")
required_packages = [
    "medmnist",
    "sklearn",   # scikit-learn
    "PIL",       # Pillow
    "pydicom",
    "tqdm",
    "yaml",      # pyyaml
    "matplotlib",
]
for pkg in required_packages:
    spec = importlib.util.find_spec(pkg)
    _check(f"{pkg} importable", spec is not None)

# ── Data mounts ───────────────────────────────────────────────────────────────

section("Data mount expectations")
workspace_data = "/workspace/data/processed/vindr_binary_roi_224"
dataset_mount = "/datasets/vindr-spinexr"
_check(
    "Processed VinDr ROI dataset present (/workspace/data/processed/vindr_binary_roi_224)",
    os.path.isdir(workspace_data),
    workspace_data,
)
_check(
    "Raw VinDr dataset mount (/datasets/vindr-spinexr) — GPU compose only",
    os.path.isdir(dataset_mount),
    dataset_mount,
    warn_only=True,
)

# ── Checkpoint mounts ─────────────────────────────────────────────────────────

section("Checkpoint mounts")
checkpoint = "/workspace/checkpoints/c006_d040_classical_anchor.pt"
_check(
    "Canonical backbone checkpoint present (c006_d040_classical_anchor.pt)",
    os.path.isfile(checkpoint),
    checkpoint,
)

# ── Output paths ─────────────────────────────────────────────────────────────

section("Output path expectations")
output_dirs = [
    "/workspace/experiments/leaderboards",
    "/workspace/experiments/results",
    "/workspace/experiments/logs",
    "/workspace/reports",
]
for d in output_dirs:
    _check(f"{d} writable", os.path.isdir(d) and os.access(d, os.W_OK), d)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "═" * 64)
print(f"  Platform : {platform.platform()}")
print(f"  Python   : {sys.version.split()[0]}")
print(f"  Failures : {len(failures)}")
print(f"  Warnings : {len(warnings)}")
print("═" * 64)

if failures:
    print(f"\n[{FAIL}] {len(failures)} check(s) failed:")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
elif warnings:
    print(f"\n[{WARN}] All required checks pass ({len(warnings)} warning(s) noted above).")
    sys.exit(0)
else:
    print(f"\n[{PASS}] All checks passed.")
    sys.exit(0)
