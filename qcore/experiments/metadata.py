"""
qcore/experiments/metadata.py

Experiment metadata capture utilities for the QStrata experiment runner (Q31 MVP).

Functions:
    generate_experiment_id() → str
        Produces a unique experiment identifier: YYYYMMDD_HHMMSS_<6-char hex>.

    capture_git_commit() → dict
        Records the current HEAD commit and dirty-tree flag.
        Returns {"commit": "unknown", "dirty": True} on any failure.

    capture_hardware() → dict
        Records GPU name, CUDA version, CPU fallback flag, and GPU memory.
        Gracefully handles environments where torch is unavailable.
"""

from __future__ import annotations

import datetime
import os
import subprocess


def generate_experiment_id() -> str:
    """
    Generate a unique experiment identifier.

    Format: YYYYMMDD_HHMMSS_<6-char hex>
    Example: 20260526_143021_a3f7c2

    The 6-char hex suffix is derived from 3 bytes of OS entropy,
    ensuring uniqueness even for experiments started in the same second.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = os.urandom(3).hex()  # 6 hex characters
    return f"{ts}_{suffix}"


def capture_git_commit() -> dict:
    """
    Capture the current HEAD commit SHA and dirty-tree status.

    Returns:
        dict with keys:
            "commit" (str): full 40-char SHA, or "unknown" on failure
            "dirty"  (bool): True if working tree or index has uncommitted changes

    On any subprocess failure (git not installed, not in a repo, etc.),
    returns {"commit": "unknown", "dirty": True} to flag reproducibility risk.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Check both unstaged and staged changes
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
        )
        dirty = (unstaged.returncode != 0) or (staged.returncode != 0)

        return {"commit": commit, "dirty": dirty}

    except Exception:
        return {"commit": "unknown", "dirty": True}


def capture_hardware() -> dict:
    """
    Capture hardware metadata at run start.

    Returns:
        dict with keys:
            "gpu_model"     (str):      GPU device name, or "cpu-only"
            "cuda_version"  (str):      CUDA version string, or "N/A"
            "cpu_fallback"  (bool):     True if no CUDA device available
            "gpu_memory_mb" (int|None): Total GPU memory in MiB, or None

    Returns safe defaults if torch is not importable or hardware query fails.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_model    = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "N/A"
            cpu_fallback = False
            gpu_memory_mb = (
                torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            )
        else:
            gpu_model     = "cpu-only"
            cuda_version  = "N/A"
            cpu_fallback  = True
            gpu_memory_mb = None

        return {
            "gpu_model":     gpu_model,
            "cuda_version":  cuda_version,
            "cpu_fallback":  cpu_fallback,
            "gpu_memory_mb": gpu_memory_mb,
        }

    except Exception:
        return {
            "gpu_model":     "unknown",
            "cuda_version":  "N/A",
            "cpu_fallback":  True,
            "gpu_memory_mb": None,
        }
