"""
qcore/experiments/metadata.py

Experiment metadata capture utilities for the QStrata experiment runner.

Functions:
    generate_experiment_id() → str
        Produces a unique experiment identifier: YYYYMMDD_HHMMSS_<6-char hex>.

    capture_git_commit() → dict
        Records the current HEAD commit and dirty-tree flag.
        Priority order for commit SHA:
          1. QSTRATA_GIT_COMMIT environment variable (if non-empty)
          2. `git rev-parse HEAD` subprocess
          3. "unknown" fallback
        Priority order for dirty flag:
          1. QSTRATA_GIT_DIRTY environment variable ("true"/"false", case-insensitive)
          2. `git diff --quiet` subprocess check
          3. True (conservative fallback — assume dirty)

    capture_hardware() → dict
        Records GPU name, CUDA version, CPU fallback flag, and GPU memory.
        Gracefully handles environments where torch is unavailable.

Environment variables:
    QSTRATA_GIT_COMMIT  — inject commit SHA when git is unavailable (e.g., inside
                          a Docker container without git installed).
                          Recommended usage:
                            docker compose exec \\
                              -e QSTRATA_GIT_COMMIT=$(git rev-parse HEAD) \\
                              -e QSTRATA_GIT_DIRTY=false \\
                              qstrata-gpu \\
                              python3 scripts/run_experiment.py --config <path>

    QSTRATA_GIT_DIRTY   — inject dirty-tree flag alongside QSTRATA_GIT_COMMIT.
                          Accepts "true" or "false" (case-insensitive).
                          If omitted when QSTRATA_GIT_COMMIT is set, defaults to False.
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

    Priority order for commit SHA:
        1. QSTRATA_GIT_COMMIT env var — checked first; used if non-empty
        2. git rev-parse HEAD — run as subprocess if git is available
        3. "unknown" — final fallback

    Priority order for dirty flag:
        1. QSTRATA_GIT_DIRTY env var — "true"/"false" (case-insensitive)
        2. git diff --quiet + git diff --cached --quiet subprocess checks
        3. True (conservative: assume dirty if nothing else can determine it)

    When QSTRATA_GIT_COMMIT is set but QSTRATA_GIT_DIRTY is absent, the dirty
    flag defaults to False (caller opted in to env-var injection, implying a
    clean commit was intentionally provided).

    Returns:
        dict with keys:
            "commit" (str):  full SHA or "unknown"
            "dirty"  (bool): True if working tree or index has uncommitted changes
    """
    env_commit = os.environ.get("QSTRATA_GIT_COMMIT", "").strip()
    env_dirty  = os.environ.get("QSTRATA_GIT_DIRTY",  "").strip().lower()

    # ── Commit SHA ────────────────────────────────────────────────────────────
    if env_commit:
        commit = env_commit
        # Dirty flag: use env var if present, else default to False (caller-provided commit)
        if env_dirty in ("true", "1"):
            dirty = True
        elif env_dirty in ("false", "0"):
            dirty = False
        else:
            dirty = False  # default when commit is injected but dirty not specified
        return {"commit": commit, "dirty": dirty}

    # ── Try git subprocess ────────────────────────────────────────────────────
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
        pass

    # ── Final fallback ────────────────────────────────────────────────────────
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
            gpu_model     = torch.cuda.get_device_name(0)
            cuda_version  = torch.version.cuda or "N/A"
            cpu_fallback  = False
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
