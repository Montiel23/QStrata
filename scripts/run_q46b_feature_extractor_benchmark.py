"""
scripts/run_q46b_feature_extractor_benchmark.py

Q46 — Feature Extractor Benchmark (Backbone Replacement Study).

Benchmarks compact backbone alternatives against the frozen C006-D040 baseline
on VinDr-SpineXR binary classification.

Scientific question:
  Does any compact feature extractor (EfficientNet-B0, MobileNetV3-Small,
  MobileNetV3-Large, ConvNeXt-Tiny) consistently outperform the current
  frozen C006-D040 ResNet baseline on mean AUROC?

Decision rule:
  WINNER: candidate achieves mean_auroc > Q45A baseline (0.7196)
  NEGATIVE: no candidate clears baseline in Phase 2 multi-seed evaluation

Candidates:
  baseline         — C006-D040 frozen ResNet (~11M params, current production)
  efficientnet_b0  — EfficientNet-B0 frozen (~5.3M params)
  mobilenetv3_small — MobileNetV3-Small frozen (~2.5M params)
  mobilenetv3_large — MobileNetV3-Large frozen (~5.5M params)
  convnext_tiny    — ConvNeXt-Tiny frozen (~28.6M params)

Evaluation phases (Q46A protocol):
  Phase 1 — Smoke:    1 seed  [42],           cap 60 min
  Phase 2 — Full:     3 seeds [42, 7, 123],   cap 120 min
  Phase 3 — Extended: 5 seeds [42,7,123,999,2025], cap 60 min (winner only)

Usage:
  # Dry-run (scaffold validation — no training):
  python scripts/run_q46b_feature_extractor_benchmark.py --dry-run

  # Smoke phase (single seed, all candidates):
  python scripts/run_q46b_feature_extractor_benchmark.py --smoke

  # Full phase (3 seeds, all candidates):
  python scripts/run_q46b_feature_extractor_benchmark.py --full

  # Single candidate, specific seed:
  python scripts/run_q46b_feature_extractor_benchmark.py --smoke --candidate efficientnet_b0 --seed 42

  # Docker:
  docker exec docker-qstrata-gpu-1 \\
      python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run

Runtime contract:
  --smoke : Phase 1 — 1 seed × 5 candidates × 4 epochs (cap: 60 min)
  --full  : Phase 2 — 3 seeds × 5 candidates × 4 epochs (cap: 120 min)
  --dry-run: Validates imports, paths, configs; no training
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Q46A protocol constants ────────────────────────────────────────────────────

SLICE_ID = "Q46"

# Seeds (Q46A protocol)
SMOKE_SEEDS  = [42]
FULL_SEEDS   = [42, 7, 123]
EXT_SEEDS    = [42, 7, 123, 999, 2025]

# Training hyperparameters (fixed from Q38A–Q45A)
EPOCHS       = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Q41-optimised DataLoader profile
_OPT_BATCH_SIZE         = 8
_OPT_NUM_WORKERS        = 4
_OPT_PIN_MEMORY         = True
_OPT_PERSISTENT_WORKERS = True
_OPT_PREFETCH_FACTOR    = 2

# Preprocessing (Q38C champion)
CLAHE_CLIP = 3.0
CLAHE_TILE = (4, 4)

# Reference baselines
Q38C_BEST_AUROC  = 0.723922
Q38C_BEST_F1     = 0.677858
Q45A_MEAN_AUROC  = 0.7196   # decision threshold
Q45A_MEAN_F1     = 0.6360

# Runtime caps (minutes)
SMOKE_CAP_MIN = 60
FULL_CAP_MIN  = 120
EXT_CAP_MIN   = 60

# Expected output paths (relative to project root)
BACKBONE_CHECKPOINT = "checkpoints/c006_d040_classical_anchor.pt"
DATASET_ROOT        = "data/processed/vindr_binary_roi_224"
HEAD_CONFIG_PATH    = "configs/experiments/q34a_classical_nas/q34a_trial_004.yaml"

SMOKE_LEADERBOARD_PATH = "experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv"
FULL_LEADERBOARD_PATH  = "experiments/leaderboards/q46b_extractor_full_leaderboard.csv"
RESULTS_JSON_PATH      = "experiments/results/q46b_feature_extractor_benchmark.json"
REPORT_PATH            = "reports/q46b_feature_extractor_benchmark.md"


# ── Leaderboard schema ─────────────────────────────────────────────────────────

# Per-seed result row
LEADERBOARD_FIELDS = [
    "rank",
    "candidate_id",
    "seed",
    "auroc",
    "f1",
    "accuracy",
    "params_backbone",
    "params_head",
    "params_total",
    "latency_ms_per_batch",
    "wall_time_s",
    "delta_auroc_vs_q45a",
    "delta_auroc_vs_q38c",
    "delta_f1_vs_q38c",
]

# Summary aggregation row (Phase 2+)
SUMMARY_FIELDS = [
    "rank",
    "candidate_id",
    "row_type",
    "mean_auroc",
    "std_auroc",
    "ci95_lo_auroc",
    "ci95_hi_auroc",
    "mean_f1",
    "std_f1",
    "ci95_lo_f1",
    "ci95_hi_f1",
    "mean_accuracy",
    "delta_mean_auroc_vs_q45a",
    "delta_mean_auroc_vs_q38c",
    "delta_mean_f1_vs_q38c",
    "seeds_auroc_beat_q45a",
    "seeds_auroc_beat_q38c",
    "decision",
]


# ── Candidate specifications ───────────────────────────────────────────────────

@dataclass
class CandidateSpec:
    candidate_id: str
    backbone_name: str
    description: str
    approx_params: int       # backbone parameter count (approx)
    feature_dim: int         # output feature dimension before head
    requires_projection: bool = False


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec(
        candidate_id="baseline",
        backbone_name="c006_d040_frozen_resnet",
        description="C006-D040 frozen ResNet (current production backbone)",
        approx_params=11_000_000,
        feature_dim=512,
        requires_projection=False,
    ),
    CandidateSpec(
        candidate_id="efficientnet_b0",
        backbone_name="efficientnet_b0",
        description="EfficientNet-B0 frozen (torchvision, ImageNet pretrained)",
        approx_params=5_288_548,
        feature_dim=1280,
        requires_projection=True,
    ),
    CandidateSpec(
        candidate_id="mobilenetv3_small",
        backbone_name="mobilenet_v3_small",
        description="MobileNetV3-Small frozen (torchvision, ImageNet pretrained)",
        approx_params=2_542_856,
        feature_dim=576,
        requires_projection=True,
    ),
    CandidateSpec(
        candidate_id="mobilenetv3_large",
        backbone_name="mobilenet_v3_large",
        description="MobileNetV3-Large frozen (torchvision, ImageNet pretrained)",
        approx_params=5_483_032,
        feature_dim=960,
        requires_projection=True,
    ),
    CandidateSpec(
        candidate_id="convnext_tiny",
        backbone_name="convnext_tiny",
        description="ConvNeXt-Tiny frozen (torchvision, ImageNet pretrained)",
        approx_params=28_589_128,
        feature_dim=768,
        requires_projection=True,
    ),
]

CANDIDATE_MAP: Dict[str, CandidateSpec] = {c.candidate_id: c for c in CANDIDATES}


# ── Dependency validation ──────────────────────────────────────────────────────

@dataclass
class DepCheckResult:
    name: str
    status: str   # OK | MISSING | WARN
    path: str
    note: str = ""


def check_dependencies(project_root: str) -> List[DepCheckResult]:
    """Validate all benchmark dependencies exist."""
    results: List[DepCheckResult] = []

    # Backbone checkpoint
    ck_path = os.path.join(project_root, BACKBONE_CHECKPOINT)
    results.append(DepCheckResult(
        "baseline_checkpoint", "OK" if os.path.isfile(ck_path) else "MISSING",
        ck_path,
    ))

    # Dataset
    ds_path = os.path.join(project_root, DATASET_ROOT)
    if os.path.isdir(ds_path):
        splits_ok = all(os.path.isdir(os.path.join(ds_path, s)) for s in ("train", "val", "test"))
        results.append(DepCheckResult(
            "dataset_root", "OK" if splits_ok else "WARN",
            ds_path,
            note="" if splits_ok else "missing train/val/test splits",
        ))
    else:
        results.append(DepCheckResult("dataset_root", "MISSING", ds_path))

    # Head config
    cfg_path = os.path.join(project_root, HEAD_CONFIG_PATH)
    results.append(DepCheckResult(
        "head_config_q34a_trial_004", "OK" if os.path.isfile(cfg_path) else "MISSING",
        cfg_path,
    ))

    # qcore package
    qcore_path = os.path.join(project_root, "qcore")
    results.append(DepCheckResult(
        "qcore_package", "OK" if os.path.isdir(qcore_path) else "MISSING",
        qcore_path,
    ))

    # Output directories (create if missing)
    for out_rel in [
        "experiments/leaderboards",
        "experiments/results",
        "reports",
    ]:
        out_path = os.path.join(project_root, out_rel)
        os.makedirs(out_path, exist_ok=True)
        results.append(DepCheckResult(
            out_rel.replace("/", "_"), "OK",
            out_path, note="created if missing",
        ))

    return results


def check_backbone_imports() -> List[DepCheckResult]:
    """Validate torchvision backbone imports."""
    results: List[DepCheckResult] = []
    try:
        import torch
        results.append(DepCheckResult("torch", "OK", "", note=torch.__version__))
    except ImportError as e:
        results.append(DepCheckResult("torch", "MISSING", "", note=str(e)))
        return results  # can't check others without torch

    try:
        import torchvision
        results.append(DepCheckResult("torchvision", "OK", "", note=torchvision.__version__))

        from torchvision.models import (
            efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, convnext_tiny
        )
        for name, fn in [
            ("efficientnet_b0", efficientnet_b0),
            ("mobilenetv3_small", mobilenet_v3_small),
            ("mobilenetv3_large", mobilenet_v3_large),
            ("convnext_tiny", convnext_tiny),
        ]:
            try:
                m = fn(weights=None)
                p = sum(q.numel() for q in m.parameters())
                results.append(DepCheckResult(name, "OK", "", note=f"{p:,} params"))
                del m
            except Exception as e:
                results.append(DepCheckResult(name, "WARN", "", note=str(e)))

    except ImportError as e:
        results.append(DepCheckResult("torchvision", "MISSING", "", note=str(e)))

    return results


# ── Backbone builder ───────────────────────────────────────────────────────────

def build_backbone_extractor(candidate: CandidateSpec, project_root: str):
    """
    Return a frozen backbone feature extractor for the given candidate.
    Returns (backbone_module, feature_dim).
    Raises ImportError if torch/torchvision not available.
    """
    import torch
    import torch.nn as nn

    if candidate.candidate_id == "baseline":
        # Load the existing C006-D040 frozen ResNet from checkpoint
        sys.path.insert(0, os.path.join(project_root, "scripts"))
        from run_q38a_preprocessing_benchmark import load_backbone
        ck = os.path.join(project_root, BACKBONE_CHECKPOINT)
        backbone = load_backbone(ck)
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone, candidate.feature_dim

    from torchvision import models

    _FACTORY = {
        "efficientnet_b0":   lambda: models.efficientnet_b0(weights="IMAGENET1K_V1"),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
        "convnext_tiny":     lambda: models.convnext_tiny(weights="IMAGENET1K_V1"),
    }

    if candidate.backbone_name not in _FACTORY:
        raise ValueError(f"Unknown backbone: {candidate.backbone_name}")

    full_model = _FACTORY[candidate.backbone_name]()

    # Strip classifier head; keep feature extraction trunk
    if candidate.backbone_name.startswith("efficientnet"):
        backbone = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
        feat_dim = candidate.feature_dim
    elif candidate.backbone_name.startswith("mobilenet_v3"):
        backbone = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
        feat_dim = candidate.feature_dim
    elif candidate.backbone_name.startswith("convnext"):
        backbone = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
        feat_dim = candidate.feature_dim
    else:
        raise ValueError(f"Unsupported backbone family: {candidate.backbone_name}")

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone, feat_dim


# ── Runtime contract summary ───────────────────────────────────────────────────

RUNTIME_CONTRACT = {
    "smoke": {
        "seeds": SMOKE_SEEDS,
        "candidates": [c.candidate_id for c in CANDIDATES],
        "epochs": EPOCHS,
        "cap_minutes": SMOKE_CAP_MIN,
        "output": SMOKE_LEADERBOARD_PATH,
        "purpose": "Single-seed viability filter — exclude clearly non-viable candidates",
    },
    "full": {
        "seeds": FULL_SEEDS,
        "candidates": [c.candidate_id for c in CANDIDATES],
        "epochs": EPOCHS,
        "cap_minutes": FULL_CAP_MIN,
        "output": FULL_LEADERBOARD_PATH,
        "purpose": "3-seed multi-seed evaluation — primary decision gate",
    },
    "extended": {
        "seeds": EXT_SEEDS,
        "candidates": ["winner"],
        "epochs": EPOCHS,
        "cap_minutes": EXT_CAP_MIN,
        "output": "experiments/leaderboards/q46b_extractor_extended_leaderboard.csv",
        "purpose": "5-seed rigorous validation — only if Phase 2 winner within 0.005 AUROC of ceiling",
    },
}


# ── Dry-run validation ─────────────────────────────────────────────────────────

def run_dry_run(project_root: str, verbose: bool = True) -> bool:
    """
    Validate all dependencies, imports, and config without executing training.
    Returns True if all critical dependencies are satisfied (READY or READY_WITH_WARNINGS).
    """
    print("=" * 70)
    print(f"Q46 Feature Extractor Benchmark — DRY-RUN VALIDATION")
    print("=" * 70)

    # 1. Dependency check
    print("\n[1] Dependency inventory:")
    dep_results = check_dependencies(project_root)
    dep_failures = []
    for r in dep_results:
        sym = "✓" if r.status == "OK" else ("⚠" if r.status == "WARN" else "✗")
        note = f"  ({r.note})" if r.note else ""
        path_display = f"  → {r.path}" if r.path else ""
        print(f"  [{sym}] {r.name:<35} {r.status}{note}{path_display}")
        if r.status == "MISSING":
            dep_failures.append(r.name)

    # 2. Backbone import check
    print("\n[2] Backbone import validation:")
    import_results = check_backbone_imports()
    import_failures = []
    for r in import_results:
        sym = "✓" if r.status == "OK" else ("⚠" if r.status == "WARN" else "✗")
        note = f"  ({r.note})" if r.note else ""
        print(f"  [{sym}] {r.name:<35} {r.status}{note}")
        if r.status == "MISSING":
            import_failures.append(r.name)

    # 3. Leaderboard schema
    print("\n[3] Leaderboard schema:")
    print(f"  Per-seed fields ({len(LEADERBOARD_FIELDS)}):  {', '.join(LEADERBOARD_FIELDS[:6])} ...")
    print(f"  Summary fields  ({len(SUMMARY_FIELDS)}):  {', '.join(SUMMARY_FIELDS[:6])} ...")

    # 4. Runtime contract
    print("\n[4] Runtime contract:")
    for phase, info in RUNTIME_CONTRACT.items():
        seeds_str = str(info["seeds"]) if isinstance(info["seeds"], list) else info["seeds"]
        print(f"  {phase:<10} seeds={seeds_str}  cap={info['cap_minutes']} min  → {info['output']}")

    # 5. Candidate summary
    print("\n[5] Candidate backbone summary:")
    for c in CANDIDATES:
        proj_flag = " + projection" if c.requires_projection else ""
        print(f"  {c.candidate_id:<22} feat_dim={c.feature_dim:<5}  ~{c.approx_params:>11,} params{proj_flag}")

    # 6. Readiness classification
    print("\n[6] Readiness classification:")
    blockers = []
    warnings_list = []

    if "baseline_checkpoint" in dep_failures:
        blockers.append("Baseline checkpoint missing: " + BACKBONE_CHECKPOINT)
    if "dataset_root" in dep_failures:
        blockers.append("Dataset missing: " + DATASET_ROOT)
    if "head_config_q34a_trial_004" in dep_failures:
        blockers.append("Head config missing: " + HEAD_CONFIG_PATH)
    if "torch" in import_failures:
        blockers.append("torch not importable — must run inside docker-qstrata-gpu-1")
    if "torchvision" in import_failures:
        blockers.append("torchvision not importable — must run inside docker-qstrata-gpu-1")

    if dep_failures or import_failures:
        outside_docker = {"torch", "torchvision"}.intersection(set(import_failures))
        if outside_docker and not dep_failures and not (set(import_failures) - outside_docker):
            classification = "READY_WITH_WARNINGS"
            warnings_list.append("torch/torchvision only available inside docker-qstrata-gpu-1")
        else:
            if dep_failures:
                classification = "BLOCKED"
            else:
                classification = "READY_WITH_WARNINGS"
    else:
        classification = "READY"

    print(f"\n  Classification: {classification}")

    if blockers:
        print("\n  Blockers:")
        for b in blockers:
            print(f"    ✗ {b}")
    if warnings_list:
        print("\n  Warnings:")
        for w in warnings_list:
            print(f"    ⚠ {w}")

    if classification == "READY":
        print("\n  → Ready to proceed to Q46 benchmark execution.")
        next_slice = "Q46-FEATURE-EXTRACTOR-BENCHMARK"
    elif classification == "READY_WITH_WARNINGS":
        print("\n  → Ready with warnings. Execute inside docker-qstrata-gpu-1.")
        next_slice = "Q46-FEATURE-EXTRACTOR-BENCHMARK"
    else:
        print("\n  → Blocked. Resolve listed dependencies before execution.")
        next_slice = "Q46D-FEATURE-EXTRACTOR-BENCHMARK-DEPENDENCY-FIX"

    print(f"\n  Recommended next slice: {next_slice}")
    print()
    print("=" * 70)
    passed = classification != "BLOCKED"
    result_str = "PASS" if passed else "FAIL"
    print(f"  DRY-RUN RESULT: {result_str}  (classification: {classification})")
    print("=" * 70)
    return passed


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q46 Feature Extractor Benchmark — Backbone Replacement Study"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke", action="store_true",
        help="Phase 1: single-seed smoke run (seed=42, cap 60 min)",
    )
    mode_group.add_argument(
        "--full", action="store_true",
        help="Phase 2: 3-seed full evaluation (seeds=42,7,123, cap 120 min)",
    )
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Validate dependencies and config without executing training",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=None,
        help="Override seed(s) (e.g. --seed 42 7 123)",
    )
    parser.add_argument(
        "--candidate", type=str, nargs="+", default=None,
        choices=list(CANDIDATE_MAP.keys()),
        help="Limit to specific candidate(s)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for leaderboard CSV and report",
    )
    parser.add_argument(
        "--project-root", type=str, default=_PROJECT_ROOT,
        help="QStrata project root (default: auto-detect from script location)",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    if args.dry_run or (not args.smoke and not args.full):
        # Default to dry-run if no mode specified
        ok = run_dry_run(project_root)
        sys.exit(0 if ok else 1)

    # Resolve seeds and candidates
    if args.smoke:
        seeds = args.seed if args.seed else SMOKE_SEEDS
        cap_minutes = SMOKE_CAP_MIN
        leaderboard_path = os.path.join(
            project_root,
            args.output_dir or SMOKE_LEADERBOARD_PATH,
        )
        phase = "smoke"
    else:
        seeds = args.seed if args.seed else FULL_SEEDS
        cap_minutes = FULL_CAP_MIN
        leaderboard_path = os.path.join(
            project_root,
            args.output_dir or FULL_LEADERBOARD_PATH,
        )
        phase = "full"

    candidates = (
        [CANDIDATE_MAP[c] for c in args.candidate]
        if args.candidate
        else CANDIDATES
    )

    print("=" * 70)
    print(f"Q46 Feature Extractor Benchmark — Phase: {phase.upper()}")
    print("=" * 70)
    print(f"  Seeds:      {seeds}")
    print(f"  Candidates: {[c.candidate_id for c in candidates]}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Cap:        {cap_minutes} min")
    print(f"  Output:     {leaderboard_path}")
    print()

    # Import runtime dependencies
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision.transforms as T
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"ERROR: Cannot import torch/torchvision: {e}", flush=True)
        print("Run inside docker-qstrata-gpu-1:", flush=True)
        print("  docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke")
        sys.exit(1)

    try:
        sys.path.insert(0, os.path.join(project_root, "scripts"))
        from run_q38a_preprocessing_benchmark import (
            _clahe_torch, build_nas_head, eval_split, set_seeds,
            HEAD_CONFIG, EXPECTED_PARAMS,
        )
        from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
    except ImportError as e:
        print(f"ERROR: Cannot import qcore/project modules: {e}", flush=True)
        sys.exit(1)

    # Runtime wall-time cap
    start_wall = time.time()

    rows = []
    os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)

    for candidate in candidates:
        for seed in seeds:
            elapsed = (time.time() - start_wall) / 60.0
            if elapsed >= cap_minutes:
                print(f"\n[WALL-TIME CAP] {cap_minutes} min reached after {elapsed:.1f} min — stopping.")
                break

            print(f"\n  → {candidate.candidate_id}  seed={seed}")
            set_seeds(seed)

            # Build backbone + optional projection + head
            try:
                backbone, feat_dim = build_backbone_extractor(candidate, project_root)
            except Exception as e:
                print(f"    ERROR building backbone: {e}")
                continue

            # Optional projection to match head input dimension
            head_in_dim = HEAD_CONFIG.get("input_dim", feat_dim)
            if feat_dim != head_in_dim:
                projection = nn.Linear(feat_dim, head_in_dim)
            else:
                projection = nn.Identity()

            head = build_nas_head(HEAD_CONFIG)
            model = nn.Sequential(backbone, projection, head)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # DataLoader (Q41-optimised profile)
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Lambda(lambda x: _clahe_torch(x, CLAHE_CLIP, CLAHE_TILE)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            ds_root = os.path.join(project_root, DATASET_ROOT)
            train_ds = VinDrSpineXRBinaryDataset(root=ds_root, split="train", transform=transform)
            val_ds   = VinDrSpineXRBinaryDataset(root=ds_root, split="val",   transform=transform)

            train_loader = DataLoader(
                train_ds, batch_size=_OPT_BATCH_SIZE,
                num_workers=_OPT_NUM_WORKERS,
                pin_memory=_OPT_PIN_MEMORY,
                persistent_workers=_OPT_PERSISTENT_WORKERS,
                prefetch_factor=_OPT_PREFETCH_FACTOR,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=_OPT_BATCH_SIZE,
                num_workers=_OPT_NUM_WORKERS,
                pin_memory=_OPT_PIN_MEMORY,
                persistent_workers=_OPT_PERSISTENT_WORKERS,
                prefetch_factor=_OPT_PREFETCH_FACTOR,
                shuffle=False,
            )

            criterion = nn.BCEWithLogitsLoss()
            # Only head parameters are trained; backbone is frozen
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

            epoch_start = time.time()
            for epoch in range(EPOCHS):
                model.train()
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.float().to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(imgs).squeeze(1), labels)
                    loss.backward()
                    optimizer.step()

                    elapsed_inner = (time.time() - start_wall) / 60.0
                    if elapsed_inner >= cap_minutes:
                        print(f"    [CAP] Wall-time cap reached during epoch {epoch+1}.")
                        break

            wall_time_s = time.time() - epoch_start
            metrics = eval_split(model, val_loader, device)
            auroc    = metrics.get("auroc", 0.0)
            f1       = metrics.get("f1", 0.0)
            accuracy = metrics.get("accuracy", 0.0)

            p_backbone = sum(p.numel() for p in backbone.parameters())
            p_head     = sum(p.numel() for p in head.parameters())

            row = {
                "rank": "",
                "candidate_id": candidate.candidate_id,
                "seed": seed,
                "auroc": round(auroc, 6),
                "f1": round(f1, 6),
                "accuracy": round(accuracy, 6),
                "params_backbone": p_backbone,
                "params_head": p_head,
                "params_total": p_backbone + p_head,
                "latency_ms_per_batch": round(wall_time_s / max(len(train_loader), 1) * 1000, 2),
                "wall_time_s": round(wall_time_s, 2),
                "delta_auroc_vs_q45a": round(auroc - Q45A_MEAN_AUROC, 6),
                "delta_auroc_vs_q38c": round(auroc - Q38C_BEST_AUROC, 6),
                "delta_f1_vs_q38c":   round(f1   - Q38C_BEST_F1,    6),
            }
            rows.append(row)
            print(f"    AUROC={auroc:.4f}  F1={f1:.4f}  Δvs_q45a={auroc - Q45A_MEAN_AUROC:+.4f}  wall={wall_time_s:.1f}s")

    # Write leaderboard CSV
    if rows:
        # Sort by AUROC desc, assign rank
        rows.sort(key=lambda r: r["auroc"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        with open(leaderboard_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Leaderboard written: {leaderboard_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
