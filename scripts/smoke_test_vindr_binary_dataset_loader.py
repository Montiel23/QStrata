"""
scripts/smoke_test_vindr_binary_dataset_loader.py

Smoke test for VinDrSpineXRBinaryDataset.

Validates tensor shape, dtype, range, labels, DataLoader batching, and
metadata return for all three splits (train, val, test).

Usage:
    python scripts/smoke_test_vindr_binary_dataset_loader.py \
        --root data/processed/vindr_binary_roi_224 \
        --batch-size 8
"""

import argparse
import sys

import torch
from torch.utils.data import DataLoader

# ── Import dataset class ───────────────────────────────────────────────────
# Use sys.path insertion so the script works when invoked from repo root
# without package installation (consistent with other scripts in this project).
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset

# Required metadata keys
REQUIRED_META_KEYS = (
    "sample_id",
    "image_id",
    "split",
    "binary_label",
    "original_label",
    "output_path",
    "crop_strategy",
    "has_bbox",
    "is_pseudo_roi",
    "background_fraction",
    "fallback_used",
)

EXPECTED_LENGTHS = {
    "train": 6712,
    "val":   1677,
    "test":  2077,
}

EXPECTED_COUNTS = {
    "train": {0: 3408, 1: 3304},
    "val":   {0: 852,  1: 825},
    "test":  {0: 1070, 1: 1007},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for VinDrSpineXRBinaryDataset"
    )
    parser.add_argument(
        "--root",
        default="data/processed/vindr_binary_roi_224",
        help="Exported dataset root directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="DataLoader batch size",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


def main() -> None:
    args = parse_args()
    root = args.root
    batch_size = args.batch_size

    print("=== VinDr-SpineXR Binary Dataset Loader Smoke Test ===")
    print(f"Root: {root}")
    print()

    # ── 1. Instantiate all three splits ───────────────────────────────────────
    datasets: dict[str, VinDrSpineXRBinaryDataset] = {}
    for split in ("train", "val", "test"):
        try:
            datasets[split] = VinDrSpineXRBinaryDataset(
                root=root, split=split, return_metadata=False
            )
        except Exception as exc:
            fail(f"Could not instantiate split='{split}': {exc}")

    # ── 2. Lengths ─────────────────────────────────────────────────────────────
    print("Dataset lengths:")
    for split in ("train", "val", "test"):
        n = len(datasets[split])
        expected = EXPECTED_LENGTHS[split]
        tag = "" if n == expected else f"  *** EXPECTED {expected} ***"
        print(f"  {split}: {n}{tag}")
    print()

    # ── 3. Class counts ────────────────────────────────────────────────────────
    print("Class counts:")
    for split in ("train", "val", "test"):
        counts = datasets[split].class_counts()
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        exp0 = EXPECTED_COUNTS[split][0]
        exp1 = EXPECTED_COUNTS[split][1]
        tag0 = "" if c0 == exp0 else f"  *** EXPECTED {exp0} ***"
        tag1 = "" if c1 == exp1 else f"  *** EXPECTED {exp1} ***"
        print(f"  {split}: label0={c0}{tag0}, label1={c1}{tag1}")
    print()

    # ── 4. Single-sample checks ────────────────────────────────────────────────
    print("Sample checks:")
    for split in ("train", "val", "test"):
        img, label = datasets[split][0]

        # Shape
        if tuple(img.shape) != (1, 224, 224):
            fail(
                f"{split} sample: expected shape (1,224,224), "
                f"got {tuple(img.shape)}"
            )
        # Dtype
        if img.dtype != torch.float32:
            fail(f"{split} sample: expected float32, got {img.dtype}")
        # Range
        vmin, vmax = img.min().item(), img.max().item()
        if vmin < 0.0 or vmax > 1.0:
            fail(
                f"{split} sample: tensor range [{vmin:.4f}, {vmax:.4f}] "
                "out of [0.0, 1.0]"
            )
        # Label
        lv = label.item()
        if lv not in (0, 1):
            fail(f"{split} sample: label value {lv} not in {{0, 1}}")

        print(
            f"  {split}: image_shape={tuple(img.shape)}, "
            f"dtype={img.dtype}, "
            f"range=[{vmin:.4f}, {vmax:.4f}], "
            f"label={lv}"
        )
    print()

    # ── 5–6. DataLoader batch checks ──────────────────────────────────────────
    print("Batch checks:")
    for split in ("train", "val", "test"):
        shuffle = split == "train"
        loader = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
        )

        imgs_batch, labels_batch = next(iter(loader))

        # Determine expected batch size (may be smaller for tiny splits)
        n = len(datasets[split])
        expected_bs = min(batch_size, n)

        # Batch image shape
        exp_img_shape = (expected_bs, 1, 224, 224)
        got_img_shape = tuple(imgs_batch.shape)
        if got_img_shape != exp_img_shape:
            fail(
                f"{split} batch: expected image shape {exp_img_shape}, "
                f"got {got_img_shape}"
            )

        # Batch labels shape
        exp_lbl_shape = (expected_bs,)
        got_lbl_shape = tuple(labels_batch.shape)
        if got_lbl_shape != exp_lbl_shape:
            fail(
                f"{split} batch: expected labels shape {exp_lbl_shape}, "
                f"got {got_lbl_shape}"
            )

        # Batch label values
        uniq = set(labels_batch.tolist())
        if not uniq.issubset({0, 1}):
            fail(f"{split} batch: unexpected label values {uniq}")

        print(
            f"  {split}: batch_shape={got_img_shape}, "
            f"labels_shape={got_lbl_shape}"
        )
    print()

    # ── 7. Metadata check ─────────────────────────────────────────────────────
    ds_meta = VinDrSpineXRBinaryDataset(
        root=root, split="train", return_metadata=True
    )
    result = ds_meta[0]
    if len(result) != 3:
        fail(
            f"return_metadata=True: expected 3-tuple (img, label, meta), "
            f"got {len(result)}-tuple"
        )
    _, _, meta = result
    if not isinstance(meta, dict):
        fail(f"Metadata is not a dict: {type(meta)}")

    missing_keys = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing_keys:
        fail(f"Metadata dict missing keys: {missing_keys}")

    print("Metadata check: PASS")
    print()

    # ── Length / count validation ──────────────────────────────────────────────
    for split in ("train", "val", "test"):
        n = len(datasets[split])
        if n != EXPECTED_LENGTHS[split]:
            fail(
                f"{split} dataset length {n} != expected {EXPECTED_LENGTHS[split]}"
            )
        counts = datasets[split].class_counts()
        for lbl, exp in EXPECTED_COUNTS[split].items():
            got = counts.get(lbl, 0)
            if got != exp:
                fail(
                    f"{split} label={lbl} count {got} != expected {exp}"
                )

    print("Smoke test: PASS")


if __name__ == "__main__":
    main()
