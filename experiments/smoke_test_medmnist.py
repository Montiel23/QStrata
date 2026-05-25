"""
smoke_test_medmnist.py — Structural smoke test for MedMNIST dataset adapters.

No training. No model instantiation.

Usage:
    python experiments/smoke_test_medmnist.py --dataset pneumoniamnist
    python experiments/smoke_test_medmnist.py --dataset pathmnist --split val
"""

import argparse
import sys
import os
import numpy as np

# Ensure project root is on path when invoked from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qcore.data.registry import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MedMNIST dataset adapter smoke test")
    parser.add_argument("--dataset", required=True, help="Dataset name (pneumoniamnist or pathmnist)")
    parser.add_argument("--split",   default="train", help="Split: train, val, or test (default: train)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = get_dataset(args.dataset, args.split)

    dataset_name = dataset.dataset_name
    split        = dataset.split
    task         = dataset.task
    n_classes    = dataset.n_classes
    label_map    = dataset.label_map

    print(f"Dataset:     {dataset_name}")
    print(f"Split:       {split}")
    print(f"Task:        {task}")
    print(f"N classes:   {n_classes}")
    print(f"Label map:   {label_map}")
    print(f"Length:      {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"First label: {dataset[0][1]}")

    # ── Assertions ────────────────────────────────────────────────────────────
    image, label = dataset[0]

    if image.dtype != np.float32:
        print(f"[FAIL] Expected image dtype float32, got {image.dtype}")
        sys.exit(1)

    if not isinstance(label, int):
        print(f"[FAIL] Expected label type int, got {type(label)}")
        sys.exit(1)

    if image.min() < 0.0 or image.max() > 1.0:
        print(f"[FAIL] Image values out of [0, 1] range: min={image.min()}, max={image.max()}")
        sys.exit(1)

    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
