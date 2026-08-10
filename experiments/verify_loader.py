import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from qcore.data.npy_spine_dataset import NpySpineDataset
from torch.utils.data import DataLoader

def inspect_npy_pipeline():
    parser = argparse.ArgumentParser(
        description="Sanity check"
    )
    parser.add_argument(
        "--images_npy",
        type=str,
    )

    parser.add_argument(
        "--labels_npy",
        type=str
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("sanity check")
    print("=" * 60)
    print(f"target hardware device: {args.device}")
    print(f"Image source path: {args.images_npy}")
    print(f"labels source path: {args.labels_npy}")

    dataset = NpySpineDataset(
        images_path =args.images_npy,
        labels_path=args.labels_npy,
        to_spatial_rgb=True
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    print(f"Dataset summary")
    print(f"total loaded samples: {len(dataset)}")
    print(f"discovered classes: {dataset.get_total_classes()}")
    print(f"class label list: {dataset.unique_classes}\n")

    images, labels = next(iter(loader))

    print(f"Tensor metrics check")
    print(f"image batch shape: {images.shape} (expected [{args.batch}, 3, 224, 224])")
    print(f"labels batch shape: {labels.shape} (expected: [{args.batch}])")
    print(f"data type (images): {images.dtype}")
    print(f"data type (labels): {labels.dtype}")
    print(f"pixel range: min={images.min():.4f}, max={images.max():.4f}")

    device = torch.device(args.device)
    images_gpu = images.to(device)
    labels_gpu = labels.to(device)
    print(f"gpu allocation test: moved to tensors to {images_gpu.device}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(args.batch, 8)):
        img_np = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].axis("off")

    plt.tight_layout()
    output_path = "npy_preview.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"output visual batch preview saved to: {os.path.abspath(output_path)}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    inspect_npy_pipeline()