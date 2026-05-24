"""
smoke_test_loader.py — Slice-1 validation for SpineCascadeDataset

Expected CSV columns: image_id, xmin, ymin, xmax, ymax, class_id

Usage (real data):
    python experiments/smoke_test_loader.py \
        --csv_file /path/to/annotations.csv \
        --img_dir /path/to/dicoms/

Usage (synthetic stub — paths that don't exist trigger auto-generation):
    python experiments/smoke_test_loader.py \
        --csv_file /nonexistent/path.csv \
        --img_dir /nonexistent/dir/
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is on the path so qcore can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qcore.data.spine_dataset import SpineCascadeDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SpineCascadeDataset smoke test")
    parser.add_argument("--csv_file",        required=True,  help="Path to annotation CSV")
    parser.add_argument("--img_dir",         required=True,  help="Path to DICOM image directory")
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--patch_size",      type=int, default=28)
    parser.add_argument("--no_background",   action="store_true",
                        help="Disable background sampling")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic stub generation (Phase 3)
# ---------------------------------------------------------------------------

def generate_synthetic_stub():
    """Create a temp dir with a stub CSV and 5 minimal DICOM files."""
    try:
        import pydicom
        from pydicom.dataset import Dataset as DicomDataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import pydicom.uid
    except ImportError:
        print("[ERROR] pydicom not installed — cannot generate synthetic stub.")
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="qstrata_smoke_")
    print(f"[INFO] Data paths not found. Generating synthetic stub for loader validation.")
    print(f"[INFO] Stub directory: {tmp_dir}")

    # --- Stub CSV ---
    csv_data = {
        "image_id": ["img_001", "img_001", "img_002", "img_002", "img_003",
                     "img_003", "img_004", "img_004", "img_005", "img_005"],
        "xmin":     [10,  50,  20,  80,  15,  60,  30,  90,  25,  70],
        "ymin":     [10,  50,  20,  80,  15,  60,  30,  90,  25,  70],
        "xmax":     [60, 100,  70, 130,  65, 110,  80, 140,  75, 120],
        "ymax":     [60, 100,  70, 130,  65, 110,  80, 140,  75, 120],
        "class_id": [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
    }
    csv_path = os.path.join(tmp_dir, "stub_annotations.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"[INFO] Stub CSV written: {csv_path}")

    # --- Stub DICOMs ---
    unique_ids = ["img_001", "img_002", "img_003", "img_004", "img_005"]
    rng = np.random.default_rng(seed=42)

    for img_id in unique_ids:
        pixel_array = rng.integers(100, 2001, size=(512, 512), dtype=np.uint16)

        # Build a minimal FileDataset
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = pydicom.uid.DigitalXRayImageStorageForPresentation
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.is_implicit_VR  = False
        ds.is_little_endian = True

        ds.Rows                      = 512
        ds.Columns                   = 512
        ds.BitsAllocated             = 16
        ds.BitsStored                = 16
        ds.HighBit                   = 15
        ds.PixelRepresentation       = 0
        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope              = 1.0
        ds.RescaleIntercept          = -1000.0
        ds.PixelData                 = pixel_array.tobytes()

        dicom_path = os.path.join(tmp_dir, f"{img_id}.dicom")
        ds.save_as(dicom_path)

    print(f"[INFO] Stub DICOMs written for: {unique_ids}")
    return csv_path, tmp_dir


# ---------------------------------------------------------------------------
# Smoke test checks (Phase 4)
# ---------------------------------------------------------------------------

def run_smoke_test(csv_file, img_dir, patch_size, batch_size, include_background):
    ps = (patch_size, patch_size)
    expected_flat = patch_size * patch_size

    # ------------------------------------------------------------------
    # CHECK 1 — Instantiation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 1 — Instantiation")
    print("=" * 60)
    dataset = SpineCascadeDataset(csv_file, img_dir, patch_size=ps,
                                  include_background=include_background)
    print("Dataset instantiated OK")
    print(f"  Total samples (with background): {len(dataset)}")
    print(f"  Annotation rows: {len(dataset.bbox_annotations)}")
    print(f"  Class map: {dataset.class_to_idx}")
    print(f"  Total classes: {dataset.get_total_classes()}")

    # ------------------------------------------------------------------
    # CHECK 2 — Single item fetch
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 2 — Single item fetch")
    print("=" * 60)

    indices_to_check = [0, len(dataset) - 1]
    for i in indices_to_check:
        patch, label = dataset[i]

        if patch.shape != (expected_flat,):
            print(f"[FAIL] idx={i}: patch.shape={patch.shape}, expected ({expected_flat},)")
            sys.exit(1)
        if label.dtype != torch.int64:
            print(f"[FAIL] idx={i}: label.dtype={label.dtype}, expected torch.int64")
            sys.exit(1)
        if not (0.0 <= patch.min() and patch.max() <= 1.0):
            print(f"[FAIL] idx={i}: patch values out of [0,1] range "
                  f"(min={patch.min():.4f}, max={patch.max():.4f})")
            sys.exit(1)

        print(f"  idx={i} | patch shape: {patch.shape} | label: {label.item()} | "
              f"min: {patch.min():.4f} | max: {patch.max():.4f} | "
              f"norm: {torch.norm(patch):.4f} | zeros: {(patch == 0).sum().item()}")

    print("Single item fetch OK")

    # ------------------------------------------------------------------
    # CHECK 3 — Full batch iteration (2 batches)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 3 — Full batch iteration (2 batches)")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (batch_patches, batch_labels) in enumerate(loader):
        if batch_idx >= 2:
            break
        print(f"  Batch {batch_idx}: patches={batch_patches.shape}  labels={batch_labels.shape}")
        unique_labels, counts = torch.unique(batch_labels, return_counts=True)
        dist_str = ", ".join(f"class {l.item()}:{c.item()}" for l, c in zip(unique_labels, counts))
        print(f"    Label distribution: {dist_str}")
        print(f"    Patch mean: {batch_patches.mean():.4f}  std: {batch_patches.std():.4f}")

    print("Batch iteration OK — 2 batches consumed without error.")

    # ------------------------------------------------------------------
    # CHECK 4 — Zero-patch rate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 4 — Zero-patch rate")
    print("=" * 60)

    n_zeros = 0
    for i in range(len(dataset)):
        patch, _ = dataset[i]
        if torch.norm(patch) == 0.0:
            n_zeros += 1

    pct = 100.0 * n_zeros / len(dataset)
    print(f"Zero-patch rate: {n_zeros}/{len(dataset)} ({pct:.1f}%)")
    if pct > 10.0:
        print(f"[WARN] Zero-patch rate {pct:.1f}% exceeds 10% threshold — check DICOM paths.")

    # ------------------------------------------------------------------
    # CHECK 5 — Class balance summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CHECK 5 — Class balance summary")
    print("=" * 60)

    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item())

    label_tensor = torch.tensor(all_labels)
    unique_labels, counts = torch.unique(label_tensor, return_counts=True)

    print("Class balance:")
    for lbl, cnt in zip(unique_labels.tolist(), counts.tolist()):
        suffix = " (background)" if lbl == 0 else ""
        print(f"  class {lbl}{suffix}: {cnt} samples")

    # ------------------------------------------------------------------
    # Final pass
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== SMOKE TEST PASSED ===")
    print("Loader is valid. Ready to proceed to next slice.")
    print("=" * 60)

    return n_zeros, all_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    csv_file  = args.csv_file
    img_dir   = args.img_dir
    using_synthetic = False

    # Phase 3 — synthetic stub if paths are missing
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        csv_file, img_dir = generate_synthetic_stub()
        using_synthetic = True

    include_background = not args.no_background

    n_zeros, all_labels = run_smoke_test(
        csv_file=csv_file,
        img_dir=img_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        include_background=include_background,
    )
