"""
check_dataset.py — VinDr-SpineXR dataset presence validator.

Usage:
    python scripts/check_dataset.py --data_dir /path/to/vindr-spinexr

Exits with code 0 if dataset is found, code 1 if not found.
"""

import argparse
import os
import sys


PHYSIONET_INSTRUCTIONS = """
[ERROR] VinDr-SpineXR dataset NOT found at the specified location.

----------------------------------------------------------------------
MANUAL DOWNLOAD REQUIRED
----------------------------------------------------------------------
Dataset: VinDr-SpineXR
URL:     https://physionet.org/content/vindr-spinexr/1.0.0/

Requirements:
  - Credentialed PhysioNet account
  - Signed Data Use Agreement (DUA)

Download command:
    wget -r -N -c -np --user YOUR_USERNAME --ask-password \\
        https://physionet.org/files/vindr-spinexr/1.0.0/

Expected directory structure after download:
    <data_dir>/
        train_images/          (DICOM files, one per study)
        test_images/           (DICOM files, one per study)
        annotations/
            train.csv          (bounding box annotations)
            test.csv           (study-level labels)

After downloading, re-run:
    python scripts/check_dataset.py --data_dir /path/to/vindr-spinexr
----------------------------------------------------------------------
"""


def find_dicom_files(root):
    """Recursively find all .dicom and .dcm files under root."""
    dicom_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith((".dicom", ".dcm")):
                dicom_files.append(os.path.join(dirpath, fname))
    return dicom_files


def find_annotation_csv(root):
    """
    Look for annotation CSV:
      1. Any .csv under <root>/annotations/ (preferred)
      2. Any .csv directly under <root> (fallback)
    Returns the first match found, or None.
    """
    annotations_dir = os.path.join(root, "annotations")
    if os.path.isdir(annotations_dir):
        for fname in sorted(os.listdir(annotations_dir)):
            if fname.lower().endswith(".csv"):
                return os.path.join(annotations_dir, fname)

    # Fallback: root-level CSV
    for fname in sorted(os.listdir(root)):
        if fname.lower().endswith(".csv"):
            return os.path.join(root, fname)

    return None


def print_folder_structure(root, max_depth=2):
    """Print folder tree up to max_depth levels."""
    print(f"\n  Folder structure ({max_depth} levels):")
    for dirpath, dirnames, filenames in os.walk(root):
        # Calculate depth relative to root
        depth = dirpath.replace(root, "").count(os.sep)
        if depth >= max_depth:
            dirnames.clear()  # Don't recurse deeper
            continue
        indent = "    " + "  " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        # Show first few files at this level
        shown_files = filenames[:5]
        for fname in shown_files:
            fsize = os.path.getsize(os.path.join(dirpath, fname))
            print(f"{indent}  {fname}  ({fsize:,} bytes)")
        if len(filenames) > 5:
            print(f"{indent}  ... and {len(filenames) - 5} more files")


def main():
    parser = argparse.ArgumentParser(
        description="Check for VinDr-SpineXR dataset presence."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory to search for the dataset",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Directory does not exist: {data_dir}")
        print(PHYSIONET_INSTRUCTIONS)
        sys.exit(1)

    print(f"[INFO] Scanning: {data_dir}")

    # --- Search for DICOMs ---
    print("[INFO] Searching for DICOM files (this may take a moment) ...")
    dicom_files = find_dicom_files(data_dir)

    # --- Search for annotation CSV ---
    csv_path = find_annotation_csv(data_dir)

    # --- Verdict ---
    has_dicoms = len(dicom_files) > 0
    has_csv    = csv_path is not None

    if not has_dicoms or not has_csv:
        missing = []
        if not has_dicoms:
            missing.append("DICOM files (.dicom/.dcm)")
        if not has_csv:
            missing.append("annotation CSV")
        print(f"[ERROR] Missing: {', '.join(missing)}")
        print(PHYSIONET_INSTRUCTIONS)
        sys.exit(1)

    # --- Report ---
    print("\n" + "=" * 60)
    print("DATASET FOUND")
    print("=" * 60)
    print(f"  Total DICOM files : {len(dicom_files):,}")
    print(f"  Annotation CSV    : {csv_path}")

    # Quick CSV peek
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  CSV rows          : {len(df):,}")
        print(f"  CSV columns       : {list(df.columns)}")
    except Exception as exc:
        print(f"  [WARN] Could not read CSV for preview: {exc}")

    print_folder_structure(data_dir, max_depth=2)
    print("\n[OK] Dataset validation passed. Ready for EDA.")


if __name__ == "__main__":
    main()
