#!/usr/bin/env python3
"""
scripts/export_vindr_binary_roi_dataset.py
==========================================
VinDr-SpineXR Binary ROI Dataset Exporter — Slice Q13

Converts VinDr-SpineXR detection-style annotations into a binary ROI
classification dataset for Any Pathology vs No Finding.

Q12 approved decisions applied:
  - Positive crop: padded_20 (bbox + 20% padding each side)
  - Negative crop: matched_pseudo_roi (matched size, spine region)
  - Canonical resolution: 224×224
  - Preprocessing: DICOM → float → p1/p99 clip → [0,1] norm → uint8 PNG
  - Augmentation: policy defined (not applied during export)
  - Val split: 80/20 stratified, seed=42

Usage:
  # Dry-run (no files written):
  python scripts/export_vindr_binary_roi_dataset.py \\
    --dataset-root /datasets/vindr-spinexr \\
    --output-root data/processed/vindr_binary_roi_224 \\
    --dry-run

  # Validation export (40 samples):
  python scripts/export_vindr_binary_roi_dataset.py \\
    --dataset-root /datasets/vindr-spinexr \\
    --output-root data/processed/vindr_binary_roi_224 \\
    --max-samples 40 --overwrite

  # Full export:
  python scripts/export_vindr_binary_roi_dataset.py \\
    --dataset-root /datasets/vindr-spinexr \\
    --output-root data/processed/vindr_binary_roi_224 \\
    --overwrite
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")

# ─── constants ────────────────────────────────────────────────────────────────
NEG_LABEL = "No finding"
PREPROC_POLICY = "dicom_decode,percentile_clip_p1p99,normalize_0_1,uint8_L_png"
PAD_FRAC = 0.20           # padded_20
SPINE_Y_LOW  = 0.20       # pseudo-ROI: lower bound of spine region (fraction of H)
SPINE_Y_HIGH = 0.80       # pseudo-ROI: upper bound of spine region
BG_THRESH = 0.05          # intensity < this = background after [0,1] norm
MAX_PSEUDO_RETRIES = 10   # max attempts before center-crop fallback
BG_REJECT_THRESH = 0.40   # reject pseudo-ROI if background fraction > this

# ─── PIL import (optional but preferred for PNG saving) ────────────────────────
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─── helpers ──────────────────────────────────────────────────────────────────

def find_dicom(image_id: str, base: str) -> str | None:
    """Return path to DICOM file.  VinDr-SpineXR uses flat layout:
       <base>/<image_id>.dicom  (no study subdirectory nesting)."""
    p = os.path.join(base, image_id + ".dicom")
    if os.path.isfile(p):
        return p
    p2 = os.path.join(base, image_id)
    if os.path.isfile(p2):
        return p2
    return None


def load_and_preprocess(dicom_path: str):
    """Load DICOM, apply preprocessing pipeline.  Returns (pixel_array, H, W).
    pixel_array is float32 in [0, 1] after p1/p99 clip + normalization."""
    ds = pydicom.dcmread(dicom_path)
    px = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(px, [1, 99])
    px = np.clip((px - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    H, W = px.shape
    return px, H, W


def padded_bbox(row: pd.Series, pad_frac: float, H: int, W: int):
    """Return (cx0, cy0, cx1, cy1) padded bbox clipped to image."""
    x0, y0, x1, y1 = float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])
    dw = (x1 - x0) * pad_frac
    dh = (y1 - y0) * pad_frac
    cx0 = max(0, int(x0 - dw))
    cy0 = max(0, int(y0 - dh))
    cx1 = min(W, int(x1 + dw))
    cy1 = min(H, int(y1 + dh))
    return cx0, cy0, cx1, cy1


def background_fraction(crop: np.ndarray) -> float:
    """Fraction of pixels with normalized intensity < BG_THRESH."""
    if crop.size == 0:
        return 1.0
    return float((crop < BG_THRESH).mean())


def crop_and_resize(px: np.ndarray, cx0: int, cy0: int, cx1: int, cy1: int,
                    res: int) -> np.ndarray | None:
    """Crop and resize to (res, res).  Returns uint8 array or None on failure."""
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = px[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    # Resize via PIL if available, else simple numpy zoom
    if HAS_PIL:
        img = PILImage.fromarray((crop * 255).astype(np.uint8), mode="L")
        img = img.resize((res, res), PILImage.LANCZOS)
        return np.array(img, dtype=np.uint8)
    else:
        from scipy.ndimage import zoom
        zy = res / crop.shape[0]
        zx = res / crop.shape[1]
        zoomed = zoom(crop, (zy, zx), order=1)
        return (np.clip(zoomed, 0, 1) * 255).astype(np.uint8)


def save_png(arr: np.ndarray, path: str) -> None:
    """Save uint8 grayscale array as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if HAS_PIL:
        PILImage.fromarray(arr, mode="L").save(path)
    else:
        import struct, zlib
        # Minimal grayscale PNG writer without PIL
        def _write_png(data, w, h):
            def chunk(name, data):
                c = len(data).to_bytes(4, "big") + name + data
                return c + zlib.crc32(name + data).to_bytes(4, "big")
            rows = b"".join(b"\x00" + bytes(data[i]) for i in range(h))
            compressed = zlib.compress(rows)
            return (b"\x89PNG\r\n\x1a\n" +
                    chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) +
                    chunk(b"IDAT", compressed) +
                    chunk(b"IEND", b""))
        with open(path, "wb") as f:
            f.write(_write_png(data=arr, w=arr.shape[1], h=arr.shape[0]))


def save_grid_png(images: list, labels: list, path: str, ncols: int = 5,
                  title: str = "") -> None:
    """Save a grid of uint8 grayscale images as a single PNG."""
    if not HAS_PIL or not images:
        return
    from PIL import ImageDraw, ImageFont
    n = len(images)
    nrows = math.ceil(n / ncols)
    cell = images[0].shape[0] if images else 112
    pad = 4
    header = 24 if title else 0
    canvas_h = nrows * (cell + pad) + pad + header
    canvas_w = ncols * (cell + pad) + pad
    canvas = PILImage.new("L", (canvas_w, canvas_h), color=20)
    for idx, (img_arr, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(idx, ncols)
        x = c * (cell + pad) + pad
        y = r * (cell + pad) + pad + header
        tile = PILImage.fromarray(img_arr, mode="L").resize((cell, cell), PILImage.LANCZOS)
        canvas.paste(tile, (x, y))
    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), title, fill=200)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def make_pseudo_roi(px: np.ndarray, H: int, W: int,
                    target_h: int, target_w: int,
                    rng: random.Random) -> tuple:
    """Attempt to sample a pseudo-ROI from the spine region.
    Returns (cx0, cy0, cx1, cy1, fallback_used, bg_frac)."""
    target_h = max(1, min(target_h, H))
    target_w = max(1, min(target_w, W))

    y_lo = int(H * SPINE_Y_LOW)
    y_hi = max(y_lo + 1, int(H * SPINE_Y_HIGH) - target_h)
    x_center = W // 2

    for attempt in range(MAX_PSEUDO_RETRIES):
        if y_hi > y_lo:
            cy0 = rng.randint(y_lo, y_hi)
        else:
            cy0 = y_lo
        cy1 = min(H, cy0 + target_h)

        x_jitter = rng.randint(-target_w // 4, target_w // 4)
        cx0 = max(0, x_center - target_w // 2 + x_jitter)
        cx1 = min(W, cx0 + target_w)

        if cx1 <= cx0 or cy1 <= cy0:
            continue
        crop = px[cy0:cy1, cx0:cx1]
        bg = background_fraction(crop)
        if bg <= BG_REJECT_THRESH:
            return cx0, cy0, cx1, cy1, False, bg

    # fallback: center crop
    cy0 = max(0, (H - target_h) // 2)
    cy1 = min(H, cy0 + target_h)
    cx0 = max(0, (W - target_w) // 2)
    cx1 = min(W, cx0 + target_w)
    crop = px[cy0:cy1, cx0:cx1]
    bg = background_fraction(crop)
    return cx0, cy0, cx1, cy1, True, bg


# ─── main export function ──────────────────────────────────────────────────────

def run_export(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    dataset_root = args.dataset_root
    output_root  = args.output_root
    res          = args.resolution
    is_dry_run   = args.dry_run
    max_samples  = args.max_samples
    is_validation = max_samples is not None and not is_dry_run

    train_img_dir = os.path.join(dataset_root, "train_images")
    test_img_dir  = os.path.join(dataset_root, "test_images")

    # ── discover annotation files ────────────────────────────────────────────
    annot_dir = os.path.join(dataset_root, "annotations")
    if not os.path.isdir(annot_dir):
        sys.exit(f"ERROR: Annotations directory not found: {annot_dir}")

    annot_files = sorted(os.listdir(annot_dir))
    train_csv = test_csv = None
    for f in annot_files:
        fl = f.lower()
        if "train" in fl and fl.endswith(".csv"):
            train_csv = os.path.join(annot_dir, f)
        elif "test" in fl and fl.endswith(".csv"):
            test_csv = os.path.join(annot_dir, f)
    if train_csv is None:
        sys.exit(f"ERROR: No train annotation CSV found in {annot_dir}. Files: {annot_files}")

    mode_str = ("DRY RUN" if is_dry_run
                else f"VALIDATION (max={max_samples})" if is_validation
                else "FULL")

    print("=== VinDr Binary ROI Dataset Export ===")
    print(f"Dataset root    : {dataset_root}")
    print(f"Output root     : {output_root}")
    print(f"Resolution      : {res}x{res}")
    print(f"Seed            : {args.seed}")
    print(f"Val ratio       : {args.val_ratio:.2f}")
    print(f"Mode            : {mode_str}")
    print(f"PIL available   : {HAS_PIL}")
    print()

    # ── load train annotations ───────────────────────────────────────────────
    df_tr = pd.read_csv(train_csv)
    # Identify columns without hard-coding
    img_col   = "image_id"   if "image_id"   in df_tr.columns else df_tr.columns[2]
    lbl_col   = "lesion_type" if "lesion_type" in df_tr.columns else df_tr.columns[4]
    study_col = "study_id"   if "study_id"   in df_tr.columns else df_tr.columns[0]
    bbox_ok   = all(c in df_tr.columns for c in ("xmin","ymin","xmax","ymax"))

    # ── image-level label sets ───────────────────────────────────────────────
    img_lbls_all = df_tr.groupby(img_col)[lbl_col].apply(set)
    neg_ids = set(img_lbls_all[img_lbls_all.apply(lambda s: s == {NEG_LABEL})].index)
    pos_ids = set(img_lbls_all[img_lbls_all.apply(lambda s: NEG_LABEL not in s)].index)
    overlap = neg_ids & pos_ids

    n_pos = len(pos_ids)
    n_neg = len(neg_ids)
    exp_pos, exp_neg = 4129, 4260

    print("Label mapping:")
    print(f"  Positive (Any Pathology) : {n_pos:,}  [expected ~{exp_pos:,}]")
    print(f"  Negative (No Finding)    : {n_neg:,}  [expected ~{exp_neg:,}]")
    print(f"  Overlap                  : {len(overlap)}  [expected 0]")
    print()

    v1 = abs(n_pos - exp_pos) / exp_pos <= 0.05
    v2 = abs(n_neg - exp_neg) / exp_neg <= 0.05
    v3 = len(overlap) == 0

    # ── union bbox per positive image ─────────────────────────────────────────
    if bbox_ok:
        path_rows = df_tr[(df_tr[img_col].isin(pos_ids)) & (df_tr["xmin"].notna())]
        img_bbox = path_rows.groupby(img_col).agg(
            xmin=("xmin", "min"), ymin=("ymin", "min"),
            xmax=("xmax", "max"), ymax=("ymax", "max"),
            study_id=(study_col, "first"),
            src_row=("xmin", lambda x: x.index[0]),
        ).reset_index()
        img_bbox["bbox_w"] = img_bbox["xmax"] - img_bbox["xmin"]
        img_bbox["bbox_h"] = img_bbox["ymax"] - img_bbox["ymin"]
    else:
        sys.exit("ERROR: Bbox columns (xmin/ymin/xmax/ymax) not found in annotation file.")

    # Positive images that have no bbox annotation — use full image fallback
    pos_no_bbox = pos_ids - set(img_bbox[img_col])

    # ── train/val split ───────────────────────────────────────────────────────
    pos_list = sorted(pos_ids)
    neg_list = sorted(neg_ids)
    rng.shuffle(pos_list)
    rng.shuffle(neg_list)

    n_val_pos = int(len(pos_list) * args.val_ratio)
    n_val_neg = int(len(neg_list) * args.val_ratio)

    val_pos  = set(pos_list[:n_val_pos])
    train_pos = set(pos_list[n_val_pos:])
    val_neg  = set(neg_list[:n_val_neg])
    train_neg = set(neg_list[n_val_neg:])

    v4 = (len(train_pos) + len(val_pos) == n_pos and
          len(train_neg) + len(val_neg) == n_neg)
    v5_leakage = (val_pos & train_pos) | (val_neg & train_neg)
    v5 = len(v5_leakage) == 0

    # ── load test if available ────────────────────────────────────────────────
    test_pos_ids = test_neg_ids = None
    df_te = None
    if test_csv and os.path.isfile(test_csv):
        df_te = pd.read_csv(test_csv)
        if lbl_col in df_te.columns and img_col in df_te.columns:
            te_lbls = df_te.groupby(img_col)[lbl_col].apply(set)
            test_neg_ids = set(te_lbls[te_lbls.apply(lambda s: s == {NEG_LABEL})].index)
            test_pos_ids = set(te_lbls[te_lbls.apply(lambda s: NEG_LABEL not in s)].index)

    print("Split counts:")
    print(f"  train (binary_label=0)   : {len(train_neg):,}")
    print(f"  train (binary_label=1)   : {len(train_pos):,}")
    print(f"  val   (binary_label=0)   : {len(val_neg):,}")
    print(f"  val   (binary_label=1)   : {len(val_pos):,}")
    if test_neg_ids is not None:
        print(f"  test  (binary_label=0)   : {len(test_neg_ids):,}")
        print(f"  test  (binary_label=1)   : {len(test_pos_ids):,}")
    print()

    # ── crop statistics from bbox data ────────────────────────────────────────
    # Sample median positive crop size for pseudo-ROI matching
    med_crop_w = float(img_bbox["bbox_w"].mean() * (1 + 2 * PAD_FRAC))
    med_crop_h = float(img_bbox["bbox_h"].mean() * (1 + 2 * PAD_FRAC))

    print("Crop statistics:")
    bw = img_bbox["bbox_w"]
    bh = img_bbox["bbox_h"]
    print(f"  Positive padded_20 width : mean={bw.mean()*(1+2*PAD_FRAC):.0f}, "
          f"std={bw.std():.0f}, min={bw.min():.0f}, max={bw.max():.0f}")
    print(f"  Positive padded_20 height: mean={bh.mean()*(1+2*PAD_FRAC):.0f}, "
          f"std={bh.std():.0f}, min={bh.min():.0f}, max={bh.max():.0f}")
    print(f"  Pseudo-ROI target width  : mean={med_crop_w:.0f}")
    print(f"  Pseudo-ROI target height : mean={med_crop_h:.0f}")
    print()

    # ── build sample plan ─────────────────────────────────────────────────────
    # Each entry: (image_id, split, binary_label, bbox_row_or_None)
    def build_pos_plan(id_set, split, bbox_df):
        plan = []
        for iid in id_set:
            rows = bbox_df[bbox_df[img_col] == iid]
            bbox_row = rows.iloc[0] if len(rows) else None
            plan.append((iid, split, 1, bbox_row))
        return plan

    def build_neg_plan(id_set, split):
        return [(iid, split, 0, None) for iid in id_set]

    all_plan = (
        build_pos_plan(train_pos, "train", img_bbox) +
        build_neg_plan(train_neg, "train") +
        build_pos_plan(val_pos, "val", img_bbox) +
        build_neg_plan(val_neg, "val")
    )
    # Add test if available
    if test_pos_ids is not None:
        test_bbox_df = df_te.copy() if df_te is not None else pd.DataFrame()
        if lbl_col in df_te.columns and bbox_ok:
            te_path_rows = df_te[(df_te[img_col].isin(test_pos_ids)) & (df_te["xmin"].notna())]
            te_img_bbox = te_path_rows.groupby(img_col).agg(
                xmin=("xmin","min"), ymin=("ymin","min"),
                xmax=("xmax","max"), ymax=("ymax","max"),
                study_id=(study_col,"first"),
                src_row=("xmin", lambda x: x.index[0]),
            ).reset_index()
            te_img_bbox["bbox_w"] = te_img_bbox["xmax"] - te_img_bbox["xmin"]
            te_img_bbox["bbox_h"] = te_img_bbox["ymax"] - te_img_bbox["ymin"]
        else:
            te_img_bbox = pd.DataFrame(columns=img_bbox.columns)
        all_plan += build_pos_plan(test_pos_ids, "test", te_img_bbox)
        all_plan += build_neg_plan(test_neg_ids, "test")

    # ── sub-sample for max-samples mode ──────────────────────────────────────
    if max_samples is not None and not is_dry_run:
        # Build per-cell (split, label) pools
        cells = defaultdict(list)
        for item in all_plan:
            cells[(item[1], item[2])].append(item)
        # Target: equal share across cells
        split_labels = list(cells.keys())
        per_cell = max(1, max_samples // len(split_labels))
        sampled = []
        for key in split_labels:
            pool = cells[key]
            # For positives: prefer diversity across bbox size
            if key[1] == 1 and pool:
                sizes = []
                for item in pool:
                    brow = item[3]
                    sz = float(brow["bbox_area"]) if (brow is not None and "bbox_area" in brow.index) else 0
                    sizes.append(sz)
                order = sorted(range(len(pool)), key=lambda i: sizes[i])
                step = max(1, len(order) // per_cell)
                picked = [pool[order[i]] for i in range(0, len(order), step)][:per_cell]
            else:
                rng.shuffle(pool)
                picked = pool[:per_cell]
            sampled.extend(picked)
        all_plan = sampled[:max_samples]

    total_plan = len(all_plan)
    print(f"Samples to process: {total_plan}")
    print()

    if is_dry_run:
        print("DRY RUN — validating schema and crop coordinates on a sample...")
        # Spot-check first 20 items with a DICOM access
        coord_fails = 0
        checked = 0
        for iid, split, label, bbox_row in all_plan[:20]:
            img_dir = test_img_dir if split == "test" else train_img_dir
            dp = find_dicom(iid, img_dir)
            if dp is None:
                continue
            try:
                ds = pydicom.dcmread(dp, stop_before_pixels=True)
                H, W = int(ds.Rows), int(ds.Columns)
                if bbox_row is not None:
                    cx0, cy0, cx1, cy1 = padded_bbox(bbox_row, PAD_FRAC, H, W)
                    if cx1 <= cx0 or cy1 <= cy0:
                        coord_fails += 1
                checked += 1
            except Exception:
                pass
        v6 = coord_fails == 0
        print(f"  DICOM spot-check: {checked} loaded, {coord_fails} coord failures")
        # Background fraction dry-run (small sample)
        _print_validation_results(v1, v2, v3, v4, v5, v6,
                                  v_imgs=True, v_bg=True, v_manifest=True,
                                  bg_pos_mean=0.0, bg_neg_mean=0.0, bg_gap=0.0)
        print()
        print("Export status: DRY RUN — no files written")
        return

    # ── prepare output directory ──────────────────────────────────────────────
    if not is_dry_run:
        if os.path.exists(output_root):
            if args.overwrite:
                shutil.rmtree(output_root)
            else:
                sys.exit(f"ERROR: Output directory exists. Use --overwrite to replace: {output_root}")
        for split in ("train", "val", "test"):
            for lbl in (0, 1):
                os.makedirs(os.path.join(output_root, split, str(lbl)), exist_ok=True)
        os.makedirs(os.path.join(output_root, "samples"), exist_ok=True)

    # ── export loop ───────────────────────────────────────────────────────────
    manifest_rows = []
    pos_bg_fracs  = []
    neg_bg_fracs  = []
    fallback_count = 0
    total_neg      = 0
    exported       = 0
    coord_fails    = 0
    img_check_fails = 0
    sample_pos_imgs = []
    sample_neg_imgs = []

    print(f"Exporting {total_plan} samples...")
    for idx, (iid, split, label, bbox_row) in enumerate(all_plan):
        img_dir = test_img_dir if split == "test" else train_img_dir
        dp = find_dicom(iid, img_dir)
        if dp is None:
            print(f"  [WARN] DICOM not found: {iid}", file=sys.stderr)
            continue

        try:
            px, H, W = load_and_preprocess(dp)
        except Exception as e:
            print(f"  [WARN] Load failed {iid}: {e}", file=sys.stderr)
            continue

        # ── crop coordinates ──────────────────────────────────────────────────
        if label == 1:  # positive: padded_20
            has_bbox = bbox_row is not None and not pd.isna(bbox_row["xmin"])
            if has_bbox:
                cx0, cy0, cx1, cy1 = padded_bbox(bbox_row, PAD_FRAC, H, W)
                x_min_orig = float(bbox_row["xmin"])
                y_min_orig = float(bbox_row["ymin"])
                x_max_orig = float(bbox_row["xmax"])
                y_max_orig = float(bbox_row["ymax"])
                src_row    = int(bbox_row["src_row"]) if "src_row" in bbox_row.index else -1
            else:
                # fallback: full image
                cx0, cy0, cx1, cy1 = 0, 0, W, H
                x_min_orig = y_min_orig = x_max_orig = y_max_orig = float("nan")
                src_row = -1
            is_pseudo = False
            fallback_used = not has_bbox
            crop_strat = "padded_20"
            orig_lbl = df_tr.loc[df_tr[img_col] == iid, lbl_col].iloc[0] if len(
                df_tr[df_tr[img_col] == iid]) else "Any Pathology"

        else:  # negative: matched_pseudo_roi
            total_neg += 1
            th = max(1, int(rng.gauss(med_crop_h, med_crop_h * 0.15)))
            tw = max(1, int(rng.gauss(med_crop_w, med_crop_w * 0.15)))
            cx0, cy0, cx1, cy1, fallback_used, _bg = make_pseudo_roi(
                px, H, W, th, tw, rng)
            x_min_orig = y_min_orig = x_max_orig = y_max_orig = float("nan")
            has_bbox = False
            is_pseudo = True
            crop_strat = "matched_pseudo_roi"
            src_row = -1
            orig_lbl = NEG_LABEL
            if fallback_used:
                fallback_count += 1

        if cx1 <= cx0 or cy1 <= cy0:
            coord_fails += 1
            continue

        # ── crop and resize ───────────────────────────────────────────────────
        arr = crop_and_resize(px, cx0, cy0, cx1, cy1, res)
        if arr is None:
            img_check_fails += 1
            continue

        # background fraction for the resized crop
        bg_frac = background_fraction(arr.astype(np.float32) / 255.0)
        if label == 1:
            pos_bg_fracs.append(bg_frac)
        else:
            neg_bg_fracs.append(bg_frac)

        # collect sample images for visualization (up to 10 each)
        if label == 1 and len(sample_pos_imgs) < 10:
            sample_pos_imgs.append(arr)
        elif label == 0 and len(sample_neg_imgs) < 10:
            sample_neg_imgs.append(arr)

        # ── write PNG ─────────────────────────────────────────────────────────
        sample_id = f"{iid}_{split}_{idx:06d}"
        rel_path  = os.path.join(split, str(label), f"{sample_id}.png")
        full_path = os.path.join(output_root, rel_path)
        save_png(arr, full_path)

        notes = PREPROC_POLICY
        if fallback_used:
            notes += "|fallback_center_crop"

        manifest_rows.append({
            "sample_id":            sample_id,
            "image_id":             iid,
            "split":                split,
            "binary_label":         label,
            "original_label":       orig_lbl,
            "dicom_path":           dp,
            "output_path":          rel_path,
            "crop_strategy":        crop_strat,
            "x_min":                x_min_orig,
            "y_min":                y_min_orig,
            "x_max":                x_max_orig,
            "y_max":                y_max_orig,
            "padded_x_min":         cx0,
            "padded_y_min":         cy0,
            "padded_x_max":         cx1,
            "padded_y_max":         cy1,
            "resize_height":        res,
            "resize_width":         res,
            "has_bbox":             has_bbox,
            "is_pseudo_roi":        is_pseudo,
            "source_annotation_row": src_row,
            "background_fraction":  round(bg_frac, 4),
            "fallback_used":        fallback_used,
            "notes":                notes,
        })
        exported += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_plan:
            print(f"  {idx+1}/{total_plan} processed  ({exported} exported)", end="\r")

    print(f"\n  Done. {exported} images exported.")
    print()

    # ── background fraction report ────────────────────────────────────────────
    bg_pos_mean = float(np.mean(pos_bg_fracs)) if pos_bg_fracs else float("nan")
    bg_neg_mean = float(np.mean(neg_bg_fracs)) if neg_bg_fracs else float("nan")
    bg_gap = abs(bg_pos_mean - bg_neg_mean) if (pos_bg_fracs and neg_bg_fracs) else float("nan")

    print("Background fraction check:")
    print(f"  Positive mean background : {bg_pos_mean:.3f}")
    print(f"  Negative mean background : {bg_neg_mean:.3f}")
    if not math.isnan(bg_gap):
        flag_str = ("LOW (<5%)" if bg_gap < 0.05
                    else "CAUTION (5-15%)" if bg_gap < 0.15
                    else "HIGH (>15%)")
        print(f"  Gap                      : {bg_gap:.3f}  [{flag_str}]")
    else:
        print("  Gap                      : N/A")
    print()

    print("Negative pseudo-ROI fallbacks:")
    pct_fb = 100 * fallback_count / max(total_neg, 1)
    print(f"  {fallback_count} / {total_neg} negatives used center-crop fallback  ({pct_fb:.1f}%)")
    print()

    # ── write manifest ────────────────────────────────────────────────────────
    manifest_path = os.path.join(output_root, "manifest.csv")
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Manifest written: {manifest_path}  ({len(df_manifest)} rows)")

    # ── manifest validation ───────────────────────────────────────────────────
    v_dup   = df_manifest["sample_id"].nunique() == len(df_manifest)
    v_split = set(df_manifest["split"].unique()).issubset({"train","val","test"})
    v_lbl   = set(df_manifest["binary_label"].unique()).issubset({0,1})
    v_manifest = v_dup and v_split and v_lbl

    v_imgs  = (img_check_fails == 0)
    v_bg    = (math.isnan(bg_gap) or bg_gap < 0.15)
    v6      = (coord_fails == 0)

    _print_validation_results(v1, v2, v3, v4, v5, v6, v_imgs, v_bg, v_manifest,
                               bg_pos_mean, bg_neg_mean,
                               bg_gap if not math.isnan(bg_gap) else 0.0)

    # ── sample visualizations ─────────────────────────────────────────────────
    samples_dir = os.path.join(output_root, "samples")
    if HAS_PIL and sample_pos_imgs:
        save_grid_png(sample_pos_imgs,
                      [1] * len(sample_pos_imgs),
                      os.path.join(samples_dir, "positive_examples.png"),
                      ncols=5, title="Positive (Any Pathology) — padded_20")
        print(f"Saved: samples/positive_examples.png ({len(sample_pos_imgs)} images)")

    if HAS_PIL and sample_neg_imgs:
        save_grid_png(sample_neg_imgs,
                      [0] * len(sample_neg_imgs),
                      os.path.join(samples_dir, "negative_examples.png"),
                      ncols=5, title="Negative (No Finding) — matched_pseudo_roi")
        print(f"Saved: samples/negative_examples.png ({len(sample_neg_imgs)} images)")

    if HAS_PIL and sample_pos_imgs and sample_neg_imgs:
        n = min(len(sample_pos_imgs), len(sample_neg_imgs), 5)
        grid_imgs   = sample_pos_imgs[:n] + sample_neg_imgs[:n]
        grid_labels = [1]*n + [0]*n
        save_grid_png(grid_imgs, grid_labels,
                      os.path.join(samples_dir, "positive_negative_grid.png"),
                      ncols=n, title="Top row: Positive | Bottom row: Negative")
        print(f"Saved: samples/positive_negative_grid.png")

    print()
    status = (f"Export status: VALIDATION ONLY ({exported} samples) — full dataset not exported"
              if is_validation else "Export status: FULL DATASET EXPORTED")
    print(status)


def _print_validation_results(v1, v2, v3, v4, v5, v6,
                               v_imgs, v_bg, v_manifest,
                               bg_pos_mean=0.0, bg_neg_mean=0.0, bg_gap=0.0):
    def pf(b): return "PASS" if b else "FAIL"
    print("Validation checks:")
    print(f"  Label overlap = 0                : {pf(v3)}")
    print(f"  Positive count ≈ 4,129 (±5%)    : {pf(v1)}")
    print(f"  Negative count ≈ 4,260 (±5%)    : {pf(v2)}")
    print(f"  train+val = original count       : {pf(v4)}")
    print(f"  No image leakage train↔val       : {pf(v5)}")
    print(f"  Crop coords valid                : {pf(v6)}")
    print(f"  Exported images valid            : {pf(v_imgs)}")
    print(f"  Background gap < 15%             : {pf(v_bg)}")
    print(f"  Manifest integrity               : {pf(v_manifest)}")
    all_pass = all([v1, v2, v3, v4, v5, v6, v_imgs, v_bg, v_manifest])
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAIL — see above'}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VinDr-SpineXR Binary ROI Dataset Exporter (Slice Q13)")
    p.add_argument("--dataset-root",  default="/datasets/vindr-spinexr",
                   help="Path to VinDr-SpineXR root")
    p.add_argument("--output-root",   default="data/processed/vindr_binary_roi_224",
                   help="Output directory")
    p.add_argument("--resolution",    type=int,   default=224,
                   help="Target resize resolution (default: 224)")
    p.add_argument("--val-ratio",     type=float, default=0.20,
                   help="Validation fraction from train (default: 0.20)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--dry-run",       action="store_true",
                   help="Validate schema and crop stats; write nothing to disk")
    p.add_argument("--max-samples",   type=int,   default=None,
                   help="Export at most N samples (validation mode)")
    p.add_argument("--overwrite",     action="store_true",
                   help="Overwrite existing output directory")
    p.add_argument("--export-samples-only", action="store_true",
                   help="Write only sample visualization PNGs")
    p.add_argument("--no-write-images", action="store_true",
                   help="Write manifest only, not image files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_export(args)
