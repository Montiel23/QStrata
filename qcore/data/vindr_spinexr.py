"""
qcore/data/vindr_spinexr.py — PyTorch Dataset for the exported VinDr-SpineXR
binary ROI dataset produced by scripts/export_vindr_binary_roi_dataset.py.

Binary classification task: Any Pathology (1) vs No Finding (0).
Dataset root is data/processed/vindr_binary_roi_224/ (gitignored).

Does NOT require DICOM access or the original VinDr-SpineXR dataset.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_SPLITS = ("train", "val", "test")

REQUIRED_COLUMNS = (
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

MANIFEST_FILENAME = "manifest.csv"


# ── Dataset class ─────────────────────────────────────────────────────────────

class VinDrSpineXRBinaryDataset(Dataset):
    """
    Per-sample PyTorch Dataset for the exported VinDr-SpineXR binary ROI
    224×224 grayscale PNG dataset.

    Reads from a manifest CSV produced by export_vindr_binary_roi_dataset.py.
    Returns float32 tensors of shape [1, 224, 224] normalised to [0.0, 1.0]
    with torch.long labels in {0, 1}.

    Parameters
    ----------
    root : str
        Path to the exported dataset root containing manifest.csv and
        split sub-directories (train/, val/, test/).
    split : str
        One of "train", "val", or "test".
    transform : callable, optional
        Transform applied to the image tensor after normalisation.
    target_transform : callable, optional
        Transform applied to the label tensor.
    return_metadata : bool
        If True, __getitem__ returns (image_tensor, label, metadata_dict)
        instead of (image_tensor, label).

    Raises
    ------
    FileNotFoundError
        If ``root`` or ``manifest.csv`` inside it does not exist.
        If any output_path file for the selected split is missing.
    ValueError
        If ``split`` is not one of VALID_SPLITS.
        If required manifest columns are absent.
        If binary_label values outside {0, 1} are present.
    """

    def __init__(
        self,
        root: str = "data/processed/vindr_binary_roi_224",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()

        # ── Validate arguments ────────────────────────────────────────────────
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Valid options: {VALID_SPLITS}"
            )

        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Dataset root not found: '{root}'. "
                "Run scripts/export_vindr_binary_roi_dataset.py to export the dataset."
            )

        manifest_path = os.path.join(root, MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: '{manifest_path}'. "
                "Dataset root exists but manifest.csv is missing."
            )

        # ── Load and filter manifest ──────────────────────────────────────────
        df = pd.read_csv(manifest_path)

        # Validate required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Manifest is missing required columns: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        # Filter to requested split
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}."
            )

        # Validate binary_label values
        label_vals = set(df["binary_label"].unique())
        if not label_vals.issubset({0, 1}):
            raise ValueError(
                f"binary_label column contains unexpected values: {label_vals}. "
                "Expected only {{0, 1}}."
            )

        # ── Resolve and validate all output_path files ────────────────────────
        resolved_paths = [
            self._resolve_path(root, str(row["output_path"]))
            for _, row in df.iterrows()
        ]

        missing = [p for p in resolved_paths if not os.path.isfile(p)]
        if missing:
            n = len(missing)
            sample = missing[:5]
            raise FileNotFoundError(
                f"{n} output_path file(s) missing for split='{split}'. "
                f"First {min(n, 5)}: {sample}"
            )

        # ── Store state ───────────────────────────────────────────────────────
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata

        self._df = df
        self._resolved_paths = resolved_paths

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns
        -------
        image : torch.Tensor, float32, shape [1, 224, 224], values in [0.0, 1.0]
        label : torch.Tensor, long, scalar in {0, 1}
        metadata : dict  (only when return_metadata=True)
        """
        row = self._df.iloc[idx]
        img_path = self._resolved_paths[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img = Image.open(img_path)

        # Validate grayscale
        if img.mode != "L":
            warnings.warn(
                f"Expected mode 'L' (grayscale) but got '{img.mode}' "
                f"for {img_path}. Converting."
            )
            img = img.convert("L")

        # ── Convert to float32 tensor [1, H, W] normalised to [0.0, 1.0] ─────
        # Use torch.tensor() (copies through Python) rather than
        # torch.from_numpy() to avoid the NumPy C-API version conflict
        # present in this container (numpy 2.x, torch compiled for numpy 1.x).
        arr = np.array(img, dtype=np.float32) / 255.0     # (H, W) float32 [0,1]
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Validate shape / dtype / range
        if x.shape != (1, 224, 224):
            warnings.warn(
                f"Unexpected tensor shape {tuple(x.shape)} at idx={idx} "
                f"({img_path}). Expected (1, 224, 224)."
            )
        if x.dtype != torch.float32:
            warnings.warn(f"Tensor dtype {x.dtype} at idx={idx}; expected float32.")
        if x.min().item() < 0.0 or x.max().item() > 1.0:
            warnings.warn(
                f"Tensor values out of [0, 1] at idx={idx}: "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}."
            )

        # ── Label ─────────────────────────────────────────────────────────────
        y = torch.tensor(int(row["binary_label"]), dtype=torch.long)

        # ── Optional transforms ───────────────────────────────────────────────
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        # ── Metadata ──────────────────────────────────────────────────────────
        if self.return_metadata:
            meta: dict[str, Any] = {
                "sample_id":           str(row["sample_id"]),
                "image_id":            str(row["image_id"]),
                "split":               str(row["split"]),
                "binary_label":        int(row["binary_label"]),
                "original_label":      str(row["original_label"]),
                "output_path":         str(row["output_path"]),
                "crop_strategy":       str(row["crop_strategy"]),
                "has_bbox":            bool(row["has_bbox"]),
                "is_pseudo_roi":       bool(row["is_pseudo_roi"]),
                "background_fraction": float(row["background_fraction"]),
                "fallback_used":       bool(row["fallback_used"]),
            }
            return x, y, meta

        return x, y

    # ── Class-count helper ────────────────────────────────────────────────────

    def class_counts(self) -> dict[int, int]:
        """Return {label: count} for this split."""
        return self._df["binary_label"].value_counts().to_dict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(root: str, output_path: str) -> str:
        """
        Resolve an output_path field to an absolute file path.

        Policy (as documented in Q12/Q13 export spec):
          1. If output_path is absolute and exists → use it directly.
          2. Otherwise resolve relative to root.
          3. If resolved path does not exist → raise FileNotFoundError.

        The existence check is done by the caller (constructor or __getitem__
        uses pre-resolved paths from the constructor).
        """
        if os.path.isabs(output_path) and os.path.isfile(output_path):
            return output_path
        return os.path.join(root, output_path)
