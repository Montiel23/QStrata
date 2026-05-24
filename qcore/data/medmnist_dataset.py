"""
qcore/data/medmnist_dataset.py — Thin per-sample Dataset wrapper for MedMNIST.

Supported datasets: pneumoniamnist (binary), pathmnist (multiclass).

Aligns with the bulk-loading pattern in qcore/data/medical_loader.py but
exposes a standard PyTorch Dataset interface (__len__ / __getitem__) without
PCA, scaling, or any ML preprocessing baked in.
"""

import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset

# Datasets supported by this module. Extend here when adding more.
SUPPORTED_DATASETS = ("pneumoniamnist", "pathmnist")


class MedMNISTDataset(Dataset):
    """
    Per-sample wrapper around a MedMNIST split.

    Parameters
    ----------
    dataset_name : str
        One of SUPPORTED_DATASETS.
    split : str
        "train", "val", or "test".
    download : bool
        Whether to auto-download on first use (default True).

    Per-sample access (dataset[i]) returns:
        image  : np.ndarray, float32, shape (H, W) or (H, W, C), values in [0, 1]
        label  : int
    """

    def __init__(self, dataset_name: str, split: str, download: bool = True):
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: {SUPPORTED_DATASETS}"
            )

        self.dataset_name = dataset_name
        self.split = split

        # ── Load metadata from medmnist registry ─────────────────────────────
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])

        # ── Instantiate the underlying MedMNIST split ─────────────────────────
        # Matches the pattern already used in medical_loader.py.
        self._inner = DataClass(split=split, download=download)

        # ── Derive task / class metadata ──────────────────────────────────────
        n_classes = len(info["label"])
        self.n_classes = n_classes
        self.task = "binary" if n_classes == 2 else "multiclass"

        raw_label_map = info.get("label", {})
        if raw_label_map:
            # medmnist INFO labels are keyed by string "0", "1", …
            self.label_map = {int(k): v for k, v in raw_label_map.items()}
        else:
            self.label_map = {0: "negative", 1: "positive"}

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        image : np.ndarray, dtype=float32, values in [0, 1]
        label : int
        """
        image, label = self._inner[idx]

        # MedMNIST may return PIL images — always convert explicitly.
        image = np.array(image).astype(np.float32) / 255.0

        # Labels may arrive as lists, numpy arrays, or nested shapes like
        # [0] or [[1]]. Always normalise to a plain Python int.
        label = int(np.asarray(label).squeeze())

        return image, label
