"""
qcore/data/torch_adapter.py — Thin torch.utils.data.Dataset wrapper.

Converts plain numpy/int items returned by any MedMNISTDataset into
properly shaped torch tensors ready for a DataLoader.

Image tensor layout: CHW float32
  (H, W)   → (1, H, W)
  (H, W, 1) → (1, H, W)
  (H, W, 3) → (3, H, W)

Label: torch.long scalar tensor
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDatasetAdapter(Dataset):
    """
    Wraps any dataset whose __getitem__ returns (np.ndarray float32, int).

    Parameters
    ----------
    dataset : object with __len__ and __getitem__ returning (ndarray, int)
        Typically a MedMNISTDataset instance from qcore.data.registry.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        image, label = self._dataset[idx]

        # ── Image → CHW float32 tensor ────────────────────────────────────────
        # image arrives as np.ndarray float32 in [0, 1], shape (H,W) or (H,W,C)
        image = np.array(image, dtype=np.float32)

        if image.ndim == 2:
            # (H, W) — grayscale, no channel dim
            x = torch.from_numpy(image).unsqueeze(0)          # → (1, H, W)
        elif image.ndim == 3 and image.shape[2] == 1:
            # (H, W, 1) — grayscale with redundant channel dim
            x = torch.from_numpy(image.squeeze(2)).unsqueeze(0)  # → (1, H, W)
        elif image.ndim == 3 and image.shape[2] == 3:
            # (H, W, 3) — RGB
            x = torch.from_numpy(image).permute(2, 0, 1)      # → (3, H, W)
        else:
            raise ValueError(
                f"Unsupported image shape {image.shape}. "
                "Expected (H,W), (H,W,1), or (H,W,3)."
            )

        # ── Label → long scalar tensor ────────────────────────────────────────
        y = torch.tensor(int(label), dtype=torch.long)

        return x, y
