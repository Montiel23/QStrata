import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NpySpineDataset(Dataset):

    def __init__(
            self,
            images_path,
            labels_path,
            target_size=(224, 224),
            to_spatial_rgb=True,
    ):

        if not os.path.exists(images_path):
            raise FileNotFoundError(
                f"Images file not found  at: {images_path}"
            )

        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels file not found at: {labels_path}"
            )

        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.target_size = target_size
        self.to_spatial_rgb = to_spatial_rgb

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
            )

        # self.unique_classes = np.unique(self.labels)
        self.unique_classes = sorted(list(np.unique(self.labels)))
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.unique_classes)
        }

    def get_total_classes(self):
        return len(self.unique_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = np.array(self.images[idx], dtype=np.float32)
        # label = int(self.labels[idx])
        raw_label = self.labels[idx]

        if isinstance(raw_label, (str, np.str_)):
            label = self.class_to_idx[raw_label]
        else:
            label = int(raw_label)

        if patch.ndim == 1:
            side = int(np.sqrt(patch.shape[0]))
            patch = patch.reshape((side, side))

        elif patch.ndim == 3:
            if patch.shape[0] not in [1, 3]:
                patch = np.mean(patch, axis=0)
            elif patch.shape[-1] not in [1,3]:
                patch = np.mean(patch, axis=-1)
            elif patch.shape[0] in [1, 3]:
                patch = np.mean(patch, axis=0)

        if patch.shape != self.target_size:
            import cv2
            patch = cv2.resize(patch, self.target_size)

        p_min, p_max = patch.min(), patch.max()
        if p_max > p_min:
            patch = (patch - p_min) / (p_max - p_min)

        if self.to_spatial_rgb:
            rgb_patch = np.stack([patch, patch, patch], axis=0)
            return torch.tensor(
                rgb_patch, dtype=torch.float32
            ), torch.tensor(label, dtype=torch.long)

        return torch.tensor(
            patch.flatten(), dtype=torch.float32
        ), torch.tensor(label, dtype=torch.long)