import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import cv2
import pydicom

class SpineCascadeDataset(Dataset):
    def __init__(self, csv_file, img_dir, path_size=(28, 28), include_background=True):
        """
        loader parsing multiclass annotations, variable slice channels,
        pairing ground-truth lesion boxes with random background control windows"
        """

        raw_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.patch_size = self.patch_size
        self.include_background = include_background

        #isolate lesion instances containing active bounding box coordinates
        self.bbox_annotations = raw_df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax']).reset_index(drop=True)

        #dynamically create clean continuous tracking maps
        self.unique_classes = sorted(self.bbox_annotations['class_id'].unique())
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.unique_classes)}
        self.class_to_idx[0] = 0 #normal/background class baseline

    def get_total_classes(self):
        return len(self.class_to_idx)
    

    def __len__(self):
        if self.include_background:
            return len(self.bbox_annotations)
        return len(self.bbox_annotations)
    
    def __getitem__(self, idx):
        is_background = idx >= len(self.bbox_annotations)
        row_idx = idx % len(self.bbox_annotations)
        row = self.bbox_annotations.iloc[row_idx]

        dicom_path = os.path.join(self.img_dir, f"{row['image_id']}.dicom")
        if not os.path.exists(dicom_path):
            return torch.zeros(self.patch_size[0] * self.patch_size[1]), torch.tensor(0, dtype=torch.long)
        
        #load raw medical image vectors
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)

        #multidimensional channel control
        if pixel_array.ndim == 3:
            pixel_array = np.mean(pixel_array, axis=0)

        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array

        #scale raw density thresholds to a stable [0, 1] intensity range
        p_min, p_max = np.min(pixel_array), np.max(pixel_array)
        if p_max > p_min:
            pixel_array = (pixel_array - p_min) / (p_max - p_min)

        h, w = pixel_array.shape


        if not is_background:
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = self.class_to_idx[int(row['class_id'])]

        else:
            #generate random background bounding box proposals for localization baseline
            box_w, box_h = int(row['xmax'] - row['xmin']), int(row['ymax'] - row['ymin'])
            xmin = np.random.randint(0, max(1, w - box_w))
            ymin = np.random.randint(0, max(1, h - box_h))
            xmax, ymax = xmin + box_w, ymin + box_h
            label = 0 #background/normal patch

        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        patch = pixel_array[ymin:ymax, xmin:xmax]
        if patch.size == 0 or xmax <= xmin or ymax <= ymin:
            patch = np.zeros(self.patch_size, dtype=np.float32)

        #highlight local bone structure using CLAHE
        patch_8bit = (patch * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        patch_enhanced = clahe.apply(patch_8bit)

        final_patch = cv2.resize(patch_enhanced, self.patch_size).astype(np.float32) / 255.0
        return torch.tensor(final_patch.flatten(), dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        