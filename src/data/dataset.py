"""
PyTorch Dataset for VisionInspect AI.

Reads one of our train/val/test CSVs (filepath, label) and
serves (image_tensor, label) pairs for training.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class BottleDefectDataset(Dataset):
    def __init__(self, csv_path, transform=None, classes=None):
        """
        csv_path: path to train.csv / val.csv / test.csv
        transform: a torchvision transform (resize, normalize, etc.)
        classes: optional list of class names, in fixed order. Pass this
                 in for val/test so they reuse train's label mapping
                 instead of each CSV building its own from whatever
                 labels happen to appear in it.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.classes = classes if classes is not None else sorted(self.df["label"].astype(str).unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        # PyTorch calls this to know how many samples make up one epoch
        return len(self.df)

    def __getitem__(self, idx):
        # PyTorch calls this once per index when building a batch
        row = self.df.iloc[idx]

        image = Image.open(row["filepath"]).convert("RGB")
        label = self.class_to_idx[str(row["label"])]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)