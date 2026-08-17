import torch
from torch.utils.data import DataLoader
from src.data.dataset import BottleDefectDataset
from src.data.transforms import train_transform
from src.config.config import TRAIN_CSV, BATCH_SIZE, NUM_WORKERS

train_dataset = BottleDefectDataset(TRAIN_CSV, transform=train_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Labels in batch: {labels}")