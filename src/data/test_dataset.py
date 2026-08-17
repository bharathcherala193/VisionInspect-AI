import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import config
from src.data.dataset import BottleDefectDataset

ds = BottleDefectDataset(config.TRAIN_CSV)
print(f"Dataset size: {len(ds)}")

image, label = ds[0]
print(f"First sample -> image type: {type(image)}, size: {image.size}, label: {label}")