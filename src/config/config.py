import os
from pathlib import Path

CATEGORY = os.environ.get("MVTEC_CATEGORY", "bottle")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / CATEGORY
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / CATEGORY

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / f"resnet50_{CATEGORY}.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42

# Used only by predict.py — 0.5 is the "no bias" default, equivalent to
# plain argmax. Raise it (e.g. 0.6) to reduce false "defect" alarms;
# lower it (e.g. 0.4) to catch more real defects at the cost of more
# false alarms. 0.3 (an earlier experiment) was too aggressive — it
# flagged over half of genuinely good units as defective.
DEFECT_THRESHOLD = 0.5