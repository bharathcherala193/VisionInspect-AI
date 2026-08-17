"""
Scans data/raw/bottle, builds a (filepath, label, defect_type) table from
train/good + all test/* folders, and writes stratified train/val/test CSVs
into data/processed/.

Run once: python -m src.data.prepare_splits
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import config


def collect_images():
    """Walk train/good and test/* and return list of (path, label, defect_type)."""
    records = []

    # train/good -> label "good", defect_type "good"
    good_train_dir = config.DATA_RAW / "train" / "good"
    for img_path in good_train_dir.glob("*.png"):
        records.append((str(img_path), "good", "good"))

    # test/* -> label "good" for the good subfolder, "defect" for every defect subfolder
    test_dir = config.DATA_RAW / "test"
    for subfolder in test_dir.iterdir():
        if not subfolder.is_dir():
            continue
        label = "good" if subfolder.name == "good" else "defect"
        for img_path in subfolder.glob("*.png"):
            records.append((str(img_path), label, subfolder.name))

    return records


def stratified_split(records, train_ratio, val_ratio, seed):
    """Split records into train/val/test, preserving the ratio of each
    defect_type in every split (so 'contamination' images, for example,
    don't all end up in one split by chance)."""
    random.seed(seed)

    by_type = defaultdict(list)
    for rec in records:
        by_type[rec[2]].append(rec)

    train, val, test = [], [], []
    for defect_type, items in by_type.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_csv(records, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "defect_type"])
        writer.writerows(records)
    print(f"Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    records = collect_images()
    print(f"Found {len(records)} total images.")

    counts = defaultdict(int)
    for _, _, defect_type in records:
        counts[defect_type] += 1
    print("Breakdown by type:", dict(counts))

    train, val, test = stratified_split(
        records, config.TRAIN_RATIO, config.VAL_RATIO, config.RANDOM_SEED
    )

    write_csv(train, config.TRAIN_CSV)
    write_csv(val, config.VAL_CSV)
    write_csv(test, config.TEST_CSV)