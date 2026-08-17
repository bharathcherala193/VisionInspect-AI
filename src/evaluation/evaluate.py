"""
Evaluation script for VisionInspect AI.

Loads the trained ResNet50 weights and runs inference over the held-out
test set, reporting accuracy, per-class precision/recall/F1, and a
confusion matrix.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import BottleDefectDataset
from src.data.transforms import eval_transform
from src.models.model import build_model
from src.config.config import TRAIN_CSV, TEST_CSV, BATCH_SIZE, NUM_WORKERS, MODEL_PATH, CATEGORY, DEFECT_THRESHOLD


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Category: {CATEGORY} | Using device: {device}")
    print(f"Loading checkpoint: {MODEL_PATH}")
    print(f"Defect decision threshold: {DEFECT_THRESHOLD}")

    train_ds = BottleDefectDataset(TRAIN_CSV, transform=eval_transform)
    test_ds = BottleDefectDataset(TEST_CSV, transform=eval_transform, classes=train_ds.classes)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Class order: {train_ds.classes}")
    defect_idx = train_ds.classes.index("defect")

    model = build_model(
        num_classes=len(train_ds.classes),
        freeze_backbone=True,
        unfreeze_last_block=True,
        unfreeze_layer3=True,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Test-time augmentation: average predictions from the
            # original image and its horizontal flip. Smooths out a
            # few borderline cases without touching the model or the
            # underlying data — the model itself is unchanged.
            outputs_orig = model(images)
            outputs_flip = model(torch.flip(images, dims=[3]))

            probs = (F.softmax(outputs_orig, dim=1) + F.softmax(outputs_flip, dim=1)) / 2

            defect_probs = probs[:, defect_idx]
            preds = torch.where(
                defect_probs > DEFECT_THRESHOLD,
                torch.full_like(defect_probs, defect_idx, dtype=torch.long),
                torch.full_like(defect_probs, 1 - defect_idx, dtype=torch.long),
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()