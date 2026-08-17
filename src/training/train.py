import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import BottleDefectDataset
from src.data.transforms import train_transform, eval_transform
from src.models.model import build_model
from src.config.config import TRAIN_CSV, VAL_CSV, BATCH_SIZE, NUM_WORKERS, MODEL_PATH, CATEGORY
import random

SEED = 42  # change this number to try a different run

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def per_class_accuracy(preds, labels, classes):
    result = {}
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    for idx, cls in enumerate(classes):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        result[cls] = (preds[mask] == idx).mean()
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Category: {CATEGORY} | Using device: {device}")

    train_ds = BottleDefectDataset(TRAIN_CSV, transform=train_transform)
    val_ds = BottleDefectDataset(VAL_CSV, transform=eval_transform, classes=train_ds.classes)

    # Build a per-sample weight so each epoch draws roughly equal numbers
    # of "good" and "defect" images, instead of the natural imbalanced mix.
    class_counts = train_ds.df["label"].value_counts()
    sample_weights = train_ds.df["label"].map(lambda lbl: 1.0 / np.sqrt(class_counts[lbl])).values
    class_counts = train_ds.df["label"].value_counts()
    sample_weights = train_ds.df["label"].map(lambda lbl: 1.0 / class_counts[lbl]).values
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print(f"Class counts: {dict(class_counts)}")
    print("Using WeightedRandomSampler to balance batches (plain CrossEntropyLoss).")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_ds.classes)
    model = build_model(
        num_classes=num_classes,
        freeze_backbone=True,
        unfreeze_last_block=True,
        unfreeze_layer3=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.layer3.parameters(), "lr": 5e-6},
    ])

    num_epochs = 30
    patience = 8
    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()

                all_val_preds.append(preds)
                all_val_labels.append(labels)

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        per_class = per_class_accuracy(
            torch.cat(all_val_preds), torch.cat(all_val_labels), train_ds.classes
        )
        per_class_str = ", ".join(f"{k}: {v:.2f}" for k, v in per_class.items())

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Val per-class: [{per_class_str}]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  -> New best (val_acc={val_acc:.4f}), checkpoint saved in memory")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> No improvement for {patience} epochs, stopping early.")
                break

    if best_state is None:
        print("WARNING: val accuracy never improved from 0.0 — saving final epoch instead.")
        best_state = model.state_dict()

    torch.save(best_state, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} (best val_acc={best_val_acc:.4f}).")


if __name__ == "__main__":
    main()