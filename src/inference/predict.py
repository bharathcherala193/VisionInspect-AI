"""
Single-image inference for VisionInspect AI.

Usage: python -m src.inference.predict path/to/image.png
"""
import sys
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import BottleDefectDataset
from src.data.transforms import eval_transform
from src.models.model import build_model
from src.config.config import TRAIN_CSV, MODEL_PATH, DEFECT_THRESHOLD


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recover the class ordering used during training
    train_ds = BottleDefectDataset(TRAIN_CSV, transform=eval_transform)
    classes = train_ds.classes
    defect_idx = classes.index("defect")

    model = build_model(num_classes=len(classes), freeze_backbone=True, unfreeze_last_block=True, unfreeze_layer3=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    defect_prob = probs[0, defect_idx].item()
    is_defect = defect_prob > DEFECT_THRESHOLD
    predicted_class = "defect" if is_defect else "good"
    # Report confidence as distance from the threshold, so it stays
    # meaningful even though we're not using plain argmax anymore.
    confidence = defect_prob if is_defect else (1 - defect_prob)

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"P(defect): {defect_prob:.2%}  (threshold: {DEFECT_THRESHOLD})")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.inference.predict <image_path>")
        sys.exit(1)

    predict(sys.argv[1])