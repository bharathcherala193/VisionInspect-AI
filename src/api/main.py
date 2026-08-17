"""
FastAPI inference server for VisionInspect AI.

Run: uvicorn src.api.main:app --reload
Then POST an image to http://127.0.0.1:8000/predict
"""
import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.data.dataset import BottleDefectDataset
from src.data.transforms import eval_transform
from src.models.model import build_model
from src.config.config import TRAIN_CSV, MODEL_PATH, DEFECT_THRESHOLD

app = FastAPI(title="VisionInspect AI")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = BottleDefectDataset(TRAIN_CSV, transform=eval_transform)
classes = train_ds.classes
defect_idx = classes.index("defect")

model = build_model(
    num_classes=len(classes),
    freeze_backbone=True,
    unfreeze_last_block=True,
    unfreeze_layer3=True,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


@app.get("/")
def root():
    return {"status": "VisionInspect AI is running", "classes": classes, "threshold": DEFECT_THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    defect_prob = probs[0, defect_idx].item()
    is_defect = defect_prob > DEFECT_THRESHOLD
    prediction = "defect" if is_defect else "good"
    confidence = defect_prob if is_defect else (1 - defect_prob)

    return {
        "prediction": prediction,
        "p_defect": round(defect_prob, 4),
        "confidence": round(confidence, 4),
        "threshold": DEFECT_THRESHOLD,
    }