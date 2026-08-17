"""
Image transforms for VisionInspect AI.

train_transform: resize + augmentation + normalize (used only on training data)
eval_transform:  resize + normalize, no randomness (used on val/test data)
"""
import torchvision.transforms as T
from src.config import config

# ImageNet mean/std - required because we're using an ImageNet-pretrained
# ResNet50. These are NOT arbitrary; they're the exact values used when
# the pretrained weights were originally trained.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),                                  # PIL (H,W,C) 0-255 -> tensor (C,H,W) 0-1
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])