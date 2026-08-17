import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(num_classes, freeze_backbone=True, unfreeze_last_block=False, unfreeze_layer3=False):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True

    if unfreeze_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model