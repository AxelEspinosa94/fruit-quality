import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return m


def head_parameters(model: nn.Module, arch: str):
    return model.fc.parameters() if arch == "resnet50" else model.classifier.parameters()


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_parameters(model, arch):
        p.requires_grad = True
