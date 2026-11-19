
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights="DEFAULT" if pretrained else None
        )
        m.classifier[-1] = nn.Linear(
            m.classifier[-1].in_features,
            num_classes
        )

    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights="DEFAULT" if pretrained else None
        )
        m.classifier[-1] = nn.Linear(
            m.classifier[-1].in_features,
            num_classes
        )

    elif arch == "resnet50":
        m = models.resnet50(
            weights="DEFAULT" if pretrained else None
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

    return m


def head_parameters(model: nn.Module, arch: str):
    if arch == "resnet50":
        return model.fc.parameters()

    return model.classifier.parameters()


def set_trainable_stage(
    model: nn.Module,
    arch: str,
    stage: int
) -> None:

    # Congelar toda la arquitectura.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # La cabeza siempre permanece entrenable.
    head = list(head_parameters(model, arch))

    for parameter in head:
        parameter.requires_grad = True

    if stage == 1:
        return

    if arch == "mobilenet_v3_large":

        # Etapa 2: últimos tres bloques.
        if stage == 2:
            blocks = model.features[-3:]

        # Etapa 3: últimos seis bloques.
        elif stage == 3:
            blocks = model.features[-6:]

        else:
            raise ValueError(f"Etapa no soportada: {stage}")

        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True

    else:
        # Para las otras arquitecturas se conserva
        # el comportamiento de fine-tuning completo.
        for parameter in model.parameters():
            parameter.requires_grad = True


def trainable_parameters(model: nn.Module):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

