
"""Entrena el clasificador utilizando PyTorch DirectML.

Uso:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch_directml

from tqdm import tqdm

from fruitq.config import load_config
from fruitq.data import make_loader
from fruitq.model import (
    build_model,
    set_backbone_trainable
)


def run_epoch(model, loader, criterion, device, optimizer=None):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    correct = 0
    n = 0

    with torch.set_grad_enabled(training):

        for x, y in tqdm(
            loader,
            leave=False,
            desc="train" if training else "val"
        ):

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            if training:

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            total_loss += loss.item() * x.size(0)

            correct += (
                logits.argmax(1) == y
            ).sum().item()

            n += x.size(0)

    return total_loss / n, correct / n


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/config.yaml"
    )

    args = ap.parse_args()

    cfg = load_config(args.config)

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    # Dispositivo DirectML
    device = torch_directml.device()

    torch.manual_seed(dcfg["seed"])

    print("=" * 55)
    print("FRUIT QUALITY - ENTRENAMIENTO")
    print("=" * 55)

    print(f"PyTorch: {torch.__version__}")
    print(f"Dispositivo: {device}")
    print(f"Arquitectura: {mcfg['arch']}")

    processed = Path(dcfg["processed_dir"])

    train_ds, train_dl = make_loader(
        processed / "train",
        mcfg["img_size"],
        tcfg["batch_size"],
        tcfg["num_workers"],
        train=True
    )

    val_ds, val_dl = make_loader(
        processed / "val",
        mcfg["img_size"],
        tcfg["batch_size"],
        tcfg["num_workers"],
        train=False
    )

    classes = train_ds.classes

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            "Las clases de entrenamiento y validación no coinciden."
        )

    if len(classes) != 6:
        raise ValueError(
            f"Se esperaban 6 clases, pero se encontraron {len(classes)}: {classes}"
        )

    print(f"Clases: {classes}")
    print(f"Imágenes entrenamiento: {len(train_ds)}")
    print(f"Imágenes validación: {len(val_ds)}")

    model = build_model(
        mcfg["arch"],
        len(classes),
        mcfg["pretrained"]
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    out_dir = Path(tcfg["out_dir"])

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_acc = -1.0
    history = []

    optimizer = None

    for epoch in range(1, tcfg["epochs"] + 1):

        frozen = epoch <= tcfg["freeze_epochs"]

        if epoch == 1 or epoch == tcfg["freeze_epochs"] + 1:

            set_backbone_trainable(
                model,
                mcfg["arch"],
                trainable=not frozen
            )

            lr = (
                tcfg["lr"]
                if frozen
                else tcfg["lr"] * 0.1
            )

            optimizer = torch.optim.AdamW(
                (
                    p for p in model.parameters()
                    if p.requires_grad
                ),
                lr=lr,
                weight_decay=tcfg["weight_decay"]
            )

        tr_loss, tr_acc = run_epoch(
            model,
            train_dl,
            criterion,
            device,
            optimizer
        )

        va_loss, va_acc = run_epoch(
            model,
            val_dl,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc
        })

        print(
            f"[{epoch:02d}/{tcfg['epochs']}] "
            f"{'(cabeza)' if frozen else '(completo)'} "
            f"train_loss={tr_loss:.4f} "
            f"train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.4f}"
        )

        if va_acc > best_acc:

            best_acc = va_acc

            # Guardar los pesos en CPU
            state_dict = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

            torch.save(
                {
                    "state_dict": state_dict,
                    "classes": classes,
                    "arch": mcfg["arch"],
                    "img_size": mcfg["img_size"]
                },
                out_dir / "best.pt"
            )

            print("Nuevo mejor modelo guardado.")

    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    print("=" * 55)
    print(f"Mejor val accuracy: {best_acc:.4f}")
    print(f"Modelo guardado: {out_dir / 'best.pt'}")
    print("=" * 55)


if __name__ == "__main__":
    main()

