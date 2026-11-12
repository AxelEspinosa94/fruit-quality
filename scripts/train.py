"""Entrena el clasificador con transfer learning y guarda el mejor checkpoint.

Uso: python scripts/train.py [--config configs/config.yaml]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from fruitq.config import load_config
from fruitq.data import make_loader
from fruitq.model import build_model, set_backbone_trainable


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(dcfg["seed"])
    print(f"Dispositivo: {device}")

    processed = Path(dcfg["processed_dir"])
    train_ds, train_dl = make_loader(processed / "train", mcfg["img_size"], tcfg["batch_size"],
                                     tcfg["num_workers"], train=True)
    _, val_dl = make_loader(processed / "val", mcfg["img_size"], tcfg["batch_size"],
                            tcfg["num_workers"], train=False)
    classes = train_ds.classes
    print(f"Clases ({len(classes)}): {classes}")

    model = build_model(mcfg["arch"], len(classes), mcfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, tcfg["epochs"] + 1):
        # Primeras épocas: solo la cabeza. Después: toda la red con lr más bajo.
        frozen = epoch <= tcfg["freeze_epochs"]
        if epoch == 1 or epoch == tcfg["freeze_epochs"] + 1:
            set_backbone_trainable(model, mcfg["arch"], trainable=not frozen)
            lr = tcfg["lr"] if frozen else tcfg["lr"] * 0.1
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=lr, weight_decay=tcfg["weight_decay"])

        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"[{epoch:02d}] {'(cabeza)' if frozen else '(completo)'} "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "arch": mcfg["arch"], "img_size": mcfg["img_size"]},
                       out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Mejor val acc: {best_acc:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
