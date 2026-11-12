"""Evalúa el checkpoint sobre un conjunto (test por defecto): reporte por clase
y matriz de confusión en PNG.

Uso:
  python scripts/evaluate.py                          # data/processed/test
  python scripts/evaluate.py --data data/propias      # sus fotos de súper (misma estructura por carpetas)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from fruitq.config import load_config
from fruitq.data import make_loader
from fruitq.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="models/best.pt")
    ap.add_argument("--data", default=None, help="carpeta con subcarpetas por clase")
    ap.add_argument("--tag", default="test", help="nombre para los archivos de salida")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    data_dir = Path(args.data) if args.data else Path(cfg["data"]["processed_dir"]) / "test"
    ds, dl = make_loader(data_dir, ckpt["img_size"], cfg["train"]["batch_size"],
                         cfg["train"]["num_workers"], train=False)
    if ds.classes != classes:
        # Permite evaluar con un subconjunto de clases (ej. solo manzanas propias)
        remap = {i: classes.index(c) for i, c in enumerate(ds.classes)}
    else:
        remap = None

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            preds = model(x.to(device)).argmax(1).cpu().tolist()
            y_pred += preds
            y_true += [remap[t] for t in y.tolist()] if remap else y.tolist()

    labels = sorted(set(y_true) | set(y_pred))
    names = [classes[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, digits=4)
    print(report)

    out_dir = Path(cfg["train"]["out_dir"])
    (out_dir / f"report_{args.tag}.txt").write_text(report)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Matriz de confusión — {args.tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{args.tag}.png", dpi=150)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    (out_dir / f"metrics_{args.tag}.json").write_text(json.dumps({"accuracy": acc, "n": len(y_true)}, indent=2))
    print(f"Accuracy={acc:.4f}  ->  {out_dir}/confusion_{args.tag}.png")


if __name__ == "__main__":
    main()
