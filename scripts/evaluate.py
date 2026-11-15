"""Evalúa uno o varios checkpoints sobre el mismo conjunto de imágenes.

Con un checkpoint: reporte por clase + matriz de confusión.
Con varios: además, una tabla comparativa (accuracy, F1 macro, recall por clase).

Uso:
  # Test interno del split (data/processed/test)
  python scripts/evaluate.py

  # Fotos propias, nunca usadas para entrenar
  python scripts/evaluate.py --data data/propias_test --tag propias

  # Comparar modelo base vs. modelo reentrenado con fotos propias
  python scripts/evaluate.py --checkpoint models/baseline.pt models/best.pt \
                             --data data/propias_test --tag propias

La carpeta --data debe tener una subcarpeta por clase con los mismos nombres
que el entrenamiento (puede traer solo algunas clases).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

from fruitq.config import load_config
from fruitq.data import make_loader
from fruitq.model import build_model


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_folder(model, ckpt, data_dir: Path, batch_size: int, num_workers: int, device: str):
    ds, dl = make_loader(data_dir, ckpt["img_size"], batch_size, num_workers, train=False)
    classes = ckpt["classes"]
    unknown = [c for c in ds.classes if c not in classes]
    if unknown:
        raise SystemExit(f"Clases en {data_dir} que el modelo no conoce: {unknown}\n"
                         f"Clases del modelo: {classes}")
    # Índice de la carpeta -> índice de la clase en el modelo
    remap = [classes.index(c) for c in ds.classes]

    y_true, y_pred = [], []
    for x, y in dl:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += [remap[t] for t in y.tolist()]
    return y_true, y_pred


def save_confusion(y_true, y_pred, labels, names, title, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", nargs="+", default=["models/best.pt"],
                    help="uno o varios .pt; con varios se genera la comparación")
    ap.add_argument("--data", default=None, help="carpeta con subcarpetas por clase")
    ap.add_argument("--tag", default="test", help="nombre del conjunto en los archivos de salida")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data) if args.data else Path(cfg["data"]["processed_dir"]) / "test"
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for ck_path in map(Path, args.checkpoint):
        name = ck_path.stem
        model, ckpt = load_checkpoint(ck_path, device)
        y_true, y_pred = predict_folder(model, ckpt, data_dir, cfg["train"]["batch_size"],
                                        cfg["train"]["num_workers"], device)
        classes = ckpt["classes"]
        labels = sorted(set(y_true))              # solo clases presentes en el conjunto
        names = [classes[i] for i in labels]

        report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                       digits=4, zero_division=0)
        print(f"\n=== {name} sobre '{args.tag}' ({len(y_true)} imágenes) ===\n{report}")

        (out_dir / f"report_{args.tag}_{name}.txt").write_text(report)
        save_confusion(y_true, y_pred, labels, names, f"{name} — {args.tag}",
                       out_dir / f"confusion_{args.tag}_{name}.png")

        recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        results[name] = {
            "checkpoint": str(ck_path),
            "n": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "recall_por_clase": dict(zip(names, map(float, recalls))),
        }

    (out_dir / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if len(results) > 1:
        print_comparison(results, args.tag)
    print(f"\nArchivos en {out_dir}/ (report_*, confusion_*, metrics_{args.tag}.json)")


def print_comparison(results: dict, tag: str):
    names = list(results)
    classes = sorted({c for r in results.values() for c in r["recall_por_clase"]})
    w = max(14, *(len(n) + 2 for n in names))
    first = names[0]

    print(f"\n=== Comparación sobre '{tag}' ===")
    header = f"{'métrica':22s}" + "".join(f"{n:>{w}s}" for n in names)
    if len(names) == 2:
        header += f"{'Δ':>10s}"
    print(header)
    print("-" * len(header))

    rows = [("accuracy", lambda r: r["accuracy"]), ("f1_macro", lambda r: r["f1_macro"])]
    rows += [(f"recall {c}", lambda r, c=c: r["recall_por_clase"].get(c, float("nan"))) for c in classes]
    for label, get in rows:
        vals = [get(results[n]) for n in names]
        line = f"{label:22s}" + "".join(f"{v:>{w}.4f}" for v in vals)
        if len(names) == 2:
            line += f"{(vals[1] - vals[0]) * 100:>+9.1f}%"
        print(line)

    if len(names) == 2:
        a, b = results[first]["accuracy"], results[names[1]]["accuracy"]
        print(f"\nAccuracy: {a:.1%} ({first}) -> {b:.1%} ({names[1]})")


if __name__ == "__main__":
    main()