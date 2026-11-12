"""Junta todas las carpetas de clase en data/raw y genera un split estratificado
train/val/test en data/processed.

La clase de cada imagen es el nombre de la carpeta que la contiene (en minúsculas),
así que puedes agregar fotos propias en data/raw/propias/<clase>/ y se mezclan solas.

Uso: python scripts/prepare_data.py [--config configs/config.yaml]
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from fruitq.config import load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(raw_dir: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            by_class[p.parent.name.lower()].append(p)
    return by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)["data"]

    raw, out = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    split = cfg["split"]
    rng = random.Random(cfg["seed"])

    by_class = collect(raw)
    if not by_class:
        raise SystemExit(f"No encontré imágenes en {raw}. ¿Corriste download_data.sh?")

    for part in ("train", "val", "test"):  # borra splits previos, conserva .gitkeep
        if (out / part).exists():
            shutil.rmtree(out / part)

    for cls, files in sorted(by_class.items()):
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_train, n_val = int(n * split["train"]), int(n * split["val"])
        parts = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for part, items in parts.items():
            dst = out / part / cls
            dst.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                shutil.copy2(src, dst / f"{i:05d}{src.suffix.lower()}")
        print(f"{cls:20s} total={n:5d}  " + "  ".join(f"{k}={len(v)}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
