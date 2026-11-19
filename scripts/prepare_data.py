"""Combina Kaggle Train + Mendeley, conserva Kaggle Test y elimina
duplicados exactos de Mendeley contra todo Kaggle y entre sí.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

from fruitq.config import load_config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"No existe la carpeta: {folder}")

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_images(files: list[Path], destination: Path, prefix: str):
    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(files):
        target = destination / f"{prefix}_{index:05d}{source.suffix.lower()}"
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/processed_combined")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["data"].get("seed", 42)

    kaggle = Path("data/raw/dataset")
    kaggle_train = kaggle / "train"
    kaggle_test = kaggle / "test"
    mendeley = Path("data/raw/mendeley/images")
    output = Path(args.output)

    print("===== VALIDANDO DATASETS =====")

    # Leer Kaggle y registrar hashes de Train y Test.
    kaggle_files = {}
    known_hashes = {}

    for cls in CLASSES:
        train_files = images_in(kaggle_train / cls)
        test_files = images_in(kaggle_test / cls)

        if not train_files or not test_files:
            raise SystemExit(f"Faltan imágenes de Kaggle para: {cls}")

        kaggle_files[cls] = (train_files, test_files)

        for path in train_files + test_files:
            known_hashes.setdefault(sha256(path), cls)

        print(
            f"{cls:16s} Kaggle Train={len(train_files):4d} "
            f"Kaggle Test={len(test_files):4d}"
        )

    # Incorporar únicamente imágenes Mendeley no duplicadas.
    mendeley_unique = {}
    duplicates = 0

    for cls in CLASSES:
        accepted = []

        for path in images_in(mendeley / cls):
            digest = sha256(path)

            if digest in known_hashes:
                duplicates += 1
                continue

            known_hashes[digest] = cls
            accepted.append(path)

        mendeley_unique[cls] = accepted

    print(f"\nDuplicados Mendeley descartados: {duplicates}")

    # Preparar una salida nueva, sin tocar data/processed ni models/best.pt.
    if output.exists():
        shutil.rmtree(output)

    rng = random.Random(seed)
    print(f"\n===== GENERANDO {output} =====")

    total_train = total_val = total_test = 0

    for cls in CLASSES:
        kaggle_train_files, kaggle_test_files = kaggle_files[cls]
        combined = [
            (path, "kaggle") for path in kaggle_train_files
        ] + [
            (path, "mendeley") for path in mendeley_unique[cls]
        ]

        rng.shuffle(combined)

        # 85 % entrenamiento y 15 % validación.
        val_count = max(1, round(len(combined) * 0.15))
        train_items = combined[val_count:]
        val_items = combined[:val_count]

        train_dir = output / "train" / cls
        val_dir = output / "val" / cls
        test_dir = output / "test" / cls

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for index, (source, origin) in enumerate(train_items):
            target = train_dir / f"{origin}_{index:05d}{source.suffix.lower()}"
            shutil.copy2(source, target)

        for index, (source, origin) in enumerate(val_items):
            target = val_dir / f"{origin}_{index:05d}{source.suffix.lower()}"
            shutil.copy2(source, target)

        # Test es exactamente el conjunto de prueba original de Kaggle.
        copy_images(kaggle_test_files, test_dir, "kaggle_test")

        total_train += len(train_items)
        total_val += len(val_items)
        total_test += len(kaggle_test_files)

        print(
            f"{cls:16s} "
            f"train={len(train_items):4d} "
            f"val={len(val_items):4d} "
            f"test={len(kaggle_test_files):4d} "
            f"(Mendeley incorporadas={len(mendeley_unique[cls])})"
        )

    print("\n===== RESUMEN =====")
    print("Train:", total_train)
    print("Val:  ", total_val)
    print("Test: ", total_test)
    print("Total:", total_train + total_val + total_test)
    print("Salida:", output.resolve())


if __name__ == "__main__":
    main()
