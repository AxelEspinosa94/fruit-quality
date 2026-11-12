"""Predice una o n imágenes y muestra el resultado en JSON (mismo formato que usará la API).

Uso: python scripts/predict.py foto1.jpg foto2.jpg [--checkpoint models/best.pt]
"""
import argparse
import json

from PIL import Image

from fruitq.inference import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="models/best.pt")
    args = ap.parse_args()

    predictor = Predictor(args.checkpoint)
    results = predictor.predict([Image.open(p) for p in args.images])
    for path, r in zip(args.images, results):
        r["archivo"] = path
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
