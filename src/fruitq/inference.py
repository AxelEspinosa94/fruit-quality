"""Inferencia reutilizable: la usan scripts/predict.py y, después, la API."""
from pathlib import Path

import torch
from PIL import Image

from .data import eval_transforms
from .model import build_model


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.classes: list[str] = ckpt["classes"]
        self.model = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.tf = eval_transforms(ckpt["img_size"])

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Recibe una o n imágenes PIL y devuelve un dict por imagen."""
        batch = torch.stack([self.tf(img.convert("RGB")) for img in images]).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1).cpu()
        out = []
        for p in probs:
            conf, idx = p.max(dim=0)
            out.append({
                "clase": self.classes[idx],
                "confianza": round(conf.item(), 4),
                "probabilidades": {c: round(v.item(), 4) for c, v in zip(self.classes, p)},
            })
        return out
