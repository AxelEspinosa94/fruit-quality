"""Inferencia reutilizable: la usan scripts/predict.py y, después, la API."""
from pathlib import Path

import torch
import torch_directml
from PIL import Image

from .data import eval_transforms
from .model import build_model


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | torch.device | None = None, device_index: int = 1):
        if device is not None:
            self.device = device
            self.device_name = str(device)
        elif torch_directml.is_available():
            self.device = torch_directml.device(device_index)
            self.device_name = f"DirectML:{device_index} ({torch_directml.device_name(device_index)})"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = "cuda"
        else:
            self.device = torch.device("cpu")
            self.device_name = "cpu"

        # Cargar primero en CPU (los pesos de best.pt ya se guardaron en CPU) y luego mover el modelo a DirectML
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.classes: list[str] = ckpt["classes"]
        self.model = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.tf = eval_transforms(ckpt["img_size"])

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Recibe una o n imágenes PIL y devuelve un dict por imagen."""
        batch = torch.stack([self.tf(img.convert("RGB")) for img in images]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).cpu()
        out = []
        for p in probs:
            conf, idx = p.max(dim=0)
            out.append({
                "clase": self.classes[idx],
                "confianza": round(conf.item(), 4),
                "probabilidades": {c: round(v.item(), 4) for c, v in zip(self.classes, p)},
            })
        return out
