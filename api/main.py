import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from fruitq.inference import Predictor

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga el modelo en GPU (usa "cuda" automáticamente si está disponible)
    ml_models["predictor"] = Predictor(checkpoint="models/best.pt", device_index=1)
    print(f"Modelo cargado en dispositivo: {ml_models['predictor'].device_name}")
    yield
    ml_models.clear()

app = FastAPI(
    title="Fruit Quality API",
    description="Clasificador de calidad de fruta (fresca / podrida)",
    version="1.0.0",
    lifespan=lifespan
)

# Habilitar CORS por si conectas la interfaz de la carpeta web/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    predictor = ml_models.get("predictor")
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": predictor.device_name if predictor else None,
        "classes": predictor.classes if predictor else []
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predictor: Predictor = ml_models["predictor"]
        # predict() espera una lista [image] y devuelve una lista de resultados
        resultados = predictor.predict([image])

        return {
            "filename": file.filename,
            **resultados[0]  # Devuelve directamente: clase, confianza y probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Procesa varias imágenes en un solo batch en la GPU."""
    images = []
    filenames = []

    try:
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{file.filename} no es una imagen válida.")
            contents = await file.read()
            images.append(Image.open(io.BytesIO(contents)).convert("RGB"))
            filenames.append(file.filename)

        predictor: Predictor = ml_models["predictor"]
        resultados = predictor.predict(images)

        return [
            {"filename": fname, **res}
            for fname, res in zip(filenames, resultados)
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia por lote: {str(e)}")
