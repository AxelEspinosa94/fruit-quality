# fruit-quality

Clasificador de calidad de fruta (fresca / podrida) para supermercados, con API y web.

## Datos base

[Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
(Kaggle): manzana, plátano y naranja, cada una en fresca y podrida (6 clases).
Después se agregan fotos propias tomadas en súper local en `data/raw/propias/<clase>/`.

## Estructura

```
fruit-quality/
├── configs/config.yaml     # hiperparámetros y rutas
├── src/fruitq/             # paquete reutilizable (la API lo importa)
│   ├── config.py
│   ├── data.py             # transforms y DataLoaders
│   ├── model.py            # MobileNetV3 / EfficientNet / ResNet con cabeza nueva
│   └── inference.py        # Predictor: una o n imágenes -> dict
├── scripts/
│   ├── download_data.sh    # baja el dataset de Kaggle
│   ├── prepare_data.py     # split estratificado 70/15/15
│   ├── train.py            # entrena y guarda models/best.pt
│   ├── evaluate.py         # reporte y matriz de confusión
│   └── predict.py          # inferencia desde terminal
├── data/{raw,processed}/   # ignorados por git
├── models/                 # ignorado por git
├── api/                    # FastAPI
├── web/                    # interfaz
├── tests/
└── deploy/                 # Dockerfile, despliegue
```

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # para poder importar fruitq desde scripts/ y api/

bash scripts/download_data.sh
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py ruta/a/foto.jpg
```

Para medir qué tan bien funciona fuera del dataset, guarden sus fotos en
`data/propias/<clase>/` (mismos nombres de clase) y corran:

```bash
python scripts/evaluate.py --data data/propias --tag propias
```
