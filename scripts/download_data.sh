#!/usr/bin/env bash
# Descarga el dataset base de Kaggle a data/raw.
# Requiere token de Kaggle: https://www.kaggle.com/settings -> "Create New Token"
# y guardar kaggle.json en ~/.kaggle/kaggle.json (chmod 600).
set -euo pipefail

mkdir -p data/raw
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p data/raw --unzip
echo "Listo. Revisa data/raw y luego corre: python scripts/prepare_data.py"
