# Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Proyecto de Machine Learning — Grupo 2

Sistema que identifica el **cultivo** y la **enfermedad** presente en una hoja a partir de una imagen, usando redes neuronales convolucionales. Incluye dos modelos entrenados y comparados (una CNN propia y Transfer Learning con ResNet50), una API REST en Flask, un frontend web, y despliegue vía Docker.

## Resultados

Dataset: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — 38 clases, ~70k imágenes de entrenamiento, ~17k de validación.

| Métrica | CNN propia | ResNet50 (Transfer Learning) |
|---|---|---|
| Accuracy | 99.7% | 76.9% |
| Precisión (weighted) | 99.7% | 80.7% |
| Recall (weighted) | 99.7% | 76.9% |
| F1-score (weighted) | 99.7% | 75.8% |

Gráficos, matrices de confusión y curvas ROC completas en [`reports/`](reports/).

## Estructura del proyecto

```
api/                 API Flask (endpoint /predict, /health, /model/info)
frontend/             Interfaz web (HTML/CSS/JS) para probar el modelo
kaggle_notebook/       Scripts del pipeline adaptados para correr en Kaggle (GPU gratuita)
scripts/                Scripts del pipeline para ejecución local
models/                 Modelos entrenados (*.keras) — no incluidos en el repo por tamaño
reports/                Resultados: EDA, métricas, matrices de confusión, curvas ROC
presentacion/           Presentación del proyecto (LaTeX/Beamer)
deploy/                  Configuración de despliegue (systemd)
Dockerfile, .dockerignore    Imagen de contenedor para la API
```

## Fases del proyecto

1. **Exploración (EDA)** — análisis de clases, balance, resolución de imágenes.
2. **Preprocesamiento** — normalización y Data Augmentation.
3. **CNN propia** — arquitectura convolucional diseñada desde cero.
4. **Transfer Learning (ResNet50)** — preentrenada en ImageNet, fine-tuning de las últimas capas.
5. **Evaluación comparativa** — accuracy, precisión, recall, F1, matriz de confusión, curvas ROC.
6. **API + despliegue** — Flask + Docker / systemd.
7. **Frontend** — interfaz web para subir una imagen y ver el resultado.

Documentación técnica completa (instalación, ejecución del pipeline, API reference, preguntas de análisis) en [`DOCUMENTACION_COMPLETA.txt`](DOCUMENTACION_COMPLETA.txt).

## Uso rápido

### Con Docker

```bash
docker build -t cnn-api .
docker run -d -p 5000:5000 --name cnn-api cnn-api
```

La API queda disponible en `http://localhost:5000` (frontend incluido en `/`).

### Local

```bash
pip install -r requirements.txt
python api/app.py
```

### API — `POST /predict`

```bash
curl -X POST -F "image=@hoja.jpg" http://localhost:5000/predict
```

```json
{
  "cultivo": "Tomate",
  "enfermedad": "Early blight",
  "probabilidad": 98.7
}
```

## Nota sobre los modelos

Los archivos `.keras` (modelos entrenados) no se incluyen en este repositorio por tamaño (varios cientos de MB). Se entrenaron en Kaggle (GPU gratuita) usando los scripts de `kaggle_notebook/`.
