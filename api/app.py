"""
Fase 6: API Flask - Endpoint /predict
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

POST /predict - Content-Type: multipart/form-data - Body: image=<archivo>
Respuesta: {"cultivo": "...", "enfermedad": "...", "probabilidad": 98.7}
"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='/static')
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model = None
class_names = []
IMG_SIZE = (224, 224)


def load_class_mapping():
    """Carga el mapeo de clases."""
    global class_names
    mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        classes_dict = data.get("classes", {})
        class_names = [classes_dict[str(i)]["folder_name"]
                       for i in range(len(classes_dict))]
        logger.info(f"Cargadas {len(class_names)} clases desde class_mapping.json")
    else:
        logger.warning("class_mapping.json no encontrado. Intentando auto-detectar.")
        train_dir = None
        dataset_dir = os.path.join(BASE_DIR, "dataset")
        for root, dirs, files in os.walk(dataset_dir):
            if 'train' in os.path.basename(root).lower():
                for f in files[:3]:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        train_dir = root
                        break
            if train_dir:
                break
        if train_dir:
            class_names = sorted([d for d in os.listdir(train_dir)
                                  if os.path.isdir(os.path.join(train_dir, d))])
            logger.info(f"Clases detectadas: {len(class_names)}")


def load_trained_model():
    """Carga el modelo entrenado."""
    global model

    best_models = [
        os.path.join(MODELS_DIR, 'resnet50_best.keras'),
        os.path.join(MODELS_DIR, 'cnn_custom_best.keras'),
        os.path.join(MODELS_DIR, 'resnet50_final.keras'),
        os.path.join(MODELS_DIR, 'cnn_custom_final.keras'),
    ]

    for model_path in best_models:
        if os.path.exists(model_path):
            model = load_model(model_path)
            logger.info(f"Modelo cargado: {model_path}")
            return True

    logger.error("No se encontró ningún modelo entrenado en models/")
    return False


CROP_TRANSLATIONS = {
    "Apple": "Manzana",
    "Blueberry": "Arándano",
    "Cherry (including sour)": "Cereza",
    "Corn (maize)": "Maíz",
    "Grape": "Uva",
    "Orange": "Naranja",
    "Peach": "Durazno",
    "Pepper, bell": "Pimiento",
    "Potato": "Papa",
    "Raspberry": "Frambuesa",
    "Soybean": "Soya",
    "Squash": "Calabaza",
    "Strawberry": "Fresa",
    "Tomato": "Tomate",
}


def parse_class_name(folder_name):
    """Extrae cultivo (traducido al español) y enfermedad del nombre de carpeta."""
    if "___" in folder_name:
        crop_raw, disease_raw = folder_name.split("___", 1)
    else:
        crop_raw, disease_raw = folder_name, "Unknown"

    crop_key = crop_raw.replace("_", " ").strip()
    crop = CROP_TRANSLATIONS.get(crop_key, crop_key)

    disease = disease_raw.replace("_", " ").strip()
    if disease.lower() == "healthy":
        disease = "Healthy"

    return crop, disease


def preprocess_image(img_file):
    """Preprocesa la imagen para predicción."""
    img = Image.open(io.BytesIO(img_file.read()))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def serve_frontend():
    """Sirve el frontend."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción.
    POST /predict
    Content-Type: multipart/form-data
    Body: image=<archivo de imagen>

    Respuesta:
    {
        "cultivo": "Tomate",
        "enfermedad": "Tomato Early Blight",
        "probabilidad": 98.7
    }
    """
    start_time = datetime.now()
    logger.info(f"Petición /predict recibida desde {request.remote_addr}")

    if 'image' not in request.files:
        logger.warning("No se proporcionó imagen")
        return jsonify({
            "error": "No se proporcionó imagen",
            "mensaje": "Use el campo 'image' para enviar un archivo de imagen"
        }), 400

    img_file = request.files['image']

    if img_file.filename == '':
        return jsonify({
            "error": "Nombre de archivo vacío",
            "mensaje": "Seleccione un archivo de imagen válido"
        }), 400

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    ext = os.path.splitext(img_file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({
            "error": f"Formato no soportado: {ext}",
            "mensaje": f"Formatos permitidos: {', '.join(allowed_extensions)}"
        }), 400

    try:
        if model is None:
            return jsonify({
                "error": "Modelo no disponible",
                "mensaje": "El modelo no ha sido cargado. Verifique que existe en models/"
            }), 500

        img_file.seek(0)
        img_array = preprocess_image(img_file)

        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx]) * 100

        if pred_idx < len(class_names):
            folder_name = class_names[pred_idx]
            crop, disease = parse_class_name(folder_name)
        else:
            crop = "Desconocido"
            disease = "Desconocido"

        response = {
            "cultivo": crop,
            "enfermedad": disease,
            "probabilidad": round(confidence, 1)
        }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Predicción: {crop} - {disease} ({confidence:.1f}%) "
            f"en {elapsed:.3f}s"
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            "error": "Error procesando imagen",
            "mensaje": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Información del modelo cargado."""
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    return jsonify({
        "model_name": model.name,
        "num_classes": len(class_names),
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_params": model.count_params()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Recurso no encontrado"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500


def create_app():
    """Factory function para crear la app."""
    load_class_mapping()
    load_trained_model()
    return app


if __name__ == '__main__':
    logger.info("Iniciando API de Detección de Enfermedades en Cultivos")
    load_class_mapping()
    load_trained_model()
    logger.info(f"Clases cargadas: {len(class_names)}")
    app.run(host='0.0.0.0', port=5000, debug=False)
