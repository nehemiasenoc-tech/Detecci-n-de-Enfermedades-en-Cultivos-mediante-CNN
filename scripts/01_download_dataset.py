"""
Fase 0: Descarga del dataset desde Kaggle
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN
"""
import os
import shutil
import kagglehub

print("=" * 60)
print("DESCARGA DEL DATASET - Plant Diseases Dataset")
print("=" * 60)

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"Path a archivos del dataset: {path}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_DIR):
    shutil.copytree(path, DATASET_DIR)
    print(f"Dataset copiado a: {DATASET_DIR}")
else:
    print(f"Dataset ya existe en: {DATASET_DIR}")

print("\nContenido del directorio del dataset:")
for item in os.listdir(DATASET_DIR):
    item_path = os.path.join(DATASET_DIR, item)
    if os.path.isdir(item_path):
        print(f"  [DIR]  {item}")
        sub_items = os.listdir(item_path)
        print(f"         Contiene {len(sub_items)} elementos")
        for sub in sub_items[:5]:
            print(f"           - {sub}")
        if len(sub_items) > 5:
            print(f"           ... y {len(sub_items) - 5} más")
    else:
        print(f"  [FILE] {item}")

print("\nDescarga completada exitosamente.")
