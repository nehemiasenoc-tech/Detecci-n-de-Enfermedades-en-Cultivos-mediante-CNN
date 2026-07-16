"""
Fase 2: Preprocesamiento y Data Augmentation
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

- Resize, normalización, split train/validation/test
- Data Augmentation con ImageDataGenerator
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42


def find_data_dirs(dataset_dir):
    """Encuentra los directorios de train, validation y test."""
    train_dir = val_dir = test_dir = None

    for root, dirs, files in os.walk(dataset_dir):
        folder_name = os.path.basename(root).lower()
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files[:3]):
            if 'train' in folder_name and not train_dir:
                train_dir = root
            elif 'valid' in folder_name and not val_dir:
                val_dir = root
            elif 'test' in folder_name and not test_dir:
                test_dir = root

    if not train_dir:
        for root, dirs, files in os.walk(dataset_dir):
            if len(dirs) > 5:
                sub_has_images = False
                for d in dirs:
                    d_path = os.path.join(root, d)
                    for f in os.listdir(d_path)[:3]:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            sub_has_images = True
                            break
                if sub_has_images:
                    for d in dirs:
                        d_path = os.path.join(root, d)
                        subsub = [x for x in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, x))]
                        if len(subsub) > 5:
                            if 'train' in d.lower():
                                train_dir = d_path
                            elif 'val' in d.lower():
                                val_dir = d_path
                            elif 'test' in d.lower():
                                test_dir = d_path

    return train_dir, val_dir, test_dir


def get_class_names(directory):
    """Obtiene los nombres de las carpetas/clases."""
    if directory is None:
        return []
    return sorted([d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))])


def create_class_mapping(class_names):
    """Crea mapeo de clases a cultivo/enfermedad."""
    mapping = {}
    for idx, cls in enumerate(class_names):
        parts = cls.replace("___", "|").replace("_", " ").split("|", 1)
        crop = parts[0].strip()
        disease = parts[1].strip() if len(parts) > 1 else "Unknown"
        mapping[idx] = {
            "folder_name": cls,
            "cultivo": crop,
            "enfermedad": disease,
            "label": f"{crop} - {disease}"
        }
    return mapping


def create_augmented_generators(train_dir, val_dir, class_names):
    """Crea generadores con Data Augmentation para entrenamiento."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False,
        seed=RANDOM_SEED
    )

    return train_generator, val_generator, train_datagen


def visualize_augmentation(train_dir, class_names, output_dir):
    """Visualiza efectos del Data Augmentation."""
    if not class_names:
        return

    sample_class = class_names[0]
    class_path = os.path.join(train_dir, sample_class)
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        return

    img_path = os.path.join(class_path, images[0])
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    aug_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3]
    )

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Data Augmentation - Ejemplo: {sample_class}', fontsize=14)

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    aug_iter = aug_datagen.flow(img_array, batch_size=1)
    for i in range(1, 10):
        ax = axes[i // 5, i % 5]
        aug_img = next(aug_iter)[0].astype('uint8')
        ax.imshow(aug_img)
        ax.set_title(f'Aug #{i}')
        ax.axis('off')

    plt.tight_layout()
    aug_path = os.path.join(output_dir, "data_augmentation_example.png")
    plt.savefig(aug_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualización de augmentation guardada: {aug_path}")


def visualize_class_distribution(train_generator, output_dir):
    """Visualiza la distribución de clases después del preprocesamiento."""
    class_counts = train_generator.classes
    unique, counts = np.unique(class_counts, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    class_names = list(train_generator.class_indices.keys())
    class_names_short = [c[:25] for c in class_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
    ax1.bar(range(len(unique)), counts, color=colors)
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels(class_names_short, rotation=90, fontsize=6)
    ax1.set_ylabel('Número de Imágenes')
    ax1.set_title('Distribución de Clases (Train - con Augmentation)')

    ax2 = axes[1]
    crop_counts = {}
    for name, count in zip(class_names, counts):
        crop = name.split('___')[0].replace('_', ' ')
        crop_counts[crop] = crop_counts.get(crop, 0) + count
    ax2.pie(crop_counts.values(), labels=crop_counts.keys(), autopct='%1.1f%%')
    ax2.set_title('Distribución por Cultivo')

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "class_distribution_preprocessed.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribución guardada: {dist_path}")


def main():
    print("=" * 70)
    print("  FASE 2: PREPROCESAMIENTO Y DATA AUGMENTATION")
    print("  Sistema de Detección de Enfermedades en Cultivos")
    print("=" * 70)

    train_dir, val_dir, test_dir = find_data_dirs(DATASET_DIR)

    print(f"\nDirectorio train: {train_dir}")
    print(f"Directorio validación: {val_dir}")

    if train_dir is None or val_dir is None:
        print("ERROR: No se encontraron los directorios de datos.")
        return

    class_names = get_class_names(train_dir)
    print(f"\nNúmero de clases: {len(class_names)}")

    class_mapping = create_class_mapping(class_names)

    print(f"\nParámetros de preprocesamiento:")
    print(f"  - Tamaño de imagen: {IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Normalización: rescale 1/255")

    print(f"\nData Augmentation configurado:")
    print(f"  - Rotación: ±25°")
    print(f"  - Shift horizontal/vertical: 15%")
    print(f"  - Shear: 15%")
    print(f"  - Zoom: ±20%")
    print(f"  - Flip horizontal")
    print(f"  - Brillo: [0.8, 1.2]")
    print(f"  - Channel shift: 0.1")

    print(f"\nCreando generadores con augmentation...")
    train_gen, val_gen, aug_datagen = create_augmented_generators(
        train_dir, val_dir, class_names
    )

    print(f"\nGeneradores creados:")
    print(f"  Train: {train_gen.samples} imágenes, {train_gen.samples // BATCH_SIZE} batches")
    print(f"  Validation: {val_gen.samples} imágenes, {val_gen.samples // BATCH_SIZE} batches")
    print(f"  Clases detectadas: {train_gen.num_classes}")

    print(f"\nVisualizando augmentation...")
    visualize_augmentation(train_dir, class_names, REPORTS_DIR)

    print(f"Visualizando distribución de clases...")
    visualize_class_distribution(train_gen, REPORTS_DIR)

    mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
    mapping_data = {
        "num_classes": len(class_names),
        "img_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "classes": {str(k): v for k, v in class_mapping.items()}
    }
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f"Mapeo de clases guardado: {mapping_path}")

    print(f"\n{'='*70}")
    print("RESUMEN DEL PREPROCESAMIENTO")
    print(f"{'='*70}")
    print(f"✓ Imágenes redimensionadas a {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"✓ Normalización [0, 1] aplicada (rescale 1/255)")
    print(f"✓ Data Augmentation activado para entrenamiento")
    print(f"✓ Validation set sin augmentation (solo rescale)")
    print(f"✓ Mapeo de clases guardado")

    info_path = os.path.join(REPORTS_DIR, "preprocessing_info.json")
    info = {
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "normalization": "rescale 1/255",
        "augmentation": {
            "rotation_range": 25,
            "width_shift_range": 0.15,
            "height_shift_range": 0.15,
            "shear_range": 0.15,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "brightness_range": [0.8, 1.2],
            "channel_shift_range": 0.1
        },
        "train_samples": train_gen.samples,
        "val_samples": val_gen.samples,
        "num_classes": train_gen.num_classes
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nPreprocesamiento completado exitosamente.")
    return train_gen, val_gen, class_names, class_mapping


if __name__ == "__main__":
    main()
