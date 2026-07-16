"""
Celda 3 - Fase 3: CNN Propia - Diseño y Entrenamiento
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Red convolucional custom sin modelos preentrenados.
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                      Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = "/kaggle/input"
MODELS_DIR = "/kaggle/working/models"
REPORTS_DIR = "/kaggle/working/reports"
LOGS_DIR = "/kaggle/working/logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_SEED = 42
LEARNING_RATE = 0.001


def find_data_dirs(dataset_dir):
    """Encuentra carpetas train/valid: contienen subcarpetas de clases con imágenes."""
    train_dir = val_dir = None
    for root, dirs, files in os.walk(dataset_dir):
        folder_name = os.path.basename(root).lower()
        class_subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(class_subdirs) < 2:
            continue

        has_images = False
        for d in class_subdirs[:3]:
            d_path = os.path.join(root, d)
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(d_path)[:5]):
                has_images = True
                break
        if not has_images:
            continue

        if 'train' in folder_name and not train_dir:
            train_dir = root
        elif 'valid' in folder_name and not val_dir:
            val_dir = root
    return train_dir, val_dir


def get_num_classes(directory):
    """Cuenta número de clases."""
    return len([d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))])


def build_custom_cnn(num_classes):
    """Construye una CNN personalizada para clasificación de enfermedades."""
    model = Sequential(name="Custom_PlantDisease_CNN")

    # Bloque 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Bloque 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Bloque 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Bloque 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Clasificador
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def create_generators(train_dir, val_dir):
    """Crea generadores de datos con augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=RANDOM_SEED
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=RANDOM_SEED
    )

    return train_gen, val_gen


def plot_training_history(history, output_path):
    """Grafica las curvas de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CNN Propia - Curvas de Entrenamiento', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gráficas guardadas: {output_path}")


print("=" * 70)
print("  FASE 3: CNN PROPIA - DISEÑO Y ENTRENAMIENTO")
print("  Sistema de Detección de Enfermedades en Cultivos")
print("=" * 70)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_dir, val_dir = find_data_dirs(DATASET_DIR)

num_classes = get_num_classes(train_dir)
print(f"\nNúmero de clases: {num_classes}")

print(f"\nCreando generadores de datos...")
train_gen, val_gen = create_generators(train_dir, val_dir)
print(f"  Train: {train_gen.samples} imágenes")
print(f"  Validation: {val_gen.samples} imágenes")

print(f"\nConstruyendo modelo CNN...")
model = build_custom_cnn(num_classes)

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nArquitectura del modelo:")
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODELS_DIR, 'cnn_custom_best.keras'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join(LOGS_DIR, 'cnn_custom'),
        histogram_freq=1
    )
]

print(f"\nIniciando entrenamiento...")
print(f"  Épocas máximas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")

start_time = time.time()

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTiempo de entrenamiento: {training_time/60:.1f} minutos")

model_path = os.path.join(MODELS_DIR, 'cnn_custom_final.keras')
model.save(model_path)
print(f"Modelo guardado: {model_path}")

plot_path = os.path.join(REPORTS_DIR, "cnn_custom_training.png")
plot_training_history(history, plot_path)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"\n{'='*70}")
print("RESULTADOS DEL ENTRENAMIENTO - CNN PROPIA")
print(f"{'='*70}")
print(f"  Épocas entrenadas: {len(history.history['accuracy'])}")
print(f"  Mejor época: {best_epoch}")
print(f"  Train Accuracy final: {final_train_acc:.4f}")
print(f"  Val Accuracy final: {final_val_acc:.4f}")
print(f"  Mejor Val Accuracy: {best_val_acc:.4f}")
print(f"  Train Loss final: {final_train_loss:.4f}")
print(f"  Val Loss final: {final_val_loss:.4f}")
print(f"  Tiempo total: {training_time/60:.1f} min")

gap = final_train_acc - final_val_acc
if gap > 0.1:
    print(f"\nALERTA: Posible overfitting (gap train-val: {gap:.4f})")
    print("  Se recomienda aumentar dropout o reducir complejidad.")
else:
    print(f"\nEl modelo no presenta overfitting significativo (gap: {gap:.4f})")

results = {
    "model": "Custom CNN",
    "num_classes": num_classes,
    "epochs_trained": len(history.history['accuracy']),
    "best_epoch": best_epoch,
    "final_train_accuracy": float(final_train_acc),
    "final_val_accuracy": float(final_val_acc),
    "best_val_accuracy": float(best_val_acc),
    "final_train_loss": float(final_train_loss),
    "final_val_loss": float(final_val_loss),
    "training_time_minutes": float(training_time / 60),
    "overfitting_gap": float(gap),
    "total_params": model.count_params()
}

results_path = os.path.join(REPORTS_DIR, "cnn_custom_results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResultados guardados: {results_path}")

print(f"\nEntrenamiento CNN propia completado exitosamente.")
