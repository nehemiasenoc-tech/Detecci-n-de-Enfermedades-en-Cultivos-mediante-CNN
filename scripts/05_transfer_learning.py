"""
Fase 4: Transfer Learning con ResNet50
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Comparación con modelo preentrenado ResNet50.
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, BatchNormalization,
                                      GlobalAveragePooling2D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_SEED = 42
LEARNING_RATE = 0.0001


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


def build_resnet50_transfer(num_classes):
    """Construye modelo ResNet50 con Transfer Learning."""
    input_layer = Input(shape=(224, 224, 3))

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=input_layer
    )

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name="ResNet50_PlantDisease")

    return model, base_model


def create_generators(train_dir, val_dir):
    """Crea generadores de datos."""
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
    """Grafica curvas de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('ResNet50 Transfer Learning - Curvas de Entrenamiento',
                 fontsize=14, fontweight='bold')

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
    plt.close()
    print(f"Gráficas guardadas: {output_path}")


def compare_models(cnn_results_path, resnet_results_path, output_path):
    """Compara resultados de CNN propia vs ResNet50."""
    with open(cnn_results_path) as f:
        cnn_res = json.load(f)
    with open(resnet_results_path) as f:
        resnet_res = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparación: CNN Propia vs ResNet50', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    models = ['CNN Propia', 'ResNet50']
    metrics = ['best_val_accuracy', 'final_val_accuracy']
    metric_labels = ['Mejor Val Accuracy', 'Final Val Accuracy']

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, [cnn_res.get(m, 0) for m in metrics],
                    width, label='CNN Propia', color='#2196F3')
    bars2 = ax1.bar(x + width/2, [resnet_res.get(m, 0) for m in metrics],
                    width, label='ResNet50', color='#FF9800')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Comparación de Accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]
    time_data = [cnn_res.get('training_time_minutes', 0),
                 resnet_res.get('training_time_minutes', 0)]
    params_data = [cnn_res.get('total_params', 0) / 1e6,
                   resnet_res.get('total_params', 0) / 1e6]

    ax2_twin = ax2.twinx()
    bars_time = ax2.bar(x - width/2, time_data, width,
                        label='Tiempo (min)', color='#4CAF50')
    bars_params = ax2_twin.bar(x + width/2, params_data, width,
                               label='Parámetros (M)', color='#E91E63')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Tiempo (minutos)')
    ax2_twin.set_ylabel('Parámetros (millones)')
    ax2.set_title('Recursos')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparación guardada: {output_path}")


def main():
    print("=" * 70)
    print("  FASE 4: TRANSFER LEARNING - ResNet50")
    print("  Sistema de Detección de Enfermedades en Cultivos")
    print("=" * 70)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_dir, val_dir = find_data_dirs(DATASET_DIR)
    if not train_dir or not val_dir:
        print("ERROR: Directorios de datos no encontrados.")
        return

    num_classes = get_num_classes(train_dir)
    print(f"\nNúmero de clases: {num_classes}")

    print(f"\nCreando generadores de datos...")
    train_gen, val_gen = create_generators(train_dir, val_dir)
    print(f"  Train: {train_gen.samples} imágenes")
    print(f"  Validation: {val_gen.samples} imágenes")

    print(f"\nConstruyendo modelo ResNet50 con Transfer Learning...")
    model, base_model = build_resnet50_transfer(num_classes)

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params

    print(f"\nResNet50 - Parámetros:")
    print(f"  Total: {total_params:,}")
    print(f"  Entrenables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Congelados: {non_trainable_params:,} ({non_trainable_params/total_params*100:.1f}%)")

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nArquitectura ResNet50 Transfer Learning:")
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'resnet50_best.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(BASE_DIR, 'logs', 'resnet50_transfer'),
            histogram_freq=1
        )
    ]

    print(f"\nIniciando entrenamiento ResNet50...")
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

    model_path = os.path.join(MODELS_DIR, 'resnet50_final.keras')
    model.save(model_path)
    print(f"Modelo guardado: {model_path}")

    plot_path = os.path.join(REPORTS_DIR, "resnet50_training.png")
    plot_training_history(history, plot_path)

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"\n{'='*70}")
    print("RESULTADOS DEL ENTRENAMIENTO - ResNet50")
    print(f"{'='*70}")
    print(f"  Épocas entrenadas: {len(history.history['accuracy'])}")
    print(f"  Mejor época: {best_epoch}")
    print(f"  Train Accuracy final: {final_train_acc:.4f}")
    print(f"  Val Accuracy final: {final_val_acc:.4f}")
    print(f"  Mejor Val Accuracy: {best_val_acc:.4f}")
    print(f"  Train Loss final: {final_train_loss:.4f}")
    print(f"  Val Loss final: {final_val_loss:.4f}")
    print(f"  Tiempo total: {training_time/60:.1f} min")
    print(f"  Parámetros totales: {total_params:,}")

    results = {
        "model": "ResNet50 Transfer Learning",
        "num_classes": num_classes,
        "epochs_trained": len(history.history['accuracy']),
        "best_epoch": best_epoch,
        "final_train_accuracy": float(final_train_acc),
        "final_val_accuracy": float(final_val_acc),
        "best_val_accuracy": float(best_val_acc),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "training_time_minutes": float(training_time / 60),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(non_trainable_params)
    }

    results_path = os.path.join(REPORTS_DIR, "resnet50_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados: {results_path}")

    # Comparación con CNN propia
    cnn_results_path = os.path.join(REPORTS_DIR, "cnn_custom_results.json")
    if os.path.exists(cnn_results_path):
        print(f"\nGenerando comparación CNN vs ResNet50...")
        compare_path = os.path.join(REPORTS_DIR, "model_comparison.png")
        compare_models(cnn_results_path, results_path, compare_path)

    print(f"\nTransfer Learning con ResNet50 completado exitosamente.")


if __name__ == "__main__":
    main()
