"""
Celda 1 - Fase 1: Exploración del Dataset (EDA)
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Análisis: Clases, cantidad de imágenes, balance, resolución, cultivos representados
"""
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = "/kaggle/input"
REPORTS_DIR = "/kaggle/working/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def find_image_dirs(dataset_dir):
    """Busca carpetas train/valid/test: contienen subcarpetas de clases con imágenes."""
    train_dir = None
    val_dir = None
    test_dir = None

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
        elif 'test' in folder_name and not test_dir:
            test_dir = root

    return train_dir, val_dir, test_dir


def count_classes_and_images(directory):
    """Cuenta clases e imágenes en un directorio."""
    classes = {}
    if directory is None:
        return classes
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            classes[folder] = len(images)
    return classes


def analyze_resolutions(directory, sample_size=100):
    """Analiza resoluciones de imágenes."""
    resolutions = []
    if directory is None:
        return resolutions

    all_classes = os.listdir(directory)
    images_per_class = max(1, sample_size // len(all_classes)) if all_classes else 0

    for cls in all_classes:
        cls_path = os.path.join(directory, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        sampled = imgs[:images_per_class]
        for img_name in sampled:
            try:
                img = Image.open(os.path.join(cls_path, img_name))
                resolutions.append(img.size)
                img.close()
            except Exception:
                pass
    return resolutions


def parse_class_name(folder_name):
    """Extrae cultivo y enfermedad del nombre de carpeta."""
    name = folder_name.replace("___", " - ").replace("_", " ")
    return name


print("=" * 70)
print("  FASE 1: EXPLORACIÓN DEL DATASET (EDA)")
print("  Sistema de Detección de Enfermedades en Cultivos")
print("=" * 70)

train_dir, val_dir, test_dir = find_image_dirs(DATASET_DIR)

print(f"\nDirectorio de entrenamiento: {train_dir}")
print(f"Directorio de validación:    {val_dir}")
print(f"Directorio de prueba:        {test_dir}")

train_classes = count_classes_and_images(train_dir)
val_classes = count_classes_and_images(val_dir) if val_dir else {}
test_classes = count_classes_and_images(test_dir) if test_dir else {}

print(f"\n{'='*70}")
print("RESUMEN GENERAL")
print(f"{'='*70}")
print(f"Total de clases: {len(train_classes)}")
print(f"Total imágenes de entrenamiento: {sum(train_classes.values()):,}")
print(f"Total imágenes de validación: {sum(val_classes.values()):,}")
if test_classes:
    print(f"Total imágenes de prueba: {sum(test_classes.values()):,}")

total = sum(train_classes.values()) + sum(val_classes.values()) + sum(test_classes.values())
print(f"Total imágenes en dataset: {total:,}")

print(f"\n{'='*70}")
print("CLASES POR CULTIVO Y ENFERMEDAD")
print(f"{'='*70}")

crops = defaultdict(list)
for cls_name in sorted(train_classes.keys()):
    parsed = parse_class_name(cls_name)
    parts = parsed.split(" - ", 1)
    crop = parts[0].strip()
    disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    crops[crop].append((cls_name, disease, train_classes[cls_name]))

for crop, diseases in sorted(crops.items()):
    total_crop = sum(d[2] for d in diseases)
    print(f"\nCultivo: {crop} ({total_crop:,} imágenes de entrenamiento)")
    for cls_name, disease, count in diseases:
        print(f"    - {disease}: {count:,} imágenes")

print(f"\n{'='*70}")
print("BALANCE DE CLASES")
print(f"{'='*70}")

counts = list(train_classes.values())
min_count = min(counts)
max_count = max(counts)
mean_count = np.mean(counts)
std_count = np.std(counts)

print(f"Mínimo de imágenes por clase: {min_count:,}")
print(f"Máximo de imágenes por clase: {max_count:,}")
print(f"Promedio de imágenes por clase: {mean_count:,.1f}")
print(f"Desviación estándar: {std_count:,.1f}")
print(f"Ratio max/min: {max_count/min_count:.2f}x")

if max_count / min_count > 2:
    print("ALERTA: Existe desbalance significativo entre clases.")
    print("  Se recomienda usar class_weight o sobremuestreo.")
else:
    print("Las clases están relativamente balanceadas.")

print(f"\n{'='*70}")
print("ANÁLISIS DE RESOLUCIONES (MUESTRA)")
print(f"{'='*70}")

resolutions = analyze_resolutions(train_dir, sample_size=300)
if resolutions:
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    unique_res = Counter(resolutions)

    print(f"Muestra analizada: {len(resolutions)} imágenes")
    print(f"Ancho mínimo: {min(widths)}px | máximo: {max(widths)}px")
    print(f"Alto mínimo: {min(heights)}px | máximo: {max(heights)}px")
    print(f"Resoluciones únicas: {len(unique_res)}")
    print("\nTop 5 resoluciones más comunes:")
    for res, count in unique_res.most_common(5):
        print(f"  {res[0]}x{res[1]}: {count} imágenes ({count/len(resolutions)*100:.1f}%)")

print(f"\n{'='*70}")
print("GRÁFICOS DEL EDA")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Exploración del Dataset - Plant Diseases', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
class_names_short = [parse_class_name(c)[:20] for c in sorted(train_classes.keys())]
class_counts = [train_classes[c] for c in sorted(train_classes.keys())]
colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
ax1.barh(range(len(class_counts)), class_counts, color=colors)
ax1.set_yticks(range(len(class_counts)))
ax1.set_yticklabels(class_names_short, fontsize=6)
ax1.set_xlabel('Número de Imágenes')
ax1.set_title('Distribución de Clases (Entrenamiento)')
ax1.axvline(x=mean_count, color='red', linestyle='--', label=f'Promedio: {mean_count:.0f}')
ax1.legend()

ax2 = axes[0, 1]
crop_totals = {crop: sum(d[2] for d in diseases) for crop, diseases in crops.items()}
ax2.pie(crop_totals.values(), labels=crop_totals.keys(), autopct='%1.1f%%',
        colors=plt.cm.Paired(np.linspace(0, 1, len(crop_totals))))
ax2.set_title('Distribución por Cultivo (Entrenamiento)')

ax3 = axes[1, 0]
if val_classes:
    sets_data = {
        'Train': sum(train_classes.values()),
        'Validation': sum(val_classes.values()),
    }
    if test_classes:
        sets_data['Test'] = sum(test_classes.values())
    ax3.bar(sets_data.keys(), sets_data.values(),
            color=['#2196F3', '#FF9800', '#4CAF50'][:len(sets_data)])
    ax3.set_ylabel('Número de Imágenes')
    ax3.set_title('Distribución por Conjunto de Datos')
    for i, (k, v) in enumerate(sets_data.items()):
        ax3.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

ax4 = axes[1, 1]
if resolutions:
    ax4.hist2d(widths, heights, bins=30, cmap='YlOrRd')
    ax4.set_xlabel('Ancho (px)')
    ax4.set_ylabel('Alto (px)')
    ax4.set_title('Distribución de Resoluciones')
    plt.colorbar(ax4.collections[0], ax=ax4)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "eda_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Gráfico guardado: {plot_path}")

report = {
    "dataset_summary": {
        "total_classes": len(train_classes),
        "total_images": total,
        "train_images": sum(train_classes.values()),
        "val_images": sum(val_classes.values()),
        "test_images": sum(test_classes.values()) if test_classes else 0,
        "crops": list(crops.keys()),
        "num_crops": len(crops)
    },
    "class_balance": {
        "min_images": min_count,
        "max_images": max_count,
        "mean_images": float(mean_count),
        "std_images": float(std_count),
        "imbalance_ratio": float(max_count / min_count)
    },
    "resolutions": {
        "sample_size": len(resolutions),
        "width_range": [min(widths), max(widths)] if resolutions else [],
        "height_range": [min(heights), max(heights)] if resolutions else []
    }
}

report_path = os.path.join(REPORTS_DIR, "eda_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Reporte guardado: {report_path}")

print(f"\n{'='*70}")
print("CONCLUSIONES DEL EDA")
print(f"{'='*70}")
print(f"1. El dataset contiene {len(train_classes)} clases de enfermedades/salud")
print(f"   distribuidas en {len(crops)} cultivos diferentes.")
print(f"2. Total de imágenes: {total:,}")
print(f"3. {'Hay desbalance de clases.' if max_count/min_count > 2 else 'Las clases están balanceadas.'}")
print(f"4. Se recomienda resize a 224x224 para entrenamiento.")
print(f"5. Data Augmentation es recomendado para las clases con menos imágenes.")
print(f"\nAnálisis exploratorio completado exitosamente.")
