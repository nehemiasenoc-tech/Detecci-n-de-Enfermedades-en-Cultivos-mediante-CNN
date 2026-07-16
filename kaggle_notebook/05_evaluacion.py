"""
Celda 5 - Fase 5: Evaluación Completa del Modelo
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Métricas: Accuracy, Precision, Recall, F1-Score, Matriz de Confusión, Curvas
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = "/kaggle/input"
MODELS_DIR = "/kaggle/working/models"
REPORTS_DIR = "/kaggle/working/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42


def find_data_dirs(dataset_dir):
    """Encuentra carpetas train/valid/test: contienen subcarpetas de clases con imágenes."""
    train_dir = val_dir = test_dir = None
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


def get_class_names(directory):
    """Obtiene nombres de las clases."""
    return sorted([d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))])


def parse_class_name(folder_name):
    """Extrae cultivo y enfermedad."""
    parts = folder_name.replace("___", "|").replace("_", " ").split("|", 1)
    crop = parts[0].strip()
    disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    return crop, disease


def plot_confusion_matrix(cm, class_names, output_path, title="Matriz de Confusión"):
    """Grafica la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(18, 15))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=False, yticklabels=False, ax=ax)

    ax.set_xlabel('Predicho', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Matriz de confusión guardada: {output_path}")

    fig2, ax2 = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicho', fontsize=10)
    ax2.set_ylabel('Real', fontsize=10)
    ax2.set_title(f'{title} (Valores Absolutos)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    detail_path = output_path.replace('.png', '_detailed.png')
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Matriz detallada guardada: {detail_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Grafica métricas por clase."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)

    metrics_to_plot = ['precision', 'recall', 'f1-score']
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Métricas por Clase', fontsize=14, fontweight='bold')

    short_names = [parse_class_name(c)[1][:20] for c in class_names]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [report.get(c, {}).get(metric, 0) for c in class_names]
        colors = plt.cm.RdYlGn(np.array(values))
        ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(short_names, fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_title(metric.capitalize())
        ax.axvline(x=np.mean(values), color='red', linestyle='--',
                   label=f'Promedio: {np.mean(values):.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Métricas por clase guardadas: {output_path}")

    return report


def plot_roc_curves(y_true_bin, y_pred_proba, class_names, output_path):
    """Grafica curvas ROC para cada clase."""
    n_classes = y_true_bin.shape[1]
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        short_name = parse_class_name(class_names[i])[1][:25]
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                label=f'{short_name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Curvas ROC - One vs Rest', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Curvas ROC guardadas: {output_path}")


def plot_top_confusions(cm, class_names, output_path, top_n=10):
    """Muestra las confusiones más frecuentes."""
    np.fill_diagonal(cm, 0)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i][j] > 0:
                crop_i, disease_i = parse_class_name(class_names[i])
                crop_j, disease_j = parse_class_name(class_names[j])
                confusions.append({
                    'real': disease_i,
                    'predicho': disease_j,
                    'veces': int(cm[i][j])
                })

    confusions.sort(key=lambda x: x['veces'], reverse=True)

    print(f"\nTop {top_n} confusiones más frecuentes:")
    for c in confusions[:top_n]:
        print(f"  Real: {c['real'][:30]:30s} -> Predicho: {c['predicho'][:30]:30s} ({c['veces']} veces)")

    return confusions[:top_n]


print("=" * 70)
print("  FASE 5: EVALUACIÓN COMPLETA DEL MODELO")
print("  Sistema de Detección de Enfermedades en Cultivos")
print("=" * 70)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_dir, val_dir, test_dir = find_data_dirs(DATASET_DIR)

class_names = get_class_names(train_dir)
num_classes = len(class_names)
print(f"\nNúmero de clases: {num_classes}")

eval_dir = test_dir if test_dir else val_dir
print(f"Directorio de evaluación: {eval_dir}")

datagen = ImageDataGenerator(rescale=1.0 / 255.0)
eval_gen = datagen.flow_from_directory(
    eval_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False, seed=RANDOM_SEED
)

models_to_evaluate = []
cnn_path = os.path.join(MODELS_DIR, 'cnn_custom_best.keras')
resnet_path = os.path.join(MODELS_DIR, 'resnet50_best.keras')

if os.path.exists(cnn_path):
    models_to_evaluate.append(("CNN Propia", cnn_path))
if os.path.exists(resnet_path):
    models_to_evaluate.append(("ResNet50", resnet_path))

if not models_to_evaluate:
    print("ERROR: No se encontraron modelos entrenados.")
    print("Ejecuta primero las celdas de entrenamiento.")
else:
    all_results = {}

    for model_name, model_path in models_to_evaluate:
        print(f"\n{'='*70}")
        print(f"  EVALUANDO: {model_name}")
        print(f"{'='*70}")

        model = load_model(model_path)

        y_pred_proba = model.predict(eval_gen, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = eval_gen.classes

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"\n--- Métricas Globales ---")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision (weighted): {precision:.4f}")
        print(f"  Recall (weighted):    {recall:.4f}")
        print(f"  F1-Score (weighted):  {f1:.4f}")
        print(f"  Precision (macro):    {precision_macro:.4f}")
        print(f"  Recall (macro):       {recall_macro:.4f}")
        print(f"  F1-Score (macro):     {f1_macro:.4f}")

        print(f"\n--- Reporte de Clasificación ---")
        short_class_names = [parse_class_name(c)[1][:30] for c in class_names]
        report_text = classification_report(y_true, y_pred,
                                             target_names=short_class_names,
                                             zero_division=0)
        print(report_text)

        cm = confusion_matrix(y_true, y_pred)

        model_slug = model_name.lower().replace(' ', '_')

        cm_path = os.path.join(REPORTS_DIR, f"{model_slug}_confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, cm_path,
                             title=f'{model_name} - Matriz de Confusión')

        metrics_path = os.path.join(REPORTS_DIR, f"{model_slug}_per_class_metrics.png")
        report_dict = plot_per_class_metrics(y_true, y_pred, class_names, metrics_path)

        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        if num_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        roc_path = os.path.join(REPORTS_DIR, f"{model_slug}_roc_curves.png")
        plot_roc_curves(y_true_bin, y_pred_proba, class_names, roc_path)

        top_confusions = plot_top_confusions(cm, class_names, None)

        metrics_detail = {}
        for i, cls in enumerate(class_names):
            crop, disease = parse_class_name(cls)
            cls_report = report_dict.get(short_class_names[i], {})
            metrics_detail[cls] = {
                "cultivo": crop,
                "enfermedad": disease,
                "precision": cls_report.get('precision', 0),
                "recall": cls_report.get('recall', 0),
                "f1-score": cls_report.get('f1-score', 0),
                "support": cls_report.get('support', 0)
            }

        results = {
            "model_name": model_name,
            "evaluated_on": "test" if test_dir else "validation",
            "num_classes": num_classes,
            "total_images": len(y_true),
            "metrics": {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro)
            },
            "per_class_metrics": metrics_detail,
            "top_confusions": top_confusions
        }

        results_path = os.path.join(REPORTS_DIR, f"{model_slug}_evaluation.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResultados guardados: {results_path}")

        all_results[model_name] = results

    if len(models_to_evaluate) > 1:
        print(f"\n{'='*70}")
        print("  COMPARACIÓN FINAL DE MODELOS")
        print(f"{'='*70}")

        comparison_path = os.path.join(REPORTS_DIR, "final_model_comparison.json")
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Comparación Final de Modelos', fontsize=14, fontweight='bold')

        ax1 = axes[0]
        model_names = list(all_results.keys())
        metrics_names = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        x = np.arange(len(metrics_names))
        width = 0.35
        for i, mname in enumerate(model_names):
            values = [all_results[mname]['metrics'][mn] for mn in metrics_names]
            ax1.bar(x + i * width, values, width, label=mname)
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        ax1.set_ylabel('Score')
        ax1.set_title('Métricas Comparativas')
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        best_model = max(all_results.keys(),
                        key=lambda k: all_results[k]['metrics']['accuracy'])
        best_metrics = all_results[best_model]['metrics']
        ax2.bar(best_metrics.keys(), best_metrics.values(),
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(best_metrics))))
        ax2.set_title(f'Mejor Modelo: {best_model}')
        ax2.set_ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        comparison_plot = os.path.join(REPORTS_DIR, "final_comparison.png")
        plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Gráfica comparativa guardada: {comparison_plot}")

    print(f"\nEvaluación completada exitosamente.")
