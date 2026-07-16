"""
Pipeline Principal - Ejecuta todas las fases del proyecto
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Uso: python 00_pipeline_completo.py
"""
import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    ("01_download_dataset.py", "Descarga del Dataset"),
    ("02_exploracion_dataset.py", "Exploración del Dataset (EDA)"),
    ("03_preprocesamiento.py", "Preprocesamiento y Data Augmentation"),
    ("04_entrenar_cnn.py", "Entrenamiento CNN Propia"),
    ("05_transfer_learning.py", "Transfer Learning (ResNet50)"),
    ("06_evaluacion.py", "Evaluación Completa"),
]


def run_phase(script_name, description, phase_num):
    """Ejecuta una fase del pipeline."""
    print(f"\n{'#'*70}")
    print(f"  FASE {phase_num}: {description}")
    print(f"  Script: {script_name}")
    print(f"{'#'*70}\n")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script no encontrado: {script_path}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        capture_output=False
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ Fase {phase_num} completada ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"\n✗ Fase {phase_num} falló con código: {result.returncode}")
        return False


def main():
    print("=" * 70)
    print("  PIPELINE COMPLETO")
    print("  Sistema Inteligente para la Detección de")
    print("  Enfermedades en Cultivos mediante CNN")
    print("=" * 70)
    print(f"\nDirectorio base: {BASE_DIR}")
    print(f"Script actual: {os.path.abspath(__file__)}")

    total_start = time.time()
    results = {}

    for i, (script, desc) in enumerate(PHASES, 1):
        success = run_phase(script, desc, i)
        results[f"Phase {i}"] = {"script": script, "description": desc, "success": success}
        if not success:
            print(f"\nDeteniendo pipeline en fase {i}.")
            break

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print("  RESUMEN DEL PIPELINE")
    print(f"{'='*70}")

    for phase, info in results.items():
        status = "✓" if info["success"] else "✗"
        print(f"  {status} {phase}: {info['description']}")

    print(f"\nTiempo total: {total_time/60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
