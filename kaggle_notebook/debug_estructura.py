"""
Celda de diagnóstico: imprime la estructura real de /kaggle/input
Correr esto y pegar el resultado si find_data_dirs no encuentra train/valid.
"""
import os

DATASET_DIR = "/kaggle/input"

def print_tree(root, max_depth=5, prefix=""):
    if max_depth < 0:
        return
    try:
        entries = sorted(os.listdir(root))
    except Exception as e:
        print(f"{prefix}[ERROR leyendo {root}: {e}]")
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(root, e))]

    print(f"{prefix}{os.path.basename(root)}/  -> {len(dirs)} carpetas, {len(files)} archivos")

    for d in dirs[:5]:
        print_tree(os.path.join(root, d), max_depth - 1, prefix + "  ")
    if len(dirs) > 5:
        print(f"{prefix}  ... y {len(dirs) - 5} carpetas más")


print_tree(DATASET_DIR, max_depth=6)
