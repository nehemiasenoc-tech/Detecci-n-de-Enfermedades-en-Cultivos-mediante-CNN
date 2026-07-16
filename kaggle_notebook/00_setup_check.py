"""
Celda 0: Verificación de entorno en Kaggle Notebook
Sistema Inteligente para la Detección de Enfermedades en Cultivos mediante CNN

Antes de correr esta celda:
1. Add Input -> buscar "New Plant Diseases Dataset" -> agregar
2. Settings -> Accelerator -> GPU T4 x2 (o P100)
"""
import os
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

print("\nContenido de /kaggle/input:")
for item in os.listdir("/kaggle/input"):
    print(" -", item)

print("\nContenido de /kaggle/working (salida, aquí quedan modelos y reportes):")
print(os.listdir("/kaggle/working"))
