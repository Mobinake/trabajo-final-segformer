# -*- coding: utf-8 -*-
"""
Filtrado secundario de teselas: elimina las que contienen bordes negros.

Mueve a carpeta `descartadas/` cualquier tesela con al menos un píxel
RGB [0, 0, 0] absoluto. Usa PIL (no OpenCV) para forzar conversión a RGB
y evitar ambigüedad BGR. Soporta múltiples formatos.

Uso:
    Editar `input_dir` y ejecutar:
        python remove_black_tiles.py

Autor: Mobin Enrique Akhtar Khavari Escobar
Proyecto: TFG — Mapeo Automático de Caña de Azúcar en Guairá
"""

import os
import shutil
import numpy as np
from PIL import Image

# ============================================
# CONFIGURACIÓN
# ============================================
input_dir = '/ruta/a/teselas'

# Carpeta de seguridad (cuarentena) para las descartadas
quarantine_dir = os.path.join(input_dir, 'descartadas')
os.makedirs(quarantine_dir, exist_ok=True)

archivos_procesados = 0
archivos_removidos = 0

print(f"Escaneando archivos con píxeles negros absolutos [0,0,0] en: {os.path.abspath(input_dir)}\n")

for root, _, files in os.walk(input_dir):
    # Evitar revisar la carpeta de cuarentena en bucle
    if 'descartadas' in root:
        continue

    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            file_path = os.path.join(root, file_name)

            try:
                # Convertir a RGB para asegurar 3 canales
                img = Image.open(file_path).convert('RGB')
                arr = np.array(img)

                # Buscar al menos UN píxel [0, 0, 0]
                has_black = np.any(np.all(arr == [0, 0, 0], axis=-1))

                if has_black:
                    img.close()

                    dest_path = os.path.join(quarantine_dir, file_name)

                    # Anti-colisión: si ya existe un archivo con el mismo nombre
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(file_name)
                        dest_path = os.path.join(quarantine_dir, f"{base}_del_{archivos_removidos}{ext}")

                    shutil.move(file_path, dest_path)
                    print(f"  Retirada: {file_name} (contenía bordes/zonas vacías)")
                    archivos_removidos += 1
                else:
                    img.close()
                    archivos_procesados += 1

            except Exception as e:
                print(f"  Error leyendo {file_name}: {e}")

print(f"\n==========================================")
print(f"ANÁLISIS FINALIZADO")
print(f"  Imágenes aprobadas (sin cambios): {archivos_procesados}")
print(f"  Imágenes movidas a descartadas:   {archivos_removidos}")
print(f"  Cuarentena: {os.path.abspath(quarantine_dir)}")
print(f"==========================================")
