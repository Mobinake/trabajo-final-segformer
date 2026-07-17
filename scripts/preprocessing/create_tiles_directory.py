# -*- coding: utf-8 -*-
"""
Generación de teselas (tiles) a partir de imágenes JPG con sliding window.

Parámetros fijos del pipeline:
    TILE_SIZE = 128 px
    OVERLAP = 8 px (stride = 120)
    BLACK_THRESHOLD = 10 (escala de grises)
    MAX_BLACK_PERCENT = 0.75 (descarta teselas con >75% píxeles negros)

Crea un subdirectorio por imagen para trazabilidad.

Uso (interactivo):
    python create_tiles_directory.py
    → Ingresa ruta de entrada y salida cuando se solicite.

Autor: Mobin Enrique Akhtar Khavari Escobar
Proyecto: TFG — Mapeo Automático de Caña de Azúcar en Guairá
"""

import cv2
import numpy as np
import os
import sys

# --- CONFIGURACIÓN FIJA DE PROCESAMIENTO ---
TILE_SIZE = 128
OVERLAP = 8
BLACK_THRESHOLD = 10
MAX_BLACK_PERCENT = 0.75


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg"))


def process_image(input_path, target_root):
    if not os.path.isfile(input_path):
        print(f"Error: No se encontró el archivo en {input_path}")
        return 0, 0

    base_filename_with_ext = os.path.basename(input_path)
    base_name = os.path.splitext(base_filename_with_ext)[0]

    output_folder = os.path.join(target_root, base_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Directorio creado: {output_folder}")
    else:
        print(f"El directorio ya existe, los archivos se guardarán en: {output_folder}")

    print(f"\nCargando imagen: {base_filename_with_ext}...")
    img = cv2.imread(input_path)

    if img is None:
        print("Error: No se pudo cargar la imagen. Verifica que sea un JPEG válido.")
        return 0, 0

    h, w, _ = img.shape
    stride = TILE_SIZE - OVERLAP

    count_saved = 0
    count_discarded = 0

    print("Iniciando recorte...")

    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]

            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray_tile <= BLACK_THRESHOLD)
            total_pixels = TILE_SIZE * TILE_SIZE
            black_ratio = black_pixels / total_pixels

            if black_ratio <= MAX_BLACK_PERCENT:
                filename = f"{base_name}_x{x}_y{y}.jpg"
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, tile, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                count_saved += 1
            else:
                count_discarded += 1

    print(f"Tiles guardados: {count_saved}, descartados: {count_discarded}")
    return count_saved, count_discarded


def slice_images():
    input_dir = input("Indica la ruta del directorio de entrada: ").strip()
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio de entrada no existe o no es válido: {input_dir}")
        return

    target_root = input("Indica la ruta del directorio de salida: ").strip()
    if not os.path.exists(target_root):
        create = input("El directorio de salida no existe. ¿Crearlo? (s/n): ").strip().lower()
        if create == "s":
            os.makedirs(target_root)
            print(f"Directorio creado: {target_root}")
        else:
            print("Proceso cancelado.")
            return

    files = [f for f in os.listdir(input_dir) if is_image_file(f)]
    if not files:
        print("No se encontraron imágenes .jpeg/.jpg en el directorio de entrada.")
        return

    print(f"\nSe encontraron {len(files)} imágenes. Procesando...\n")

    total_saved = 0
    total_discarded = 0

    for f in sorted(files):
        input_path = os.path.join(input_dir, f)
        saved, discarded = process_image(input_path, target_root)
        total_saved += saved
        total_discarded += discarded
        print()

    print("========================================")
    print("PROCESO COMPLETADO")
    print(f"Total tiles guardados:     {total_saved}")
    print(f"Total tiles descartados:   {total_discarded}")
    print(f"Imágenes procesadas:       {len(files)}")
    print("========================================")


if __name__ == "__main__":
    slice_images()
