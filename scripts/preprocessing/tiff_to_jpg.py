# -*- coding: utf-8 -*-
"""
Conversión batch TIF 8-bit → JPG para pipeline SegFormer.

Lee las primeras 3 bandas (R, G, B) y guarda como JPG de alta calidad.
Preserva la estructura de subcarpetas.

Uso:
    Editar INPUT_FOLDER y OUTPUT_FOLDER y ejecutar:
        python tiff_to_jpg.py

Autor: Mobin Enrique Akhtar Khavari Escobar
Proyecto: TFG — Mapeo Automático de Caña de Azúcar en Guairá
"""

import os
from pathlib import Path
import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio no está instalado.")
    print("Instalalo con: pip install rasterio numpy pillow")
    raise

try:
    from PIL import Image
except ImportError:
    print("ERROR: pillow no está instalado.")
    print("Instalalo con: pip install pillow")
    raise


# ============================================
# CONFIGURACIÓN — EDITÁ ESTAS RUTAS
# ============================================
INPUT_FOLDER = "/ruta/a/imagenes/8bit"
OUTPUT_FOLDER = "/ruta/a/imagenes/jpg"

# Bandas a leer para el JPG (1-indexed)
# El script convertir_s2_a8bit.py guarda R, G, B en bandas 1, 2, 3.
BANDAS_JPG = [1, 2, 3]

# Calidad JPG (1-100).
# 100 = la más alta posible, sin chroma subsampling (4:4:4).
JPG_QUALITY = 100


def tif_a_jpg(path_in: Path, path_out: Path):
    """Convierte un TIF multibanda a JPG de 3 canales."""

    with rasterio.open(path_in) as src:
        n_bandas = src.count

        if n_bandas >= 3:
            rgb = src.read(BANDAS_JPG[:3])
            rgb = rgb.transpose(1, 2, 0)
        else:
            gray = src.read(1)
            rgb = np.stack([gray, gray, gray], axis=-1)

    img = Image.fromarray(rgb)
    img.save(path_out, quality=JPG_QUALITY)


def main():
    in_dir = Path(INPUT_FOLDER)
    out_dir = Path(OUTPUT_FOLDER)

    if not in_dir.exists():
        print(f"ERROR: no existe la carpeta de entrada: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(in_dir.rglob("*.tif")) + sorted(in_dir.rglob("*.tiff"))

    print(f"\n=== Conversión TIF 8-bit → JPG ===")
    print(f"Entrada:    {in_dir}")
    print(f"Salida:     {out_dir}")
    print(f"Calidad:    {JPG_QUALITY}")
    print(f"Encontrados: {len(tifs)} TIFs\n")

    if not tifs:
        print("No se encontraron .tif. Revisá la ruta INPUT_FOLDER.")
        return

    ok, fail = 0, 0
    for tif in tifs:
        rel = tif.relative_to(in_dir)
        out_path = out_dir / rel.parent / f"{tif.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            tif_a_jpg(tif, out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"  OK: {tif.name}  →  {out_path.name}  ({size_kb:.0f} KB)")
            ok += 1
        except Exception as e:
            print(f"  ERROR en {tif.name}: {e}")
            fail += 1

    print(f"\n=== Listo: {ok} OK, {fail} con error ===")
    print(f"Salida en: {out_dir}")


if __name__ == "__main__":
    main()
