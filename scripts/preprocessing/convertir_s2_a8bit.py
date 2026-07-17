# -*- coding: utf-8 -*-
"""
Conversión batch Sentinel-2 L2A: 16-bit → 8-bit

Estiramiento POR BANDA con percentiles p2-p98 automáticos.
Output: GeoTIFF uint8, comprimido LZW, listo para conversión a JPG.

Uso:
    Editar INPUT_FOLDER y OUTPUT_FOLDER y ejecutar:
        python convertir_s2_a8bit.py

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
    print("Instalalo con: pip install rasterio numpy")
    raise


# ============================================
# CONFIGURACIÓN — EDITÁ ESTAS 2 RUTAS
# ============================================
INPUT_FOLDER = "/ruta/a/imagenes/16bit"
OUTPUT_FOLDER = "/ruta/a/imagenes/8bit"

# Percentiles para el estiramiento (por banda)
# 2 y 98 = estándar, ignora outliers brillantes/oscuros
# Subí P_MIN (ej. 5) si te queda lavada en zonas oscuras
# Bajá P_MAX (ej. 95) si te queda lavada en zonas brillantes
P_MIN, P_MAX = 2, 98

# Bandas a leer y orden de salida:
#   None        = todas las bandas (multiespectral)
#   [3, 2, 1]   = RGB natural para Sentinel-2 (lee B4, B3, B2
#                 y las guarda en ese orden → R, G, B en
#                 el output). RECOMENDADO para el TFG.
#   [1, 2, 3]   = solo si tu TIF YA está en orden R, G, B
BANDAS = [3, 2, 1]


def convertir(path_in: Path, path_out: Path):
    """Convierte un TIF de 16 bits a 8 bits con estiramiento
    percentílico INDEPENDIENTE por banda."""

    with rasterio.open(path_in) as src:
        if BANDAS is None:
            datos = src.read()
            n_bandas = src.count
        else:
            datos = src.read(BANDAS)
            n_bandas = len(BANDAS)

        meta = src.meta.copy()
        crs = src.crs
        transform = src.transform
        nodata = src.nodata

    # ----------------------------------------------------------
    # CALCULAR PERCENTILES POR BANDA (clave del fix)
    # ----------------------------------------------------------
    p2_por_banda = []
    p98_por_banda = []

    for i in range(n_bandas):
        b = datos[i]
        p2 = float(np.percentile(b, P_MIN))
        p98 = float(np.percentile(b, P_MAX))

        if p98 <= p2:
            raise ValueError(
                f"Banda {i+1}: p{P_MAX} ({p98:.0f}) <= "
                f"p{P_MIN} ({p2:.0f}). Imagen constante o vacía."
            )

        p2_por_banda.append(p2)
        p98_por_banda.append(p98)

    print(f"  Percentiles por banda:")
    for i in range(n_bandas):
        print(f"    Banda {i+1}: p{P_MIN}={p2_por_banda[i]:.0f}  "
              f"p{P_MAX}={p98_por_banda[i]:.0f}")

    # ----------------------------------------------------------
    # ESTIRAR POR BANDA (cada una usa su propio rango)
    # ----------------------------------------------------------
    datos_8 = np.zeros(datos.shape, dtype=np.uint8)

    for i in range(n_bandas):
        p2 = p2_por_banda[i]
        p98 = p98_por_banda[i]
        rango = p98 - p2

        datos_8[i] = np.clip(
            (datos[i] - p2) / rango * 255.0,
            0, 255
        ).astype(np.uint8)

    # Tratar NoData
    if nodata is not None:
        if BANDAS is None:
            mask_nodata = (datos == nodata)
        else:
            mask_nodata = (datos[0] == nodata)
        for i in range(n_bandas):
            datos_8[i][mask_nodata] = 0

    # Metadata de salida
    out_meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0 if nodata is not None else None,
        'width': meta['width'],
        'height': meta['height'],
        'count': n_bandas,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }

    if n_bandas in (3, 4):
        out_meta['photometric'] = 'rgb'

    with rasterio.open(path_out, 'w', **out_meta) as dst:
        dst.write(datos_8)

    return p2_por_banda, p98_por_banda


def main():
    in_dir = Path(INPUT_FOLDER)
    out_dir = Path(OUTPUT_FOLDER)

    if not in_dir.exists():
        print(f"ERROR: no existe la carpeta de entrada: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(in_dir.rglob("*.tif")) + sorted(in_dir.rglob("*.tiff"))
    tifs = [t for t in tifs if "_8bit" not in t.stem]

    print(f"\n=== Conversión 16-bit → 8-bit (per-band stretch) ===")
    print(f"Entrada:  {in_dir}")
    print(f"Salida:   {out_dir}")
    print(f"BANDAS:   {BANDAS if BANDAS else 'todas'}")
    print(f"Encontradas: {len(tifs)} imágenes para procesar\n")

    if not tifs:
        print("No se encontraron .tif. Revisá la ruta INPUT_FOLDER.")
        return

    ok, fail = 0, 0
    for tif in tifs:
        rel = tif.relative_to(in_dir)
        out_path = out_dir / rel.parent / f"{tif.stem}_8bit.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{tif.name}]")
        try:
            p2_list, p98_list = convertir(tif, out_path)
            print(f"  → {out_path.name}  OK\n")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}\n")
            fail += 1

    print(f"=== Listo: {ok} OK, {fail} con error ===")


if __name__ == "__main__":
    main()
