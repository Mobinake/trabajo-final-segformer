# -*- coding: utf-8 -*-
"""
Obtener imágenes satelitales Sentinel-2 L2A vía Google Earth Engine.

Diseñado para ejecutarse en Google Colab. Descarga imágenes RGB (B2, B3, B4)
de la colección COPERNICUS/S2_SR_HARMONIZED para un distrito específico de Guairá,
Paraguay. Aplica filtro de nubosidad a nivel de metadatos y máscara de nubes
a nivel de píxel con la banda SCL.

Uso:
    1. Subir el JSON de service account a /content/ en Colab.
    2. Cambiar la variable `ubicacion` por el distrito deseado.
    3. Ejecutar celda por celda.

Autor: Mobin Enrique Akhtar Khavari Escobar
Proyecto: TFG — Mapeo Automático de Caña de Azúcar en Guairá
"""

# ============================================================
# Paso 0: Instalación y autenticación
# ============================================================
!pip install -q earthengine-api geemap

import ee
import geemap
import os
import time
import shutil
from google.oauth2 import service_account

# Service account JSON (subir a /content/ antes de ejecutar)
KEY = '/content/tesismobin-d1c7c231981d.json'
creds = service_account.Credentials.from_service_account_file(
    KEY, scopes=['https://www.googleapis.com/auth/earthengine']
)
ee.Initialize(creds, project='tesismobin')

# Test de conexión
print(ee.Image("USGS/SRTMGL1_003").getInfo()['id'])

# ============================================================
# Paso 1: Definir AOI (Área de Interés)
# ============================================================
# Colección FAO/GAUL nivel 2 (distritos/municipios)
# Cambiar `ubicacion` por el distrito a descargar:
#   - 'Tebicuary'      (entrenamiento)
#   - 'Itape'           (entrenamiento)
#   - 'Cnel Martinez'   (entrenamiento)
#   - 'San Salvador'    (validación externa, opcional)

ubicacion = 'Tebicuary'

countries = ee.FeatureCollection("FAO/GAUL/2015/level2")
guaira_aoi = countries.filter(ee.Filter.eq('ADM0_NAME', 'Paraguay')) \
                      .filter(ee.Filter.eq('ADM2_NAME', ubicacion))
aoi = guaira_aoi.geometry()

# Visualización (opcional, en Colab)
Map = geemap.Map()
Map.centerObject(guaira_aoi, 12)
Map.addLayer(guaira_aoi, {'color': 'green'}, 'Área de Interés')
Map

# ============================================================
# Paso 2: Cargar y filtrar colección Sentinel-2
# ============================================================
# Colección: COPERNICUS/S2_SR_HARMONIZED (L2A, reflectancia de superficie)
# Filtros:
#   - Espacial: filterBounds(aoi)
#   - Temporal: filterDate (ventana configurable)
#   - Nubosidad: CLOUDY_PIXEL_PERCENTAGE < 20 (descarta la escena entera)

fecha_inicio = '2024-01-01'
fecha_fin = '2026-12-30'

s2_coleccion = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi)
                .filterDate(fecha_inicio, fecha_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

print(f"Imágenes encontradas (pre-máscara): {s2_coleccion.size().getInfo()}")

# ============================================================
# Paso 3: Máscara de nubes a nivel de píxel (banda SCL)
# ============================================================
# SCL (Scene Classification Layer) de Sen2Cor:
#   3 = sombra de nubes, 8 = nube media prob., 9 = nube alta prob.,
#   10 = cirrus, 11 = nieve/hielo. Se vuelven transparentes (NoData).

def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(mask)

s2_procesada = s2_coleccion.map(mask_s2_clouds)

# ============================================================
# Paso 4: Verificar imágenes a exportar
# ============================================================
lista_info = s2_procesada.aggregate_array('system:index').getInfo()
fechas_info = s2_procesada.aggregate_array('system:time_start').getInfo()

print(f"Imágenes listas para descargar: {len(lista_info)}")
for i in range(min(5, len(lista_info))):
    fecha = ee.Date(fechas_info[i]).format('yyyy-MM-dd').getInfo()
    print(f"  ID: {lista_info[i]} | Fecha: {fecha}")

# ============================================================
# Paso 5: Descarga (GeoTIFF 16-bit, EPSG:32721)
# ============================================================
# Bandas: B2 (azul), B3 (verde), B4 (rojo) → RGB visibles, 10 m
# CRS: UTM 21S (EPSG:32721) — proyección target para Guairá
# Formato: GeoTIFF 16-bit, 3 bandas en un solo archivo (file_per_band=False)

OUT_DIR = '/content/S2_images'
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ['B2', 'B3', 'B4']
image_list = s2_procesada.toList(s2_procesada.size())
num_images = s2_procesada.size().getInfo()

print(f"Descargando {num_images} imágenes de {ubicacion}...")
fallos = []

for i in range(num_images):
    image = ee.Image(image_list.get(i))
    date = ee.Date(image.get('system:time_start')).format('yyyy_MM_dd').getInfo()
    img_id = image.get('system:index').getInfo()
    file_name = f'Sentinel2_{ubicacion.replace(" ", "")}_{date}_{img_id}'

    image_to_export = image.select(BANDS).clip(aoi)

    try:
        geemap.ee_export_image(
            ee_object=image_to_export,
            filename=f'{OUT_DIR}/{file_name}.tif',
            scale=10,
            crs='EPSG:32721',
            region=aoi,
            file_per_band=False,
            timeout=600
        )
        print(f'[{i+1}/{num_images}] OK  {file_name}')
    except Exception as e:
        print(f'[{i+1}/{num_images}] FAIL {file_name}: {e}')
        fallos.append((file_name, str(e)))

    time.sleep(2)  # cortesía al servidor

print(f"\n--- Resumen ---")
print(f"Total: {num_images} | Exitosas: {num_images - len(fallos)} | Fallidas: {len(fallos)}")
if fallos:
    print("Fallos:", [f[0] for f in fallos])

# ============================================================
# Paso 6: Zip para descargar todo de una
# ============================================================
zip_path = f'/content/S2_{ubicacion.replace(" ", "")}_full.zip'
shutil.make_archive(zip_path.replace('.zip', ''), 'zip', OUT_DIR)
print(f"\nZip listo: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
