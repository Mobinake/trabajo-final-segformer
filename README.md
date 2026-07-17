# Mapeo Automático de Caña de Azúcar en Guairá con SegFormer

Trabajo Final de Grado — Ingeniería Informática, UCNSA Campus Guairá

Segmentación semántica de plantaciones de caña de azúcar utilizando imágenes satelitales Sentinel-2 y el modelo SegFormer (backbone MiT-B3), aplicado a tres distritos del departamento de Guairá, Paraguay.

---

## Tabla de contenidos

- [Resumen](#resumen)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Pipeline completo](#pipeline-completo)
- [Herramientas utilizadas](#herramientas-utilizadas)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Guía de uso](#guía-de-uso)
- [Dataset](#dataset)
- [Hiperparámetros de entrenamiento](#hiperparámetros-de-entrenamiento)
- [Autoría](#autoría)
- [Licencia](#licencia)

---

## Resumen

Paraguay es uno de los principales productores de caña de azúcar de Sudamérica. El departamento de Guairá concentra una gran proporción de la producción nacional, pero no existen herramientas dinámicas que permitan mapear las plantaciones con precisión espacial. El MAG (Ministerio de Agricultura y Ganadería) maneja estadísticas agregadas, no mapas a nivel de parcela.

Este proyecto aplica técnicas de segmentación semántica con deep learning para mapear automáticamente las plantaciones de caña de azúcar a partir de imágenes satelitales Sentinel-2 (RGB visible, 10 m de resolución). El modelo utilizado es **SegFormer con backbone MiT-B3**, preentrenado en ImageNet-1k y fine-tuned en ADE20K, reentrenado sobre un dataset de 513 imágenes etiquetadas manualmente en CVAT.

El alcance geográfico cubre tres distritos de Guairá: **Tebicuary**, **Itapé** y **Coronel Martínez**.

---

## Resultados

La configuración final (run 11, post-grid search) logró:

| Métrica | Valor |
|---|---|
| **mIoU test (sugar_cane)** | **0,91** |
| mIoU val (sugar_cane) | 0,91 |
| Mediana IoU test | 0,94 |
| Desviación estándar test | 0,08 |
| Hipótesis (mIoU >= 0,85) | Superada |

El modelo fue seleccionado tras una búsqueda de hiperparámetros (grid search 3x3 sobre `eval_steps` y `patience`) seguida de dos experimentos post-grid variando el número máximo de épocas. La run final utilizó 500 épocas máximas con early stopping en la época 284, preservando el checkpoint de la época ~151 (step 6800).

---

## Estructura del repositorio

```
trabajo-final-segformer/
├── scripts/
│   ├── data_acquisition/
│   │   └── obtener_imagenes_gee_sentinel2.py   # Descarga Sentinel-2 vía Google Earth Engine
│   ├── preprocessing/
│   │   ├── convertir_s2_a8bit.py               # Conversión 16-bit → 8-bit (percentil p2-p98 por banda)
│   │   ├── tiff_to_jpg.py                      # Conversión TIF 8-bit → JPG (quality=100)
│   │   ├── create_tiles_directory.py           # Teselado 128px con sliding window (overlap=8)
│   │   └── remove_black_tiles.py               # Filtrado de teselas con bordes negros
│   └── training/
│       └── sugarcane_segformer_v3.py           # Fine-tuning SegFormer MiT-B3 (notebook Colab)
├── ejecuciones/                                 # Logs de las 13 corridas de entrenamiento
├── docs/                                        # Documentación adicional
├── .gitignore
├── LICENSE
└── README.md
```

---

## Pipeline completo

El proyecto sigue un pipeline de 6 fases, desde la descarga de imágenes satelitales hasta la evaluación del modelo:

```
FASE 1 — Descarga (Google Colab + GEE)
  │  obtener_imagenes_gee_sentinel2.py
  │  Sentinel-2 L2A → GeoTIFF 16-bit (EPSG:32721)
  ▼
FASE 2 — Preprocesamiento (PC local)
  │  convertir_s2_a8bit.py    → GeoTIFF 8-bit (estiramiento percentílico)
  │  tiff_to_jpg.py           → JPG quality=100
  ▼
FASE 3 — Teselado y filtrado automático (PC local)
  │  create_tiles_directory.py   → Teselas 128×128, overlap=8, filtra >75% negras
  │  remove_black_tiles.py       → Cuarentena teselas con píxel [0,0,0]
  ▼
FASE 4 — Filtrado manual (PC local)
  │  Filtro 1: Coincidencia con shapefile AZPA (parcelas de caña)
  │  Filtro 2: Densidad de caña (rankar por % de caña visible)
  │  Resultado: 513 imágenes finales (de 4.471 teselas originales)
  ▼
FASE 5 — Etiquetado (CVAT)
  │  Polígonos manuales → máscaras PNG (Segmentation Mask 1.1)
  │  Verde = sugar_cane, Negro = background
  ▼
FASE 6 — Entrenamiento (Google Colab + GPU T4)
  │  sugarcane_segformer_v3.py
  │  Split 70/15/15 → 359 train / 76 val / 78 test
  │  Fine-tuning SegFormer MiT-B3 con HuggingFace Trainer
  │  Resultado: mIoU test = 0,91
  ▼
Evaluación e inferencia
     evaluate_set() sobre val y test
     Visualización: imagen original + predicción + overlay + ground truth
```

---

## Herramientas utilizadas

| Componente | Tecnología | Descripción |
|---|---|---|
| Plataforma de ejecución | **Google Colab** | GPU NVIDIA T4 para entrenamiento |
| Imágenes satelitales | **Sentinel-2** (ESA Copernicus) | Colección L2A (`COPERNICUS/S2_SR_HARMONIZED`) |
| Ingesta de datos | **Google Earth Engine** | API Python, filtrado espacial/temporal/nubosidad |
| Anotación | **CVAT** | Computer Vision Annotation Tool, exportación Segmentation Mask 1.1 |
| Framework DL | **Hugging Face Transformers** | `SegformerForSemanticSegmentation`, `Trainer`, `evaluate` |
| Modelo | **SegFormer MiT-B3** | Checkpoint `nvidia/segformer-b3-finetuned-ade-512-512` |
| Procesador | **SegformerImageProcessor** | Resize 512×512, normalización ImageNet |
| Persistencia | **Google Drive** | Checkpoints, modelo final, curvas de entrenamiento |
| Procesamiento de imágenes | **rasterio**, **OpenCV**, **Pillow** | Conversión de formatos, teselado, filtrado |
| Data augmentation | **albumentations** | Flip, rotación, brightness/contrast (solo train) |
| Lenguaje | **Python 3** | Todos los scripts del repositorio |

---

## Requisitos e instalación

### Python 3.10 o superior

```bash
# Clonar el repositorio
git clone git@github.com:Mobinake/trabajo-final-segformer.git
cd trabajo-final-segformer
```

### Dependencias por componente

**Data acquisition (Colab):**
```bash
pip install earthengine-api geemap
```

**Preprocessing (PC local):**
```bash
pip install rasterio numpy pillow opencv-python
```

**Training (Colab):**
```bash
pip install datasets transformers evaluate torch torchvision pillow matplotlib opencv-python albumentations
```

### Credenciales necesarias

- **Google Earth Engine:** service account JSON con la API de Earth Engine habilitada en un proyecto de Google Cloud.
- **Google Drive:** cuenta de Google para montar Drive en Colab y guardar checkpoints.

---

## Guía de uso

### 1. Descargar imágenes Sentinel-2

El script `obtener_imagenes_gee_sentinel2.py` se ejecuta en Google Colab.

1. Subir el JSON de service account a `/content/` en Colab.
2. Cambiar la variable `ubicacion` por el distrito deseado (`'Tebicuary'`, `'Itape'`, `'Cnel Martinez'`).
3. Ejecutar celda por celda. El script descarga GeoTIFFs de 16-bit y genera un `.zip` por distrito.
4. Descargar el `.zip` y descomprimirlo localmente.

### 2. Convertir a 8-bit

```bash
# Editar INPUT_FOLDER y OUTPUT_FOLDER en el script
python scripts/preprocessing/convertir_s2_a8bit.py
```

Esto aplica un estiramiento percentílico p2-p98 por banda y genera GeoTIFFs de 8-bit.

### 3. Convertir a JPG

```bash
# Editar INPUT_FOLDER y OUTPUT_FOLDER en el script
python scripts/preprocessing/tiff_to_jpg.py
```

### 4. Generar teselas

```bash
python scripts/preprocessing/create_tiles_directory.py
# Ingresar rutas de entrada y salida cuando se solicite
```

Genera teselas de 128×128 px con overlap de 8 px. Las teselas con más de 75% de píxeles negros se descartan automáticamente.

### 5. Filtrar teselas con bordes negros

```bash
# Editar input_dir en el script
python scripts/preprocessing/remove_black_tiles.py
```

Mueve a `descartadas/` cualquier tesela con al menos un píxel RGB [0,0,0].

### 6. Filtrado manual y etiquetado

- **Filtrado manual:** revisar visualmente las JPGs y conservar solo las que contienen caña de azúcar visible.
- **Etiquetado en CVAT:** delimitar parcelas de caña con polígonos, exportar como Segmentation Mask 1.1 (PNG con paleta de colores).

### 7. Entrenamiento

El script `sugarcane_segformer_v3.py` se ejecuta en Google Colab con GPU T4.

1. Subir imágenes y máscaras a Google Drive con la siguiente estructura:
   ```
   MiDrive/sugarcane/
   ├── images/    # teselas .jpg
   └── masks/     # máscaras .png
   ```
2. Abrir el script en Colab y ejecutar celda por celda.
3. El entrenamiento realiza split 70/15/15, data augmentation, early stopping, y guarda el modelo en Drive.
4. Las métricas finales (mIoU en val y test) se calculan a resolución original con `evaluate_set()`.

---

## Dataset

| Etapa | Cantidad | Criterio |
|---|---|---|
| Teselas originales | 4.471 | Salida del teselado sobre los 3 distritos |
| Filtro 1 (AZPA) | — | Coincidencia con shapefile de parcelas de caña |
| Filtro 2 (densidad) | 584 | Ranking por % de píxeles de caña |
| Limpieza visual | **513** | Eliminación de teselas con muy poca caña visible |
| Split train | 359 (70%) | — |
| Split val | 76 (15%) | — |
| Split test | 78 (15%) | — |

**Clases:** 0 = background (fondo), 1 = sugar_cane (caña de azúcar)
**Color de máscara:** verde = caña, negro = background
**Formato CVAT:** Segmentation Mask 1.1 (PNG con paleta de colores)
**Semilla de split:** 42 (reproducibilidad)

---

## Hiperparámetros de entrenamiento

Configuración final (run 11, seleccionada tras grid search + experimentos post-grid):

| Parámetro | Valor |
|---|---|
| Modelo | SegFormer MiT-B3 |
| Checkpoint base | `nvidia/segformer-b3-finetuned-ade-512-512` |
| Learning rate | 6 × 10⁻⁵ |
| Épocas máximas | 500 |
| Batch size (por dispositivo) | 2 |
| Batch size efectivo | 8 (gradient accumulation = 4) |
| Warmup steps | 50 |
| Eval steps | 200 |
| Early stopping patience | 30 |
| Precisión mixta | FP16 |
| Métrica de selección | mIoU |
| Gradient clipping | max_grad_norm = 1.0 |
| Data augmentation | HorizontalFlip, VerticalFlip, RandomRotate90 (p=0.5), RandomBrightnessContrast (p=0.3) |
| Semilla | 42 |
| reduce_labels | False (explícito, crítico) |

---

## Autoría

**Autor:** Mobin Enrique Akhtar Khavari Escobar

**Tutora:** Dra. Liz Báez Lovera

**Institución:** UCNSA — Campus Guairá, Unidad Académica Ciencias y Tecnología

**Carrera:** Ingeniería Informática

**Ubicación:** Villarrica del Espíritu Santo, Paraguay


---

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.
