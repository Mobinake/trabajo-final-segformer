# Mapeo Automático de Caña de Azúcar en Guairá con SegFormer

Trabajo Final de Grado — Ingeniería Informática, UCNSA Campus Guairá

Segmentación semántica de plantaciones de caña de azúcar con imágenes Sentinel-2 y SegFormer (MiT-B3), aplicado a tres distritos del departamento de Guairá, Paraguay.

## Resultados

| Métrica | Valor |
|---|---|
| **mIoU test (sugar_cane)** | **0,91** |
| mIoU val (sugar_cane) | 0,91 |
| Mediana IoU test | 0,94 |
| Desviación estándar test | 0,08 |
| Hipótesis (mIoU ≥ 0,85) | Superada |

Configuración final seleccionada tras grid search 3×3 (`eval_steps` × `patience`) + experimentos post-grid. Run final: 500 épocas, early stopping en época 284, mejor checkpoint en época ~151 (step 6.800).

## Estructura

```
trabajo-final-segformer/
├── scripts/
│   ├── data_acquisition/
│   │   └── obtener_imagenes_gee_sentinel2.py   # Descarga Sentinel-2 vía GEE
│   ├── preprocessing/
│   │   ├── convertir_s2_a8bit.py               # 16-bit → 8-bit (percentil p2-p98 por banda)
│   │   ├── tiff_to_jpg.py                      # TIF 8-bit → JPG (quality=100)
│   │   ├── create_tiles_directory.py           # Teselado 128px (overlap=8, filtra >75% negras)
│   │   └── remove_black_tiles.py               # Cuarentena teselas con píxel [0,0,0]
│   └── training/
│       └── sugarcane_segformer_v3.py           # Fine-tuning SegFormer MiT-B3 (Colab)
├── ejecuciones/                                 # Logs de las 13 corridas
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

```
Descarga (Colab + GEE)        obtener_imagenes_gee_sentinel2.py
  → Sentinel-2 L2A, GeoTIFF 16-bit
Conversión (PC local)         convertir_s2_a8bit.py → tiff_to_jpg.py
  → JPG 8-bit RGB
Teselado (PC local)           create_tiles_directory.py → remove_black_tiles.py
  → 4.471 teselas 128×128, filtrado automático
Filtrado manual + CVAT        Shapefile AZPA + limpieza visual
  → 513 imágenes etiquetadas (verde = caña, negro = background)
Entrenamiento (Colab + GPU T4) sugarcane_segformer_v3.py
  → Split 70/15/15 (359/76/78), SegFormer MiT-B3, mIoU test = 0,91
```

## Dataset

| Etapa | Cantidad |
|---|---|
| Teselas originales | 4.471 |
| Tras filtros AZPA + densidad + limpieza visual | **513** |
| Train / Val / Test (70/15/15, seed=42) | 359 / 76 / 78 |

**Clases:** 0 = background, 1 = sugar_cane · **Color máscara CVAT:** verde RGB(88,207,74)

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| Modelo | SegFormer MiT-B3 (`nvidia/segformer-b3-finetuned-ade-512-512`) |
| Learning rate | 6 × 10⁻⁵ |
| Épocas máximas | 500 |
| Batch size efectivo | 8 (2 por dispositivo × accum 4) |
| Eval / save steps | 200 |
| Warmup steps | 50 |
| Early stopping patience | 30 |
| Precisión mixta | FP16 |
| Gradient clipping | max_grad_norm = 1.0 |
| Data augmentation | HorizontalFlip, VerticalFlip, RandomRotate90 (p=0,5), RandomBrightnessContrast (p=0,3) |
| reduce_labels | False (explícito) |
| Semilla | 42 |

## Requisitos

```bash
git clone git@github.com:Mobinake/trabajo-final-segformer.git
cd trabajo-final-segformer
pip install -r requirements.txt
```

**Data acquisition + training:** Google Colab con GPU T4 · service account JSON de Google Earth Engine · Google Drive

**Preprocessing (PC local):** Python 3.10+ con `rasterio numpy pillow opencv-python`

Stack: Hugging Face Transformers · albumentations · Google Earth Engine · CVAT · Sentinel-2 L2A

## Autoría

**Autor:** Mobin Enrique Akhtar Khavari Escobar

**Tutora:** Dra. Liz Báez Lovera

**Institución:** UCNSA — Campus Guairá

**Carrera:** Ingeniería Informática

## Licencia

MIT
