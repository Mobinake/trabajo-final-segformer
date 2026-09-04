# Mapeo Automático de Caña de Azúcar en Guairá con SegFormer

Trabajo Final de Grado — Ingeniería Informática · UCNSA, Campus Guairá

Segmentación semántica de plantaciones de caña de azúcar con imágenes Sentinel-2 L2A y SegFormer (MiT-B3), aplicada a tres distritos del departamento de Guairá, Paraguay: **Tebicuary, Itapé y Coronel Martínez**.

## Resultados (run 12, configuración final)

| Métrica | Validación (76) | Prueba (78) |
|---|---|---|
| **mIoU (sugar_cane)** | **0,90** | **0,90** |
| Mediana IoU | 0,94 | 0,94 |
| Desviación estándar | 0,09 | 0,08 |
| Accuracy | 0,9640 | 0,9760 |
| Precision | 0,9457 | 0,9593 |
| Recall | 0,9591 | 0,9649 |
| F1-score | 0,9524 | 0,9621 |

> Hipótesis del trabajo: mIoU ≥ 0,85 en el conjunto de prueba → **superada** (0,90 ≥ 0,85).

Métricas calculadas con `evaluate_set()` a resolución original (128×128 px), IoU de la clase caña promediado por imagen. La corrida final usó 500 épocas máx. con *early stopping* activado en la época 311 (paso 14.000); mejor *checkpoint* en el paso 8.000 (época ~178). Entrenamiento: ~221 min en GPU T4 (Colab).

## Pipeline

```
Descarga (Colab + GEE)          obtener_imagenes_gee_sentinel2.py
  → Sentinel-2 L2A RGB (B4,B3,B2), GeoTIFF 16-bit, DN crudos
Conversión (PC local)           convertir_s2_a8bit.py → tiff_to_jpg.py
  → estiramiento p2–p98 por banda, JPG 8-bit
Teselado (PC local)             create_tiles_directory.py → remove_black_tiles.py
  → ~8.000 teselas en bruto, 128×128 (solape 8 px) → 4.471 tras filtro
Filtrado + etiquetado (CVAT)    AZPA + limpieza visual
  → 513 pares imagen-máscara (verde = caña, negro = fondo)
Entrenamiento (Colab + T4)      sugarcane_segformer_v3.py
  → split 70/15/15 (359/76/78), mIoU prueba = 0,90
```

## Dataset

| Etapa | Cantidad |
|---|---|
| Teselas generadas (sliding window) | ~8.000 |
| Tras filtro >75 % píxeles negros | 4.471 |
| Tras filtros AZPA + densidad + limpieza visual | **513** |
| Train / Val / Test (70/15/15, seed=42) | 359 / 76 / 78 |

**Clases:** 0 = background, 1 = sugar_cane · **Máscara CVAT:** verde RGB(88, 207, 74)

## Estructura

```
trabajo-final-segformer/
├── scripts/
│   ├── data_acquisition/
│   │   └── obtener_imagenes_gee_sentinel2.py   # Descarga Sentinel-2 vía GEE (Colab)
│   ├── preprocessing/
│   │   ├── convertir_s2_a8bit.py               # 16-bit → 8-bit (percentil p2–p98 por banda)
│   │   ├── tiff_to_jpg.py                      # TIF 8-bit → JPG (quality=100)
│   │   ├── create_tiles_directory.py           # Teselado 128 px (overlap=8, filtra >75 % negras)
│   │   └── remove_black_tiles.py               # Cuarentena teselas con píxel [0,0,0]
│   └── training/
│       └── sugarcane_segformer_v3.py           # Fine-tuning SegFormer MiT-B3 (Colab, formato notebook)
├── ejecuciones/                                 # Logs de las 12 corridas (grid + post-grid)
├── docs/                                        # Guía de uso del modelo en local
├── requirements.txt
├── LICENSE
└── README.md
```

## Experimentos

Búsqueda en dos etapas: **grid search** de sensibilidad al criterio de parada (runs 01–09: `eval_steps` ∈ {50, 100, 200} × `patience` ∈ {10, 20, 30}, 100 épocas máx.) y **post-grid** (runs 10–12: 200, 300 y 500 épocas máx. con la mejor config). Detalle completo por corrida en [`ejecuciones/`](ejecuciones/README.md).

| Run | Épocas máx. | Early stop | mIoU val | mIoU test |
|---|---|---|---|---|
| 01–09 (grid) | 100 | 2 de 9 | 0,89 – 0,90 | 0,88 – 0,89 |
| 10 | 200 | No | 0,9052 | 0,8992 |
| 11 | 300 | No | 0,9131 | 0,9050 |
| **12 (final)** | **500** | **Sí (época 311)** | **0,9038** | **0,9041** |

Todas las corridas usaron una única semilla (42); las diferencias entre runs se interpretan como variabilidad del criterio de parada, no como mejoras significativas. La run 12 se seleccionó por operar con el margen de épocas más amplio, activar el *early stopping* (validando el criterio) y tener la menor desviación estándar en prueba.

## Hiperparámetros (run final)

| Parámetro | Valor |
|---|---|
| Modelo | SegFormer MiT-B3 (`nvidia/segformer-b3-finetuned-ade-512-512`) |
| Learning rate | 6 × 10⁻⁵ |
| Épocas máximas | 500 (early stopping patience=30) |
| Batch size efectivo | 8 (2 por dispositivo × acumulación 4) |
| Eval / save steps | 200 (conserva los 3 mejores por mIoU) |
| Warmup steps | 50 |
| Precisión mixta | FP16 |
| Gradient clipping | max_grad_norm = 1.0 |
| Data augmentation | HorizontalFlip, VerticalFlip, RandomRotate90 (p=0,5), RandomBrightnessContrast (p=0,3) — solo train |
| reduce_labels | False (explícito) |
| Semilla | 42 |

## Requisitos y uso

```bash
git clone https://github.com/Mobinake/trabajo-final-segformer.git
cd trabajo-final-segformer
pip install -r requirements.txt
```

- **Descarga de imágenes + entrenamiento:** Google Colab con GPU T4, *service account* JSON de Google Earth Engine y Google Drive.
- **Preprocesamiento (PC local):** Python 3.10+ con `rasterio numpy pillow opencv-python`.

Para inferencia con el modelo entrenado en tu máquina (sin Colab), ver la **[guía de uso local](docs/uso-modelo-local.md)**.

### Notas de troubleshooting

- `reduce_labels=False` **es obligatorio** en `SegformerImageProcessor`: el checkpoint de ADE20K puede traer `reduce_labels=True`, que resta 1 a los labels y deja el entrenamiento con una sola clase efectiva (falla silenciosa).
- La lectura de máscaras usa una única función `mask_rgb_to_class()` (G > 100, R < 150, B < 150) en dataset, inferencia y evaluación; no usar umbrales inline distintos (p. ej. R < 50), que no detectan el verde RGB(88, 207, 74) de CVAT.

## Manuscrito

El manuscrito completo del Trabajo Final de Grado (en español) se redactó en Overleaf y no está incluido en este repositorio. Este repo contiene el código reproducible del pipeline: los valores técnicos del manuscrito son consistentes con los logs de [`ejecuciones/`](ejecuciones/README.md).

## Autoría

- **Autor:** Mobin Enrique Akhtar Khavari Escobar
- **Tutora:** Dra. Liz Báez Lovera
- **Institución:** Universidad Católica Nuestra Señora de la Asunción — Campus Guairá
- **Carrera:** Ingeniería Informática

## Licencia

MIT
