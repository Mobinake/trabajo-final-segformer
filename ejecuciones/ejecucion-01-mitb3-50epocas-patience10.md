---
type: project
subtype: experiment-log
ejecucion: 01
fecha: 2026-06-20
modelo: SegFormer MiT-B3
epocas_max: 50
patience: 10
---

# Ejecución 01 — MiT-B3, 50 épocas, patience 10

> Primera ejecución de entrenamiento con dataset multi-distrito (513 imágenes). Fecha: 20 de junio de 2026. Script: `sugarcane_segformer_v3.py` ejecutado en Google Colab con GPU T4.

## Parámetros

| Parámetro | Valor |
|---|---|
| Modelo | SegFormer MiT-B3 (`nvidia/segformer-b3-finetuned-ade-512-512`) |
| Learning rate | 6 × 10⁻⁵ |
| Épocas máximas | 50 |
| Patience early stopping | 10 |
| Batch size train (per device) | 2 |
| Batch size eval (per device) | 4 |
| Gradient accumulation | 4 pasos |
| Batch size efectivo | 8 |
| Warmup steps | 50 |
| Evaluación | cada 50 pasos |
| Guardado | cada 50 pasos, top-3 |
| Precisión | FP16 (mixta) |
| Métrica selección | mIoU (`greater_is_better=True`) |
| Semilla split | 42 |
| Optimizador | AdamW (default HuggingFace) |
| Scheduler | warmup lineal (default) |
| max_grad_norm | 1.0 |
| Data augmentation | albumentations: HorizontalFlip, VerticalFlip, RandomRotate90 (p=0.5 c/u), RandomBrightnessContrast (p=0.3). Solo en train. |
| reduce_labels | False (explícito) |
| ignore_mismatched_sizes | True |
| collate_fn | torch.stack explícito |
| dataloader_num_workers | 2 |

## Dataset

- Total etiquetado: 513 imágenes (de 4.471 tiles originales → 584 post-filtros → 513 etiquetadas finales)
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42)
  - Train: 359 imgs
  - Val: 76 imgs
  - Test: 78 imgs
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss final | 0.5416 |
| Runtime | 3.382 s (~56 min) |
| Épocas ejecutadas | 43.33 (early stopping activado) |
| Pasos totales | 1.950 |

## Métricas de validación (76 imgs)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8727** |
| Median IoU | 0.9136 |
| Std | 0.1052 |
| Min | 0.4752 |
| Max | 0.9804 |

## Métricas de test (78 imgs, nunca vistas)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8535** |
| Median IoU | 0.9082 |
| Std | 0.1411 |
| Min | 0.1593 |
| Max | 0.9781 |

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8535 ≥ 0.85.
- La diferencia entre val (0.8727) y test (0.8535) es pequeña, indicando buena generalización.
- El std más alto en test (0.1411 vs 0.1052) sugiere mayor variabilidad entre distritos.
- Early stopping activado a época 43.33: el modelo dejó de mejorar en mIoU de validación antes de llegar a las 50 épocas máximas.

## Errores identificados (análisis visual de 3 muestras del val set)

1. Falsos positivos: caminos rurales y suelo desnudo con alta reflectancia pueden confundirse con caña.
2. Falsos negativos: parcelas recién plantadas o en etapas tempranas con firma espectral similar a pasturas.
3. Variabilidad entre distritos: el std más alto en test sugiere que el modelo tiene mayor dificultad en algunos distritos. Pendiente análisis por distrito.

## Comparación con trabajos relacionados

| Trabajo | Modelo | Resultado |
|---|---|---|
| Poortinga et al. 2021 | CNN tradicionales + Sentinel-2 series temporales | Accuracy > 90% (no segmentación densa) |
| Yuan et al. 2024 | DSCA-PSPNet (CNN+transformer híbrido) | mIoU 0.82 |
| **Esta tesis (ejecución 01)** | **SegFormer MiT-B3** | **mIoU 0.8535 (test multi-distrito)** |

→ Resultado superior a Yuan et al. y comparable a Poortinga (aunque con metodología distinta: segmentación densa vs clasificación de píxeles).

## Baseline histórico (referencia, no esta ejecución)

> Conservado para referencia. Es el resultado del modelo viejo entrenado en 1 solo distrito (Tebicuary, 98 teselas), antes de la ampliación a multi-distrito.

| Métrica | Valor |
|---|---|
| mIoU (sin fondo, sugar_cane) | 0.929 |
| Pixel Accuracy | 0.967 |
| Precision sugar_cane | 0.941 |
| Recall sugar_cane | 0.918 |
| F1 sugar_cane | 0.929 |

## 🔗 Notas relacionadas
- [[04 - Resultados y Validación]] — estos mismos resultados consolidados para el manuscrito
- [[08 - Configuración Técnica]] — hiperparámetros detallados del script
- [[ejecuciones/README]] — índice de todas las ejecuciones
- [[ejecucion-02-mitb3-100epocas-patience50]] — siguiente ejecución (100 épocas, patience 50)
