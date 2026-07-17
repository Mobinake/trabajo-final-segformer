---
type: project
subtype: experiment-log
ejecucion: 06
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 10
eval_steps: 100
estado: completo
---

# Ejecución 06 — Run 06: eval_steps=100, patience=10, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Cuarta run del grid 3×3, primera del bloque steps=100. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat. Las celdas de `trainer.evaluate(val_ds/test_ds)` fueron limpiadas (sin outputs), pero `evaluate_set` y `TrainOutput` sí están disponibles.

## Parámetros

> Cambios respecto a ejecución 05: eval_steps sube de 50 a 100. patience vuelve a 10 (como ejec. 03). num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 100 |
| save_steps | 100 |
| Patience early stopping | 10 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| Warmup steps | 50 |
| FP16 | True |
| Seed | 42 |
| Data augmentation | on (flip, rotate90, brightness/contrast) |
| reduce_labels | False |
| Checkpoint base | `nvidia/mit-b3` |
| Dataset | 513 imgs (3 distritos) |
| Split | 70/15/15 → 359/76/78 |

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42)
  - Train: 359 imgs
  - Val: 76 imgs
  - Test: 78 imgs
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.3858 |
| Training loss (step final 3500, de tabla) | 0.1787 |
| Runtime | 3687.2s (~1:01:24, ~61 min) |
| Épocas ejecutadas | 77.78 |
| Pasos totales | 3.500 |
| Early stopping activado | **Sí** (step 3500, época 77.78) |
| Mejor step (Trainer mIoU) | 2500 (época ~55.56, mIoU Trainer = 0.921030) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 77.78 épocas ≈ 3500 pasos. Cuadra.

> Early stopping: el mejor mIoU del Trainer fue 0.921030 en step 2500. Desde ahí, 10 evaluaciones consecutivas (steps 2600-3500) sin superar ese valor. Con patience=10, el early stopping disparó en step 3500 (la 10ª evaluación sin mejora). El step 3400 (mIoU=0.921005) estuvo muy cerca pero no superó a 0.921030.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 100 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 100 | 1.405494 | 0.442282 | 0.633280 | 0.764389 | 0.792688 |
| 200 | 1.052236 | 0.320818 | 0.741483 | 0.868881 | 0.854928 |
| 300 | 0.936035 | 0.269329 | 0.792337 | 0.886230 | 0.889975 |
| 400 | 0.700049 | 0.197571 | 0.846138 | 0.917668 | 0.921244 |
| 500 | 0.629132 | 0.185597 | 0.849382 | 0.923898 | 0.922378 |
| 600 | 0.553985 | 0.178622 | 0.869833 | 0.931101 | 0.934326 |
| 700 | 0.461935 | 0.182054 | 0.871773 | 0.930086 | 0.935673 |
| 800 | 0.427013 | 0.172613 | 0.878017 | 0.937166 | 0.938566 |
| 900 | 0.401293 | 0.154439 | 0.883515 | 0.944261 | 0.941065 |
| 1000 | 0.394715 | 0.146996 | 0.891133 | 0.941325 | 0.945950 |
| 1100 | 0.322178 | 0.136495 | 0.898873 | 0.947040 | 0.949862 |
| 1200 | 0.297671 | 0.129744 | 0.900553 | 0.949215 | 0.950615 |
| 1300 | 0.265177 | 0.152678 | 0.902729 | 0.951814 | 0.951619 |
| 1400 | 0.283989 | 0.141812 | 0.903195 | 0.950131 | 0.952052 |
| 1500 | 0.267518 | 0.132415 | 0.906721 | 0.952559 | 0.953845 |
| 1600 | 0.302728 | 0.149818 | 0.903710 | 0.950780 | 0.952286 |
| 1700 | 0.260551 | 0.130165 | 0.906344 | 0.952937 | 0.953593 |
| 1800 | 0.237406 | 0.117161 | 0.914510 | 0.956932 | 0.957875 |
| 1900 | 0.213047 | 0.129530 | 0.912783 | 0.954898 | 0.957079 |
| 2000 | 0.234554 | 0.155988 | 0.910157 | 0.954504 | 0.955626 |
| 2100 | 0.208640 | 0.128535 | 0.914540 | 0.957478 | 0.957845 |
| 2200 | 0.193450 | 0.138807 | 0.913386 | 0.956662 | 0.957265 |
| 2300 | 0.204387 | 0.128201 | 0.915452 | 0.957455 | 0.958360 |
| 2400 | 0.171710 | 0.135949 | 0.915811 | 0.958510 | 0.958474 |
| **2500** | **0.192516** | **0.114299** | **0.921030** | **0.959605** | **0.961299** |
| 2600 | 0.195224 | 0.135754 | 0.917353 | 0.958073 | 0.959375 |
| 2700 | 0.187924 | 0.132244 | 0.917847 | 0.958937 | 0.959580 |
| 2800 | 0.176954 | 0.148686 | 0.916612 | 0.958292 | 0.958941 |
| 2900 | 0.179222 | 0.128163 | 0.919844 | 0.959994 | 0.960609 |
| 3000 | 0.165690 | 0.146861 | 0.916878 | 0.958519 | 0.959072 |
| 3100 | 0.169453 | 0.145595 | 0.917330 | 0.958609 | 0.959317 |
| 3200 | 0.181249 | 0.145690 | 0.919589 | 0.959776 | 0.960484 |
| 3300 | 0.160810 | 0.146165 | 0.919009 | 0.959788 | 0.960160 |
| 3400 | 0.156683 | 0.136815 | 0.921005 | 0.960752 | 0.961194 |
| 3500 | 0.178714 | 0.143450 | 0.920112 | 0.960613 | 0.960709 |

> Mejor Trainer mIoU: **0.921030** en step 2500 (época ~55.56). Después de eso, 10 evaluaciones consecutivas (steps 2600-3500) sin mejora. Con patience=10 y eval_steps=100, el early stopping disparó en step 3500 (la 10ª evaluación sin mejora = 1000 pasos después del best).

## Métricas del Trainer (evaluate, best model cargado)

> ⚠️ El usuario limpió los outputs de `trainer.evaluate(val_ds)` y `trainer.evaluate(test_ds)`. Las métricas del Trainer a 512×512 NO están disponibles para esta run. Solo evaluate_set (resolución original) está disponible.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | — | — |
| eval_mean_iou | — | — |
| eval_mean_accuracy | — | — |
| eval_overall_accuracy | — | — |

> ⏳ Pendiente: si se necesitan las métricas del Trainer para esta run, el usuario puede re-ejecutar `trainer.evaluate(val_ds)` y `trainer.evaluate(test_ds)` en Colab con el modelo guardado.

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8914** |
| Median IoU | 0.9250 |
| Std | 0.0943 |
| Min | 0.5036 |
| Max | 0.9817 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8845** |
| Median IoU | 0.9162 |
| Std | 0.0900 |
| Min | 0.5824 |
| Max | 0.9766 |

- ✅ Hipótesis superada: mIoU en test set = 0.8845 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8845 ≥ 0.85.
- **Early stopping activado:** el modelo detuvo en step 3500 (época 77.78). El mejor checkpoint fue step 2500 (época ~55.56, mIoU Trainer = 0.921030). Desde ahí, 10 evaluaciones consecutivas (steps 2600-3500) sin mejora. Con eval_steps=100, patience=10 → 1000 pasos de margen. El step 3400 (mIoU=0.921005) estuvo a 0.000025 de superar el best pero no lo logró.
- **Comparación con ejec. 03 (eval_steps=50, patience=10):** misma patience pero eval_steps doble. Ejec. 03 detuvo en step 3050 (época 67.78); esta run detuvo en step 3500 (época 77.78). mIoU test: ejec. 03 = 0.8828 vs esta run = 0.8845 (+0.0017). La diferencia es mínima. Con eval_steps=100, el early stopping tardó más en disparar (necesita 1000 pasos de margen vs 500 con eval_steps=50), permitiendo 450 pasos más de entrenamiento.
- **Comparación con ejec. 05 (eval_steps=50, patience=30):** ejec. 05 corrió 100 épocas completas (4500 pasos); esta run detuvo en 3500. mIoU test: ejec. 05 = 0.8861 vs esta run = 0.8845 (-0.0016). Diferencia mínima. Esta run fue más eficiente (61 min vs 91 min, -30 min).
- **Eficiencia:** esta run fue la más rápida del grid hasta ahora (61 min vs 69-100 min de las anteriores), gracias al early stopping en step 3500.
- **Convergencia:** el mejor mIoU del Trainer se encontró temprano (step 2500, época ~55.56). Los siguientes 1000 pasos no aportaron mejora. Esto refuerza la conclusión de que el modelo converge alrededor de step 2500.
- **Variabilidad:** Std en test = 0.0900 (la más baja del grid hasta ahora: ejec. 03: 0.1048, ejec. 04: 0.1019, ejec. 05: 0.0910).
- **Min IoU en test:** 0.5824 — comparable a ejec. 05 (0.5851) y mejor que ejec. 03 (0.2808) y ejec. 04 (0.3708).
- **Δ val-test:** 0.8914 - 0.8845 = 0.0069 (similar a ejec. 05: 0.0080, mejor que ejec. 03: 0.0114 y ejec. 04: 0.0111).
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 27 jun.

## Comparación con ejecuciones 01-05

| Métrica | Ejec. 01 (50 ep, p=10, s=50) | Ejec. 02 (100 ep, p=50, s=50) | Ejec. 03 (100 ep, p=10, s=50) | Ejec. 04 (100 ep, p=20, s=50) | Ejec. 05 (100 ep, p=30, s=50) | **Ejec. 06 (100 ep, p=10, s=100)** | Δ 06 vs 03 | Δ 06 vs 05 |
|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | 20 | 30 | 10 | — | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | 100 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 77.78 | +10.0 | -22.22 |
| Early stopping | Sí (ép 43.33) | No | Sí (ép 67.78) | No | No | **Sí (ép 77.78)** | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 3.500 | +450 | -1.000 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | 0.1787 | — | +0.0187 |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | **~61 min** | -8 min | -30 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | 0.8941 | **0.8914** | -0.0028 | -0.0027 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | 0.9250 | +0.0001 | -0.0050 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | 0.0943 | +0.0066 | +0.0003 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | 0.5036 | -0.0445 | -0.0523 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | 0.9817 | +0.0022 | -0.0011 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | 0.8861 | **0.8845** | +0.0017 | -0.0016 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | 0.9162 | -0.0034 | -0.0005 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | 0.0900 | -0.0148 | -0.0010 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | 0.5824 | +0.3016 | -0.0027 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | 0.9766 | -0.0017 | -0.0023 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | 0.0069 | -0.0045 | -0.0011 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |

> **Conclusión parcial del bloque steps=100 (run 06):** con eval_steps=100 y patience=10, el early stopping disparó en step 3500 (época 77.78), 1000 pasos después del best (step 2500). El mIoU test (0.8845) es comparable al bloque steps=50 (0.8816-0.8861). La principal diferencia es que con eval_steps=100, cada evaluación cubre más pasos, dando más margen al early stopping (1000 pasos vs 500 con eval_steps=50). Esto permitió entrenar 450 pasos más que ejec. 03 (mismo patience=10) sin beneficio significativo. Ejec. 02 (patience=50, fuera del grid) sigue siendo la mejor en test (0.8903).

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, mismo patience pero eval_steps=50
- [[ejecucion-04-mitb3-100epocas-patience20-steps50]] — ejecución 04, patience=20
- [[ejecucion-05-mitb3-100epocas-patience30-steps50]] — ejecución 05, patience=30
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
