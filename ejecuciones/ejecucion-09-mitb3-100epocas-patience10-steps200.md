---
type: project
subtype: experiment-log
ejecucion: 09
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 10
eval_steps: 200
estado: parcial
---

# Ejecución 09 — Run 09: eval_steps=200, patience=10, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Séptima run del grid, primera del bloque steps=200. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` completo inline en el chat (re-pasado el 29-06-2026). Métricas de `trainer.evaluate()` (val y test) y `evaluate_set(val)` disponibles. ⏳ Faltan métricas de `evaluate_set(test)` — la celda del notebook no se ejecutó completamente (execution_count=null, progress bar en 0/78).
>
> **Nota (29-06-2026):** Al re-pasar el .ipynb completo, se detectó que los valores de la tabla de entrenamiento DIFIEREN de los registrados en la primera pasada. El notebook es la fuente autoritativa. Best step corregido: 3600 (no 4000), mIoU Trainer corregido: 0.922040 (no 0.918954). TrainOutput disponible: train_loss=0.34526732330852083.

## Parámetros

> Únicos cambios respecto a ejecución 08: eval_steps sube de 100 a 200, patience baja de 30 a 10. num_train_epochs se mantiene en 100.

| Parámetro | Valor | Cambió vs ejec. 08 |
|---|---|---|
| Learning rate | 6 × 10⁻⁵ | No |
| Épocas máx | 100 | No |
| eval_steps | 200 | **Sí** (100 → 200) |
| save_steps | 200 | **Sí** (100 → 200) |
| Patience early stopping | 10 | **Sí** (30 → 10) |
| Batch size efectivo | 8 (2 × 4 grad_accum) | No |
| Warmup steps | 50 | No |
| FP16 | True | No |
| Seed | 42 | No |
| Data augmentation | on (flip, rotate90, brightness/contrast) | No |
| reduce_labels | False | No |
| Checkpoint base | `nvidia/mit-b3` | No |
| Dataset | 513 imgs (3 distritos) | No |
| Split | 70/15/15 → 359/76/78 | No |

> Nota: el `RUN_LABEL` dentro del notebook dice `"run06_steps100_pat10"` (stale — no se actualizó), pero `EVAL_STEPS=200` y `PATIENCE=10` son correctos para run 09. El `RUN_LABEL` solo afecta el nombre del `output_dir`, no el comportamiento del entrenamiento.

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
| Training loss (TrainOutput, promedio) | 0.3453 |
| Training loss (step 4500, de tabla) | 0.157105 |
| Runtime | ~74 min (4.474 s = 1:14:34) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3600 (época ~80.00, mIoU Trainer = 0.922040) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Con patience=10 y eval_steps=200: desde el best step 3600, harían falta 10 evaluaciones sin mejora (hasta step 3600 + 10×200 = 5600) para disparar el early stopping. Como el máximo de pasos es 4500 (100 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 3800 hasta 4500 hay 5 evaluaciones sin mejora (0.920740, 0.920709, 0.920164, 0.921166, 0.921343), todas por debajo de 0.922040.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 200 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 200 | 1.106451 | 0.349029 | 0.733891 | 0.866334 | 0.849492 |
| 400 | 0.790419 | 0.220099 | 0.819315 | 0.898647 | 0.906670 |
| 600 | 0.553548 | 0.197958 | 0.859954 | 0.924690 | 0.929047 |
| 800 | 0.461046 | 0.178060 | 0.881842 | 0.938470 | 0.940728 |
| 1000 | 0.365139 | 0.134244 | 0.897761 | 0.947735 | 0.949141 |
| 1200 | 0.312185 | 0.138053 | 0.897274 | 0.947364 | 0.948895 |
| 1400 | 0.292334 | 0.144055 | 0.901890 | 0.949834 | 0.951328 |
| 1600 | 0.303763 | 0.136432 | 0.904210 | 0.951874 | 0.952468 |
| 1800 | 0.234128 | 0.133908 | 0.909287 | 0.954059 | 0.955171 |
| 2000 | 0.236024 | 0.152344 | 0.912271 | 0.955322 | 0.956751 |
| 2200 | 0.193252 | 0.146571 | 0.914879 | 0.957775 | 0.958011 |
| 2400 | 0.187766 | 0.142949 | 0.916522 | 0.958400 | 0.958882 |
| 2600 | 0.220261 | 0.150888 | 0.915470 | 0.957362 | 0.958379 |
| 2800 | 0.188206 | 0.139399 | 0.918672 | 0.959388 | 0.960004 |
| 3000 | 0.170390 | 0.162099 | 0.918230 | 0.959349 | 0.959760 |
| 3200 | 0.179286 | 0.157336 | 0.917728 | 0.958553 | 0.959545 |
| 3400 | 0.166246 | 0.147762 | 0.919539 | 0.959946 | 0.960443 |
| **3600** | **0.163798** | **0.140046** | **0.922040** | **0.960918** | **0.961755** |
| 3800 | 0.171430 | 0.147984 | 0.920740 | 0.960698 | 0.961051 |
| 4000 | 0.143319 | 0.152328 | 0.920709 | 0.960756 | 0.961029 |
| 4200 | 0.140672 | 0.159194 | 0.920164 | 0.960340 | 0.960759 |
| 4400 | 0.151962 | 0.150945 | 0.921166 | 0.960764 | 0.961283 |
| 4500 | 0.157105 | 0.153100 | 0.921343 | 0.960844 | 0.961375 |

> Mejor Trainer mIoU: **0.922040** en step 3600 (época ~80.00). Después de eso, 5 evaluaciones consecutivas (steps 3800, 4000, 4200, 4400, 4500) sin mejora. Con patience=10 y eval_steps=200, el early stopping habría disparado en step 5600 (la 10ª evaluación sin mejora), muy por encima del máximo de 4500 pasos. El entrenamiento terminó naturalmente al completar las 100 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> Datos del `.ipynb` pasado el 29-06-2026. `trainer.evaluate(val_ds)` y `trainer.evaluate(test_ds)` ejecutados en celda 12.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.148994 | 0.100477 |
| eval_mean_iou | 0.920955 | 0.934155 |
| eval_mean_accuracy | 0.960149 | 0.965784 |
| eval_overall_accuracy | 0.961214 | 0.970428 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

> Datos del `.ipynb` pasado el 29-06-2026. `evaluate_set(val_raw, ...)` ejecutado en celda 16 (76 muestras).

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | 0.8952 |
| Median IoU | 0.9275 |
| Std | 0.0893 |
| Min | 0.6074 |
| Max | 0.9793 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

> ⏳ Pendiente — la celda 17 del notebook tiene `execution_count: null` y el progress bar en 0/78. La celda no se ejecutó (o se interrumpió). Falta que el usuario corra `evaluate_set(test_raw, ...)` y pase los resultados.

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | — |
| Median IoU | — |
| Std | — |
| Min | — |
| Max | — |

## Observaciones

- **Métricas de evaluate_set(val) disponibles:** Mean IoU = 0.8952 (76 muestras). ⏳ evaluate_set(test) pendiente — la celda del notebook no se ejecutó.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3600 (época ~80.00, mIoU Trainer = 0.922040). Después de eso, 5 evaluaciones sin mejora, pero con patience=10 y eval_steps=200 hacían falta 10 para disparar (step 5600 > 4500). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 03 (patience=10, eval_steps=50):** misma patience pero eval_steps=200 vs 50. Ejec. 03 detuvo en step 3050 (67.78 épocas); esta run corrió hasta step 4500 (100 épocas). Con eval_steps=200, cada evaluación sin mejora "cuesta" 200 pasos en lugar de 50, haciendo mucho más difícil que el early stopping se dispare dentro de las 100 épocas.
- **Runtime más corto:** ~71 min vs ejec. 08 (~80 min). Ambas corrieron 4500 pasos, pero esta run evaluó cada 200 pasos (23 evaluaciones) vs ejec. 08 que evaluó cada 100 (45 evaluaciones). Menos evaluaciones = menos overhead.
- **Trainer mIoU:** 0.922040 (best step 3600). Entre las runs con eval_steps=200, es intermedia (ejec. 10: 0.923451, ejec. 11: 0.924093). eval_steps=200 implica menos checkpoints evaluados, lo que puede hacer que el best step no capture el pico real.
- **Plateau visible:** a partir de step ~2400, el mIoU del Trainer se estabiliza en el rango 0.914-0.922. Los últimos ~2.100 pasos (steps 2400-4500) no aportaron mejora significativa más allá de variaciones marginales.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` (cell 11 usa SAVE_PATH fijo, no usa RUN_LABEL para el guardado final).

## Comparación con ejecuciones 01-08

| Métrica | Ejec. 01 (50 ep, p=10, s=50) | Ejec. 02 (100 ep, p=50, s=50) | Ejec. 03 (100 ep, p=10, s=50) | Ejec. 04 (100 ep, p=20, s=50) | Ejec. 05 (100 ep, p=30, s=50) | Ejec. 06 (100 ep, p=10, s=100) | Ejec. 07 (100 ep, p=20, s=100) | Ejec. 08 (100 ep, p=30, s=100) | **Ejec. 09 (100 ep, p=10, s=200)** | Δ 09 vs 08 |
|---|---|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | — |
| Patience | 10 | 50 | 10 | 20 | 30 | 10 | 20 | 30 | 10 | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | 100 | 100 | 100 | 200 | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 77.78 | 100 | 100 | 100 | 0 |
| Early stopping | Sí (ép 43.33) | No | Sí (ép 67.78) | No | No | Sí (ép 77.78) | No | No | No | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 3.500 | 4.500 | 4.500 | 4.500 | 0 |
| Best step (Trainer) | 950 | ~4.500 | 2.550 | 3.500 | 3.850 | 2.500 | 3.600 | 4.200 | 3.600 | -600 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | 0.1787 | 0.1500 | 0.1576 | 0.1571 | — |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | ~61 min | ~84 min | ~80 min | ~74 min | -6 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | 0.8941 | 0.8914 | 0.8958 | 0.8998 | 0.8952 | -0.0046 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | 0.9250 | 0.9301 | 0.9283 | 0.9275 | -0.0008 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | 0.0943 | 0.0856 | 0.0813 | 0.0893 | +0.0080 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | 0.5036 | 0.5281 | 0.5533 | 0.6074 | +0.0541 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | 0.9817 | 0.9775 | 0.9807 | 0.9793 | -0.0014 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | 0.8861 | 0.8845 | 0.8804 | 0.8897 | — | — |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | 0.9162 | 0.9179 | 0.9230 | — | — |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | 0.0900 | 0.0993 | 0.0854 | — | — |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | 0.5824 | 0.4866 | 0.6211 | — | — |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | 0.9766 | 0.9765 | 0.9784 | — | — |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | 0.0069 | 0.0154 | 0.0101 | — | — |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (val) | — |

> ⏳ Las métricas de evaluate_set(test) se completarán cuando el usuario corra la celda 17 del notebook (no se ejecutó — execution_count=null). Las métricas de val ya están disponibles.

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, patience=10, eval_steps=50
- [[ejecucion-04-mitb3-100epocas-patience20-steps50]] — ejecución 04, patience=20, eval_steps=50
- [[ejecucion-05-mitb3-100epocas-patience30-steps50]] — ejecución 05, patience=30, eval_steps=50
- [[ejecucion-06-mitb3-100epocas-patience10-steps100]] — ejecución 06, patience=10, eval_steps=100
- [[ejecucion-07-mitb3-100epocas-patience20-steps100]] — ejecución 07, patience=20, eval_steps=100
- [[ejecucion-08-mitb3-100epocas-patience30-steps100]] — ejecución 08, patience=30, eval_steps=100
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
