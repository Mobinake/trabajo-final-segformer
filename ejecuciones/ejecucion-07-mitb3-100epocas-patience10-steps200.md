---
type: project
subtype: experiment-log
ejecucion: 07
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 10
eval_steps: 200
estado: incompleto
---

# Ejecución 07 — eval_steps=200, patience=10, max_epochs=100

> Grid search 3×3. Séptima run, segunda del bloque eval_steps=200. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 06: eval_steps sube de 100 a 200. patience vuelve a 10. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 200 |
| save_steps | 200 |
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

> `RUN_LABEL="run11_steps200_pat30"` (stale — no se actualizó). `EVAL_STEPS=200` y `PATIENCE=10` son correctos.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.3453 |
| Training loss (step final 4500, de tabla) | 0.1571 |
| Runtime | 4474.0s (~1:14:15, ~74 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3600 (época ~80.00, mIoU Trainer = 0.922040) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 3600 (mIoU Trainer = 0.922040). Después de eso, 5 evaluaciones sin mejora (steps 3800-4500), pero con patience=10 hacían falta 10 para disparar (step 5600 > 4500). El entrenamiento terminó naturalmente.

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

> Mejor Trainer mIoU: **0.922040** en step 3600 (época ~80.00). Después de eso, 5 evaluaciones sin mejora. Con patience=10, el early stopping habría disparado en step 5600, muy por encima del máximo de 4500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.148994 | 0.100477 |
| eval_mean_iou | 0.920955 | 0.934155 |
| eval_mean_accuracy | 0.960149 | 0.965784 |
| eval_overall_accuracy | 0.961214 | 0.970428 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8952** |
| Median IoU | 0.9275 |
| Std | 0.0893 |
| Min | 0.6074 |
| Max | 0.9793 |

## Métricas de test (78 imgs, evaluate_set a resolución original)

> ⚠️ **PENDIENTE:** el notebook se cortó antes de completar `evaluate_set()` sobre el test set. Solo está disponible `trainer.evaluate(test_ds)` (sobre 512×512).

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **⏳ Pendiente** |
| Median IoU | ⏳ Pendiente |
| Std | ⏳ Pendiente |
| Min | ⏳ Pendiente |
| Max | ⏳ Pendiente |

## Observaciones

- ⏳ **evaluate_set(test) pendiente:** el notebook se cortó en la celda 17 (evaluate_set test) antes de completar la evaluación a resolución original. Solo está disponible el mIoU del Trainer sobre test (0.934155 a 512×512).
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3600 (época ~80.00, mIoU Trainer = 0.922040).
- **Δ val-test (Trainer):** 0.920955 - 0.934155 = -0.013200 — el test supera al val en el Trainer, patrón consistente con otras runs.
- **Training loss:** bajó de 1.106 (step 200) a 0.157 (step 4500). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 29 jun a las 02:06.

## 🔗 Notas relacionadas
- ejecucion-05-mitb3-100epocas-patience10-steps200 — ejecución 05, mismo eval_steps y patience
- ejecucion-08-mitb3-100epocas-patience20-steps200 — ejecución 08, mismo eval_steps pero patience=20
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
