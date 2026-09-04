---
type: project
subtype: experiment-log
ejecucion: 05
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 10
eval_steps: 200
estado: completo
---

# Ejecución 05 — eval_steps=200, patience=10, max_epochs=100

> Grid search 3×3. Quinta run, primera del bloque eval_steps=200. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 04: eval_steps sube de 100 a 200. patience vuelve a 10. num_train_epochs se mantiene en 100.

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
| Training loss (TrainOutput, promedio) | 0.3421 |
| Training loss (step final 4500, de tabla) | 0.1579 |
| Runtime | 4740.7s (~1:18:42, ~79 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 4000 (época ~88.89, mIoU Trainer = 0.928746) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 4000 (mIoU Trainer = 0.928746). Después de eso, 3 evaluaciones sin mejora (steps 4200, 4400, 4500), pero con patience=10 hacían falta 10 para disparar (step 6000 > 4500). El entrenamiento terminó naturalmente.

### Tabla de entrenamiento completa (Trainer, cada 200 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 200 | 1.039879 | 0.324845 | 0.746149 | 0.869359 | 0.858464 |
| 400 | 0.767392 | 0.212414 | 0.823261 | 0.896808 | 0.909854 |
| 600 | 0.654807 | 0.181260 | 0.861555 | 0.930040 | 0.929285 |
| 800 | 0.479618 | 0.152259 | 0.879086 | 0.937263 | 0.939206 |
| 1000 | 0.381721 | 0.153157 | 0.886865 | 0.943947 | 0.943109 |
| 1200 | 0.301724 | 0.140248 | 0.892663 | 0.945558 | 0.946378 |
| 1400 | 0.298148 | 0.137964 | 0.901228 | 0.949801 | 0.950948 |
| 1600 | 0.299099 | 0.140346 | 0.902611 | 0.951107 | 0.951619 |
| 1800 | 0.231258 | 0.130189 | 0.909973 | 0.954487 | 0.955523 |
| 2000 | 0.238623 | 0.120012 | 0.915952 | 0.957090 | 0.958672 |
| 2200 | 0.203200 | 0.112549 | 0.919159 | 0.958805 | 0.960323 |
| 2400 | 0.182762 | 0.116525 | 0.919916 | 0.960369 | 0.960619 |
| 2600 | 0.219836 | 0.113233 | 0.922894 | 0.961059 | 0.962217 |
| 2800 | 0.189158 | 0.109740 | 0.924035 | 0.962172 | 0.962763 |
| 3000 | 0.162316 | 0.118121 | 0.924128 | 0.961341 | 0.962876 |
| 3200 | 0.172601 | 0.115323 | 0.925294 | 0.961798 | 0.963484 |
| 3400 | 0.161257 | 0.112683 | 0.925289 | 0.962499 | 0.963429 |
| 3600 | 0.161117 | 0.119359 | 0.925242 | 0.961971 | 0.963442 |
| 3800 | 0.160528 | 0.123602 | 0.924544 | 0.962559 | 0.963014 |
| **4000** | **0.147434** | **0.109582** | **0.928746** | **0.964172** | **0.965202** |
| 4200 | 0.141148 | 0.118575 | 0.926953 | 0.963382 | 0.964277 |
| 4400 | 0.150444 | 0.117551 | 0.926865 | 0.963339 | 0.964232 |
| 4500 | 0.157899 | 0.119514 | 0.927371 | 0.963627 | 0.964489 |

> Mejor Trainer mIoU: **0.928746** en step 4000 (época ~88.89). Después de eso, 3 evaluaciones sin mejora. Con patience=10, el early stopping habría disparado en step 6000, muy por encima del máximo de 4500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.116112 | 0.123264 |
| eval_mean_iou | 0.927411 | 0.926339 |
| eval_mean_accuracy | 0.963433 | 0.961510 |
| eval_overall_accuracy | 0.964524 | 0.966768 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8995** |
| Median IoU | 0.9239 |
| Std | 0.0784 |
| Min | 0.5575 |
| Max | 0.9796 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8837** |
| Median IoU | 0.9134 |
| Std | 0.0863 |
| Min | 0.5954 |
| Max | 0.9781 |

- ✅ Hipótesis superada: mIoU en test set = 0.8837 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8837 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 4000 (época ~88.89, mIoU Trainer = 0.928746).
- **mIoU val más alto del grid hasta este punto:** 0.8995 — supera a ejec. 04 (0.8958), ejec. 03 (0.8941), ejec. 02 (0.8927), ejec. 01 (0.8942).
- **Std val más bajo del grid:** 0.0784 — indica buena consistencia entre imágenes de validación.
- **Δ val-test:** 0.8995 - 0.8837 = 0.0158 — alto, sugiriendo ligero overfitting al val set.
- **Training loss:** bajó de 1.040 (step 200) a 0.158 (step 4500). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 3 jul a las 22:47.

## 🔗 Notas relacionadas
- ejecucion-04-mitb3-100epocas-patience20-steps100 — ejecución 04, eval_steps=100, patience=20
- ejecucion-06-mitb3-100epocas-patience30-steps100 — ejecución 06, eval_steps=100, patience=30
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
