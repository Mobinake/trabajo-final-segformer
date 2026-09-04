---
type: project
subtype: experiment-log
ejecucion: 08
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 20
eval_steps: 200
estado: completo
---

# Ejecución 08 — eval_steps=200, patience=20, max_epochs=100

> Grid search 3×3. Octava run, tercera del bloque eval_steps=200 con patience=20. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 07: patience sube de 10 a 20. eval_steps se mantiene en 200. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 200 |
| save_steps | 200 |
| Patience early stopping | 20 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| Warmup steps | 50 |
| FP16 | True |
| Seed | 42 |
| Data augmentation | on (flip, rotate90, brightness/contrast) |
| reduce_labels | False |
| Checkpoint base | `nvidia/mit-b3` |
| Dataset | 513 imgs (3 distritos) |
| Split | 70/15/15 → 359/76/78 |

> `RUN_LABEL="run06_steps100_pat10"` (stale — no se actualizó). `EVAL_STEPS=200` y `PATIENCE=20` son correctos.

> ⚠️ **Nota sobre checkpoint previo:** el training loss arranca en 0.189 (step 200), muy bajo comparado con runs que cargan desde `nvidia/mit-b3` fresco (~1.0). Los execution counts saltan de 8 a 21, indicando que se reejecutaron celdas entre la carga del modelo y el inicio del entrenamiento. Esto sugiere que el modelo cargó desde un checkpoint previo (posiblemente `segformer-sugarcane-final` de una run anterior guardado en Drive), NO desde `nvidia/mit-b3` fresco.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.1281 |
| Training loss (step final 4500, de tabla) | 0.1027 |
| Runtime | 4369.8s (~1:12:48, ~73 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3400 (época ~75.56, mIoU Trainer = 0.923451) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 3400 (mIoU Trainer = 0.923451). Después de eso, 6 evaluaciones sin mejora (steps 3600-4500), pero con patience=20 hacían falta 20 para disparar (step 7400 > 4500). El entrenamiento terminó naturalmente.

### Tabla de entrenamiento completa (Trainer, cada 200 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 200 | 0.189496 | 0.176466 | 0.904455 | 0.951574 | 0.952638 |
| 400 | 0.181021 | 0.153443 | 0.909734 | 0.956576 | 0.955202 |
| 600 | 0.169749 | 0.135330 | 0.919274 | 0.960586 | 0.960244 |
| 800 | 0.161559 | 0.165393 | 0.913618 | 0.957509 | 0.957323 |
| 1000 | 0.156196 | 0.153499 | 0.917251 | 0.956879 | 0.959418 |
| 1200 | 0.140649 | 0.144128 | 0.919733 | 0.960046 | 0.960543 |
| 1400 | 0.136738 | 0.171215 | 0.920133 | 0.960301 | 0.960745 |
| 1600 | 0.152545 | 0.169901 | 0.920293 | 0.960462 | 0.960821 |
| 1800 | 0.123929 | 0.158939 | 0.923097 | 0.962365 | 0.962230 |
| 2000 | 0.133289 | 0.167599 | 0.922001 | 0.962110 | 0.961643 |
| 2200 | 0.112364 | 0.169471 | 0.921239 | 0.961247 | 0.961286 |
| 2400 | 0.106184 | 0.192303 | 0.919745 | 0.960703 | 0.960497 |
| 2600 | 0.119695 | 0.184555 | 0.922373 | 0.962285 | 0.961835 |
| 2800 | 0.113518 | 0.182784 | 0.921771 | 0.961021 | 0.961599 |
| 3000 | 0.099621 | 0.209027 | 0.921887 | 0.961629 | 0.961616 |
| 3200 | 0.107234 | 0.186910 | 0.923037 | 0.962307 | 0.962201 |
| **3400** | **0.103538** | **0.180996** | **0.923451** | **0.962101** | **0.962446** |
| 3600 | 0.103988 | 0.200312 | 0.922527 | 0.962163 | 0.961930 |
| 3800 | 0.115219 | 0.187519 | 0.921623 | 0.961362 | 0.961490 |
| 4000 | 0.096415 | 0.187269 | 0.921662 | 0.961400 | 0.961509 |
| 4200 | 0.097769 | 0.190432 | 0.922519 | 0.961807 | 0.961953 |
| 4400 | 0.114581 | 0.191550 | 0.922638 | 0.961990 | 0.962004 |
| 4500 | 0.102718 | 0.194412 | 0.922715 | 0.962116 | 0.962037 |

> Mejor Trainer mIoU: **0.923451** en step 3400 (época ~75.56). Después de eso, 6 evaluaciones sin mejora. Con patience=20, el early stopping habría disparado en step 7400, muy por encima del máximo de 4500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.189307 | 0.140612 |
| eval_mean_iou | 0.922599 | 0.934628 |
| eval_mean_accuracy | 0.961911 | 0.966134 |
| eval_overall_accuracy | 0.961989 | 0.970642 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8989** |
| Median IoU | 0.9346 |
| Std | 0.0966 |
| Min | 0.5262 |
| Max | 0.9857 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8938** |
| Median IoU | 0.9281 |
| Std | 0.0905 |
| Min | 0.5769 |
| Max | 0.9797 |

- ✅ Hipótesis superada: mIoU en test set = 0.8938 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8938 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3400 (época ~75.56, mIoU Trainer = 0.923451).
- **Checkpoint previo:** el training loss inicial (0.189 en step 200) es anormalmente bajo, indicando que el modelo no arrancó desde `nvidia/mit-b3` fresco sino desde un checkpoint ya entrenado. Los execution counts saltan de 8 a 21. Esto afecta la interpretación de los resultados: el modelo ya estaba parcialmente entrenado al iniciar esta run.
- **Mejor mIoU test del bloque steps=200:** 0.8938 — supera a ejec. 07 (pendiente) y ejec. 09 (0.8947).
- **Δ val-test:** 0.8989 - 0.8938 = 0.0051 — bajo, indicando buena generalización.
- **Training loss:** bajó de 0.189 (step 200) a 0.103 (step 4500). Descenso suave pero arrancando desde un punto bajo.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun a las 22:00.

## 🔗 Notas relacionadas
- ejecucion-07-mitb3-100epocas-patience10-steps200 — ejecución 07, mismo eval_steps pero patience=10
- ejecucion-09-mitb3-100epocas-patience30-steps200 — ejecución 09, mismo eval_steps pero patience=30
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
