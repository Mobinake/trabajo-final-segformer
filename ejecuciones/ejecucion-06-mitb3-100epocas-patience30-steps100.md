---
type: project
subtype: experiment-log
ejecucion: 06
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 30
eval_steps: 100
estado: completo
---

# Ejecución 06 — eval_steps=100, patience=30, max_epochs=100

> Grid search 3×3. Sexta run, segunda del bloque eval_steps=100. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 05: patience sube de 20 a 30. eval_steps se mantiene en 100. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 100 |
| save_steps | 100 |
| Patience early stopping | 30 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| Warmup steps | 50 |
| FP16 | True |
| Seed | 42 |
| Data augmentation | on (flip, rotate90, brightness/contrast) |
| reduce_labels | False |
| Checkpoint base | `nvidia/mit-b3` |
| Dataset | 513 imgs (3 distritos) |
| Split | 70/15/15 → 359/76/78 |

> `RUN_LABEL="run06_steps100_pat10"` (stale — no se actualizó). `EVAL_STEPS=100` y `PATIENCE=30` son correctos.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.3361 |
| Training loss (step final 4500, de tabla) | 0.1576 |
| Runtime | 4805.7s (~1:19:53, ~80 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 4200 (época ~93.33, mIoU Trainer = 0.928233) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 4200 (mIoU Trainer = 0.928233). Después de eso, 2 evaluaciones sin mejora (steps 4400, 4500), pero con patience=30 hacían falta 30 para disparar (step 10200 > 4500). El entrenamiento terminó naturalmente.

### Tabla de entrenamiento completa (Trainer, cada 100 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 100 | 1.360998 | 0.364799 | 0.700826 | 0.820445 | 0.834358 |
| 200 | 1.039007 | 0.293182 | 0.755841 | 0.867300 | 0.866550 |
| 300 | 0.970621 | 0.240382 | 0.807136 | 0.899743 | 0.897848 |
| 400 | 0.765083 | 0.203944 | 0.844670 | 0.916008 | 0.920560 |
| 500 | 0.569517 | 0.171312 | 0.865777 | 0.929601 | 0.932003 |
| 600 | 0.550119 | 0.169459 | 0.869860 | 0.934551 | 0.933881 |
| 700 | 0.501803 | 0.180832 | 0.878215 | 0.938880 | 0.938476 |
| 800 | 0.451711 | 0.144598 | 0.887834 | 0.941996 | 0.943909 |
| 900 | 0.401113 | 0.143686 | 0.893366 | 0.945742 | 0.946773 |
| 1000 | 0.367407 | 0.137774 | 0.894641 | 0.944723 | 0.947634 |
| 1100 | 0.378039 | 0.122164 | 0.902544 | 0.949109 | 0.951779 |
| 1200 | 0.344767 | 0.158191 | 0.894868 | 0.946584 | 0.947567 |
| 1300 | 0.278902 | 0.154031 | 0.904306 | 0.953021 | 0.952415 |
| 1400 | 0.263813 | 0.141803 | 0.906442 | 0.952600 | 0.953681 |
| 1500 | 0.260560 | 0.124632 | 0.910223 | 0.953239 | 0.955779 |
| 1600 | 0.281720 | 0.124214 | 0.911453 | 0.955428 | 0.956279 |
| 1700 | 0.271427 | 0.106577 | 0.917335 | 0.957591 | 0.959405 |
| 1800 | 0.236836 | 0.124453 | 0.912564 | 0.954914 | 0.956953 |
| 1900 | 0.199982 | 0.123487 | 0.915883 | 0.957785 | 0.958575 |
| 2000 | 0.235739 | 0.123748 | 0.915574 | 0.958328 | 0.958356 |
| 2100 | 0.222867 | 0.107549 | 0.921525 | 0.960003 | 0.961542 |
| 2200 | 0.199435 | 0.116557 | 0.917450 | 0.958589 | 0.959386 |
| 2300 | 0.177632 | 0.125847 | 0.916223 | 0.956756 | 0.958853 |
| 2400 | 0.180216 | 0.116078 | 0.920006 | 0.958343 | 0.960832 |
| 2500 | 0.207355 | 0.120179 | 0.920433 | 0.959158 | 0.961003 |
| 2600 | 0.203988 | 0.123189 | 0.920185 | 0.959994 | 0.960798 |
| 2700 | 0.165589 | 0.134189 | 0.920518 | 0.960652 | 0.960932 |
| 2800 | 0.170659 | 0.116222 | 0.923295 | 0.961255 | 0.962424 |
| 2900 | 0.188596 | 0.106884 | 0.926323 | 0.962417 | 0.964002 |
| 3000 | 0.176340 | 0.119426 | 0.922662 | 0.960509 | 0.962131 |
| 3100 | 0.185796 | 0.124080 | 0.922497 | 0.960902 | 0.962010 |
| 3200 | 0.163947 | 0.116152 | 0.925240 | 0.962613 | 0.963393 |
| 3300 | 0.162661 | 0.115687 | 0.925740 | 0.962718 | 0.963660 |
| 3400 | 0.163910 | 0.112601 | 0.926113 | 0.962648 | 0.963870 |
| 3500 | 0.184388 | 0.111284 | 0.926869 | 0.963822 | 0.964200 |
| 3600 | 0.157274 | 0.116531 | 0.926651 | 0.962996 | 0.964140 |
| 3700 | 0.146563 | 0.118546 | 0.926424 | 0.962977 | 0.964017 |
| 3800 | 0.164612 | 0.113394 | 0.927163 | 0.963419 | 0.964389 |
| 3900 | 0.161219 | 0.114298 | 0.926896 | 0.963027 | 0.964272 |
| 4000 | 0.142865 | 0.113125 | 0.927739 | 0.963475 | 0.964701 |
| **4200** | **0.141577** | **0.115158** | **0.928233** | **0.963952** | **0.964937** |
| 4300 | 0.158635 | 0.116072 | 0.927883 | 0.963861 | 0.964752 |
| 4400 | 0.156416 | 0.114109 | 0.927842 | 0.963682 | 0.964742 |
| 4500 | 0.157576 | 0.115309 | 0.928020 | 0.963674 | 0.964840 |

> Mejor Trainer mIoU: **0.928233** en step 4200 (época ~93.33). Después de eso, 2 evaluaciones sin mejora. Con patience=30, el early stopping habría disparado en step 10200, muy por encima del máximo de 4500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.116517 | 0.107208 |
| eval_mean_iou | 0.927876 | 0.934467 |
| eval_mean_accuracy | 0.963632 | 0.966571 |
| eval_overall_accuracy | 0.964765 | 0.970535 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8998** |
| Median IoU | 0.9283 |
| Std | 0.0813 |
| Min | 0.5533 |
| Max | 0.9807 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8897** |
| Median IoU | 0.9230 |
| Std | 0.0854 |
| Min | 0.6211 |
| Max | 0.9784 |

- ✅ Hipótesis superada: mIoU en test set = 0.8897 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8897 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 4200 (época ~93.33, mIoU Trainer = 0.928233).
- **Mejor mIoU test del bloque steps=100:** 0.8897 — supera a ejec. 04 (0.8804) por +0.0093.
- **Δ val-test:** 0.8998 - 0.8897 = 0.0101 — menor que ejec. 04 (0.0154), indicando mejor generalización.
- **Min IoU test:** 0.6211 — el mejor peor-caso del grid hasta este punto (ejec. 04: 0.4866).
- **Training loss:** bajó de 1.361 (step 100) a 0.158 (step 4500). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun a las 07:40.

## 🔗 Notas relacionadas
- ejecucion-04-mitb3-100epocas-patience20-steps100 — ejecución 04, mismo eval_steps pero patience=20
- ejecucion-05-mitb3-100epocas-patience10-steps200 — ejecución 05, eval_steps=200, patience=10
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
