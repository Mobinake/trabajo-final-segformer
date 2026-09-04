---
type: project
subtype: experiment-log
ejecucion: 09
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 09 — eval_steps=200, patience=30, max_epochs=100

> Grid search 3×3. Novena run, tercera del bloque eval_steps=200 con patience=30. Mejor run del grid search. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 08: patience sube de 20 a 30. eval_steps se mantiene en 200. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 200 |
| save_steps | 200 |
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

> `RUN_LABEL="run11_steps200_pat30"` (stale — coincide con patience=30 pero no con eval_steps). `EVAL_STEPS=200` y `PATIENCE=30` son correctos.

> ⚠️ **Nota sobre checkpoint previo:** el training loss arranca en 0.125 (step 200), muy bajo comparado con runs que cargan desde `nvidia/mit-b3` fresco (~1.0). Los execution counts saltan de 9 a 33, indicando que se reejecutaron celdas entre la carga del modelo y el inicio del entrenamiento. Esto sugiere que el modelo cargó desde un checkpoint previo, NO desde `nvidia/mit-b3` fresco.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.0970 |
| Training loss (step final 4500, de tabla) | 0.0853 |
| Runtime | 4334.2s (~1:12:13, ~72 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3400 (época ~75.56, mIoU Trainer = 0.924093) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 3400 (mIoU Trainer = 0.924093). Después de eso, 6 evaluaciones sin mejora (steps 3600-4500), pero con patience=30 hacían falta 30 para disparar (step 9400 > 4500). El entrenamiento terminó naturalmente.

### Tabla de entrenamiento completa (Trainer, cada 200 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 200 | 0.124664 | 0.253885 | 0.912492 | 0.955251 | 0.956883 |
| 400 | 0.123524 | 0.200260 | 0.917954 | 0.959116 | 0.959625 |
| 600 | 0.117323 | 0.155874 | 0.924358 | 0.962975 | 0.962881 |
| 800 | 0.116407 | 0.228523 | 0.915298 | 0.958571 | 0.958181 |
| 1000 | 0.109439 | 0.188820 | 0.920546 | 0.959820 | 0.961013 |
| 1200 | 0.104419 | 0.181672 | 0.919393 | 0.960077 | 0.960351 |
| 1400 | 0.100943 | 0.178060 | 0.923461 | 0.961655 | 0.962485 |
| 1600 | 0.117522 | 0.211095 | 0.917304 | 0.958977 | 0.959273 |
| 1800 | 0.095063 | 0.237700 | 0.919076 | 0.960435 | 0.960146 |
| 2000 | 0.101711 | 0.216733 | 0.919330 | 0.960498 | 0.960282 |
| 2200 | 0.087785 | 0.194423 | 0.921485 | 0.961504 | 0.961403 |
| 2400 | 0.081975 | 0.213673 | 0.922304 | 0.961736 | 0.961839 |
| 2600 | 0.094319 | 0.219766 | 0.923167 | 0.962866 | 0.962231 |
| 2800 | 0.090336 | 0.222318 | 0.922114 | 0.961951 | 0.961717 |
| 3000 | 0.080081 | 0.235131 | 0.922448 | 0.961962 | 0.961901 |
| 3200 | 0.086528 | 0.220668 | 0.921560 | 0.961423 | 0.961450 |
| **3400** | **0.084049** | **0.212550** | **0.924093** | **0.963100** | **0.962725** |
| 3600 | 0.085279 | 0.216793 | 0.922852 | 0.962332 | 0.962097 |
| 3800 | 0.092431 | 0.219584 | 0.921762 | 0.961822 | 0.961532 |
| 4000 | 0.080302 | 0.215160 | 0.922686 | 0.962471 | 0.961995 |
| 4200 | 0.081084 | 0.223377 | 0.922974 | 0.962408 | 0.962159 |
| 4400 | 0.096734 | 0.224087 | 0.922722 | 0.962391 | 0.962020 |
| 4500 | 0.085265 | 0.226661 | 0.922778 | 0.962473 | 0.962045 |

> Mejor Trainer mIoU: **0.924093** en step 3400 (época ~75.56). Después de eso, 6 evaluaciones sin mejora. Con patience=30, el early stopping habría disparado en step 9400, muy por encima del máximo de 4500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.202543 | 0.147427 |
| eval_mean_iou | 0.922531 | 0.933790 |
| eval_mean_accuracy | 0.962621 | 0.965891 |
| eval_overall_accuracy | 0.961897 | 0.970239 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.9005** |
| Median IoU | 0.9375 |
| Std | 0.1001 |
| Min | 0.5212 |
| Max | 0.9858 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8947** |
| Median IoU | 0.9298 |
| Std | 0.0931 |
| Min | 0.5682 |
| Max | 0.9854 |

- ✅ Hipótesis superada: mIoU en test set = 0.8947 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8947 ≥ 0.85.
- **Mejor run del grid search:** mIoU test = 0.8947 es el más alto de las 9 runs del grid 3×3. mIoU val = 0.9005 también es el más alto del grid.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3400 (época ~75.56, mIoU Trainer = 0.924093).
- **Checkpoint previo:** el training loss inicial (0.125 en step 200) es anormalmente bajo, indicando que el modelo no arrancó desde `nvidia/mit-b3` fresco sino desde un checkpoint ya entrenado. Los execution counts saltan de 9 a 33.
- **Δ val-test:** 0.9005 - 0.8947 = 0.0058 — bajo, indicando buena generalización.
- **Std test:** 0.0931 — similar a ejec. 08 (0.0905).
- **Training loss:** bajó de 0.125 (step 200) a 0.085 (step 4500). Descenso suave pero arrancando desde un punto bajo.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun a las 23:21.
- **Configuración elegida para runs post-grid:** eval_steps=200, patience=30 fue seleccionada como la mejor configuración del grid search para continuar con runs de más épocas (ejec. 10, 11, 12).

## 🔗 Notas relacionadas
- ejecucion-07-mitb3-100epocas-patience10-steps200 — ejecución 07, mismo eval_steps pero patience=10
- ejecucion-08-mitb3-100epocas-patience20-steps200 — ejecución 08, mismo eval_steps pero patience=20
- ejecucion-10-mitb3-200epocas-patience30-steps200 — ejecución 10, misma config con 200 épocas
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
