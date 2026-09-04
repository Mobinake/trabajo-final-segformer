---
type: project
subtype: experiment-log
ejecucion: 04
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 20
eval_steps: 100
estado: completo
---

# Ejecución 04 — eval_steps=100, patience=20, max_epochs=100

> Grid search 3×3. Cuarta run, primera del bloque eval_steps=100. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 05: eval_steps sube de 50 a 100. patience sube de 10 a 20. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 100 |
| save_steps | 100 |
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

> `RUN_LABEL="run06_steps100_pat10"` (stale — no se actualizó). `EVAL_STEPS=100` y `PATIENCE=20` son correctos. El `RUN_LABEL` solo afecta el nombre del `output_dir`.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.3393 |
| Training loss (step final 4500, de tabla) | 0.1500 |
| Runtime | 5025.3s (~1:23:40, ~84 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3600 (época ~80.00, mIoU Trainer = 0.928121) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 3600 (mIoU Trainer = 0.928121). Después de eso, 9 evaluaciones sin mejora (steps 3700-4500), pero con patience=20 hacían falta 20 para disparar (step 7600 > 4500). El entrenamiento terminó naturalmente.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled).

### Tabla de entrenamiento completa (Trainer, cada 100 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 100 | 1.401075 | 0.404988 | 0.657430 | 0.784736 | 0.807685 |
| 200 | 1.047218 | 0.340153 | 0.737761 | 0.864994 | 0.852775 |
| 300 | 0.942921 | 0.246693 | 0.796809 | 0.885867 | 0.893377 |
| 400 | 0.785543 | 0.237930 | 0.817210 | 0.897337 | 0.905473 |
| 500 | 0.656938 | 0.178530 | 0.859333 | 0.927701 | 0.928206 |
| 600 | 0.578131 | 0.181809 | 0.858498 | 0.922757 | 0.928420 |
| 700 | 0.542320 | 0.156588 | 0.873550 | 0.933783 | 0.936263 |
| 800 | 0.439411 | 0.160696 | 0.877843 | 0.936967 | 0.938485 |
| 900 | 0.396357 | 0.156922 | 0.877390 | 0.936296 | 0.938293 |
| 1000 | 0.390299 | 0.158744 | 0.885428 | 0.940760 | 0.942613 |
| 1100 | 0.332663 | 0.150497 | 0.889905 | 0.942846 | 0.945045 |
| 1200 | 0.339122 | 0.142409 | 0.891258 | 0.943266 | 0.945800 |
| 1300 | 0.280933 | 0.159532 | 0.892401 | 0.945039 | 0.946280 |
| 1400 | 0.277306 | 0.125665 | 0.902835 | 0.949963 | 0.951861 |
| 1500 | 0.256960 | 0.122727 | 0.906351 | 0.952696 | 0.953620 |
| 1600 | 0.289166 | 0.129906 | 0.903875 | 0.950988 | 0.952360 |
| 1700 | 0.261468 | 0.144916 | 0.903737 | 0.951504 | 0.952231 |
| 1800 | 0.224387 | 0.130788 | 0.909799 | 0.954540 | 0.955419 |
| 1900 | 0.225716 | 0.141404 | 0.907578 | 0.953592 | 0.954239 |
| 2000 | 0.232731 | 0.124409 | 0.910586 | 0.955354 | 0.955794 |
| 2100 | 0.221732 | 0.113333 | 0.917902 | 0.958597 | 0.959638 |
| 2200 | 0.186939 | 0.113446 | 0.917782 | 0.958299 | 0.959595 |
| 2300 | 0.189288 | 0.132728 | 0.914651 | 0.957585 | 0.957899 |
| 2400 | 0.178302 | 0.124248 | 0.914761 | 0.956197 | 0.958079 |
| 2500 | 0.198624 | 0.119372 | 0.920150 | 0.960177 | 0.960764 |
| 2600 | 0.199681 | 0.125427 | 0.917954 | 0.958906 | 0.959642 |
| 2700 | 0.172038 | 0.117223 | 0.920180 | 0.960506 | 0.960755 |
| 2800 | 0.187378 | 0.117502 | 0.920611 | 0.960605 | 0.960987 |
| 2900 | 0.181017 | 0.114231 | 0.922554 | 0.960382 | 0.962082 |
| 3000 | 0.174355 | 0.127407 | 0.920033 | 0.959832 | 0.960727 |
| 3100 | 0.171501 | 0.117689 | 0.923092 | 0.960793 | 0.962347 |
| 3200 | 0.176084 | 0.114423 | 0.925925 | 0.962518 | 0.963777 |
| 3300 | 0.158345 | 0.113057 | 0.925964 | 0.962989 | 0.963763 |
| 3400 | 0.162198 | 0.114948 | 0.925503 | 0.962639 | 0.963536 |
| 3500 | 0.191769 | 0.117050 | 0.924375 | 0.962308 | 0.962940 |
| **3600** | **0.160240** | **0.109181** | **0.928121** | **0.962926** | **0.964950** |
| 3700 | 0.158387 | 0.118490 | 0.924405 | 0.961829 | 0.962992 |
| 3800 | 0.167070 | 0.118352 | 0.924707 | 0.961787 | 0.963162 |
| 3900 | 0.157092 | 0.116698 | 0.925466 | 0.962175 | 0.963550 |
| 4000 | 0.141492 | 0.116171 | 0.926381 | 0.962769 | 0.964009 |
| 4100 | 0.169234 | 0.118341 | 0.925683 | 0.962564 | 0.963640 |
| 4200 | 0.137244 | 0.120057 | 0.925601 | 0.962505 | 0.963600 |
| 4300 | 0.162533 | 0.120867 | 0.924952 | 0.962250 | 0.963262 |
| 4400 | 0.146268 | 0.122170 | 0.923478 | 0.961830 | 0.962481 |
| 4500 | 0.150006 | 0.122167 | 0.924684 | 0.962421 | 0.963102 |

> Mejor Trainer mIoU: **0.928121** en step 3600 (época ~80.00). Después de eso, 9 evaluaciones consecutivas (steps 3700-4500) sin mejora. Con patience=20, el early stopping habría disparado en step 7600, muy por encima del máximo de 4500 pasos.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.118332 | 0.118099 |
| eval_mean_iou | 0.924825 | 0.927602 |
| eval_mean_accuracy | 0.962021 | 0.962879 |
| eval_overall_accuracy | 0.963209 | 0.967315 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8958** |
| Median IoU | 0.9301 |
| Std | 0.0856 |
| Min | 0.5281 |
| Max | 0.9775 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8804** |
| Median IoU | 0.9179 |
| Std | 0.0993 |
| Min | 0.4866 |
| Max | 0.9765 |

- ✅ Hipótesis superada: mIoU en test set = 0.8804 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8804 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3600 (época ~80.00, mIoU Trainer = 0.928121). Después de eso, 9 evaluaciones sin mejora, pero patience=20 requería 20 (step 7600 > 4500).
- **Δ val-test:** 0.8958 - 0.8804 = 0.0154 — el más alto del grid hasta este punto, sugiriendo ligero overfitting al val set con patience=20.
- **Training loss:** bajó de 1.401 (step 100) a 0.150 (step 4500). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun a las 06:06.

## 🔗 Notas relacionadas
- ejecucion-03-mitb3-100epocas-patience30-steps50 — ejecución 03, mismo max_epochs pero eval_steps=50
- ejecucion-05-mitb3-100epocas-patience10-steps200 — ejecución 05, eval_steps=200, patience=10
- grid-search-steps-patience — script y grilla completa del grid search
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
