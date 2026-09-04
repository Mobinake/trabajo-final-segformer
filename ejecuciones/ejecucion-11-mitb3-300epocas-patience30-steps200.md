---
type: project
subtype: experiment-log
ejecucion: 11
fecha: 2026-07-27
modelo: SegFormer MiT-B3
epocas_max: 300
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 11 — eval_steps=200, patience=30, max_epochs=300

> Post-grid. Toma la mejor config del grid (ejec. 09: eval_steps=200, patience=30) y sube épocas máx a 300. Fecha: 27 de julio de 2026. Notebook ejecutado en Google Colab (GPU T4).

> **Nota sobre RUN_LABEL:** el script interno usa `RUN_LABEL="run10_steps200_pat30_epochs300"` pero corresponde a la ejecución 11 (300 épocas). El label no fue actualizado pero los parámetros confirman que es esta run.

## Parámetros

> Únicos cambios respecto a ejec. 10: `num_train_epochs` sube de 200 a 300. Resto igual.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 300 |
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

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0,1797 |
| Training loss (step final 13500, de tabla) | 0,0838 |
| Runtime | 13607,4s (~3:46:47, ~227 min) |
| Épocas ejecutadas | 300 |
| Pasos totales | 13.500 |
| Early stopping activado | No (corrió las 300 épocas completas) |
| Mejor step (Trainer mIoU) | 11200 (época ~248,9, mIoU Trainer = 0,937146) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 300 épocas = 13.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 11200 (mIoU Trainer = 0,937146). Después de eso, 12 evaluaciones sin mejora (steps 11400-13500), pero con patience=30 hacían falta 30 para disparar (step 17200 > 13500). El entrenamiento terminó naturalmente al completar las 300 épocas.

### Tabla de entrenamiento completa (Trainer, cada 200 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 200 | 1,075254 | 0,294958 | 0,764687 | 0,879850 | 0,870604 |
| 400 | 0,722444 | 0,228046 | 0,822107 | 0,900857 | 0,908154 |
| 600 | 0,518632 | 0,208694 | 0,855687 | 0,920305 | 0,927013 |
| 800 | 0,492651 | 0,168043 | 0,879803 | 0,936500 | 0,939738 |
| 1000 | 0,365521 | 0,137790 | 0,897929 | 0,948363 | 0,949174 |
| 1200 | 0,336403 | 0,166703 | 0,891472 | 0,943577 | 0,945892 |
| 1400 | 0,294942 | 0,139125 | 0,902704 | 0,950357 | 0,951747 |
| 1600 | 0,277922 | 0,164485 | 0,898169 | 0,948729 | 0,949277 |
| 1800 | 0,243825 | 0,152971 | 0,909987 | 0,955171 | 0,955470 |
| 2000 | 0,253415 | 0,154733 | 0,912014 | 0,955845 | 0,956560 |
| 2200 | 0,201891 | 0,124940 | 0,915711 | 0,956526 | 0,958585 |
| 2400 | 0,185592 | 0,112544 | 0,922443 | 0,960724 | 0,961993 |
| 2600 | 0,209822 | 0,127790 | 0,921088 | 0,960010 | 0,961299 |
| 2800 | 0,183592 | 0,137570 | 0,920113 | 0,960488 | 0,960719 |
| 3000 | 0,164819 | 0,130600 | 0,922232 | 0,961457 | 0,961820 |
| 3200 | 0,174695 | 0,140559 | 0,921776 | 0,961024 | 0,961601 |
| 3400 | 0,150985 | 0,132188 | 0,925436 | 0,962292 | 0,963525 |
| 3600 | 0,153341 | 0,150189 | 0,922491 | 0,961637 | 0,961950 |
| 3800 | 0,153158 | 0,129479 | 0,928844 | 0,965085 | 0,965190 |
| 4000 | 0,138337 | 0,124647 | 0,929719 | 0,964999 | 0,965674 |
| 4200 | 0,130570 | 0,139892 | 0,926781 | 0,963815 | 0,964152 |
| 4400 | 0,134989 | 0,127991 | 0,927893 | 0,963395 | 0,964791 |
| 4600 | 0,141947 | 0,145105 | 0,925721 | 0,962166 | 0,963691 |
| 4800 | 0,130891 | 0,163644 | 0,924931 | 0,962850 | 0,963206 |
| 5000 | 0,122982 | 0,152864 | 0,928241 | 0,964918 | 0,964873 |
| 5200 | 0,124010 | 0,175990 | 0,923398 | 0,961937 | 0,962428 |
| 5400 | 0,118295 | 0,145850 | 0,928636 | 0,964554 | 0,965115 |
| 5600 | 0,116932 | 0,126137 | 0,932225 | 0,965959 | 0,966970 |
| 5800 | 0,109398 | 0,132468 | 0,931907 | 0,965987 | 0,966795 |
| 6000 | 0,109399 | 0,125371 | 0,933278 | 0,966205 | 0,967523 |
| 6200 | 0,130117 | 0,136981 | 0,930953 | 0,965324 | 0,966322 |
| 6400 | 0,110633 | 0,131654 | 0,932482 | 0,965929 | 0,967111 |
| 6600 | 0,112011 | 0,129141 | 0,935007 | 0,967007 | 0,968403 |
| 6800 | 0,106658 | 0,140767 | 0,930706 | 0,965944 | 0,966146 |
| 7000 | 0,109074 | 0,124782 | 0,934894 | 0,967150 | 0,968333 |
| 7200 | 0,106429 | 0,142292 | 0,932324 | 0,966532 | 0,966985 |
| 7400 | 0,113882 | 0,124387 | 0,936873 | 0,967566 | 0,969369 |
| 7600 | 0,095902 | 0,127638 | 0,935282 | 0,967143 | 0,968542 |
| 7800 | 0,115985 | 0,133451 | 0,933445 | 0,966058 | 0,967623 |
| 8000 | 0,107538 | 0,134939 | 0,933942 | 0,966828 | 0,967840 |
| 8200 | 0,098070 | 0,131123 | 0,934934 | 0,967025 | 0,968362 |
| 8400 | 0,110033 | 0,137772 | 0,934661 | 0,966916 | 0,968222 |
| 8600 | 0,102270 | 0,140069 | 0,933157 | 0,966065 | 0,967467 |
| 8800 | 0,092892 | 0,138283 | 0,933754 | 0,966786 | 0,967742 |
| 9000 | 0,104317 | 0,144393 | 0,933299 | 0,966695 | 0,967502 |
| 9200 | 0,094685 | 0,153755 | 0,931972 | 0,966253 | 0,966813 |
| 9400 | 0,092896 | 0,136302 | 0,935801 | 0,967609 | 0,968791 |
| 9600 | 0,097686 | 0,138714 | 0,934853 | 0,966655 | 0,968342 |
| 9800 | 0,105457 | 0,156398 | 0,932061 | 0,966216 | 0,966863 |
| 10000 | 0,096390 | 0,147823 | 0,933052 | 0,966390 | 0,967389 |
| 10200 | 0,091882 | 0,147674 | 0,932569 | 0,966225 | 0,967138 |
| 10400 | 0,093451 | 0,151089 | 0,933710 | 0,967321 | 0,967683 |
| 10600 | 0,087500 | 0,143788 | 0,934528 | 0,967444 | 0,968116 |
| 10800 | 0,088298 | 0,140239 | 0,935324 | 0,967769 | 0,968524 |
| 11000 | 0,088146 | 0,140639 | 0,936299 | 0,968153 | 0,969024 |
| **11200** | **0,093747** | **0,136593** | **0,937146** | **0,968213** | **0,969474** |
| 11400 | 0,075941 | 0,150659 | 0,934433 | 0,967333 | 0,968072 |
| 11600 | 0,088430 | 0,140903 | 0,936813 | 0,968304 | 0,969290 |
| 11800 | 0,078679 | 0,143103 | 0,935715 | 0,967988 | 0,968720 |
| 12000 | 0,087013 | 0,140889 | 0,936518 | 0,968207 | 0,969138 |
| 12200 | 0,083663 | 0,142512 | 0,936225 | 0,968101 | 0,968988 |
| 12400 | 0,078767 | 0,149313 | 0,935405 | 0,967917 | 0,968559 |
| 12600 | 0,087410 | 0,149281 | 0,934687 | 0,967495 | 0,968199 |
| 12800 | 0,085152 | 0,145510 | 0,935506 | 0,967808 | 0,968619 |
| 13000 | 0,089198 | 0,144529 | 0,935953 | 0,968058 | 0,968844 |
| 13200 | 0,081314 | 0,144612 | 0,935422 | 0,967665 | 0,968584 |
| 13400 | 0,087472 | 0,145519 | 0,935889 | 0,967831 | 0,968824 |
| 13500 | 0,083801 | 0,143724 | 0,935750 | 0,967858 | 0,968748 |

> Mejor Trainer mIoU: **0,937146** en step 11200 (época ~248,9). Después de eso, 12 evaluaciones sin mejora (steps 11400-13500). Con patience=30, el early stopping habría disparado en step 17200, muy por encima del máximo de 13500.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0,141830 | 0,143611 |
| eval_mean_iou | 0,935675 | 0,940935 |
| eval_mean_accuracy | 0,967721 | 0,969010 |
| eval_overall_accuracy | 0,968716 | 0,973600 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0,9131** |
| Median IoU | 0,9401 |
| Std | 0,0769 |
| Min | 0,5705 |
| Max | 0,9883 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0,9050** |
| Median IoU | 0,9333 |
| Std | 0,0838 |
| Min | 0,5686 |
| Max | 0,9813 |

- ✅ Hipótesis superada: mIoU en test set = 0,9050 ≥ 0,85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0,9050 ≥ 0,85.
- **Sin early stopping:** el modelo corrió las 300 épocas completas (13.500 pasos). El mejor checkpoint fue step 11200 (época ~248,9, mIoU Trainer = 0,937146).
- **mIoU val:** 0,9131 — el más alto del post-grid hasta este punto.
- **mIoU test:** 0,9050 — supera a ejec. 10 (0,8992) por +0,0058.
- **Std test:** 0,0838 — menor que ejec. 10 (0,0908).
- **Δ val-test:** 0,9131 - 0,9050 = 0,0081 — bajo, indicando buena generalización.
- **Runtime:** ~227 min (~3h47m) — significativamente más largo que ejec. 10 (~145 min) por las 100 épocas adicionales.
- **Training loss:** bajó de 1,075 (step 200) a 0,084 (step 13500). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 27 jul a las 23:34.

## Comparación con ejecuciones 10 y 12

| Métrica | Ejec. 09 (100ep) | Ejec. 10 (200ep) | **Ejec. 11 (300ep)** | Ejec. 12 (500ep) |
|---|---|---|---|---|
| Épocas máx | 100 | 200 | 300 | 500 |
| Épocas ejecutadas | 100 | 200 | 300 | 284,44 |
| Early stopping | No | No | No | Sí (step 12800) |
| Pasos totales | 4.500 | 9.000 | 13.500 | 12.800 |
| Runtime | ~72 min | ~145 min | ~227 min | ~232 min |
| **mIoU val** | 0,9005 | 0,9052 | **0,9131** | 0,9061 |
| **mIoU test** | 0,8947 | 0,8992 | **0,9050** | 0,9078 |
| Δ val-test | 0,0058 | 0,0060 | 0,0081 | -0,0017 |
| Hipótesis (≥0,85) | ✅ | ✅ | ✅ | ✅ |

## 🔗 Notas relacionadas
- ejecucion-09-mitb3-100epocas-patience30-steps200 — ejecución 09, misma config con 100 épocas (mejor run del grid)
- ejecucion-10-mitb3-200epocas-patience30-steps200 — ejecución 10, misma config con 200 épocas
- ejecucion-12-mitb3-500epocas-patience30-steps200 — ejecución 12, misma config con 500 épocas (mejor ejecución, early stopping)
- README — índice de todas las ejecuciones
- 04 - Resultados y Validación — consolidación final para el manuscrito
- 08 - Configuración Técnica — parámetros del script
