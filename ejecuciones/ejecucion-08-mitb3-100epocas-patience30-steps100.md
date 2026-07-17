---
type: project
subtype: experiment-log
ejecucion: 08
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 30
eval_steps: 100
estado: completo
---

# Ejecución 08 — Run 08: eval_steps=100, patience=30, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Sexta run del grid 3×3, tercera y última del bloque steps=100. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat con todas las métricas (TrainOutput + tabla de progreso + Trainer evaluate + evaluate_set).

## Parámetros

> Únicos cambios respecto a ejecución 07: patience sube de 20 a 30. eval_steps se mantiene en 100. num_train_epochs se mantiene en 100.

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

> Nota: el `RUN_LABEL` dentro del notebook dice `"run06_steps100_pat10"` (stale — no se actualizó), pero `EVAL_STEPS=100` y `PATIENCE=30` son correctos para run 08. El `RUN_LABEL` solo afecta el nombre del `output_dir`, no el comportamiento del entrenamiento.

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
| Training loss (TrainOutput, promedio) | 0.3361 |
| Training loss (step final 4500, de tabla) | 0.157576 |
| Runtime | 4805.7s (~1:19:53, ~80 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 4200 (época ~93.33, mIoU Trainer = 0.928233) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Con patience=30: desde el best step 4200, harían falta 30 evaluaciones sin mejora (hasta step 4200 + 30×100 = 7200) para disparar el early stopping. Como el máximo de pasos es 4500 (100 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 4300 hasta 4500 hay 3 evaluaciones sin mejora, todas por debajo de 0.928233.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

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
| 4100 | 0.156253 | 0.119586 | 0.926820 | 0.963366 | 0.964205 |
| **4200** | **0.141577** | **0.115158** | **0.928233** | **0.963952** | **0.964937** |
| 4300 | 0.158635 | 0.116072 | 0.927883 | 0.963861 | 0.964752 |
| 4400 | 0.156416 | 0.114109 | 0.927842 | 0.963682 | 0.964742 |
| 4500 | 0.157576 | 0.115309 | 0.928020 | 0.963674 | 0.964840 |

> Mejor Trainer mIoU: **0.928233** en step 4200 (época ~93.33). Después de eso, 3 evaluaciones consecutivas (steps 4300-4500) sin mejora. Con patience=30 y eval_steps=100, el early stopping habría disparado en step 7200 (la 30ª evaluación sin mejora), muy por encima del máximo de 4500 pasos. El entrenamiento terminó naturalmente al completar las 100 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.116517 | 0.107208 |
| eval_mean_iou | 0.927876 | 0.934467 |
| eval_mean_accuracy | 0.963632 | 0.966571 |
| eval_overall_accuracy | 0.964765 | 0.970535 |

> Nota: el mIoU del Trainer en test (0.934467) es mayor que en val (0.927876), consistente con el patrón observado en ejecs. 05 y 07. El eval_loss en test (0.107208) es menor que en val (0.116517), también consistente.

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
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 4200 (época ~93.33, mIoU Trainer = 0.928233). Después de eso, 3 evaluaciones sin mejora, pero con patience=30 hacían falta 30 para disparar (step 7200 > 4500). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 07 (patience=20):** la única diferencia es patience (30 vs 20). Ambas runs corrieron 100 épocas completas (4.500 pasos). mIoU test: ejec. 07 = 0.8804 vs esta run = 0.8897 (+0.0093). Esta run es mejor en test a pesar de tener el mismo patience mayor. El best step fue más tarde (4200 vs 3600), y el mIoU Trainer fue ligeramente superior (0.928233 vs 0.928121).
- **Comparación con ejec. 06 (patience=10):** ejec. 06 detuvo en step 3500 (época 77.78); esta run corrió hasta step 4500 (época 100). mIoU test: ejec. 06 = 0.8845 vs esta run = 0.8897 (+0.0052). La run más larga obtuvo mejor test.
- **mIoU val más alto del grid:** 0.8998 — el más alto de todas las runs del grid (ejec. 03: 0.8942, ejec. 04: 0.8927, ejec. 05: 0.8941, ejec. 06: 0.8914, ejec. 07: 0.8958).
- **mIoU test:** 0.8897 — el segundo más alto del grid (ejec. 03: 0.8828, ejec. 04: 0.8816, ejec. 05: 0.8861, ejec. 06: 0.8845, ejec. 07: 0.8804). Solo superado por ejec. 05 (0.8861) si excluimos esta. En realidad 0.8897 es el más alto del grid search hasta ahora.
- **Δ val-test:** 0.8998 - 0.8897 = 0.0101 — dentro del rango normal del grid (ejec. 03: 0.0114, ejec. 04: 0.0111, ejec. 05: 0.0080, ejec. 06: 0.0069, ejec. 07: 0.0154). Menor que ejec. 07, indicando mejor generalización.
- **Variabilidad:** Std en test = 0.0854 — el más bajo del bloque steps=100 (ejec. 06: 0.0900, ejec. 07: 0.0993) y el más bajo de todo el grid junto con ejec. 05 (0.0910). Indica predicciones más consistentes.
- **Min IoU en test:** 0.6211 — el más alto del bloque steps=100 (ejec. 06: 0.5824, ejec. 07: 0.4866) y el más alto de todo el grid (ejec. 03: 0.2808, ejec. 04: 0.3708, ejec. 05: 0.5851). La peor predicción de esta run es mejor que la peor de cualquier otra run del grid.
- **Plateau visible:** a partir de step ~2800, el mIoU del Trainer se estabiliza en el rango 0.920-0.928. Los últimos ~1.700 pasos (steps 2800-4500) no aportaron mejora significativa más allá de variaciones marginales.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun.

## Comparación con ejecuciones 01-07

| Métrica | Ejec. 01 (50 ep, p=10, s=50) | Ejec. 02 (100 ep, p=50, s=50) | Ejec. 03 (100 ep, p=10, s=50) | Ejec. 04 (100 ep, p=20, s=50) | Ejec. 05 (100 ep, p=30, s=50) | Ejec. 06 (100 ep, p=10, s=100) | Ejec. 07 (100 ep, p=20, s=100) | **Ejec. 08 (100 ep, p=30, s=100)** | Δ 08 vs 07 | Δ 08 vs 05 |
|---|---|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | 20 | 30 | 10 | 20 | 30 | — | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | 100 | 100 | 100 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 77.78 | 100 | 100 | 0 | 0 |
| Early stopping | Sí (ép 43.33) | No | Sí (ép 67.78) | No | No | Sí (ép 77.78) | No (corrió todas) | No (corrió todas) | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 3.500 | 4.500 | 4.500 | 0 | 0 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | 0.1787 | 0.1500 | 0.1576 | — | — |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | ~61 min | ~84 min | ~80 min | -4 min | -11 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | 0.8941 | 0.8914 | 0.8958 | **0.8998** | +0.0040 | +0.0057 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | 0.9250 | 0.9301 | 0.9283 | -0.0018 | -0.0017 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | 0.0943 | 0.0856 | 0.0813 | -0.0043 | -0.0127 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | 0.5036 | 0.5281 | 0.5533 | +0.0252 | -0.0026 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | 0.9817 | 0.9775 | 0.9807 | +0.0032 | -0.0021 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | 0.8861 | 0.8845 | 0.8804 | **0.8897** | +0.0093 | +0.0036 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | 0.9162 | 0.9179 | 0.9230 | +0.0051 | +0.0063 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | 0.0900 | 0.0993 | 0.0854 | -0.0139 | -0.0056 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | 0.5824 | 0.4866 | 0.6211 | +0.1345 | +0.0360 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | 0.9766 | 0.9765 | 0.9784 | +0.0019 | -0.0005 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | 0.0069 | 0.0154 | 0.0101 | -0.0053 | +0.0021 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |

> **Conclusión parcial del bloque steps=100 (runs 06, 07, 08):** con eval_steps=100, patience=10 detuvo en step 3500 (mIoU test 0.8845); patience=20 corrió 100 épocas (mIoU test 0.8804); patience=30 corrió 100 épocas (mIoU test 0.8897). A diferencia del bloque steps=50 donde patience no marcaba diferencia clara, en steps=100 el patience mayor (30) sí produjo el mejor resultado del bloque. Ejec. 08 tiene el mIoU test más alto del grid search hasta ahora (0.8897), superando a ejec. 05 (0.8861) que era la anterior mejor. También tiene el Std test más bajo (0.0854) y el Min test más alto (0.6211), indicando la mayor consistencia del grid. Ejec. 02 (patience=50, fuera del grid) sigue siendo la mejor en test absoluto (0.8903), pero por un margen mínimo (0.0006).

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, patience=10, eval_steps=50
- [[ejecucion-04-mitb3-100epocas-patience20-steps50]] — ejecución 04, patience=20, eval_steps=50
- [[ejecucion-05-mitb3-100epocas-patience30-steps50]] — ejecución 05, patience=30, eval_steps=50
- [[ejecucion-06-mitb3-100epocas-patience10-steps100]] — ejecución 06, patience=10, eval_steps=100
- [[ejecucion-07-mitb3-100epocas-patience20-steps100]] — ejecución 07, patience=20, eval_steps=100
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
