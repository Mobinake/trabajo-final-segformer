---
type: project
subtype: experiment-log
ejecucion: 03
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 10
eval_steps: 50
estado: completo
---

# Ejecución 03 — Run 03: eval_steps=50, patience=10, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Primera run del grid 3×3. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 02: patience baja de 50 a 10, eval_steps se mantiene en 50 (igual que ejec. 01 y 02). num_train_epochs sube a 100 (igual que ejec. 02).

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 50 |
| save_steps | 50 |
| Patience early stopping | 10 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| Warmup steps | 50 |
| FP16 | True |
| Seed | 42 |
| Data augmentation | on (flip, rotate90, brightness/contrast) |
| reduce_labels | False |
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
| Training loss final | 0.4236 (step 3050) |
| Runtime | 4.135 s (~69 min) |
| Épocas ejecutadas | 67.78 |
| Pasos totales | 3.050 |
| Early stopping activado | Sí (patience=10, best en step 2550, stop en 3050) |
| Mejor step (Trainer mIoU) | 2550 (época ~56.67, mIoU Trainer = 0.9265) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 67.78 épocas = 3.050 pasos. Cuadra.

> El mejor checkpoint fue el del step 2550 (mIoU Trainer = 0.926495). Después de eso, 10 evaluaciones consecutivas (steps 2600-3050) sin superar ese valor → early stopping disparó en step 3050. `load_best_model_at_end=True` cargó el checkpoint del step 2550.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 50 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 50 | 2.012002 | 0.451321 | 0.627298 | 0.776630 | 0.778627 |
| 100 | 1.368087 | 0.400899 | 0.661582 | 0.784070 | 0.812999 |
| 150 | 1.189985 | 0.304267 | 0.742372 | 0.852885 | 0.859612 |
| 200 | 1.030379 | 0.302515 | 0.762923 | 0.878637 | 0.869507 |
| 250 | 1.012893 | 0.287248 | 0.758402 | 0.858445 | 0.871185 |
| 300 | 0.909076 | 0.259173 | 0.797632 | 0.889647 | 0.893097 |
| 350 | 0.794890 | 0.212067 | 0.823908 | 0.906804 | 0.908226 |
| 400 | 0.730717 | 0.204272 | 0.835313 | 0.908536 | 0.915675 |
| 450 | 0.680535 | 0.212000 | 0.846101 | 0.919502 | 0.920920 |
| 500 | 0.656310 | 0.233583 | 0.841130 | 0.917631 | 0.917952 |
| 550 | 0.525847 | 0.184983 | 0.866751 | 0.932250 | 0.932249 |
| 600 | 0.519855 | 0.191879 | 0.867383 | 0.929786 | 0.932983 |
| 650 | 0.484763 | 0.193179 | 0.867687 | 0.932041 | 0.932861 |
| 700 | 0.528658 | 0.152911 | 0.884601 | 0.940624 | 0.942133 |
| 750 | 0.451252 | 0.186187 | 0.868740 | 0.928391 | 0.934031 |
| 800 | 0.425021 | 0.153093 | 0.882550 | 0.939586 | 0.941021 |
| 850 | 0.427839 | 0.200045 | 0.876909 | 0.938951 | 0.937671 |
| 900 | 0.436238 | 0.160739 | 0.885448 | 0.942629 | 0.942408 |
| 950 | 0.400772 | 0.163315 | 0.883668 | 0.940107 | 0.941633 |
| 1000 | 0.366232 | 0.144749 | 0.893325 | 0.945887 | 0.946733 |
| 1050 | 0.401994 | 0.139037 | 0.895698 | 0.944020 | 0.948334 |
| 1100 | 0.326157 | 0.155527 | 0.895608 | 0.947715 | 0.947883 |
| 1150 | 0.350555 | 0.150622 | 0.897278 | 0.947125 | 0.948922 |
| 1200 | 0.314078 | 0.119202 | 0.906445 | 0.953409 | 0.953607 |
| 1250 | 0.352262 | 0.136640 | 0.901832 | 0.948716 | 0.951407 |
| 1300 | 0.324409 | 0.124558 | 0.907010 | 0.952578 | 0.954008 |
| 1350 | 0.343086 | 0.125593 | 0.907696 | 0.954069 | 0.954263 |
| 1400 | 0.293925 | 0.149836 | 0.900682 | 0.948534 | 0.950759 |
| 1450 | 0.291444 | 0.131404 | 0.906873 | 0.952560 | 0.953932 |
| 1500 | 0.246160 | 0.120586 | 0.911538 | 0.953994 | 0.956456 |
| 1550 | 0.240428 | 0.147506 | 0.905126 | 0.952651 | 0.952921 |
| 1600 | 0.286205 | 0.121016 | 0.911729 | 0.955006 | 0.956473 |
| 1650 | 0.256549 | 0.133622 | 0.910110 | 0.955829 | 0.955481 |
| 1700 | 0.254184 | 0.112572 | 0.915305 | 0.957129 | 0.958306 |
| 1750 | 0.246197 | 0.117075 | 0.915037 | 0.956721 | 0.958190 |
| 1800 | 0.247247 | 0.121939 | 0.914448 | 0.957617 | 0.957782 |
| 1850 | 0.246570 | 0.149628 | 0.910410 | 0.955167 | 0.955710 |
| 1900 | 0.208522 | 0.118772 | 0.915747 | 0.957595 | 0.958514 |
| 1950 | 0.223323 | 0.122030 | 0.913966 | 0.957712 | 0.957502 |
| 2000 | 0.237216 | 0.108549 | 0.921051 | 0.959912 | 0.961286 |
| 2050 | 0.213253 | 0.126470 | 0.913821 | 0.956323 | 0.957539 |
| 2100 | 0.212583 | 0.114310 | 0.919258 | 0.959599 | 0.960314 |
| 2150 | 0.245235 | 0.119271 | 0.916162 | 0.958152 | 0.958700 |
| 2200 | 0.195198 | 0.119619 | 0.917239 | 0.957557 | 0.959354 |
| 2250 | 0.234030 | 0.104489 | 0.922652 | 0.959539 | 0.962202 |
| 2300 | 0.221071 | 0.120221 | 0.917425 | 0.957921 | 0.959427 |
| 2350 | 0.184383 | 0.113403 | 0.921236 | 0.960685 | 0.961328 |
| 2400 | 0.181045 | 0.107862 | 0.922685 | 0.961220 | 0.962089 |
| 2450 | 0.202033 | 0.122004 | 0.918356 | 0.959123 | 0.959849 |
| 2500 | 0.194585 | 0.117379 | 0.919203 | 0.958295 | 0.960389 |
| **2550** | **0.212346** | **0.102710** | **0.926495** | **0.963079** | **0.964048** |
| 2600 | 0.209141 | 0.126876 | 0.918827 | 0.959685 | 0.960067 |
| 2650 | 0.223091 | 0.124807 | 0.921612 | 0.960863 | 0.961523 |
| 2700 | 0.180733 | 0.123474 | 0.920910 | 0.961555 | 0.961080 |
| 2750 | 0.174714 | 0.122574 | 0.921905 | 0.960968 | 0.961677 |
| 2800 | 0.181640 | 0.112529 | 0.924976 | 0.961710 | 0.963315 |
| 2850 | 0.166580 | 0.114001 | 0.924724 | 0.962576 | 0.963112 |
| 2900 | 0.187480 | 0.120944 | 0.922161 | 0.961268 | 0.961795 |
| 2950 | 0.167006 | 0.113044 | 0.925866 | 0.962864 | 0.963719 |
| 3000 | 0.168959 | 0.118723 | 0.925340 | 0.962758 | 0.963437 |
| 3050 | 0.167547 | 0.119624 | 0.923679 | 0.961796 | 0.962594 |

> Mejor Trainer mIoU: **0.926495** en step 2550 (época ~56.67). Después de eso, 10 evaluaciones consecutivas sin mejora → early stopping en step 3050 (época 67.78).

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.114518 | 0.102646 |
| eval_mean_iou | 0.923743 | 0.930019 |
| eval_mean_accuracy | 0.961653 | 0.964905 |
| eval_overall_accuracy | 0.962640 | 0.968405 |

> Nota: el mIoU del Trainer en test (0.930019) es mayor que en val (0.923743). Esto es inusual y puede deberse a que el test set tiene imágenes ligeramente más fáciles o con mayor proporción de caña (menor desbalance por imagen). La métrica del manuscrito es la de evaluate_set (resolución original).

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8942** |
| Median IoU | 0.9249 |
| Std | 0.0877 |
| Min | 0.5481 |
| Max | 0.9795 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8828** |
| Median IoU | 0.9196 |
| Std | 0.1048 |
| Min | 0.2808 |
| Max | 0.9783 |

- ✅ Hipótesis superada: mIoU en test set = 0.8828 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8828 ≥ 0.85.
- **Early stopping activado en step 3050 (época 67.78):** el mejor checkpoint fue step 2550 (época ~56.67). Después de eso, 10 evaluaciones consecutivas (steps 2600-3050) sin superar el mIoU de 0.9265 → patience=10 agotado. Comportamiento esperado y correcto.
- **Comparación con ejec. 01 (50 ep, patience=10, eval_steps=50):** la única diferencia es que ejec. 01 tenía num_train_epochs=50 y esta run tiene 100. Ejec. 01 paró en época 43.33 (early stopping), esta run paró en época 67.78 (early stopping). Al permitir más épocas, el modelo encontró un mejor punto de corte → mIoU test subió de 0.8535 a 0.8828 (+0.0293).
- **Comparación con ejec. 02 (100 ep, patience=50, eval_steps=50):** ejec. 02 corrió las 100 épocas completas (4.500 pasos) porque patience=50 nunca se agotó. Esta run paró antes (3.050 pasos) gracias a patience=10. mIoU test: ejec. 02 = 0.8903 vs esta run = 0.8828 (-0.0075). La diferencia es mínima y esta run fue más eficiente (3.050 vs 4.500 pasos, ~69 min vs ~90 min).
- **Variabilidad:** Std en test = 0.1048 (entre ejec. 01: 0.1411 y ejec. 02: 0.0869). Mejor que ejec. 01 pero peor que ejec. 02.
- **Min IoU en test:** 0.2808 — el peor caso es mejor que ejec. 01 (0.1593) pero peor que ejec. 02 (0.6115).
- **Δ val-test:** 0.8942 - 0.8828 = 0.0114 (pequeña; mejor que ejec. 01: 0.0192, peor que ejec. 02: 0.0048).
- **Training loss:** bajó de 2.012 (step 50) a 0.168 (step 3050). Descenso suave y estable.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 27 jun a las 21:05.

## Comparación con ejecuciones 01 y 02

| Métrica | Ejec. 01 (50 ep, p=10) | Ejec. 02 (100 ep, p=50) | **Ejec. 03 (100 ep, p=10)** | Δ 03 vs 01 | Δ 03 vs 02 |
|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | — | — |
| eval_steps | 50 | 50 | 50 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | +24.45 | -32.22 |
| Early stopping | Sí (época 43.33) | No (corrió todas) | Sí (época 67.78) | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | +1.100 | -1.450 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | -0.1180 | +0.2677 |
| Runtime | ~56 min | ~90 min | ~69 min | +13 min | -21 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | **0.8942** | +0.0215 | -0.0009 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | +0.0113 | -0.0054 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | -0.0175 | -0.0081 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | +0.0729 | +0.0106 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | -0.0009 | -0.0015 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | **0.8828** | +0.0293 | -0.0075 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | +0.0114 | +0.0004 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | -0.0363 | +0.0179 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | +0.1215 | -0.3307 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | +0.0002 | +0.0001 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | -0.0078 | +0.0066 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | — | — |

> **Conclusión:** con patience=10 y max_epochs=100 (vs ejec. 01 que tenía max_epochs=50), el modelo entrena 24 épocas más y mejora mIoU test en +0.029. Sin embargo, no supera a ejec. 02 (patience=50) que obtuvo 0.8903 al correr las 100 épocas completas. La diferencia es pequeña (-0.008 en test) y esta run fue más eficiente: 3.050 pasos en ~69 min vs 4.500 pasos en ~90 min. El std en val (0.0877) es el más bajo de las 3 ejecuciones, indicando buena consistencia entre imágenes de validación.

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, mismo patience pero 50 épocas máx
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
