---
type: project
subtype: experiment-log
ejecucion: 05
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 30
eval_steps: 50
estado: completo
---

# Ejecución 05 — Run 05: eval_steps=50, patience=30, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Tercera run del grid 3×3. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat con todas las métricas (TrainOutput + tabla de progreso + Trainer evaluate + evaluate_set).

## Parámetros

> Únicos cambios respecto a ejecución 04: patience sube de 20 a 30. eval_steps se mantiene en 50. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 50 |
| save_steps | 50 |
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

> Nota: el `RUN_LABEL` dentro del notebook dice `"run03_steps50_pat10"` (stale — no se actualizó), pero `EVAL_STEPS=50` y `PATIENCE=30` son correctos para run 05. El `RUN_LABEL` solo afecta el nombre del `output_dir`, no el comportamiento del entrenamiento.

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
| Training loss (TrainOutput, promedio) | 0.3381 |
| Training loss (step final 4500, de tabla) | 0.1600 |
| Runtime | 5454.7s (~1:30:44, ~91 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3850 (época ~85.56, mIoU Trainer = 0.921044) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Con patience=30: desde el best step 3850, harían falta 30 evaluaciones sin mejora (hasta step 3850 + 30×50 = 5350) para disparar el early stopping. Como el máximo de pasos es 4500 (100 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 3900 hasta 4500 hay 13 evaluaciones sin mejora, todas por debajo de 0.921044.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 50 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 50 | 1.978574 | 0.461260 | 0.637492 | 0.791224 | 0.783746 |
| 100 | 1.393087 | 0.360460 | 0.697114 | 0.819552 | 0.831174 |
| 150 | 1.212062 | 0.317142 | 0.737149 | 0.859981 | 0.853378 |
| 200 | 1.051094 | 0.346153 | 0.731706 | 0.868168 | 0.847492 |
| 250 | 0.918050 | 0.255226 | 0.789649 | 0.886751 | 0.887864 |
| 300 | 0.910339 | 0.208588 | 0.828178 | 0.909024 | 0.910742 |
| 350 | 0.798271 | 0.212484 | 0.834612 | 0.912256 | 0.914517 |
| 400 | 0.790811 | 0.213324 | 0.832987 | 0.906561 | 0.914484 |
| 450 | 0.626033 | 0.180735 | 0.852122 | 0.921727 | 0.924483 |
| 500 | 0.606157 | 0.211641 | 0.848792 | 0.922770 | 0.922164 |
| 550 | 0.507883 | 0.172903 | 0.866924 | 0.931347 | 0.932479 |
| 600 | 0.577202 | 0.168127 | 0.870741 | 0.931303 | 0.934862 |
| 650 | 0.453855 | 0.202141 | 0.863523 | 0.932122 | 0.930243 |
| 700 | 0.517690 | 0.174483 | 0.876318 | 0.940133 | 0.937170 |
| 750 | 0.425859 | 0.175896 | 0.871527 | 0.932954 | 0.935125 |
| 800 | 0.411488 | 0.157681 | 0.881663 | 0.938654 | 0.940598 |
| 850 | 0.441236 | 0.161807 | 0.883015 | 0.941993 | 0.941018 |
| 900 | 0.424492 | 0.157246 | 0.885948 | 0.943649 | 0.942594 |
| 950 | 0.408972 | 0.143873 | 0.895152 | 0.947636 | 0.947623 |
| 1000 | 0.377477 | 0.160553 | 0.889567 | 0.943812 | 0.944735 |
| 1050 | 0.398001 | 0.139865 | 0.895871 | 0.945440 | 0.948278 |
| 1100 | 0.343803 | 0.120068 | 0.905093 | 0.952770 | 0.952891 |
| 1150 | 0.334889 | 0.162558 | 0.891692 | 0.944138 | 0.945960 |
| 1200 | 0.301115 | 0.154296 | 0.894584 | 0.948167 | 0.947234 |
| 1250 | 0.376181 | 0.142759 | 0.899663 | 0.948495 | 0.950171 |
| 1300 | 0.282096 | 0.158077 | 0.896262 | 0.947551 | 0.948283 |
| 1350 | 0.311493 | 0.153852 | 0.899592 | 0.951213 | 0.949859 |
| 1400 | 0.286348 | 0.167265 | 0.900227 | 0.948789 | 0.950469 |
| 1450 | 0.292592 | 0.128373 | 0.905088 | 0.951705 | 0.952989 |
| 1500 | 0.272941 | 0.126003 | 0.906172 | 0.952343 | 0.953550 |
| 1550 | 0.234846 | 0.140731 | 0.904745 | 0.951521 | 0.952810 |
| 1600 | 0.278930 | 0.120327 | 0.910599 | 0.955362 | 0.955800 |
| 1650 | 0.243602 | 0.125173 | 0.909700 | 0.955763 | 0.955254 |
| 1700 | 0.254212 | 0.129071 | 0.907885 | 0.953416 | 0.954431 |
| 1750 | 0.255315 | 0.132604 | 0.906589 | 0.953146 | 0.953714 |
| 1800 | 0.237231 | 0.128208 | 0.909804 | 0.953537 | 0.955513 |
| 1850 | 0.234460 | 0.131597 | 0.909410 | 0.954600 | 0.955192 |
| 1900 | 0.220579 | 0.119505 | 0.913451 | 0.955684 | 0.957386 |
| 1950 | 0.220868 | 0.133087 | 0.911979 | 0.954551 | 0.956655 |
| 2000 | 0.237977 | 0.147294 | 0.912905 | 0.956449 | 0.957011 |
| 2050 | 0.213686 | 0.143536 | 0.909822 | 0.954005 | 0.955480 |
| 2100 | 0.209252 | 0.157432 | 0.908345 | 0.954294 | 0.954613 |
| 2150 | 0.238434 | 0.172089 | 0.907692 | 0.954495 | 0.954222 |
| 2200 | 0.196202 | 0.142380 | 0.912762 | 0.956794 | 0.956901 |
| 2250 | 0.220083 | 0.146739 | 0.912632 | 0.956648 | 0.956840 |
| 2300 | 0.185219 | 0.164276 | 0.914086 | 0.957574 | 0.957582 |
| 2350 | 0.198248 | 0.148268 | 0.911717 | 0.954888 | 0.956477 |
| 2400 | 0.176496 | 0.159349 | 0.911987 | 0.955932 | 0.956537 |
| 2450 | 0.201523 | 0.161228 | 0.913946 | 0.957384 | 0.957519 |
| 2500 | 0.231150 | 0.143257 | 0.916038 | 0.957544 | 0.958682 |
| 2550 | 0.210569 | 0.150932 | 0.913286 | 0.956895 | 0.957188 |
| 2600 | 0.212662 | 0.149369 | 0.914908 | 0.957844 | 0.958022 |
| 2650 | 0.189000 | 0.152902 | 0.915052 | 0.957116 | 0.958165 |
| 2700 | 0.179481 | 0.167216 | 0.912956 | 0.957857 | 0.956920 |
| 2750 | 0.182925 | 0.149639 | 0.916788 | 0.958958 | 0.958985 |
| 2800 | 0.183182 | 0.149032 | 0.917196 | 0.959150 | 0.959198 |
| 2850 | 0.165289 | 0.152541 | 0.917605 | 0.960064 | 0.959353 |
| 2900 | 0.182899 | 0.143234 | 0.917886 | 0.959196 | 0.959580 |
| 2950 | 0.163874 | 0.141896 | 0.918050 | 0.959314 | 0.959662 |
| 3000 | 0.167472 | 0.153098 | 0.917721 | 0.959469 | 0.959466 |
| 3050 | 0.169132 | 0.143344 | 0.918254 | 0.959284 | 0.959779 |
| 3100 | 0.183485 | 0.151622 | 0.918264 | 0.959352 | 0.959779 |
| 3150 | 0.166181 | 0.151165 | 0.918476 | 0.960053 | 0.959841 |
| 3200 | 0.175762 | 0.164791 | 0.917469 | 0.959728 | 0.959304 |
| 3250 | 0.173714 | 0.169748 | 0.917624 | 0.960256 | 0.959348 |
| 3300 | 0.172497 | 0.157968 | 0.918118 | 0.959768 | 0.959664 |
| 3350 | 0.180262 | 0.156547 | 0.918094 | 0.959314 | 0.959687 |
| 3400 | 0.154609 | 0.150577 | 0.919163 | 0.959840 | 0.960242 |
| 3450 | 0.178190 | 0.138741 | 0.920731 | 0.960788 | 0.961039 |
| 3500 | 0.179240 | 0.164681 | 0.916002 | 0.958866 | 0.958552 |
| 3550 | 0.167356 | 0.149352 | 0.920175 | 0.960674 | 0.960739 |
| 3600 | 0.166113 | 0.155547 | 0.918949 | 0.960044 | 0.960106 |
| 3650 | 0.160154 | 0.158792 | 0.918218 | 0.959769 | 0.959720 |
| 3700 | 0.155183 | 0.149735 | 0.920069 | 0.960370 | 0.960704 |
| 3750 | 0.166304 | 0.146802 | 0.920686 | 0.961063 | 0.960993 |
| 3800 | 0.164966 | 0.147272 | 0.920843 | 0.961178 | 0.961072 |
| **3850** | **0.161109** | **0.148484** | **0.921044** | **0.960999** | **0.961197** |
| 3900 | 0.160124 | 0.155113 | 0.920875 | 0.960946 | 0.961107 |
| 3950 | 0.153871 | 0.164938 | 0.919071 | 0.960588 | 0.960131 |
| 4000 | 0.146998 | 0.168313 | 0.919455 | 0.960596 | 0.960344 |
| 4050 | 0.163540 | 0.163039 | 0.919369 | 0.960501 | 0.960304 |
| 4100 | 0.170863 | 0.160032 | 0.920293 | 0.960996 | 0.960779 |
| 4150 | 0.155104 | 0.157502 | 0.920188 | 0.960742 | 0.960741 |
| 4200 | 0.139920 | 0.163524 | 0.919730 | 0.960616 | 0.960496 |
| 4250 | 0.155196 | 0.165560 | 0.919163 | 0.960064 | 0.960223 |
| 4300 | 0.152464 | 0.164708 | 0.919481 | 0.960430 | 0.960372 |
| 4350 | 0.147283 | 0.160654 | 0.919940 | 0.960576 | 0.960616 |
| 4400 | 0.147083 | 0.157805 | 0.919661 | 0.960545 | 0.960463 |
| 4450 | 0.150716 | 0.164937 | 0.919674 | 0.960570 | 0.960469 |
| 4500 | 0.160049 | 0.162109 | 0.919805 | 0.960559 | 0.960542 |

> Mejor Trainer mIoU: **0.921044** en step 3850 (época ~85.56). Después de eso, 13 evaluaciones consecutivas (steps 3900-4500) sin mejora. Con patience=30, el early stopping habría disparado en step 5350 (la 30ª evaluación sin mejora), muy por encima del máximo de 4500 pasos. El entrenamiento terminó naturalmente al completar las 100 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.162595 | 0.118476 |
| eval_mean_iou | 0.919622 | 0.929753 |
| eval_mean_accuracy | 0.960516 | 0.963902 |
| eval_overall_accuracy | 0.960444 | 0.968336 |

> Nota: el mIoU del Trainer en test (0.929753) es mayor que en val (0.919622). Esto puede deberse a que el test set tiene imágenes con patrones más fáciles de segmentar, o a que el modelo generaliza bien a datos no vistos. El eval_loss en test (0.118476) también es menor que en val (0.162595), consistente con esto.

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8941** |
| Median IoU | 0.9300 |
| Std | 0.0940 |
| Min | 0.5559 |
| Max | 0.9828 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8861** |
| Median IoU | 0.9167 |
| Std | 0.0910 |
| Min | 0.5851 |
| Max | 0.9789 |

- ✅ Hipótesis superada: mIoU en test set = 0.8861 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8861 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3850 (época ~85.56, mIoU Trainer = 0.921044). Después de eso, 13 evaluaciones sin mejora, pero con patience=30 hacían falta 30 para disparar (step 5350 > 4500). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 04 (patience=20):** la única diferencia es patience (30 vs 20). Ambas corrieron 100 épocas completas. mIoU test: ejec. 04 = 0.8816 vs esta run = 0.8861 (+0.0045). La mejora es marginal y dentro de la variabilidad esperada. Best step: ejec. 04 = 3500 vs esta run = 3850 (+350 pasos). El patience mayor permitió encontrar un checkpoint ligeramente mejor en el Trainer (0.921044 vs 0.923316 — de hecho el de ejec. 04 fue superior en Trainer mIoU).
- **Comparación con ejec. 03 (patience=10):** ejec. 03 detuvo en step 3050 (época 67.78); esta run corrió hasta step 4500 (época 100). mIoU test: ejec. 03 = 0.8828 vs esta run = 0.8861 (+0.0033). Diferencia mínima que no justifica los 1.450 pasos adicionales (~22 min más).
- **Comparación con ejec. 02 (patience=50):** ejec. 02 también corrió 100 épocas completas. mIoU test: ejec. 02 = 0.8903 vs esta run = 0.8861 (-0.0042). Ejec. 02 sigue siendo la mejor en test.
- **Conclusión del bloque steps=50 (runs 03, 04, 05):** las tres runs con eval_steps=50 producen mIoU test en el rango 0.8816-0.8861, con diferencia máxima de 0.0045 entre ellas. La run con patience=10 (ejec. 03) ya logra resultados equivalentes deteniendo en step 3050, mientras que patience=20 y 30 corren 100 épocas completas sin mejora significativa. Esto confirma que el modelo converge alrededor de step 2500-3000 y entrenar más allá no aporta beneficio. Aumentar patience con eval_steps=50 no mejora el rendimiento, solo prolonga el entrenamiento.
- **Plateau visible:** a partir de step ~2500, el mIoU del Trainer se estabiliza en el rango 0.912-0.921. Los últimos 2.000 pasos (steps 2500-4500) no aportaron mejora significativa.
- **Variabilidad:** Std en test = 0.0910 (entre ejec. 03: 0.1048, ejec. 04: 0.1019, ejec. 02: 0.0869). Esta run tiene la std más baja del bloque steps=50.
- **Min IoU en test:** 0.5851 — mejor que ejec. 03 (0.2808) y ejec. 04 (0.3708), pero peor que ejec. 02 (0.6115).
- **Δ val-test:** 0.8941 - 0.8861 = 0.0080 (similar a ejec. 03: 0.0114, ejec. 04: 0.0111). La menor diferencia val-test sugiere buena generalización.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 27 jun.

## Comparación con ejecuciones 01, 02, 03 y 04

| Métrica | Ejec. 01 (50 ep, p=10) | Ejec. 02 (100 ep, p=50) | Ejec. 03 (100 ep, p=10) | Ejec. 04 (100 ep, p=20) | **Ejec. 05 (100 ep, p=30)** | Δ 05 vs 04 | Δ 05 vs 03 | Δ 05 vs 02 |
|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | — | — | — |
| Patience | 10 | 50 | 10 | 20 | 30 | — | — | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | — | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 0 | +32.22 | 0 |
| Early stopping | Sí (época 43.33) | No (corrió todas) | Sí (época 67.78) | No (corrió todas) | No (corrió todas) | — | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 0 | +1.450 | 0 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | -0.0012 | — | +0.0041 |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | -9 min | +22 min | +1 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | **0.8941** | +0.0014 | -0.0001 | -0.0010 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | +0.0008 | +0.0051 | -0.0003 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | +0.0002 | +0.0063 | -0.0018 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | -0.0205 | +0.0078 | +0.0184 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | +0.0031 | +0.0033 | +0.0018 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | **0.8861** | +0.0045 | +0.0033 | -0.0042 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | -0.0004 | -0.0029 | -0.0025 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | -0.0109 | -0.0138 | +0.0041 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | +0.2143 | +0.3043 | -0.0264 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | -0.0005 | +0.0006 | +0.0007 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | -0.0031 | -0.0034 | +0.0032 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |

> **Conclusión del bloque steps=50:** con eval_steps=50, las tres variantes de patience (10, 20, 30) producen resultados equivalentes en test (0.8816-0.8861, Δ < 0.005). Patience=10 ya encuentra un buen punto de corte en step 3050; aumentar patience solo prolonga el entrenamiento sin mejora significativa. Ejec. 02 (patience=50, fuera del grid pero con mismos parámetros base) sigue siendo la mejor en test (0.8903), sugiriendo que la variabilidad entre runs es del orden de ±0.005 y que no hay un efecto claro de patience sobre el rendimiento final cuando eval_steps=50.

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, patience=10 (bloque steps=50)
- [[ejecucion-04-mitb3-100epocas-patience20-steps50]] — ejecución 04, patience=20 (bloque steps=50)
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
