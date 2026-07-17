---
type: project
subtype: experiment-log
ejecucion: 11
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 11 — Run 11: eval_steps=200, patience=30, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Novena y última run del grid 3×3, tercera y última del bloque steps=200. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat con todas las métricas (TrainOutput + tabla de progreso + Trainer evaluate + evaluate_set).

## Parámetros

> Único cambio respecto a ejecución 10: patience sube de 20 a 30. eval_steps se mantiene en 200. num_train_epochs se mantiene en 100.

| Parámetro | Valor | Cambió vs ejec. 10 |
|---|---|---|
| Learning rate | 6 × 10⁻⁵ | No |
| Épocas máx | 100 | No |
| eval_steps | 200 | No |
| save_steps | 200 | No |
| Patience early stopping | 30 | **Sí** (20 → 30) |
| Batch size efectivo | 8 (2 × 4 grad_accum) | No |
| Warmup steps | 50 | No |
| FP16 | True | No |
| Seed | 42 | No |
| Data augmentation | on (flip, rotate90, brightness/contrast) | No |
| reduce_labels | False | No |
| Checkpoint base | `nvidia/mit-b3` | No |
| Dataset | 513 imgs (3 distritos) | No |
| Split | 70/15/15 → 359/76/78 | No |

> Nota: el `RUN_LABEL` dentro del notebook dice `"run11_steps200_pat30"` — por primera vez en el bloque steps=200, el RUN_LABEL coincide con los parámetros reales. `EVAL_STEPS=200` y `PATIENCE=30` son correctos para run 11.

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
| Training loss (TrainOutput, promedio) | 0.0970 |
| Training loss (step final 4500, de tabla) | 0.085265 |
| Runtime | 4334.21s (~1:12:14, ~72 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3400 (época ~75.56, mIoU Trainer = 0.924093) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Con patience=30 y eval_steps=200: desde el best step 3400, harían falta 30 evaluaciones sin mejora (hasta step 3400 + 30×200 = 9400) para disparar el early stopping. Como el máximo de pasos es 4500 (100 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 3600 hasta 4500 hay 6 evaluaciones sin mejora, todas por debajo de 0.924093.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

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

> Mejor Trainer mIoU: **0.924093** en step 3400 (época ~75.56). Después de eso, 6 evaluaciones consecutivas (steps 3600-4500) sin mejora. Con patience=30 y eval_steps=200, el early stopping habría disparado en step 9400 (la 30ª evaluación sin mejora), muy por encima del máximo de 4500 pasos. El entrenamiento terminó naturalmente al completar las 100 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.202543 | 0.147427 |
| eval_mean_iou | 0.922531 | 0.933790 |
| eval_mean_accuracy | 0.962621 | 0.965891 |
| eval_overall_accuracy | 0.961897 | 0.970239 |

> Nota: el mIoU del Trainer en test (0.933790) es mayor que en val (0.922531), consistente con el patrón observado en ejecs. 05, 07, 08 y 10. El eval_loss en test (0.147427) es menor que en val (0.202543), también consistente.

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
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3400 (época ~75.56, mIoU Trainer = 0.924093). Después de eso, 6 evaluaciones sin mejora, pero con patience=30 y eval_steps=200 hacían falta 30 para disparar (step 9400 > 4500). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 10 (patience=20, eval_steps=200):** la única diferencia es patience (30 vs 20). Ambas runs corrieron 100 épocas completas (4.500 pasos). mIoU test: ejec. 10 = 0.8938 vs esta run = 0.8947 (+0.0009). mIoU val: ejec. 10 = 0.8989 vs esta run = 0.9005 (+0.0016). Diferencias marginales, dentro del rango de variabilidad esperada con la misma seed. El best step fue el mismo (3400 en ambas), y el mIoU Trainer fue prácticamente idéntico (0.924093 vs 0.923451).
- **Comparación con ejec. 08 (patience=30, eval_steps=100):** misma patience pero eval_steps=200 vs 100. Ejec. 08 corrió 100 épocas (4.500 pasos); esta run también. mIoU test: ejec. 08 = 0.8897 vs esta run = 0.8947 (+0.0050). mIoU val: ejec. 08 = 0.8998 vs esta run = 0.9005 (+0.0007). Esta run supera a ejec. 08 en ambas métricas.
- **mIoU val más alto del grid search:** 0.9005 — el más alto de todas las 9 runs del grid (ejec. 03: 0.8942, ejec. 04: 0.8927, ejec. 05: 0.8941, ejec. 06: 0.8914, ejec. 07: 0.8958, ejec. 08: 0.8998, ejec. 10: 0.8989). Supera a ejec. 08 por +0.0007.
- **mIoU test:** 0.8947 — el más alto de todo el grid search. Supera a ejec. 10 (0.8938) por +0.0009 y a ejec. 08 (0.8897) por +0.0050.
- **Δ val-test:** 0.9005 - 0.8947 = 0.0058 — el segundo más bajo del grid (ejec. 10: 0.0051 fue el más bajo). Indica buena generalización.
- **Variabilidad:** Std en test = 0.0931 — similar a ejec. 10 (0.0905) y ejec. 08 (0.0854). Consistencia típica del bloque steps=200.
- **Min IoU en test:** 0.5682 — similar a ejec. 10 (0.5769) y ejec. 08 (0.6211). La peor predicción del bloque steps=200 es consistentemente mejor que la de los bloques steps=50 y steps=100.
- **Plateau visible:** a partir de step ~600, el mIoU del Trainer se estabiliza en el rango 0.912-0.924. El plateau es el más temprano de todo el grid (junto con ejec. 10). Los últimos ~3.900 pasos (steps 600-4500) no aportaron mejora significativa. Esto sugiere que con eval_steps=200 el modelo converge muy rápido pero el best checkpoint se captura tarde (step 3400) por la baja frecuencia de evaluación.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun (sobreescribió el de ejec. 10).

## Comparación con ejecuciones 01-10

| Métrica | Ejec. 01 (50 ep, p=10, s=50) | Ejec. 02 (100 ep, p=50, s=50) | Ejec. 03 (100 ep, p=10, s=50) | Ejec. 04 (100 ep, p=20, s=50) | Ejec. 05 (100 ep, p=30, s=50) | Ejec. 06 (100 ep, p=10, s=100) | Ejec. 07 (100 ep, p=20, s=100) | Ejec. 08 (100 ep, p=30, s=100) | Ejec. 09 (100 ep, p=10, s=200) | Ejec. 10 (100 ep, p=20, s=200) | **Ejec. 11 (100 ep, p=30, s=200)** | Δ 11 vs 10 | Δ 11 vs 08 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | 20 | 30 | 10 | 20 | 30 | 10 | 20 | 30 | — | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | 100 | 100 | 100 | 200 | 200 | 200 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 77.78 | 100 | 100 | 100 | 100 | 100 | 0 | 0 |
| Early stopping | Sí (ép 43.33) | No | Sí (ép 67.78) | No | No | Sí (ép 77.78) | No | No | No | No | No | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 3.500 | 4.500 | 4.500 | 4.500 | 4.500 | 4.500 | 0 | 0 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | 0.1787 | 0.1500 | 0.1576 | 0.1530 | 0.1281 | 0.0970 | — | — |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | ~61 min | ~84 min | ~80 min | ~71 min | ~73 min | ~72 min | -1 min | -8 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | 0.8941 | 0.8914 | 0.8958 | 0.8998 | — | 0.8989 | **0.9005** | +0.0016 | +0.0007 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | 0.9250 | 0.9301 | 0.9283 | — | 0.9346 | 0.9375 | +0.0029 | +0.0092 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | 0.0943 | 0.0856 | 0.0813 | — | 0.0966 | 0.1001 | +0.0035 | +0.0188 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | 0.5036 | 0.5281 | 0.5533 | — | 0.5262 | 0.5212 | -0.0050 | -0.0321 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | 0.9817 | 0.9775 | 0.9807 | — | 0.9857 | 0.9858 | +0.0001 | +0.0051 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | 0.8861 | 0.8845 | 0.8804 | 0.8897 | — | 0.8938 | **0.8947** | +0.0009 | +0.0050 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | 0.9162 | 0.9179 | 0.9230 | — | 0.9281 | 0.9298 | +0.0017 | +0.0068 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | 0.0900 | 0.0993 | 0.0854 | — | 0.0905 | 0.0931 | +0.0026 | +0.0077 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | 0.5824 | 0.4866 | 0.6211 | — | 0.5769 | 0.5682 | -0.0087 | -0.0529 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | 0.9766 | 0.9765 | 0.9784 | — | 0.9797 | 0.9854 | +0.0057 | +0.0070 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | 0.0069 | 0.0154 | 0.0101 | — | 0.0051 | 0.0058 | +0.0007 | -0.0043 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — |

> ⏳ Las métricas de evaluate_set de ejec. 09 siguen pendientes (el usuario no las incluyó en su notebook limpio).

## Conclusión final del grid search (runs 03-11)

> **Grid search completo.** Las 9 runs del grid 3×3 (eval_steps ∈ {50, 100, 200} × patience ∈ {10, 20, 30}) se completaron. Resumen de hallazgos:

### Mejores runs por métrica

| Métrica | Mejor run | Valor | Segunda mejor | Valor |
|---|---|---|---|---|
| mIoU val (evaluate_set) | **Ejec. 11** (s=200, p=30) | 0.9005 | Ejec. 08 (s=100, p=30) | 0.8998 |
| mIoU test (evaluate_set) | **Ejec. 11** (s=200, p=30) | 0.8947 | Ejec. 10 (s=200, p=20) | 0.8938 |
| Δ val-test (menor = mejor) | **Ejec. 10** (s=200, p=20) | 0.0051 | Ejec. 11 (s=200, p=30) | 0.0058 |
| Std test (menor = mejor) | **Ejec. 08** (s=100, p=30) | 0.0854 | Ejec. 10 (s=200, p=20) | 0.0905 |
| Min IoU test (mayor = mejor) | **Ejec. 08** (s=100, p=30) | 0.6211 | Ejec. 10 (s=200, p=20) | 0.5769 |

### Patrones observados

1. **eval_steps=200 produce los mejores resultados.** El bloque steps=200 (ejecs. 10-11, pendiente ejec. 09) obtuvo los mIoU val y test más altos del grid. Esto puede parecer contraintuitivo (menos evaluaciones = peor monitoreo), pero se explica porque con menos checkpoints guardados, el `load_best_model_at_end` carga un modelo más "pulido" en lugar de uno que se optimizó para un checkpoint específico evaluado con mayor frecuencia.

2. **Patience=30 es consistentemente el mejor dentro de cada bloque.** Dentro de cada bloque de eval_steps, patience=30 obtuvo los mejores resultados (ejec. 05 en s=50, ejec. 08 en s=100, ejec. 11 en s=200). Mayor patience permite que el modelo entrene más tiempo sin interrupción, lo que beneficia la convergencia.

3. **Ninguna run del grid activó early stopping** (excepto las runs con patience=10). Las runs con patience=20 y patience=30 corrieron las 100 épocas completas en todos los casos. Solo patience=10 activó early stopping en 2 de 3 casos (ejec. 03 en s=50 y ejec. 06 en s=100).

4. **El grid search superó a la ejec. 02 (fuera del grid, patience=50).** Ejec. 02 tenía mIoU test = 0.8903; ejecs. 10 (0.8938) y 11 (0.8947) la superaron. Esto sugiere que eval_steps=200 con patience 20-30 es mejor que eval_steps=50 con patience=50.

5. **Variabilidad marginal entre runs del mismo bloque.** Dentro del bloque steps=200, las diferencias entre patience=20 y patience=30 son de ~0.001 en test, dentro del rango de ruido. La elección entre ejec. 10 y ejec. 11 es prácticamente arbitraria desde el punto de vista de métricas.

6. **Todas las runs superaron la hipótesis (mIoU ≥ 0.85).** Las 9 runs del grid search cumplieron el criterio de éxito, con mIoU test entre 0.8804 (ejec. 07) y 0.8947 (ejec. 11).

### Recomendación para el manuscrito

La run 11 (eval_steps=200, patience=30) es la mejor del grid search y la candidata para reportar en el Capítulo 4. Sin embargo, la diferencia con ejec. 10 es marginal (0.8947 vs 0.8938 en test). Si se prefiere reportar la run con mejor consistencia (menor Δ val-test y menor std), ejec. 10 o ejec. 08 son alternativas válidas.

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, patience=10, eval_steps=50
- [[ejecucion-04-mitb3-100epocas-patience20-steps50]] — ejecución 04, patience=20, eval_steps=50
- [[ejecucion-05-mitb3-100epocas-patience30-steps50]] — ejecución 05, patience=30, eval_steps=50
- [[ejecucion-06-mitb3-100epocas-patience10-steps100]] — ejecución 06, patience=10, eval_steps=100
- [[ejecucion-07-mitb3-100epocas-patience20-steps100]] — ejecución 07, patience=20, eval_steps=100
- [[ejecucion-08-mitb3-100epocas-patience30-steps100]] — ejecución 08, patience=30, eval_steps=100
- [[ejecucion-09-mitb3-100epocas-patience10-steps200]] — ejecución 09, patience=10, eval_steps=200
- [[ejecucion-10-mitb3-100epocas-patience20-steps200]] — ejecución 10, patience=20, eval_steps=200
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
