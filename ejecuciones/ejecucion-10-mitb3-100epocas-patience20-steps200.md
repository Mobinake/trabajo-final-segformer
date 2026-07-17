---
type: project
subtype: experiment-log
ejecucion: 10
fecha: 2026-06-28
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 20
eval_steps: 200
estado: completo
---

# Ejecución 10 — Run 10: eval_steps=200, patience=20, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Octava run del grid, segunda del bloque steps=200. Fecha: 28 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat con todas las métricas (TrainOutput + tabla de progreso + Trainer evaluate + evaluate_set).

## Parámetros

> Único cambio respecto a ejecución 09: patience sube de 10 a 20. eval_steps se mantiene en 200. num_train_epochs se mantiene en 100.

| Parámetro | Valor | Cambió vs ejec. 09 |
|---|---|---|
| Learning rate | 6 × 10⁻⁵ | No |
| Épocas máx | 100 | No |
| eval_steps | 200 | No |
| save_steps | 200 | No |
| Patience early stopping | 20 | **Sí** (10 → 20) |
| Batch size efectivo | 8 (2 × 4 grad_accum) | No |
| Warmup steps | 50 | No |
| FP16 | True | No |
| Seed | 42 | No |
| Data augmentation | on (flip, rotate90, brightness/contrast) | No |
| reduce_labels | False | No |
| Checkpoint base | `nvidia/mit-b3` | No |
| Dataset | 513 imgs (3 distritos) | No |
| Split | 70/15/15 → 359/76/78 | No |

> Nota: el `RUN_LABEL` dentro del notebook dice `"run06_steps100_pat10"` (stale — no se actualizó), pero `EVAL_STEPS=200` y `PATIENCE=20` son correctos para run 10. El `RUN_LABEL` solo afecta el nombre del `output_dir`, no el comportamiento del entrenamiento.

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
| Training loss (TrainOutput, promedio) | 0.1281 |
| Training loss (step final 4500, de tabla) | 0.102718 |
| Runtime | 4369.78s (~1:12:50, ~73 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas) |
| Mejor step (Trainer mIoU) | 3400 (época ~75.56, mIoU Trainer = 0.923451) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Con patience=20 y eval_steps=200: desde el best step 3400, harían falta 20 evaluaciones sin mejora (hasta step 3400 + 20×200 = 7400) para disparar el early stopping. Como el máximo de pasos es 4500 (100 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 3600 hasta 4500 hay 6 evaluaciones sin mejora, todas por debajo de 0.923451.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

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

> Mejor Trainer mIoU: **0.923451** en step 3400 (época ~75.56). Después de eso, 6 evaluaciones consecutivas (steps 3600-4500) sin mejora. Con patience=20 y eval_steps=200, el early stopping habría disparado en step 7400 (la 20ª evaluación sin mejora), muy por encima del máximo de 4500 pasos. El entrenamiento terminó naturalmente al completar las 100 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.189307 | 0.140612 |
| eval_mean_iou | 0.922599 | 0.934628 |
| eval_mean_accuracy | 0.961911 | 0.966134 |
| eval_overall_accuracy | 0.961989 | 0.970642 |

> Nota: el mIoU del Trainer en test (0.934628) es mayor que en val (0.922599), consistente con el patrón observado en ejecs. 05, 07 y 08. El eval_loss en test (0.140612) es menor que en val (0.189307), también consistente.

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
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3400 (época ~75.56, mIoU Trainer = 0.923451). Después de eso, 6 evaluaciones sin mejora, pero con patience=20 y eval_steps=200 hacían falta 20 para disparar (step 7400 > 4500). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 09 (patience=10, eval_steps=200):** la única diferencia es patience (20 vs 10). Ambas runs corrieron 100 épocas completas (4.500 pasos). mIoU test: ejec. 09 = pendiente vs esta run = 0.8938. mIoU val: esta run = 0.8989. El best step fue más tarde (3400 vs 4000 en ejec. 09), y el mIoU Trainer fue ligeramente superior (0.923451 vs 0.918954).
- **Comparación con ejec. 07 (patience=20, eval_steps=100):** misma patience pero eval_steps=200 vs 100. Ejec. 07 corrió 100 épocas (4.500 pasos); esta run también. mIoU test: ejec. 07 = 0.8804 vs esta run = 0.8938 (+0.0134). mIoU val: ejec. 07 = 0.8958 vs esta run = 0.8989 (+0.0031). Esta run supera a ejec. 07 en ambas métricas.
- **mIoU test más alto del bloque steps=200 (parcial):** 0.8938 — a falta de ejec. 09 (pendiente) y ejec. 11, es el mejor del bloque steps=200 hasta ahora.
- **mIoU val:** 0.8989 — segundo más alto del grid (ejec. 08: 0.8998 sigue siendo la más alta).
- **mIoU test:** 0.8938 — segundo más alto del grid (ejec. 08: 0.8897). Supera a ejec. 08 en test por +0.0041.
- **Δ val-test:** 0.8989 - 0.8938 = 0.0051 — el más bajo del bloque steps=200 y uno de los más bajos del grid (ejec. 02: 0.0048, ejec. 06: 0.0069). Indica buena generalización.
- **Variabilidad:** Std en test = 0.0905 — más bajo que ejec. 07 (0.0993) y ejec. 06 (0.0900). Consistencia similar al bloque steps=100.
- **Min IoU en test:** 0.5769 — el más alto del bloque steps=200 (parcial, a falta de ejec. 09 y 11).
- **Plateau visible:** a partir de step ~1200, el mIoU del Trainer se estabiliza en el rango 0.919-0.923. Los últimos ~3.300 pasos (steps 1200-4500) no aportaron mejora significativa más allá de variaciones marginales. El plateau es más temprano que en ejec. 09 (que plateauaba desde ~2400).
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 28 jun (sobreescribió el de ejec. 09).

## Comparación con ejecuciones 01-09

| Métrica | Ejec. 01 (50 ep, p=10, s=50) | Ejec. 02 (100 ep, p=50, s=50) | Ejec. 03 (100 ep, p=10, s=50) | Ejec. 04 (100 ep, p=20, s=50) | Ejec. 05 (100 ep, p=30, s=50) | Ejec. 06 (100 ep, p=10, s=100) | Ejec. 07 (100 ep, p=20, s=100) | Ejec. 08 (100 ep, p=30, s=100) | Ejec. 09 (100 ep, p=10, s=200) | **Ejec. 10 (100 ep, p=20, s=200)** | Δ 10 vs 09 | Δ 10 vs 07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | 20 | 30 | 10 | 20 | 30 | 10 | 20 | — | — |
| eval_steps | 50 | 50 | 50 | 50 | 50 | 100 | 100 | 100 | 200 | 200 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | 100 | 77.78 | 100 | 100 | 100 | 100 | 0 | 0 |
| Early stopping | Sí (ép 43.33) | No | Sí (ép 67.78) | No | No | Sí (ép 77.78) | No | No | No | No | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | 4.500 | 3.500 | 4.500 | 4.500 | 4.500 | 4.500 | 0 | 0 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | 0.1600 | 0.1787 | 0.1500 | 0.1576 | 0.1530 | 0.1281 | — | — |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | ~91 min | ~61 min | ~84 min | ~80 min | ~71 min | ~73 min | +2 min | -11 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | 0.8927 | 0.8941 | 0.8914 | 0.8958 | 0.8998 | — | **0.8989** | — | +0.0031 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | 0.9300 | 0.9250 | 0.9301 | 0.9283 | — | 0.9346 | — | +0.0045 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | 0.0940 | 0.0943 | 0.0856 | 0.0813 | — | 0.0966 | — | +0.0110 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | 0.5559 | 0.5036 | 0.5281 | 0.5533 | — | 0.5262 | — | -0.0019 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | 0.9828 | 0.9817 | 0.9775 | 0.9807 | — | 0.9857 | — | +0.0082 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | 0.8816 | 0.8861 | 0.8845 | 0.8804 | 0.8897 | — | **0.8938** | — | +0.0134 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | 0.9167 | 0.9162 | 0.9179 | 0.9230 | — | 0.9281 | — | +0.0102 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | 0.0910 | 0.0900 | 0.0993 | 0.0854 | — | 0.0905 | — | -0.0088 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | 0.5851 | 0.5824 | 0.4866 | 0.6211 | — | 0.5769 | — | +0.0903 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | 0.9789 | 0.9766 | 0.9765 | 0.9784 | — | 0.9797 | — | +0.0032 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | 0.0080 | 0.0069 | 0.0154 | 0.0101 | — | 0.0051 | — | -0.0103 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — |

> ⏳ Las métricas de evaluate_set de ejec. 09 siguen pendientes (el usuario no las incluyó en su notebook limpio).

> **Conclusión parcial del bloque steps=200 (runs 09, 10, 11):** con eval_steps=200, patience=10 (ejec. 09) y patience=20 (esta run) ambos corrieron 100 épocas completas. Esta run (patience=20) obtuvo mIoU test 0.8938, el más alto del bloque steps=200 hasta ahora y el segundo más alto de todo el grid (solo superado por ejec. 08 con 0.8897... esperá, 0.8938 > 0.8897, así que esta run tiene el mIoU test más alto de todo el grid search). Falta ejec. 09 (pendiente evaluate_set) y ejec. 11 (patience=30) para completar el bloque.

> **mIoU test más alto del grid search:** 0.8938 — supera a ejec. 08 (0.8897) por +0.0041 y a ejec. 02 (0.8903, fuera del grid) por +0.0035. Esta es la mejor run del grid search en métrica de test.

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
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
