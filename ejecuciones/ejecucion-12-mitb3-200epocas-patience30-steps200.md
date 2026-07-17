---
type: project
subtype: experiment-log
ejecucion: 12
fecha: 2026-07-04
modelo: SegFormer MiT-B3
epocas_max: 200
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 12 — Run 10: eval_steps=200, patience=30, max_epochs=200

> Run posterior al grid search. Toma la mejor configuración del grid (run 09: eval_steps=200, patience=30) y duplica las épocas máximas de 100 a 200 para evaluar si el modelo sigue mejorando con más entrenamiento. Fecha: 4 de julio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline en el chat con todas las métricas.

## Parámetros

> Únicos cambios respecto a ejec. 11 (run 09): `num_train_epochs` sube de 100 a 200. `eval_steps` y `patience` se mantienen en 200 y 30 respectivamente.

| Parámetro | Valor | Cambió vs ejec. 11 |
|---|---|---|
| Learning rate | 6 × 10⁻⁵ | No |
| Épocas máx | 200 | **Sí** (100 → 200) |
| eval_steps | 200 | No |
| save_steps | 200 | No |
| Patience early stopping | 30 | No |
| Batch size efectivo | 8 (2 × 4 grad_accum) | No |
| Warmup steps | 50 | No |
| FP16 | True | No |
| Seed | 42 | No |
| Data augmentation | on (flip, rotate90, brightness/contrast) | No |
| reduce_labels | False | No |
| Checkpoint base | `nvidia/mit-b3` | No |
| Dataset | 513 imgs (3 distritos) | No |
| Split | 70/15/15 → 359/76/78 | No |

> Nota: el `RUN_LABEL` dentro del notebook dice `"run10_steps200_pat30_epochs200"` — coincide con los parámetros reales.

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
| Training loss (TrainOutput, promedio) | 0.1966 |
| Runtime | 8712.69s (~2:25:12, ~145 min) |
| Épocas ejecutadas | 200 |
| Pasos totales | 9.000 |
| Early stopping activado | No (corrió las 200 épocas completas) |
| Mejor step (Trainer mIoU) | 7800 (época ~173.3, mIoU Trainer = 0.932452) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 200 épocas = 9.000 pasos. Cuadra.

> Con patience=30 y eval_steps=200: desde el best step 7800, harían falta 30 evaluaciones sin mejora (hasta step 7800 + 30×200 = 13800) para disparar el early stopping. Como el máximo de pasos es 9000 (200 épocas), el entrenamiento terminó naturalmente antes de que el patience se agotara. Desde step 8000 hasta 9000 hay 6 evaluaciones sin mejora, todas por debajo de 0.932452.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento (Trainer, cada 200 pasos)

| Step | Train Loss | Val Loss | Mean Iou | Mean Acc | Overall Acc |
|---|---|---|---|---|---|
| 200 | 0.806 | 0.256 | 0.797 | 0.900 | 0.891 |
| 400 | 0.635 | 0.188 | 0.845 | 0.913 | 0.921 |
| 600 | 0.499 | 0.150 | 0.879 | 0.937 | 0.939 |
| 800 | 0.387 | 0.140 | 0.888 | 0.940 | 0.944 |
| 1000 | 0.344 | 0.174 | 0.886 | 0.940 | 0.943 |
| 1200 | 0.307 | 0.140 | 0.896 | 0.948 | 0.948 |
| 1400 | 0.272 | 0.137 | 0.901 | 0.948 | 0.951 |
| 1800 | 0.227 | 0.119 | 0.914 | 0.956 | 0.958 |
| 2000 | 0.219 | 0.106 | 0.921 | 0.960 | 0.961 |
| 2400 | 0.173 | 0.106 | 0.924 | 0.961 | 0.963 |
| 3000 | 0.155 | 0.138 | 0.919 | 0.960 | 0.960 |
| 3400 | 0.149 | 0.111 | 0.927 | 0.963 | 0.964 |
| 3600 | 0.151 | 0.107 | 0.930 | 0.964 | 0.966 |
| 4000 | 0.129 | 0.126 | 0.925 | 0.962 | 0.963 |
| 4400 | 0.123 | 0.122 | 0.929 | 0.964 | 0.965 |
| 5000 | 0.120 | 0.111 | 0.932 | 0.965 | 0.967 |
| 6000 | 0.110 | 0.122 | 0.931 | 0.965 | 0.966 |
| 7000 | 0.108 | 0.132 | 0.931 | 0.965 | 0.967 |
| 7600 | 0.101 | 0.131 | 0.932 | 0.965 | 0.967 |
| **7800** | **0.126** | **0.127** | **0.9325** | **0.966** | **0.967** |
| 8000 | 0.092 | 0.135 | 0.931 | 0.965 | 0.966 |
| 9000 | 0.111 | 0.145 | 0.930 | 0.965 | 0.966 |

> Mejor Trainer mIoU: **0.932452** en step 7800 (época ~173.3). Después de eso, 6 evaluaciones consecutivas (steps 8000-9000) sin mejora. El entrenamiento terminó naturalmente al completar las 200 épocas.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.139245 | 0.118521 |
| eval_mean_iou | 0.930070 | 0.940492 |
| eval_mean_accuracy | 0.964848 | 0.969928 |
| eval_overall_accuracy | 0.965875 | 0.973332 |

> Nota: el mIoU del Trainer en test (0.940492) es mayor que en val (0.930070), consistente con el patrón observado en todas las runs anteriores.

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.9052** |
| Median IoU | 0.9368 |
| Std | 0.0851 |
| Min | 0.5047 |
| Max | 0.9836 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8992** |
| Median IoU | 0.9317 |
| Std | 0.0908 |
| Min | 0.4758 |
| Max | 0.9782 |

- ✅ Hipótesis superada: mIoU en test set = 0.8992 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8992 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 200 épocas completas (9.000 pasos). El mejor checkpoint fue step 7800 (época ~173.3, mIoU Trainer = 0.932452). Después de eso, 6 evaluaciones sin mejora, pero con patience=30 y eval_steps=200 hacían falta 30 para disparar (step 13800 > 9000). El entrenamiento terminó naturalmente.
- **Comparación con ejec. 11 (run 09, 100 épocas, misma config):** la única diferencia es épocas máx (200 vs 100). mIoU test: ejec. 11 = 0.8947 vs esta run = 0.8992 (+0.0045). mIoU val: ejec. 11 = 0.9005 vs esta run = 0.9052 (+0.0047). Duplicar las épocas produjo una mejora marginal de ~0.005 en ambas métricas.
- **Mejor mIoU test hasta ahora:** 0.8992 — supera a ejec. 11 (0.8947) por +0.0045 y a ejec. 10 (0.8938) por +0.0054. Es el mIoU test más alto de todas las ejecuciones registradas.
- **Mejor mIoU val hasta ahora:** 0.9052 — supera a ejec. 11 (0.9005) por +0.0047. También es el mIoU val más alto de todas las ejecuciones.
- **Δ val-test:** 0.9052 - 0.8992 = 0.0060 — similar a ejec. 11 (0.0058). Indica buena generalización, consistente con el resto del bloque steps=200.
- **Variabilidad:** Std en test = 0.0908 — similar a ejec. 11 (0.0931). Consistencia típica.
- **Runtime:** ~145 min (2h25m) — el doble aproximadamente que ejec. 11 (~72 min), como era de esperarse al duplicar las épocas.
- **Plateau visible:** el mIoU del Trainer se estabiliza en el rango 0.92-0.93 a partir de step ~3600 (época ~80). Los últimos ~5.400 pasos (steps 3600-9000) no aportaron mejora significativa. El best checkpoint (step 7800) se capturó tardíamente, pero la diferencia entre el mejor y el peor step del plateau es de solo ~0.003 en mIoU del Trainer.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 4 jul (sobreescribió el de ejec. 11).

## Comparación con ejecuciones 11 y 10 (bloque steps=200, patience=30/20)

| Métrica | Ejec. 10 (100ep, p=20) | Ejec. 11 (100ep, p=30) | **Ejec. 12 (200ep, p=30)** | Δ 12 vs 11 | Δ 12 vs 10 |
|---|---|---|---|---|---|
| Épocas máx | 100 | 100 | 200 | +100 | +100 |
| Épocas ejecutadas | 100 | 100 | 200 | +100 | +100 |
| Pasos totales | 4.500 | 4.500 | 9.000 | +4.500 | +4.500 |
| Training loss final | 0.1281 | 0.0970 | 0.1966 | +0.0996 | +0.0685 |
| Runtime | ~73 min | ~72 min | ~145 min | +73 min | +72 min |
| **mIoU val (evaluate_set)** | 0.8989 | 0.9005 | **0.9052** | +0.0047 | +0.0063 |
| mIoU val median | 0.9346 | 0.9375 | 0.9368 | -0.0007 | +0.0022 |
| mIoU val std | 0.0966 | 0.1001 | 0.0851 | -0.0150 | -0.0115 |
| mIoU val min | 0.5262 | 0.5212 | 0.5047 | -0.0165 | -0.0215 |
| mIoU val max | 0.9857 | 0.9858 | 0.9836 | -0.0022 | -0.0021 |
| **mIoU test (evaluate_set)** | 0.8938 | 0.8947 | **0.8992** | +0.0045 | +0.0054 |
| mIoU test median | 0.9281 | 0.9298 | 0.9317 | +0.0019 | +0.0036 |
| mIoU test std | 0.0905 | 0.0931 | 0.0908 | -0.0023 | +0.0003 |
| mIoU test min | 0.5769 | 0.5682 | 0.4758 | -0.0924 | -0.1011 |
| mIoU test max | 0.9797 | 0.9854 | 0.9782 | -0.0072 | -0.0015 |
| Δ val-test | 0.0051 | 0.0058 | 0.0060 | +0.0002 | +0.0009 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | — | — |

> Nota: el min IoU en test bajó de 0.5682 (ejec. 11) a 0.4758 (esta run). Aunque el mIoU promedio mejoró, la peor predicción empeoró. Esto sugiere que 200 épocas pueden causar ligero overfitting en casos extremos, pero el impacto general es positivo.

## Conclusión

- **Ejec. 12 es la mejor ejecución registrada** tanto en mIoU val (0.9052) como en mIoU test (0.8992).
- Duplicar las épocas de 100 a 200 produjo una mejora de +0.0045 en test y +0.0047 en val respecto a ejec. 11 (la mejor del grid search).
- El runtime se duplicó (~72 min → ~145 min) por el doble de épocas.
- El plateau del mIoU del Trainer se estabiliza a partir de step ~3600 (época ~80), sugiriendo que las ~120 épocas adicionales (steps 3600-9000) aportaron mejora marginal pero medible.
- **Candidata a ser reportada en el manuscrito** como configuración final, superando a ejec. 11 (la mejor del grid search).

## 🔗 Notas relacionadas
- [[ejecucion-10-mitb3-100epocas-patience20-steps200]] — ejecución 10, misma config con 100 épocas y patience=20
- [[ejecucion-11-mitb3-100epocas-patience30-steps200]] — ejecución 11, misma config con 100 épocas y patience=30
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
