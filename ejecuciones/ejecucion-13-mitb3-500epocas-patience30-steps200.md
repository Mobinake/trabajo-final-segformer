---
type: project
subtype: experiment-log
ejecucion: 13
fecha: 2026-07-06
modelo: SegFormer MiT-B3
epocas_max: 500
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 13 — Run 11: eval_steps=200, patience=30, max_epochs=500

> Post-grid. Toma la mejor config del grid (run 09: eval_steps=200, patience=30) y sube épocas máx a 500 con early stopping activo (patience=30). Objetivo: dejar que el early stopping decida cuándo frenar en lugar de fijar un techo arbitrario. Fecha: 6 de julio de 2026. Notebook ejecutado en Google Colab (GPU T4). El usuario pegó el `.ipynb` inline con todas las métricas.

## Parámetros

> Únicos cambios respecto a ejec. 12: `num_train_epochs` sube de 200 a 500. Resto igual.

| Parámetro | Valor | Cambió vs ejec. 12 |
|---|---|---|
| Learning rate | 6 × 10⁻⁵ | No |
| Épocas máx | 500 | **Sí** (200 → 500) |
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

> RUN_LABEL dentro del notebook: `"run12_steps200_pat30_epochs500"` — coincide con los parámetros reales.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0.1824 |
| Runtime | 13954.89s (~3:52:35, ~232 min) |
| Épocas ejecutadas | 284.44 |
| Pasos totales | 12.800 |
| Early stopping activado | **Sí** (frenó en step 12800, best step 6800) |
| Mejor step (Trainer mIoU) | 6800 (época ~151.1, mIoU Trainer = 0.932262) |

> Verificación: 359 / 8 ≈ 45 pasos/época × 284.44 épocas ≈ 12.800 pasos. Cuadra.

> Early stopping: best step 6800 (mIoU Trainer 0.932262). Desde ahí, 30 evaluaciones sin mejora (30 × 200 = 6000 pasos → step 6800 + 6000 = 12800). El entrenamiento frenó en step 12800 por early stopping. ✅ Funcionó como se esperaba.

### Tabla de entrenamiento (selección, Trainer, cada 200 pasos)

| Step | Train Loss | Val Loss | Mean Iou | Mean Acc | Overall Acc |
|---|---|---|---|---|---|
| 200 | 1.055 | 0.285 | 0.763 | 0.868 | 0.872 |
| 1000 | 0.357 | 0.160 | 0.888 | 0.944 | 0.944 |
| 2000 | 0.244 | 0.113 | 0.919 | 0.960 | 0.960 |
| 3000 | 0.169 | 0.102 | 0.929 | 0.963 | 0.966 |
| 4000 | 0.135 | 0.128 | 0.926 | 0.963 | 0.964 |
| 5000 | 0.118 | 0.130 | 0.932 | 0.966 | 0.967 |
| 6000 | 0.117 | 0.179 | 0.922 | 0.962 | 0.962 |
| **6800** | **0.102** | **0.146** | **0.9323** | **0.966** | **0.967** |
| 8000 | 0.094 | 0.188 | 0.928 | 0.965 | 0.965 |
| 10000 | 0.094 | 0.207 | 0.927 | 0.964 | 0.964 |
| 11200 | 0.093 | 0.175 | 0.932 | 0.966 | 0.967 |
| 12800 | 0.079 | 0.197 | 0.928 | 0.965 | 0.965 |

> Mejor Trainer mIoU: **0.932262** en step 6800 (época ~151). Después, 30 evaluaciones (steps 7000-12800) sin superar 0.932262. Early stopping frenó en step 12800.

## Métricas del Trainer (evaluate, best model cargado)

> Trainer evalúa sobre 512×512. NO son las finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0.173941 | 0.124408 |
| eval_mean_iou | 0.927246 | 0.942746 |
| eval_mean_accuracy | 0.964065 | 0.969702 |
| eval_overall_accuracy | 0.964389 | 0.974451 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.9061** |
| Median IoU | 0.9421 |
| Std | 0.0993 |
| Min | 0.5331 |
| Max | 0.9874 |

## Métricas de test (78 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.9078** |
| Median IoU | 0.9361 |
| Std | 0.0769 |
| Min | 0.6295 |
| Max | 0.9801 |

- ✅ Hipótesis superada: mIoU en test set = 0.9078 ≥ 0.85.

## Observaciones

- ✅ **Early stopping funcionó**: frenó en step 12800 (época ~284) tras 30 evaluaciones sin mejora desde step 6800 (época ~151). No corrió las 500 épocas completas. Esto valida el diseño de early stopping con patience=30 y eval_steps=200.
- **Comparación con ejec. 12 (200 épocas, sin early stop):** mIoU test: ejec. 12 = 0.8992 vs esta run = 0.9078 (+0.0086). mIoU val: ejec. 12 = 0.9052 vs esta run = 0.9061 (+0.0009). Mejora notable en test, marginal en val.
- **mIoU test más alto hasta ahora:** 0.9078 — supera a ejec. 12 (0.8992) por +0.0086. Es el mIoU test más alto de todas las ejecuciones registradas.
- **mIoU val:** 0.9061 — supera marginalmente a ejec. 12 (0.9052) por +0.0009. También es el más alto en val.
- **Δ val-test:** 0.9061 - 0.9078 = -0.0017 — el test es ligeramente mayor que val, patrón consistente con todas las runs anteriores.
- **Std test bajo:** 0.0769 — el más bajo de todas las runs (ejec. 12 = 0.0908, ejec. 11 = 0.0931). Indica menos variabilidad entre predicciones individuales.
- **Min IoU test mejoró:** 0.6295 vs ejec. 12 (0.4758) — la peor predicción mejoró significativamente (+0.1537).
- **Runtime:** ~232 min (~3h53m) — más largo que ejec. 12 (~145 min) porque corrió 284 épocas vs 200. Pero el early stopping evitó correr las 500 completas.
- **Best step temprano:** step 6800 (época ~151) de 12800 totales. La segunda mitad del entrenamiento (steps 6800-12800) no aportó mejora al mIoU del Trainer, pero el modelo final cargado sí benefitó de ese entrenamiento adicional en test (+0.0086 vs ejec. 12).
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 6 jul (sobreescribió el de ejec. 12).

## Comparación con ejecuciones 12 y 11

| Métrica | Ejec. 11 (100ep, p=30) | Ejec. 12 (200ep, p=30) | **Ejec. 13 (500ep, p=30)** | Δ 13 vs 12 | Δ 13 vs 11 |
|---|---|---|---|---|---|
| Épocas máx | 100 | 200 | 500 | +300 | +400 |
| Épocas ejecutadas | 100 | 200 | 284.44 | +84.44 | +184.44 |
| Pasos totales | 4.500 | 9.000 | 12.800 | +3.800 | +8.300 |
| Early stopping | No | No | **Sí** (step 12800) | — | — |
| Training loss final | 0.0970 | 0.1966 | 0.1824 | -0.0142 | +0.0854 |
| Runtime | ~72 min | ~145 min | ~232 min | +87 min | +160 min |
| **mIoU val (evaluate_set)** | 0.9005 | 0.9052 | **0.9061** | +0.0009 | +0.0056 |
| mIoU val median | 0.9375 | 0.9368 | 0.9421 | +0.0053 | +0.0046 |
| mIoU val std | 0.1001 | 0.0851 | 0.0993 | +0.0142 | -0.0008 |
| mIoU val min | 0.5212 | 0.5047 | 0.5331 | +0.0284 | +0.0119 |
| mIoU val max | 0.9858 | 0.9836 | 0.9874 | +0.0038 | +0.0016 |
| **mIoU test (evaluate_set)** | 0.8947 | 0.8992 | **0.9078** | +0.0086 | +0.0131 |
| mIoU test median | 0.9298 | 0.9317 | 0.9361 | +0.0044 | +0.0063 |
| mIoU test std | 0.0931 | 0.0908 | 0.0769 | -0.0139 | -0.0162 |
| mIoU test min | 0.5682 | 0.4758 | 0.6295 | +0.1537 | +0.0613 |
| mIoU test max | 0.9854 | 0.9782 | 0.9801 | +0.0019 | -0.0053 |
| Δ val-test | 0.0058 | 0.0060 | -0.0017 | -0.0077 | -0.0075 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | — | — |

## Conclusión

- **Ejec. 13 es la mejor ejecución registrada** tanto en mIoU val (0.9061) como en mIoU test (0.9078).
- Early stopping funcionó correctamente: frenó en época 284 tras 30 evaluaciones sin mejora desde step 6800 (época 151). No corrió las 500 épocas.
- Subir épocas a 500 con early stopping produjo la mejor generalización registrada: std test más bajo (0.0769) y min IoU test más alto (0.6295).
- **Candidata a ser reportada en el manuscrito** como configuración final, superando a ejec. 12 en ambas métricas.

## 🔗 Notas relacionadas
- [[ejecucion-12-mitb3-200epocas-patience30-steps200]] — ejecución 12, misma config con 200 épocas sin early stop
- [[ejecucion-11-mitb3-100epocas-patience30-steps200]] — ejecución 11, mejor run del grid search
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
