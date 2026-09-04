---
type: project
subtype: experiment-log
ejecucion: 12
fecha: 2026-08-16
modelo: SegFormer MiT-B3
epocas_max: 500
patience: 30
eval_steps: 200
estado: completo
---

# Ejecución 12 — run 12: eval_steps=200, patience=30, max_epochs=500 (re-ejecutada)

> **Configuración final re-ejecutada.** Mismos hiperparámetros que la corrida original (06-07-2026). Re-ejecutada en Google Colab (GPU T4) el 15-ago-2026 (~23:37 UTC). Los valores de esta corrida reemplazan a los de la versión original, para que las métricas del manuscrito (mIoU, matriz de confusión, Accuracy/Precision/Recall/F1) salgan de una única corrida. El usuario la llamó temporalmente "run12-b"; queda registrada como run 12.

> ⚠️ **Efecto de la re-ejecución:** el entrenamiento es estocástico (FP16, orden de reducción en GPU, dataloader con `num_workers=2`), así que esta corrida difiere de la original: early stopping frenó en step 14.000 (época 311) con best step 8.000, y el mIoU de `evaluate_set()` bajó de 0,9061/0,9078 a 0,9038/0,9041 (→ 0,90). La run 11 (300 épocas) quedó con mIoU marginalmente mayor (val 0,9131 / test 0,9050). La run 12 se mantiene como configuración final por el early stopping y por su desviación estándar de prueba más baja (0,08 frente a 0,09).

## Parámetros

> Idénticos a la corrida original (ningún hiperparámetro cambió).

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 500 |
| eval_steps | 200 |
| save_steps | 200 |
| Patience early stopping | 30 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| per_device_eval_batch_size | 4 |
| Warmup steps | 50 |
| max_grad_norm | 1,0 |
| FP16 | True |
| Seed | 42 |
| reduce_labels | False |
| Checkpoint base | `nvidia/segformer-b3-finetuned-ade-512-512` |
| Dataset | 513 imgs (3 distritos) |
| Split | 70/15/15 → 359/76/78 |

> RUN_LABEL dentro del notebook: `"run12_steps200_pat30_epochs500"`.

## Dataset

- Total: 513 imágenes
- Distritos: Tebicuary, Itapé, Coronel Martínez
- Split: 70/15/15 (seed=42) → Train: 359 / Val: 76 / Test: 78
- Clases: 0 = background, 1 = sugar_cane

## Resultados de entrenamiento

| Métrica | Valor |
|---|---|
| Training loss (TrainOutput, promedio) | 0,1861 |
| Runtime | 13.259,5 s (~3:41, ~221 min) |
| Épocas ejecutadas | 311,11 |
| Pasos totales (global_step) | 14.000 |
| Early stopping activado | Sí (frenó en step 14.000, best step 8.000) |
| Mejor step (Trainer mIoU) | 8.000 (época ~177,8, mIoU Trainer = 0,930165) |

> Verificación: 45 pasos/época × 311,11 ≈ 14.000 pasos. Early stopping: best step 8.000 + 30×200 = 14.000. ✅

## Métricas del Trainer (evaluate, best model cargado, 512×512)

> NO son las finales del manuscrito.

| Métrica | Val (76 imgs) | Test (78 imgs) |
|---|---|---|
| eval_loss | 0,256514 | 0,141535 |
| eval_mean_iou | 0,923620 | 0,943122 |
| eval_mean_accuracy | 0,961515 | 0,971435 |
| eval_overall_accuracy | 0,962583 | 0,974544 |

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| Mean IoU (sugar_cane) | 0,9038 |
| Median IoU | 0,9401 |
| Std | 0,0931 |
| Min | 0,6214 |
| Max | 0,9866 |

## Métricas de test (78 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| Mean IoU (sugar_cane) | 0,9041 |
| Median IoU | 0,9374 |
| Std | 0,0793 |
| Min | 0,6258 |
| Max | 0,9824 |

- ✅ Hipótesis superada: mIoU test = 0,9041 ≥ 0,85.

## Matriz de confusión + Accuracy/Precision/Recall/F1 (resolución original)

> Clase positiva = caña. Filas = real, columnas = predicho.

### Validación (76 imágenes)

```
            pred. bg      pred. caña
real bg        752.492       25.699
real caña       19.111      447.882
```

| Métrica | Valor |
|---|---|
| Accuracy | 0,9640 |
| Precision | 0,9457 |
| Recall | 0,9591 |
| F1-score | 0,9524 |

### Prueba (78 imágenes)

```
            pred. bg      pred. caña
real bg        857.378       16.525
real caña       14.173      389.876
```

| Métrica | Valor |
|---|---|
| Accuracy | 0,9760 |
| Precision | 0,9593 |
| Recall | 0,9649 |
| F1-score | 0,9621 |

## Reconciliación de accuracy

- `eval_overall_accuracy` del Trainer (512×512) test = 0,974544 → era el valor 0,9745 de `tab:comparison`.
- Accuracy a resolución original (esta celda) = 0,9760 test / 0,9640 val.
- El manuscrito quedó con la accuracy a resolución original (0,9760), consistente con el mIoU (también a resolución original).

## Observaciones

- Re-ejecución con resultados levemente distintos a pesar de seed=42 (estocasticidad FP16/GPU). Early stop más tardío (época 311 vs 284) y mIoU ~0,003 menor.
- La run 11 (300 épocas) quedó con mIoU marginalmente mayor; la run 12 se mantiene como final por el early stopping (corta solo cuando deja de mejorar) y la std de prueba más baja (0,08).
- Min val mejoró (0,6214 vs 0,5331 de la corrida original).
- Por primera vez se tienen Precision/Recall/F1 y matriz de confusión (val y test) a resolución original → lo que faltaba para C4-10 y C4-11 del manuscrito.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 15-ago (sobreescribió el de la corrida original).

## 🔗 Notas relacionadas
- ejecucion-11-mitb3-300epocas-patience30-steps200 — post-grid 300 épocas
- ejecucion-10-mitb3-200epocas-patience30-steps200 — post-grid 200 épocas
- grid-search-steps-patience — grid 1-9
- README — índice
- 07 - Pendientes y Notas para Expandir — C4-10 / C4-11
