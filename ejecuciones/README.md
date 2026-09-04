---
type: project
subtype: experiment-log
fecha: 2026-06-22
proposito: Registro de ejecuciones de entrenamiento del modelo SegFormer MiT-B3 con distintos hiperparámetros. Cada ejecución tiene su propio archivo .md dentro de este directorio.
---

# Ejecuciones de entrenamiento — SegFormer MiT-B3

> **Convención:** cada ejecución de entrenamiento (con parámetros distintos) se registra como un archivo separado en este directorio. El nombre del archivo indica el modelo, el número de épocas máx y el patience del early stopping.

> **Fuente de los datos:** los parámetros y métricas se extraen del log de entrenamiento que el usuario descarga de Colab (archivo `.txt` o `.json` en `~/Downloads/`) y pasa al agente, o que el agente lee directamente si está en la VM. Los resultados también se consolidan en `04 - Resultados y Validación` y `08 - Configuración Técnica`, pero ese registro es el "resumen final" para el manuscrito; este directorio es el "log de laboratorio" completo con todas las ejecuciones, incluyendo las que no entraron al manuscrito.

> **Renumeración 2026-07-24:** las antiguas runs preliminares fueron borradas y la numeración actual es 1-12: runs 1-9 corresponden al grid search y runs 10-12 al bloque post-grid.

## Índice de ejecuciones

| # | Archivo | Fecha | Épocas máx | eval_steps | patience | Épocas ejecutadas | mIoU val | mIoU test | Notas |
|---|---|---|---|---|---|---|---|---|---|
| 01 | ejecucion-01-mitb3-100epocas-patience10-steps50 | 2026-06-27 | 100 | 50 | 10 | 67,78 | 0,8942 | 0,8828 | Grid search. Early stopping en step 3050 (best 2550). |
| 02 | ejecucion-02-mitb3-100epocas-patience20-steps50 | 2026-06-27 | 100 | 50 | 20 | 100 | 0,8927 | 0,8816 | Grid search. Sin early stopping. Best step 3500. |
| 03 | ejecucion-03-mitb3-100epocas-patience30-steps50 | 2026-06-27 | 100 | 50 | 30 | 100 | 0,8941 | 0,8861 | Grid search. Sin early stopping. Best step 3850. |
| 04 | ejecucion-04-mitb3-100epocas-patience20-steps100 | 2026-06-27 | 100 | 100 | 20 | 100 | 0,8958 | 0,8804 | Grid search. Sin early stopping. Best step 3600. |
| 05 | ejecucion-05-mitb3-100epocas-patience10-steps200 | 2026-06-28 | 100 | 200 | 10 | 100 | 0,8995 | 0,8837 | Grid search. Sin early stopping. Best step 4000. |
| 06 | ejecucion-06-mitb3-100epocas-patience30-steps100 | 2026-06-28 | 100 | 100 | 30 | 100 | 0,8998 | 0,8897 | Grid search. Sin early stopping. Best step 4200. |
| 07 | ejecucion-07-mitb3-100epocas-patience10-steps200 | 2026-06-29 | 100 | 200 | 10 | 100 | 0,8952 | — | Grid search. Trainer test=0,934155; evaluate_set(test) pendiente. |
| 08 | ejecucion-08-mitb3-100epocas-patience20-steps200 | 2026-06-29 | 100 | 200 | 20 | 100 | 0,8989 | 0,8938 | Grid search. Cargó checkpoint previo. Best step 3400. |
| 09 | ejecucion-09-mitb3-100epocas-patience30-steps200 | 2026-06-29 | 100 | 200 | 30 | 100 | 0,9005 | 0,8947 | Grid search. Cargó checkpoint previo. Best step 3400. |
| 10 | ejecucion-10-mitb3-200epocas-patience30-steps200 | 2026-07-04 | 200 | 200 | 30 | 200 | 0,9052 | 0,8992 | Post-grid. Sin early stopping. Best step 7800. |
| 11 | ejecucion-11-mitb3-300epocas-patience30-steps200 | 2026-07-27 | 300 | 200 | 30 | 300 | 0,9131 | 0,9050 | Post-grid. Sin early stopping. Best step 11200. |
| 12 | ejecucion-12-mitb3-500epocas-patience30-steps200 | 2026-08-16 | 500 | 200 | 30 | 311,11 | 0,9038 | 0,9041 | Configuración final (re-ejecutada 15-ago). Early stop step 14000 (best 8000). + matriz confusión y Acc/Prec/Rec/F1. |

## Tabla comparativa de runs

> Métricas de `evaluate_set()` a resolución original. Δ v-t = diferencia entre mIoU val y mIoU test.

| Run | eval_steps | patience | Épocas máx. | Épocas ejec. | Pasos totales | Train loss | mIoU val | mIoU test | Δ v-t | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 50 | 10 | 100 | 67,8 | 3.050 | 0,4236 | 0,8942 | 0,8828 | 0,0114 | Early stop. Best step 2550. |
| 02 | 50 | 20 | 100 | 100 | 4.500 | 0,1612 | 0,8927 | 0,8816 | 0,0111 | Sin early stop. Best step 3500. |
| 03 | 50 | 30 | 100 | 100 | 4.500 | 0,3381 | 0,8941 | 0,8861 | 0,0080 | Sin early stop. Best step 3850. |
| 04 | 100 | 20 | 100 | 100 | 4.500 | 0,3393 | 0,8958 | 0,8804 | 0,0154 | Sin early stop. Best step 3600. |
| 05 | 200 | 10 | 100 | 100 | 4.500 | 0,3421 | 0,8995 | 0,8837 | 0,0158 | Sin early stop. Best step 4000. |
| 06 | 100 | 30 | 100 | 100 | 4.500 | 0,3361 | 0,8998 | 0,8897 | 0,0101 | Sin early stop. Best step 4200. |
| 07 | 200 | 10 | 100 | 100 | 4.500 | 0,3453 | 0,8952 | — | — | Sin early stop. Trainer test=0,934155; evaluate_set(test) pendiente. |
| 08 | 200 | 20 | 100 | 100 | 4.500 | 0,1281 | 0,8989 | 0,8938 | 0,0051 | Checkpoint previo. Sin early stop. Best step 3400. |
| 09 | 200 | 30 | 100 | 100 | 4.500 | 0,0970 | 0,9005 | 0,8947 | 0,0058 | Checkpoint previo. Sin early stop. Best step 3400. |
| 10 | 200 | 30 | 200 | 200 | 9.000 | 0,1966 | 0,9052 | 0,8992 | 0,0060 | Sin early stop. Best step 7800. Post-grid. |
| 11 | 200 | 30 | 300 | 300 | 13.500 | 0,1797 | 0,9131 | 0,9050 | 0,0081 | Sin early stop. Best step 11200. Post-grid. |
| 12 | 200 | 30 | 500 | 311,1 | 14.000 | 0,1861 | 0,9038 | 0,9041 | -0,0003 | Early stop (step 14000, best 8000). Configuración final (re-ejecutada). |

> A medida que el usuario haga nuevas ejecuciones con otros parámetros, se agregan filas a esta tabla y se crean los archivos correspondientes.

## Parámetros que pueden variar entre ejecuciones

- Épocas máximas (50, 100, etc.)
- Patience del early stopping (10, 20, 50, etc.)
- Learning rate
- Batch size / gradient accumulation steps
- Data augmentation (on/off, tipos, probabilidades)
- Dataset (número de imágenes, distritos incluidos, split)
- Semilla de split
- Checkpoint base (MiT-B0, B3, B5, etc.)

## Cómo agregar una nueva ejecución

1. Crear un archivo `ejecucion-NN-<modelo>-<epocas>epocas-patience<valor>.md` en este directorio.
2. Copiar el formato de una ejecución actual con parámetros equivalentes y llenar los campos.
3. Agregar una fila a la tabla de índice arriba.
4. Si la ejecución entra al manuscrito, consolidar las métricas en `04 - Resultados y Validación` y los parámetros en `08 - Configuración Técnica`.

## 🔗 Notas relacionadas
- 04 - Resultados y Validación — métricas finales que entraron al manuscrito
- 08 - Configuración Técnica — hiperparámetros exactos del script de entrenamiento
- 03 - Propuesta y Desarrollo — narrativa del pipeline
- 12 - Pipeline de Descarga y Preparación de Imágenes — dataset
