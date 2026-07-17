---
type: project
subtype: experiment-log
fecha: 2026-06-22
proposito: Registro de ejecuciones de entrenamiento del modelo SegFormer MiT-B3 con distintos hiperparámetros. Cada ejecución tiene su propio archivo .md dentro de este directorio.
---

# Ejecuciones de entrenamiento — SegFormer MiT-B3

> **Convención:** cada ejecución de entrenamiento (con parámetros distintos) se registra como un archivo separado en este directorio. El nombre del archivo indica el modelo, el número de épocas máx y el patience del early stopping.

> **Fuente de los datos:** los parámetros y métricas se extraen del log de entrenamiento que el usuario descarga de Colab (archivo `.txt` o `.json` en `~/Downloads/`) y pasa al agente, o que el agente lee directamente si está en la VM. Los resultados también se consolidan en `[[04 - Resultados y Validación]]` y `[[08 - Configuración Técnica]]`, pero ese registro es el "resumen final" para el manuscrito; este directorio es el "log de laboratorio" completo con todas las ejecuciones, incluyendo las que no entraron al manuscrito.

> **Renumeración 2026-07-03:** las antiguas runs 01 y 02 (preliminares, usaron otra cantidad de imágenes) fueron borradas. Las antiguas runs 03-11 se renumeraron como 01-09. Los archivos .md individuales conservan sus nombres originales (ej: `ejecucion-11-*.md` corresponde a la run 09 actual), pero la tabla de índice y el manuscrito usan la numeración nueva (01-09).

## Índice de ejecuciones

| # | Archivo | Fecha | Épocas máx | Patience | Épocas ejecutadas | mIoU test | Notas |
|---|---|---|---|---|---|---|---|
| 01 | [[ejecucion-03-mitb3-100epocas-patience10-steps50]] | 2026-06-27 | 100 | 10 | 67.78 | 0.8828 | Grid search run 01. eval_steps=50, patience=10. Early stopping en step 3050 (best step 2550, época ~57). mIoU val 0.8942. |
| 02 | [[ejecucion-04-mitb3-100epocas-patience20-steps50]] | 2026-06-27 | 100 | 20 | 100 | 0.8816 | Grid search run 02. eval_steps=50, patience=20. Sin early stopping (corrió 100 épocas completas). Best step 3500. mIoU val 0.8927. |
| 03 | [[ejecucion-05-mitb3-100epocas-patience30-steps50]] | 2026-06-27 | 100 | 30 | 100 | 0.8861 | Grid search run 03. eval_steps=50, patience=30. Sin early stopping (corrió 100 épocas completas). Best step 3850. mIoU val 0.8941. |
| 04 | [[ejecucion-06-mitb3-100epocas-patience10-steps100]] | 2026-06-27 | 100 | 10 | 77.78 | 0.8845 | Grid search run 04. eval_steps=100, patience=10. Early stopping en step 3500 (best step 2500, época ~56). mIoU val 0.8914. Run más rápida (61 min). |
| 05 | [[ejecucion-07-mitb3-100epocas-patience20-steps100]] | 2026-06-27 | 100 | 20 | 100 | 0.8804 | Grid search run 05. eval_steps=100, patience=20. Sin early stopping (corrió 100 épocas completas). Best step 3600. mIoU val 0.8958. |
| 06 | [[ejecucion-08-mitb3-100epocas-patience30-steps100]] | 2026-06-28 | 100 | 30 | 100 | 0.8897 | Grid search run 06. eval_steps=100, patience=30. Sin early stopping (corrió 100 épocas completas). Best step 4200. mIoU val 0.8998. |
| 07 | [[ejecucion-09-mitb3-100epocas-patience10-steps200]] | 2026-06-28 | 100 | 10 | 100 | — | Grid search run 07. eval_steps=200, patience=10. Sin early stopping (corrió 100 épocas completas). Best step 3600 (mIoU Trainer 0.922040). ~74 min. mIoU val 0.8952. ⏳ evaluate_set(test) pendiente (celda no ejecutada). |
| 08 | [[ejecucion-10-mitb3-100epocas-patience20-steps200]] | 2026-06-28 | 100 | 20 | 100 | 0.8938 | Grid search run 08. eval_steps=200, patience=20. Sin early stopping (corrió 100 épocas completas). Best step 3400 (mIoU Trainer 0.923451). ~73 min. mIoU val 0.8989. |
| 09 | [[ejecucion-11-mitb3-100epocas-patience30-steps200]] | 2026-06-28 | 100 | 30 | 100 | 0.8947 | Grid search run 09. eval_steps=200, patience=30. Sin early stopping (corrió 100 épocas completas). Best step 3400 (mIoU Trainer 0.924093). ~72 min. mIoU val 0.9005 (más alto del grid). mIoU test 0.8947 (más alto del grid). Mejor run del grid search. |
| 10 | [[ejecucion-12-mitb3-200epocas-patience30-steps200]] | 2026-07-04 | 200 | 30 | 200 | 0.8992 | Post-grid. Duplica épocas a 200 con mejor config del grid (s=200, p=30). Sin early stopping (corrió 200 épocas completas). Best step 7800 (mIoU Trainer 0.932452). ~145 min. mIoU val 0.9052. mIoU test 0.8992. Mejor ejecución registrada. |
| 11 | [[ejecucion-13-mitb3-500epocas-patience30-steps200]] | 2026-07-06 | 500 | 30 | 284.44 | 0.9078 | Post-grid. Sube épocas a 500 con early stopping activo (patience=30). **Early stopping SÍ activado** (frenó en step 12800, best step 6800, época ~151). mIoU Trainer 0.932262. ~232 min. mIoU val 0.9061. mIoU test 0.9078. **Mejor ejecución registrada** (mIoU val y test más altos, std test más bajo). |

## Tabla comparativa de runs (TODO 2 tutora, 27-06-2026)

> Completada con datos del grid search. Métricas de `evaluate_set()` a resolución original. Δ v-t = diferencia entre mIoU val y mIoU test.

| Run | eval_steps | patience | Épocas ejec. | Pasos totales | Train loss | mIoU val | mIoU test | Δ v-t | Notas |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 50 | 10 | 67,8 | 3.050 | 0,4236 | 0,8942 | 0,8828 | 0,0114 | Early stop. Best step 2550. |
| 02 | 50 | 20 | 100 | 4.500 | 0,1612 | 0,8927 | 0,8816 | 0,0111 | Sin early stop. Best step 3500. |
| 03 | 50 | 30 | 100 | 4.500 | 0,1600 | 0,8941 | 0,8861 | 0,0080 | Sin early stop. Best step 3850. |
| 04 | 100 | 10 | 77,8 | 3.500 | 0,1787 | 0,8914 | 0,8845 | 0,0069 | Early stop. Best step 2500. Run más rápida (61 min). |
| 05 | 100 | 20 | 100 | 4.500 | 0,1500 | 0,8958 | 0,8804 | 0,0154 | Sin early stop. Best step 3600. |
| 06 | 100 | 30 | 100 | 4.500 | 0,1576 | 0,8998 | 0,8897 | 0,0101 | Sin early stop. Best step 4200. |
| 07 | 200 | 10 | 100 | 4.500 | 0,1579 | 0,8995 | 0,8837 | 0,0158 | Sin early stop. Best step 3600. ⏳ test pendiente. |
| 08 | 200 | 20 | 100 | 4.500 | 0,1281 | 0,8989 | 0,8938 | 0,0051 | Sin early stop. Best step 3400. |
| 09 | 200 | 30 | 100 | 4.500 | 0,0970 | 0,9005 | 0,8947 | 0,0058 | Sin early stop. Best step 3400. Mejor run. |
| 10 | 200 | 30 | 200 | 9.000 | 0,1966 | 0,9052 | 0,8992 | 0,0060 | Sin early stop. Best step 7800. Post-grid (200 épocas). Mejor mIoU val y test. |
| 11 | 200 | 30 | 284,4 | 12.800 | 0,1824 | 0,9061 | 0,9078 | -0,0017 | **Early stop** (step 12800, best 6800). Post-grid (500 épocas). Mejor mIoU val y test. Std test más bajo (0,0769). |

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
2. Copiar la plantilla de `[[ejecucion-02-mitb3-100epocas-patience50]]` (sección "Pendiente de datos") y llenar los campos.
3. Agregar una fila a la tabla de índice arriba.
4. Si la ejecución entra al manuscrito, consolidar las métricas en `[[04 - Resultados y Validación]]` y los parámetros en `[[08 - Configuración Técnica]]`.

## 🔗 Notas relacionadas
- [[04 - Resultados y Validación]] — métricas finales que entraron al manuscrito
- [[08 - Configuración Técnica]] — hiperparámetros exactos del script de entrenamiento
- [[03 - Propuesta y Desarrollo]] — narrativa del pipeline
- [[12 - Pipeline de Descarga y Preparación de Imágenes]] — dataset
