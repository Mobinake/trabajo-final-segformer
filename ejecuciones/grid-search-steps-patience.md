---
type: project
subtype: experiment-log
fecha: 2026-06-27
proposito: Grid search 3x3 de eval_steps (50, 100, 200) x early_stopping_patience (10, 20, 30). Total 9 runs. Script completo para correr en Google Colab.
---

# Grid Search — eval_steps x patience (Runs 03-11)

> **Tarea de la tutora (27-06-2026):** probar variaciones de steps (50, 100, 200) y paciencia (10, 20, 30). Registrar métricas en tabla comparativa.

## Grilla de experimentos

| | patience=10 | patience=20 | patience=30 |
|---|---|---|---|
| steps=50 | Run 03 | Run 04 | Run 05 |
| steps=100 | Run 06 | Run 07 | Run 8 |
| steps=200 | Run 9 | Run 10 | Run 11 |

**Todo lo demás constante:** lr=6e-5, batch=2, grad_accum=4, warmup=50, seed=42, fp16=True, num_train_epochs=100, metric_for_best_model=mean_iou.

## Script para Google Colab

> Pegar DESPUES del setup del script `sugarcane_segformer_v3.py` (dataset, compute_metrics ampliado, collate_fn ya definidos). Ver `08 - Configuración Técnica` § "compute_metrics ampliado" para la version ampliada de metricas.

```python
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer, EarlyStoppingCallback
import json, os

GRID = [
    {"eval_steps": 50,  "patience": 10, "label": "run03_steps50_pat10"},
    {"eval_steps": 50,  "patience": 20, "label": "run04_steps50_pat20"},
    {"eval_steps": 50,  "patience": 30, "label": "run05_steps50_pat30"},
    {"eval_steps": 100, "patience": 10, "label": "run06_steps100_pat10"},
    {"eval_steps": 100, "patience": 20, "label": "run07_steps100_pat20"},
    {"eval_steps": 100, "patience": 30, "label": "run08_steps100_pat30"},
    {"eval_steps": 200, "patience": 10, "label": "run09_steps200_pat10"},
    {"eval_steps": 200, "patience": 20, "label": "run10_steps200_pat20"},
    {"eval_steps": 200, "patience": 30, "label": "run11_steps200_pat30"},
]

RESULTS = []

for cfg in GRID:
    print(f"\n{'='*60}")
    print(f"  INICIANDO {cfg['label']}")
    print(f"{'='*60}\n")

    # Re-instanciar modelo fresco cada run
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=f"segformer-sugarcane-{cfg['label']}",
        learning_rate=6e-5,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["eval_steps"],
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        push_to_hub=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])],
    )

    train_result = trainer.train()

    # Evaluar en val y test
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    row = {
        "label": cfg["label"],
        "eval_steps": cfg["eval_steps"],
        "patience": cfg["patience"],
        "epochs_trained": train_result.metrics.get("epoch", None),
        "total_steps": train_result.metrics.get("total_flos", None),
        "train_loss": train_result.metrics.get("train_loss", None),
        "val_mean_iou": val_metrics.get("eval_mean_iou", None),
        "val_loss": val_metrics.get("eval_loss", None),
        "test_mean_iou": test_metrics.get("eval_mean_iou", None),
        "test_loss": test_metrics.get("eval_loss", None),
        "val_median_iou": val_metrics.get("per_image_iou_median", None),
        "val_std_iou": val_metrics.get("per_image_iou_std", None),
        "test_median_iou": test_metrics.get("per_image_iou_median", None),
        "test_std_iou": test_metrics.get("per_image_iou_std", None),
        "val_precision_sugar": val_metrics.get("sugar_cane_precision", None),
        "val_recall_sugar": val_metrics.get("sugar_cane_recall", None),
        "val_f1_sugar": val_metrics.get("sugar_cane_f1", None),
        "test_precision_sugar": test_metrics.get("sugar_cane_precision", None),
        "test_recall_sugar": test_metrics.get("sugar_cane_recall", None),
        "test_f1_sugar": test_metrics.get("sugar_cane_f1", None),
    }
    RESULTS.append(row)

    # Guardar resultados parciales (por si Colab se corta)
    with open(f"results_{cfg['label']}.json", "w") as f:
        json.dump(row, f, indent=2)

# Tabla final
import pandas as pd
df = pd.DataFrame(RESULTS)
df.to_csv("grid_search_results.csv", index=False)
print(df.to_string())
```

## Notas

- **Re-instanciar el modelo fresco en cada run es OBLIGATORIO.** Si no se hace, el modelo del run anterior queda cargado y los resultados no son comparables.
- Los resultados parciales se guardan como JSON individuales por si Colab se corta a mitad del grid.
- El CSV final (`grid_search_results.csv`) tiene toda la tabla comparativa lista para pegar en el vault.
- Cada run puede tomar entre 30 min y 2 horas dependiendo de cuándo active early stopping. El grid completo puede tardar varias horas.
- Si Colab se corta, se pueden comentar las runs ya completadas en la lista GRID y reanudar desde la siguiente.

## Resultados

> Pendiente hasta que el usuario corra el grid en Colab. Cuando lleguen los resultados, actualizar la tabla comparativa en `README.md` de este directorio y crear archivos individuales por cada run si hay hallazgos importantes.

## 🔗 Notas relacionadas
- [[README]] — tabla comparativa de todas las ejecuciones
- [[08 - Configuración Técnica]] — explicación de TrainingArguments + compute_metrics ampliado
- [[04 - Resultados y Validación]] — métricas que entraron al manuscrito
