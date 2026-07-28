# -*- coding: utf-8 -*-
"""sugarcane_segformer_v3.py

Fine-Tuning SegFormer (MIT-B3) para segmentacion de cana de azucar con Sentinel-2.

CORRECCIONES vs version original:
  1. Color detection unificado: una sola funcion mask_rgb_to_class() usada en TODAS las celdas.
     Antes mask_to_class usaba R(70-110),G(190-220),B(55-95) pero la inferencia usaba R<50,G>100,B<50
     que NO detecta el color real RGB(88,207,74) porque R=88>50 y B=74>50. Resultado: IoU=0 en eval.
  2. reduce_labels=False explicito en SegformerImageProcessor.
     Si el checkpoint nvidia/segformer-b3-finetuned-ade-512-512 tiene reduce_labels=True (viene de ADE20k), resta 1 a los labels:
     0(background)->255(ignore), 1(cana)->0. El modelo entrenaria con 1 sola clase efectiva. SILENCIOSO.
  3. Split 70/15/15 (train/val/TEST) en vez de 80/20 sin test. Plan del vault: 359/76/78.
  4. Data augmentation con albumentations (flip, rotacion, brightness). 513 imagenes lo necesitan.
  5. EarlyStoppingCallback(patience=30) para evitar overfitting en 500 epochs.
  6. max_grad_norm=1.0 y seed=42 explicitos en TrainingArguments.
  7. Docstring corregido: MIT-B3 (no B0).
  8. collate_fn explicito para mayor robustez.
  9. Seccion de evaluacion final en TEST set (nunca visto en entrenamiento).
  10. Seccion de validacion externa con San Salvador (placeholder listo para usar).

Clases: 0 = background | 1 = sugar_cane
Color mascara CVAT: verde RGB(88, 207, 74) = cana, negro = background
"""

# %% Cell 1: Instalacion de dependencias
!pip install -q datasets transformers evaluate torch torchvision pillow matplotlib opencv-python albumentations
print("Dependencias instaladas")

"""## 1. Estructura de carpetas en Drive

Antes de ejecutar, creá esta estructura en tu Drive:
```
MiDrive/
└── sugarcane/
    ├── images/           <- tiles .jpg (Sentinel-2 RGB de los 3 distritos)
    ├── masks/            <- mascaras .png (exportacion CVAT Segmentation Mask 1.1)
    ├── test_images/      <- (opcional) imagenes sueltas para inferencia rapida
    └── san_salvador/     <- (opcional) imagenes de San Salvador para validacion externa
        ├── images/
        └── masks/
```
"""

# %% Cell 2: Montar Drive y definir rutas
from google.colab import drive
drive.mount('/content/drive')

# CAMBIA ESTO si tus carpetas estan en otra ruta
BASE_DIR = "/content/drive/MyDrive/sugarcane"
IMAGE_DIR = BASE_DIR + "/images"
MASK_DIR = BASE_DIR + "/masks"
SAN_SALVADOR_DIR = BASE_DIR + "/san_salvador"  # para validacion externa (mas adelante)

import os
print(f"Imagenes: {IMAGE_DIR}")
print(f"Mascaras: {MASK_DIR}")
print(f"San Salvador: {SAN_SALVADOR_DIR}")

img_count = len(os.listdir(IMAGE_DIR)) if os.path.isdir(IMAGE_DIR) else 0
mask_count = len(os.listdir(MASK_DIR)) if os.path.isdir(MASK_DIR) else 0
print(f"Imagenes encontradas: {img_count}")
print(f"Mascaras encontradas: {mask_count}")

# %% Cell 3: Funcion compartida para convertir mascaras RGB -> clases
# ============================================================
# UNA SOLA funcion usada en TODO el notebook (dataset, inferencia, evaluacion).
# Antes habia 2 formulas distintas: la del dataset funcionaba, la de inferencia NO.
#
# Color cana: RGB(88, 207, 74). Deteccion robusta:
#   - G > 100: capta el verde (88,207,74) tiene G=207; (0,255,0) tiene G=255.
#   - R < 150 y B < 150: excluye rojos, amarillos, azules, cianes.
#   - Funciona para (88,207,74), (0,255,0) y cualquier verde razonable de CVAT.
# ============================================================
import numpy as np

def mask_rgb_to_class(mask_rgb):
    """Convierte mascara RGB a indices de clase: 0=background, 1=sugar_cane.

    Args:
        mask_rgb: numpy array (H, W, 3) en formato RGB.
    Returns:
        numpy array (H, W) int64 con valores 0 (background) o 1 (sugar_cane).
    """
    r, g, b = mask_rgb[:, :, 0], mask_rgb[:, :, 1], mask_rgb[:, :, 2]
    sugar_cane = (g > 100) & (r < 150) & (b < 150)
    out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    out[sugar_cane] = 1
    return out

# %% Cell 4: Dataset personalizado (con augmentacion opcional)
import os, cv2
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import albumentations as A

class SugarCaneDataset(Dataset):
    """Lee imagenes .jpg + mascaras .png y las matchea por nombre de archivo.

    Args:
        image_dir: carpeta con imagenes .jpg
        mask_dir: carpeta con mascaras .png
    """
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir

        # Matchear mascaras con imagenes por nombre (sin extension)
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.samples = []

        for mask_path in mask_paths:
            name = os.path.basename(mask_path)
            img_name = name.replace(".png", ".jpg")
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, mask_path))

        print(f"Pares imagen-mascara cargados: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Leer imagen (BGR -> RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Leer mascara (BGR -> RGB -> clases)
        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = mask_rgb_to_class(mask_rgb)  # funcion compartida

        return {"pixel_values": image, "label": mask}

# Verificar que el dataset carga correctamente
dataset = SugarCaneDataset(IMAGE_DIR, MASK_DIR)
s = dataset[0]
print(f"Imagen: {s['pixel_values'].shape}")
print(f"Mascara: {s['label'].shape}, clases: {np.unique(s['label'])}")

# Mostrar una muestra
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(s['pixel_values'])
axes[0].set_title("Imagen"); axes[0].axis('off')
axes[1].imshow(s['label'], cmap='viridis')
axes[1].set_title(f"Mascara (clases: {np.unique(s['label'])})"); axes[1].axis('off')
plt.show()

# %% Cell 5: Diagnostico de colores en mascaras (VERIFICAR ANTES DE ENTRENAR)
# ============================================================
# Corre esta celda para confirmar que colores tiene realmente tu mascara CVAT.
# Deberias ver: negro (0,0,0) = background y verde (88,207,74) = sugar_cane.
# Si ves otros colores, ajusta mask_rgb_to_class en Cell 3.
# ============================================================
def diagnose_mask_colors(mask_dir, num_samples=3):
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))[:num_samples]
    print(f'Analizando {len(mask_paths)} mascaras...\n')
    for mp in mask_paths:
        mask_bgr = cv2.imread(mp)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        pixels = mask_rgb.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        print(f'{os.path.basename(mp)}:')
        print(f'   Colores unicos: {len(unique_colors)}')
        for color in unique_colors[:10]:
            count = np.sum(np.all(pixels == color, axis=1))
            print(f'   - RGB{tuple(color)}: {count} pixeles')
        print()

diagnose_mask_colors(MASK_DIR, num_samples=3)

# %% Cell 6: Split 70/15/15 (train / val / test)
# ============================================================
# ANTES era 80/20 (sin test set). El plan del vault es 70/15/15:
#   513 imagenes -> 359 train / 76 val / 78 test
# El test set NUNCA se usa durante entrenamiento. Se evalua al final.
# ============================================================
import torch
from torch.utils.data import random_split

torch.manual_seed(42)
total = len(dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_raw, val_raw, test_raw = random_split(dataset, [train_size, val_size, test_size])
print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

# %% Cell 7: Procesador SegFormer + adaptador HF + collate_fn
from transformers import SegformerImageProcessor

labels = ["background", "sugar_cane"]
id2label = {0: "background", 1: "sugar_cane"}
label2id = {"background": 0, "sugar_cane": 1}

model_checkpoint = "nvidia/segformer-b3-finetuned-ade-512-512"

# CORRECCION CRITICA: reduce_labels=False EXPLICITO
# El checkpoint viene de ADE20k donde reduce_labels=True resta 1 a los labels.
# Sin este parametro, si el config del checkpoint tiene True:
#   0(background) -> 255(ignore), 1(sugar_cane) -> 0
# El modelo entrenaria con 1 sola clase efectiva. FALLA SILENCIOSAMENTE.
image_processor = SegformerImageProcessor.from_pretrained(
    model_checkpoint,
    reduce_labels=False  # <-- CRITICO
)

class HFDatasetAdapter(Dataset):
    """Adapta torch Dataset -> HuggingFace Trainer.
    Aplica image_processor (resize 512x512 + normalizacion ImageNet) a cada muestra.
    """
    def __init__(self, subset, image_processor, augment=False):
        self.subset = subset
        self.image_processor = image_processor
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        image = item["pixel_values"]
        mask = item["label"]

        # Augmentacion antes del image_processor
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # image_processor: resize 512x512 + normalizacion
        batch = self.image_processor(
            [image], [mask],
            return_tensors="pt"
        )
        return {
            "pixel_values": batch["pixel_values"].squeeze(0),
            "labels": batch["labels"].squeeze(0),
        }

# collate_fn explicito: apila tensores en batch
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

# Train CON augmentacion, val y test SIN augmentacion
train_ds = HFDatasetAdapter(train_raw, image_processor, augment=True)
val_ds = HFDatasetAdapter(val_raw, image_processor, augment=False)
test_ds = HFDatasetAdapter(test_raw, image_processor, augment=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
s = train_ds[0]
print(f"pixel_values: {s['pixel_values'].shape}, labels: {s['labels'].shape}")
print(f"Clases en labels: {torch.unique(s['labels']).tolist()}")

# %% Cell 8: Cargar modelo SegFormer MIT-B3
from transformers import SegformerForSemanticSegmentation
import torch

model = SegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # necesario: el checkpoint tiene 150 clases (ADE20k), nosotros 2
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modelo en {device}")

p = sum(p.numel() for p in model.parameters())
print(f"Parametros: {p:,}")

# %% Cell 9: Metricas (mean IoU)
import evaluate
import torch.nn as nn

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)

    # Logits vienen a 1/4 de resolucion (H/4, W/4). Upscalear a tamaño del label.
    logits_t = nn.functional.interpolate(
        logits_t, size=labels_t.shape[-2:], mode="bilinear", align_corners=False
    ).argmax(dim=1)

    preds = logits_t.cpu().numpy()
    m = metric.compute(
        predictions=preds,
        references=labels,
        num_labels=2,
        ignore_index=255,
        reduce_labels=False  # consistente con el image_processor
    )
    # Quitar metricas por categoria para limpiar logs
    m.pop("per_category_accuracy", None)
    m.pop("per_category_iou", None)
    return m

print("mean_iou configurado (ignore_index=255, 2 clases, reduce_labels=False)")

# %% Cell 10: TrainingArguments + Early Stopping + Trainer
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="segformer-sugarcane",
    learning_rate=6e-5,
    num_train_epochs=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,   # eval puede usar batch mas grande (sin gradientes)
    gradient_accumulation_steps=4,  # batch efectivo = 2*4 = 8
    warmup_steps=50,
    eval_strategy="steps",          # en transformers <4.46 usar "evaluation_strategy"
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    push_to_hub=False,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    seed=42,                # reproducibilidad
    max_grad_norm=1.0,      # gradient clipping (default, pero explicito)
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],  # para si mIoU no mejora en 30 evals
)
print("Trainer listo con Early Stopping (patience=30)")

# %% Cell 11: Entrenar
trainer.train()

# %% Cell 12: Guardar modelo en Drive
SAVE_PATH = "/content/drive/MyDrive/sugarcane/segformer-sugarcane-final"
trainer.save_model(SAVE_PATH)
image_processor.save_pretrained(SAVE_PATH)
print(f"Modelo guardado en: {SAVE_PATH}")
!ls -lh "{SAVE_PATH}"

# %% Cell 13: Visualizar curvas de entrenamiento
import matplotlib.pyplot as plt

log_history = trainer.state.log_history

# Extraer datos de loss y mIoU
train_loss = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
eval_loss = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]
eval_miou = [(x["step"], x["eval_mean_iou"]) for x in log_history if "eval_mean_iou" in x]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
if train_loss:
    axes[0].plot([x[0] for x in train_loss], [x[1] for x in train_loss], label="Train loss", alpha=0.7)
if eval_loss:
    axes[0].plot([x[0] for x in eval_loss], [x[1] for x in eval_loss], label="Val loss", marker='o', markersize=3)
axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss"); axes[0].set_title("Curva de perdida")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# mIoU
if eval_miou:
    axes[1].plot([x[0] for x in eval_miou], [x[1] for x in eval_miou], label="Val mIoU", marker='o', color='green')
    axes[1].axhline(y=0.85, color='red', linestyle='--', label="Hipotesis (0.85)")
axes[1].set_xlabel("Step"); axes[1].set_ylabel("mean IoU"); axes[1].set_title("mIoU en validacion")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/sugarcane/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Curvas guardadas en: /content/drive/MyDrive/sugarcane/training_curves.png")

# %% Cell 14: Cargar modelo entrenado + procesador
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
import matplotlib.pyplot as plt
import cv2

MODEL_PATH = "/content/drive/MyDrive/sugarcane/segformer-sugarcane-final"

model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
image_processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Modelo cargado desde {MODEL_PATH}")

# %% Cell 15: Funcion de inferencia + visualizacion
# ============================================================
# CORRECCION: usa mask_rgb_to_class() (la MISMA funcion del dataset)
# para leer el ground truth. Antes usaba R<50,G>100,B<50 que NO detecta
# el color RGB(88,207,74) porque R=88>50 y B=74>50.
# ============================================================

def predict_and_show(image_path, mask_path=None, save_path=None):
    """Ejecuta inferencia sobre una imagen y muestra resultado lado a lado."""
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    inputs = image_processor(images=img_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(pixel_values=inputs["pixel_values"]).logits

    # Upscalear a tamaño original
    logits_upsampled = torch.nn.functional.interpolate(
        logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred_mask = logits_upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

    # Overlay: verde donde predice sugar_cane
    overlay = img_rgb.copy()
    overlay[pred_mask == 1] = [0, 200, 0]

    n_cols = 4 if mask_path else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen original"); axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"Prediccion\n(cana: {(pred_mask==1).sum():,} px)"); axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (verde = cana)"); axes[2].axis("off")

    if mask_path:
        # CORRECCION: usar mask_rgb_to_class (funcion compartida) en vez de umbrales inline
        gt_bgr = cv2.imread(mask_path)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_class = mask_rgb_to_class(gt_rgb)

        axes[3].imshow(gt_class, cmap="viridis", vmin=0, vmax=1)
        axes[3].set_title(f"Ground truth\n(cana: {(gt_class==1).sum():,} px)"); axes[3].axis("off")

        # IoU de esta imagen
        intersection = ((pred_mask == 1) & (gt_class == 1)).sum()
        union = ((pred_mask == 1) | (gt_class == 1)).sum()
        iou = intersection / union if union > 0 else 0
        fig.suptitle(f"IoU cana: {iou:.3f}", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return pred_mask

# %% Cell 16: Probar con 3 imagenes al azar del val set
import random

val_indices = list(range(len(val_raw)))
random.seed(42)
test_idxs = random.sample(val_indices, min(3, len(val_indices)))

for idx in test_idxs:
    img_path = dataset.samples[val_raw.indices[idx]][0]
    mask_path = dataset.samples[val_raw.indices[idx]][1]

    print(f"\n{'='*60}")
    print(f"Muestra #{idx} - {os.path.basename(img_path)}")
    predict_and_show(img_path, mask_path)

# %% Cell 17: IoU promedio en todo el validation set
from tqdm.notebook import tqdm

def evaluate_set(subset, dataset, image_processor, model, device, set_name="val"):
    """Evalua IoU sobre un subset (val o test) del dataset."""
    ious = []
    for idx in tqdm(range(len(subset)), desc=f"Evaluando {set_name} set"):
        img_path = dataset.samples[subset.indices[idx]][0]
        mask_path = dataset.samples[subset.indices[idx]][1]

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        inputs = image_processor(images=img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(pixel_values=inputs["pixel_values"]).logits

        logits_up = torch.nn.functional.interpolate(
            logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False
        )
        pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

        # CORRECCION: mask_rgb_to_class en vez de umbrales inline
        gt_bgr = cv2.imread(mask_path)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_class = mask_rgb_to_class(gt_rgb)

        inter = ((pred == 1) & (gt_class == 1)).sum()
        union = ((pred == 1) | (gt_class == 1)).sum()
        ious.append(inter / union if union > 0 else 0)

    ious = np.array(ious)
    print(f"\nMetricas en {set_name} set ({len(ious)} muestras):")
    print(f"   Mean IoU (cana): {ious.mean():.4f}")
    print(f"   Median IoU:      {np.median(ious):.4f}")
    print(f"   Std:             {ious.std():.4f}")
    print(f"   Min:             {ious.min():.4f}")
    print(f"   Max:             {ious.max():.4f}")
    return ious

# Evaluar validation set
val_ious = evaluate_set(val_raw, dataset, image_processor, model, device, "val")

# Histograma
plt.figure(figsize=(8, 4))
plt.hist(val_ious, bins=10, edgecolor="black", alpha=0.7)
plt.xlabel("IoU (cana)"); plt.ylabel("Frecuencia")
plt.title("Distribucion de IoU en validation set")
plt.axvline(val_ious.mean(), color="red", linestyle="--", label=f"Media: {val_ious.mean():.3f}")
plt.legend(); plt.show()

# %% Cell 18: Evaluacion final en TEST set (nunca visto en entrenamiento)
# ============================================================
# NUEVO: el test set (78 imagenes) nunca se uso durante entrenamiento.
# Estas son las metricas finales que van al Capitulo 4 de la tesis.
# ============================================================
test_ious = evaluate_set(test_raw, dataset, image_processor, model, device, "test")

plt.figure(figsize=(8, 4))
plt.hist(test_ious, bins=10, edgecolor="black", alpha=0.7, color="orange")
plt.xlabel("IoU (cana)"); plt.ylabel("Frecuencia")
plt.title("Distribucion de IoU en TEST set")
plt.axvline(test_ious.mean(), color="red", linestyle="--", label=f"Media: {test_ious.mean():.3f}")
plt.axvline(0.85, color="blue", linestyle="--", label="Hipotesis (0.85)")
plt.legend(); plt.show()

print(f"\n{'='*60}")
print(f"RESULTADO FINAL:")
print(f"  Val  mIoU: {val_ious.mean():.4f}")
print(f"  Test mIoU: {test_ious.mean():.4f}")
print(f"  Hipotesis: 0.85")
if test_ious.mean() >= 0.85:
    print(f"  >>> HIPOTESIS SUPERADA en test set <<<")
else:
    print(f"  >>> Hipotesis NO superada en test set <<<")
print(f"{'='*60}")

# %% Cell 19: Validacion externa con San Salvador (OOD - Out of Distribution)
# ============================================================
# NUEVO: evalua el modelo en imagenes de San Salvador, un distrito
# NUNCA visto por el modelo durante entrenamiento.
# Esto prueba generalizacion geografica.
#
# REQUISITOS:
#   1. Subir imagenes de San Salvador a Drive/sugarcane/san_salvador/images/
#   2. Subir mascaras (si tenes) a Drive/sugarcane/san_salvador/masks/
#   3. Si no tenes mascaras, igual corre: muestra predicciones sin IoU.
# ============================================================

SS_IMAGE_DIR = SAN_SALVADOR_DIR + "/images"
SS_MASK_DIR = SAN_SALVADOR_DIR + "/masks"

if os.path.isdir(SS_IMAGE_DIR) and len(os.listdir(SS_IMAGE_DIR)) > 0:
    print(f"Imagenes de San Salvador encontradas: {len(os.listdir(SS_IMAGE_DIR))}")

    has_masks = os.path.isdir(SS_MASK_DIR) and len(os.listdir(SS_MASK_DIR)) > 0
    if has_masks:
        print(f"Mascaras de San Salvador encontradas: {len(os.listdir(SS_MASK_DIR))}")
        ss_dataset = SugarCaneDataset(SS_IMAGE_DIR, SS_MASK_DIR)

        ss_ious = []
        for idx in tqdm(range(len(ss_dataset)), desc="San Salvador"):
            img_path, mask_path = ss_dataset.samples[idx]
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            inputs = image_processor(images=img_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(pixel_values=inputs["pixel_values"]).logits

            logits_up = torch.nn.functional.interpolate(
                logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False
            )
            pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

            gt_bgr = cv2.imread(mask_path)
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            gt_class = mask_rgb_to_class(gt_rgb)

            inter = ((pred == 1) & (gt_class == 1)).sum()
            union = ((pred == 1) | (gt_class == 1)).sum()
            ss_ious.append(inter / union if union > 0 else 0)

        ss_ious = np.array(ss_ious)
        print(f"\n{'='*60}")
        print(f"VALIDACION EXTERNA - SAN SALVADOR (OOD)")
        print(f"  Muestras:        {len(ss_ious)}")
        print(f"  Mean IoU (cana): {ss_ious.mean():.4f}")
        print(f"  Median IoU:      {np.median(ss_ious):.4f}")
        print(f"  Std:             {ss_ious.std():.4f}")
        print(f"  Min:             {ss_ious.min():.4f}")
        print(f"  Max:             {ss_ious.max():.4f}")
        print(f"{'='*60}")

        # Visualizar 3 muestras
        random.seed(42)
        ss_idxs = random.sample(range(len(ss_dataset)), min(3, len(ss_dataset)))
        for idx in ss_idxs:
            img_path, mask_path = ss_dataset.samples[idx]
            print(f"\nSan Salvador muestra #{idx} - {os.path.basename(img_path)}")
            predict_and_show(img_path, mask_path)
    else:
        print("No hay mascaras de San Salvador. Mostrando solo predicciones...")
        ss_images = sorted(glob(os.path.join(SS_IMAGE_DIR, "*.jpg")))
        for img_path in ss_images[:5]:
            print(f"\nSan Salvador - {os.path.basename(img_path)}")
            predict_and_show(img_path)
else:
    print("No se encontraron imagenes de San Salvador.")
    print(f"Subi imagenes a: {SS_IMAGE_DIR}")
    print("Y vuelve a correr esta celda cuando tengas las imagenes listas.")
