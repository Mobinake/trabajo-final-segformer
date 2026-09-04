# Guía: usar el modelo entrenado en tu PC (sin Colab)

Cómo cargar el modelo fine-tuneado (`segformer-sugarcane-final`, guardado en Google Drive al terminar el entrenamiento) y ejecutar inferencia sobre imágenes nuevas en tu propia máquina, sin GPU.

## 1. Requisitos

```bash
pip install torch torchvision transformers pillow numpy opencv-python
```

- Python 3.10 o superior. Funciona en CPU: una imagen 128×128 tarda < 1 s.

## 2. Conseguir el modelo

Tras el entrenamiento, el script guarda en Drive:

```
MiDrive/sugarcane/segformer-sugarcane-final/
├── config.json               # arquitectura SegFormer + id2label
├── pytorch_model.bin         # pesos del mejor checkpoint
└── preprocessor_config.json  # SegformerImageProcessor (reduce_labels=False)
```

Descargá esa carpeta completa a tu PC, por ejemplo a `./modelo/`.

## 3. Cargar el modelo

```python
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_PATH = "./modelo"  # carpeta descargada de Drive

model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
image_processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Modelo en {device}")
```

## 4. Inferencia sobre una imagen

```python
import cv2
import numpy as np
import torch.nn.functional as F

def predecir(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    inputs = image_processor(images=img_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(pixel_values=inputs["pixel_values"]).logits

    # Volver a la resolución original (los logits salen a 1/4)
    logits_up = F.interpolate(
        logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

    # Overlay: verde donde predice caña
    overlay = img_rgb.copy()
    overlay[pred == 1] = [0, 200, 0]

    pct = (pred == 1).mean() * 100
    print(f"{image_path}: {pct:.1f} % de píxeles = caña de azúcar")
    return pred, overlay
```

## 5. Visualizar el resultado

```python
from PIL import Image

_, overlay = predecir("mi_tesela.jpg")
Image.fromarray(overlay).save("mi_tesela_prediccion.png")
```

El overlay pinta de verde los píxeles predichos como caña de azúcar, igual que en el notebook de entrenamiento.

## 6. Entrada esperada

- Imágenes JPG RGB de 128×128 px, generadas con el mismo pipeline del repo (Sentinel-2 L2A → 8-bit p2–p98 → JPG → teselas 128 px con solape 8 px). Si tus imágenes provienen de otra fuente (otro sensor, otro estiramiento de color), esperá menor fidelidad: el modelo es sensible al dominio visual.
- Si querés cubrir un área grande, teselá la escena con `scripts/preprocessing/create_tiles_directory.py`, predecí tesela por tesela y recomponé (el solape de 8 px permite pegar sin costuras visibles a simple vista).

## 7. Limitaciones

- Entrenado solo con 3 distritos del Guairá (Tebicuary, Itapé, Coronel Martínez) y una zafras; en zonas fuera de esa distribución el desempeño baja (validación externa con San Salvador quedó como trabajo futuro).
- Umbral de la hipótesis: mIoU clase caña ≥ 0,85 en prueba → alcanzado 0,90; eso no garantiza el mismo desempeño en imágenes con nubes, quema o cosecha reciente.
- El modelo distingue caña de otras coberturas por textura/color de la imagen; no usa bandas espectrales adicionales ni series temporales.
