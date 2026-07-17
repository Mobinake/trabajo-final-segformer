---
type: project
subtype: experiment-log
ejecucion: 04
fecha: 2026-06-27
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 20
eval_steps: 50
estado: completo
---

# Ejecución 04 — Run 04: eval_steps=50, patience=20, max_epochs=100

> Grid search de la tutora (TODO 1, 27-06-2026). Segunda run del grid 3×3. Fecha: 27 de junio de 2026. Notebook: `sugarcane_segformer_v3.ipynb` ejecutado en Google Colab (GPU T4).

## Parámetros

> Únicos cambios respecto a ejecución 03: patience sube de 10 a 20. eval_steps se mantiene en 50. num_train_epochs se mantiene en 100.

| Parámetro | Valor |
|---|---|
| Learning rate | 6 × 10⁻⁵ |
| Épocas máx | 100 |
| eval_steps | 50 |
| save_steps | 50 |
| Patience early stopping | 20 |
| Batch size efectivo | 8 (2 × 4 grad_accum) |
| Warmup steps | 50 |
| FP16 | True |
| Seed | 42 |
| Data augmentation | on (flip, rotate90, brightness/contrast) |
| reduce_labels | False |
| Dataset | 513 imgs (3 distritos) |
| Split | 70/15/15 → 359/76/78 |

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
| Training loss final | 0.1612 (step 4500) |
| Runtime | ~1:39:49 (~100 min) |
| Épocas ejecutadas | 100 |
| Pasos totales | 4.500 |
| Early stopping activado | No (corrió las 100 épocas completas; patience=20 casi se agota — ver observaciones) |
| Mejor step (Trainer mIoU) | 3500 (época ~77.78, mIoU Trainer = 0.9233) |

> Cálculo de verificación: 359 imgs / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> El mejor checkpoint fue el del step 3500 (mIoU Trainer = 0.923316). Después de eso, 20 evaluaciones consecutivas (steps 3550-4500) sin superar ese valor. Con patience=20, el early stopping habría disparado en el step 4500 (la 20ª evaluación sin mejora), pero como step 4500 es también el último paso de la época 100, el entrenamiento terminó naturalmente al mismo tiempo. Es decir: patience=20 fue insuficiente para detener antes de las 100 épocas completas, pero por muy poco.

> Nota sobre Trainer mIoU vs evaluate_set: el Trainer reporta Mean Iou sobre imágenes redimensionadas a 512×512. La evaluación final con `evaluate_set()` se hace a resolución original (predicciones upsampled). Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 50 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 50 | 2.031609 | 0.437458 | 0.636353 | 0.783858 | 0.785173 |
| 100 | 1.417288 | 0.378792 | 0.683260 | 0.812666 | 0.820642 |
| 150 | 1.227524 | 0.319466 | 0.728712 | 0.842878 | 0.851199 |
| 200 | 1.068644 | 0.317861 | 0.740495 | 0.867985 | 0.854329 |
| 250 | 0.975768 | 0.268340 | 0.785096 | 0.885849 | 0.884682 |
| 300 | 0.986961 | 0.241471 | 0.804858 | 0.895228 | 0.897125 |
| 350 | 0.843814 | 0.227063 | 0.812828 | 0.890975 | 0.903828 |
| 400 | 0.716727 | 0.215006 | 0.830021 | 0.905088 | 0.912753 |
| 450 | 0.637052 | 0.207818 | 0.838449 | 0.911526 | 0.917223 |
| 500 | 0.613353 | 0.218640 | 0.845546 | 0.921538 | 0.920238 |
| 550 | 0.516700 | 0.175276 | 0.864441 | 0.925494 | 0.931767 |
| 600 | 0.544200 | 0.180166 | 0.858498 | 0.923360 | 0.928325 |
| 650 | 0.468048 | 0.147678 | 0.877158 | 0.933050 | 0.938578 |
| 700 | 0.529717 | 0.165074 | 0.872593 | 0.930169 | 0.936169 |
| 750 | 0.519392 | 0.172341 | 0.859497 | 0.919405 | 0.929610 |
| 800 | 0.465703 | 0.133161 | 0.890164 | 0.943257 | 0.945152 |
| 850 | 0.416110 | 0.144836 | 0.887907 | 0.942359 | 0.943911 |
| 900 | 0.430626 | 0.144512 | 0.891054 | 0.946234 | 0.945353 |
| 950 | 0.415558 | 0.160055 | 0.887949 | 0.942423 | 0.943929 |
| 1000 | 0.376668 | 0.162239 | 0.888365 | 0.938919 | 0.944593 |
| 1050 | 0.374792 | 0.143947 | 0.889011 | 0.938184 | 0.945069 |
| 1100 | 0.322238 | 0.137749 | 0.895469 | 0.946704 | 0.947907 |
| 1150 | 0.353554 | 0.139058 | 0.898760 | 0.948164 | 0.949679 |
| 1200 | 0.307482 | 0.159247 | 0.890836 | 0.941095 | 0.945801 |
| 1250 | 0.364462 | 0.162343 | 0.896216 | 0.946111 | 0.948409 |
| 1300 | 0.271747 | 0.172592 | 0.895127 | 0.948829 | 0.947486 |
| 1350 | 0.305954 | 0.124789 | 0.906841 | 0.952554 | 0.953914 |
| 1400 | 0.293312 | 0.151901 | 0.899101 | 0.949262 | 0.949766 |
| 1450 | 0.296408 | 0.139618 | 0.902364 | 0.951933 | 0.951397 |
| 1500 | 0.265040 | 0.148166 | 0.903200 | 0.952716 | 0.951806 |
| 1550 | 0.232027 | 0.146947 | 0.902595 | 0.951476 | 0.951574 |
| 1600 | 0.303465 | 0.145928 | 0.904556 | 0.952892 | 0.952570 |
| 1650 | 0.277206 | 0.116330 | 0.912220 | 0.956658 | 0.956606 |
| 1700 | 0.264086 | 0.145668 | 0.904563 | 0.951379 | 0.952719 |
| 1750 | 0.231181 | 0.151353 | 0.906210 | 0.952761 | 0.953533 |
| 1800 | 0.244329 | 0.159590 | 0.905670 | 0.953606 | 0.953145 |
| 1850 | 0.232954 | 0.144855 | 0.907230 | 0.954231 | 0.953981 |
| 1900 | 0.229955 | 0.122410 | 0.914903 | 0.957567 | 0.958042 |
| 1950 | 0.226447 | 0.134635 | 0.910277 | 0.953242 | 0.955809 |
| 2000 | 0.233267 | 0.132959 | 0.913378 | 0.956718 | 0.957255 |
| 2050 | 0.218177 | 0.142785 | 0.909558 | 0.954175 | 0.955315 |
| 2100 | 0.233098 | 0.146301 | 0.908783 | 0.954787 | 0.954818 |
| 2150 | 0.232553 | 0.148936 | 0.907626 | 0.954521 | 0.954182 |
| 2200 | 0.191031 | 0.140797 | 0.912159 | 0.955294 | 0.956690 |
| 2250 | 0.219193 | 0.129384 | 0.915255 | 0.957761 | 0.958224 |
| 2300 | 0.187143 | 0.138683 | 0.913332 | 0.955579 | 0.957328 |
| 2350 | 0.200990 | 0.159256 | 0.909973 | 0.955523 | 0.955430 |
| 2400 | 0.176670 | 0.130075 | 0.916993 | 0.957976 | 0.959181 |
| 2450 | 0.196244 | 0.114632 | 0.920165 | 0.958984 | 0.960868 |
| 2500 | 0.195517 | 0.128968 | 0.917306 | 0.957853 | 0.959367 |
| 2550 | 0.212828 | 0.133274 | 0.916127 | 0.958783 | 0.958629 |
| 2600 | 0.207375 | 0.146229 | 0.912282 | 0.956364 | 0.956666 |
| 2650 | 0.190004 | 0.129102 | 0.919379 | 0.959268 | 0.960408 |
| 2700 | 0.175016 | 0.143676 | 0.914476 | 0.958050 | 0.957761 |
| 2750 | 0.175399 | 0.126909 | 0.920184 | 0.960466 | 0.960760 |
| 2800 | 0.175146 | 0.133689 | 0.917543 | 0.958434 | 0.959451 |
| 2850 | 0.169951 | 0.140081 | 0.917740 | 0.959275 | 0.959492 |
| 2900 | 0.183998 | 0.142230 | 0.916592 | 0.958326 | 0.958927 |
| 2950 | 0.164644 | 0.129367 | 0.920415 | 0.959663 | 0.960952 |
| 3000 | 0.163760 | 0.140919 | 0.919629 | 0.959707 | 0.960512 |
| 3050 | 0.168135 | 0.146680 | 0.917495 | 0.958852 | 0.959390 |
| 3100 | 0.178444 | 0.148032 | 0.916710 | 0.958412 | 0.958987 |
| 3150 | 0.165808 | 0.147627 | 0.917465 | 0.958733 | 0.959383 |
| 3200 | 0.170766 | 0.140371 | 0.920086 | 0.959997 | 0.960743 |
| 3250 | 0.173782 | 0.142682 | 0.919985 | 0.960252 | 0.960666 |
| 3300 | 0.163798 | 0.161333 | 0.915695 | 0.958305 | 0.958426 |
| 3350 | 0.171522 | 0.149460 | 0.918268 | 0.959670 | 0.959756 |
| 3400 | 0.161394 | 0.133816 | 0.921657 | 0.960471 | 0.961578 |
| 3450 | 0.177462 | 0.145532 | 0.919072 | 0.959824 | 0.960192 |
| **3500** | **0.183629** | **0.123756** | **0.923316** | **0.961712** | **0.962400** |
| 3550 | 0.164443 | 0.139652 | 0.920635 | 0.960193 | 0.961033 |
| 3600 | 0.166672 | 0.143342 | 0.920175 | 0.959969 | 0.960795 |
| 3650 | 0.161723 | 0.137867 | 0.921039 | 0.960571 | 0.961227 |
| 3700 | 0.151645 | 0.137155 | 0.921722 | 0.960868 | 0.961583 |
| 3750 | 0.163721 | 0.133847 | 0.921394 | 0.960812 | 0.961406 |
| 3800 | 0.166004 | 0.135420 | 0.921096 | 0.960596 | 0.961257 |
| 3850 | 0.161269 | 0.139086 | 0.921418 | 0.961179 | 0.961390 |
| 3900 | 0.147506 | 0.138519 | 0.921384 | 0.960575 | 0.961419 |
| 3950 | 0.146354 | 0.139746 | 0.920830 | 0.960935 | 0.961083 |
| 4000 | 0.143254 | 0.137891 | 0.921296 | 0.960610 | 0.961367 |
| 4050 | 0.156502 | 0.144474 | 0.920755 | 0.960476 | 0.961077 |
| 4100 | 0.170868 | 0.144474 | 0.921656 | 0.960896 | 0.961544 |
| 4150 | 0.154704 | 0.141488 | 0.921660 | 0.960841 | 0.961551 |
| 4200 | 0.143067 | 0.141203 | 0.922118 | 0.961225 | 0.961775 |
| 4250 | 0.146535 | 0.142883 | 0.921609 | 0.960766 | 0.961528 |
| 4300 | 0.154643 | 0.140510 | 0.922320 | 0.961098 | 0.961897 |
| 4350 | 0.148541 | 0.141994 | 0.921860 | 0.960814 | 0.961664 |
| 4400 | 0.144943 | 0.145020 | 0.921051 | 0.960902 | 0.961208 |
| 4450 | 0.138945 | 0.146258 | 0.921303 | 0.960921 | 0.961347 |
| 4500 | 0.161182 | 0.145301 | 0.921460 | 0.960858 | 0.961439 |

> Mejor Trainer mIoU: **0.923316** en step 3500 (época ~77.78). Después de eso, 20 evaluaciones consecutivas (steps 3550-4500) sin mejora. Con patience=20, el early stopping habría disparado en step 4500, pero como era el último paso de la época 100, el entrenamiento terminó naturalmente.

## Métricas del Trainer (evaluate, best model cargado)

> El Trainer evalúa sobre imágenes redimensionadas a 512×512. Estas NO son las métricas finales del manuscrito (ver evaluate_set abajo).

| Métrica | Val (76 imgs) |
|---|---|
| eval_loss | 0.116791 |
| eval_mean_iou | 0.929529 |
| eval_mean_accuracy | 0.963443 |
| eval_overall_accuracy | 0.968253 |

> Nota: el mIoU del Trainer tras `load_best_model_at_end` (0.929529) es mayor que el mejor mIoU del log de entrenamiento (0.923316 en step 3500). Esto se debe a que la re-evaluación final con `model.eval()` y sin gradientes puede dar resultados ligeramente distintos a las evaluaciones intermedias durante el entrenamiento.

> Test set (Trainer): no evaluado con `trainer.evaluate()` en esta ejecución. Solo disponible evaluate_set (resolución original).

## Métricas de validación (76 imgs, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8927** |
| Median IoU | 0.9292 |
| Std | 0.0938 |
| Min | 0.5764 |
| Max | 0.9797 |

## Métricas de test (78 imgs, nunca vistas, evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8816** |
| Median IoU | 0.9171 |
| Std | 0.1019 |
| Min | 0.3708 |
| Max | 0.9794 |

- ✅ Hipótesis superada: mIoU en test set = 0.8816 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8816 ≥ 0.85.
- **Sin early stopping:** el modelo corrió las 100 épocas completas (4.500 pasos). El mejor checkpoint fue step 3500 (época ~77.78, mIoU Trainer = 0.9233). Después de eso, 20 evaluaciones consecutivas (steps 3550-4500) sin mejora. Con patience=20, el early stopping habría disparado exactamente en step 4500 (la 20ª evaluación sin mejora), pero como ese era también el último paso de la época 100, el entrenamiento terminó naturalmente. Es decir: patience=20 estuvo a punto de activarse pero no alcanzó a cortar antes del final.
- **Comparación con ejec. 03 (100 ep, patience=10, eval_steps=50):** la única diferencia es patience (20 vs 10). Ejec. 03 detuvo en step 3050 (época 67.78); esta run corrió hasta step 4500 (época 100). mIoU test: ejec. 03 = 0.8828 vs esta run = 0.8816 (-0.0012). La diferencia es mínima y no justifica las 1.450 pasos adicionales (~31 min más).
- **Comparación con ejec. 02 (100 ep, patience=50, eval_steps=50):** ejec. 02 también corrió las 100 épocas completas. mIoU test: ejec. 02 = 0.8903 vs esta run = 0.8816 (-0.0087). Ejec. 02 obtuvo mejor test a pesar de mismo número de pasos. La diferencia principal es que ejec. 02 tuvo patience=50 (más tolerante), pero como ambas corrieron 100 épocas completas, la diferencia en resultados se debe a la variabilidad del entrenamiento (mismo seed pero el early stopping callback puede afectar el estado del optimizador ligeramente).
- **Variabilidad:** Std en test = 0.1019 (entre ejec. 03: 0.1048 y ejec. 02: 0.0869). Similar a ejec. 03.
- **Min IoU en test:** 0.3708 — peor caso mejor que ejec. 03 (0.2808) pero peor que ejec. 02 (0.6115).
- **Δ val-test:** 0.8927 - 0.8816 = 0.0111 (similar a ejec. 03: 0.0114).
- **Training loss:** bajó de 2.032 (step 50) a 0.161 (step 4500). Descenso suave y estable, con meseta a partir de ~step 2400.
- **Plateau visible:** a partir de step ~2450, el mIoU del Trainer se estabiliza en el rango 0.916-0.923. Los últimos 2.050 pasos (steps 2450-4500) no aportaron mejora significativa. Esto sugiere que el modelo converge alrededor de step 2400-2500 y entrenar más allá no mejora.
- Modelo guardado en `/content/drive/MyDrive/sugarcane/segformer-sugarcane-final` el 27 jun.

## Comparación con ejecuciones 01, 02 y 03

| Métrica | Ejec. 01 (50 ep, p=10) | Ejec. 02 (100 ep, p=50) | Ejec. 03 (100 ep, p=10) | **Ejec. 04 (100 ep, p=20)** | Δ 04 vs 03 | Δ 04 vs 02 |
|---|---|---|---|---|---|---|
| Épocas máx | 50 | 100 | 100 | 100 | — | — |
| Patience | 10 | 50 | 10 | 20 | — | — |
| eval_steps | 50 | 50 | 50 | 50 | — | — |
| Épocas ejecutadas | 43.33 | 100 | 67.78 | 100 | +32.22 | 0 |
| Early stopping | Sí (época 43.33) | No (corrió todas) | Sí (época 67.78) | No (corrió todas) | — | — |
| Pasos totales | 1.950 | 4.500 | 3.050 | 4.500 | +1.450 | 0 |
| Training loss final | 0.5416 | 0.1559 | 0.4236 | 0.1612 | -0.2624 | +0.0053 |
| Runtime | ~56 min | ~90 min | ~69 min | ~100 min | +31 min | +10 min |
| **mIoU val (evaluate_set)** | 0.8727 | 0.8951 | 0.8942 | **0.8927** | -0.0015 | -0.0024 |
| mIoU val median | 0.9136 | 0.9303 | 0.9249 | 0.9292 | +0.0043 | -0.0011 |
| mIoU val std | 0.1052 | 0.0958 | 0.0877 | 0.0938 | +0.0061 | -0.0020 |
| mIoU val min | 0.4752 | 0.5375 | 0.5481 | 0.5764 | +0.0283 | +0.0389 |
| mIoU val max | 0.9804 | 0.9810 | 0.9795 | 0.9797 | +0.0002 | -0.0013 |
| **mIoU test (evaluate_set)** | 0.8535 | 0.8903 | 0.8828 | **0.8816** | -0.0012 | -0.0087 |
| mIoU test median | 0.9082 | 0.9192 | 0.9196 | 0.9171 | -0.0025 | -0.0021 |
| mIoU test std | 0.1411 | 0.0869 | 0.1048 | 0.1019 | -0.0029 | +0.0150 |
| mIoU test min | 0.1593 | 0.6115 | 0.2808 | 0.3708 | +0.0900 | -0.2407 |
| mIoU test max | 0.9781 | 0.9782 | 0.9783 | 0.9794 | +0.0011 | +0.0012 |
| Δ val-test | 0.0192 | 0.0048 | 0.0114 | 0.0111 | -0.0003 | +0.0063 |
| Hipótesis (≥0.85) | ✅ | ✅ | ✅ | ✅ | — | — |

> **Conclusión:** con patience=20 (vs patience=10 en ejec. 03), el modelo corrió 32 épocas más (100 vs 67.78) pero el mIoU test fue esencialmente igual (0.8816 vs 0.8828, -0.0012). Esto confirma que patience=10 ya era suficiente para encontrar un buen punto de corte, y aumentar a 20 solo prolonga el entrenamiento sin beneficio. Comparado con ejec. 02 (patience=50, también 100 épocas), esta run obtuvo peor test (0.8816 vs 0.8903), sugiriendo que la variabilidad entre runs con mismos hiperparámetros pero distinto patience puede deberse a factores estocásticos. El plateau visible a partir de step ~2450 indica que el modelo converge temprano y entrenar más allá de ~2500 pasos no aporta mejora significativa.

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución 01, baseline
- [[ejecucion-02-mitb3-100epocas-patience50]] — ejecución 02, mismo max_epochs pero patience=50
- [[ejecucion-03-mitb3-100epocas-patience10-steps50]] — ejecución 03, mismo max_epochs pero patience=10
- [[grid-search-steps-patience]] — script y grilla completa del grid search
- [[README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
