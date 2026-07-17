---
type: project
subtype: experiment-log
ejecucion: 02
fecha: 2026-06-22
modelo: SegFormer MiT-B3
epocas_max: 100
patience: 50
estado: completo
---

# Ejecución 02 — MiT-B3, 100 épocas, patience 50

> Segunda ejecución de entrenamiento. Fecha: 22 de junio de 2026. Cambio de parámetros respecto a la ejecución 01: épocas máx suben de 50 a 100, patience sube de 10 a 50. Log descargado de Colab en `~/Downloads/segformer-sugarcane-mitb3-100epocas-50patience.txt`.

## Parámetros

> Confirmado por el usuario el 2026-06-22: los únicos parámetros que cambiaron respecto a la ejecución 01 son las épocas máx (50→100) y el patience del early stopping (10→50). Todo lo demás es idéntico.

| Parámetro | Valor ejec. 01 | Valor ejec. 02 | Cambió |
|---|---|---|---|
| Épocas máximas | 50 | 100 | ✅ |
| Patience early stopping | 10 | 50 | ✅ |
| Learning rate | 6 × 10⁻⁵ | 6 × 10⁻⁵ | No |
| Batch size efectivo | 8 | 8 | No |
| Data augmentation | on | on | No |
| Dataset | 513 imgs | 513 imgs | No |
| Semilla split | 42 | 42 | No |
| Resto de hiperparámetros | — | idéntico | No |

## Dataset

> Confirmado: idéntico a la ejecución 01 (mismo script, mismo dataset, misma semilla).

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
| Training loss final | 0.155910 (step 4500) |
| Runtime | ~90 min |
| Épocas ejecutadas | 100 (todas — early stopping NO activado) |
| Pasos totales | 4.500 |
| Early stopping activado | No (patience=50 nunca se agotó) |

> Cálculo de verificación: 359 imgs train / batch efectivo 8 ≈ 45 pasos/época × 100 épocas = 4.500 pasos. Cuadra.

> Nota sobre el mIoU del Trainer vs evaluate_set: el Trainer reporta Mean Iou durante el entrenamiento sobre imágenes redimensionadas a 512×512. El mejor Mean Iou del Trainer fue **0.920758** en el step 3650 (época ~81). Sin embargo, la evaluación final con `evaluate_set()` se hace sobre las imágenes a resolución original (predicciones upsampled), por lo que el mIoU de validación reportado abajo (0.8951) difiere del del Trainer. Esta distinción es importante para el manuscrito.

### Tabla de entrenamiento completa (Trainer, cada 50 pasos)

| Step | Training Loss | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy |
|---|---|---|---|---|---|
| 50 | 1.965694 | 0.492893 | 0.551406 | 0.695521 | 0.737794 |
| 100 | 1.417707 | 0.370420 | 0.701976 | 0.834252 | 0.830501 |
| 150 | 1.202890 | 0.314387 | 0.729272 | 0.842944 | 0.851665 |
| 200 | 1.052796 | 0.274745 | 0.773946 | 0.884356 | 0.876678 |
| 250 | 0.969391 | 0.245331 | 0.807385 | 0.894056 | 0.899193 |
| 300 | 0.911392 | 0.235045 | 0.802465 | 0.887831 | 0.897065 |
| 350 | 0.828252 | 0.214136 | 0.836728 | 0.915858 | 0.915321 |
| 400 | 0.794217 | 0.224417 | 0.812696 | 0.885501 | 0.905164 |
| 450 | 0.614988 | 0.178963 | 0.854288 | 0.920279 | 0.926116 |
| 500 | 0.579590 | 0.172609 | 0.864196 | 0.928603 | 0.931152 |
| 550 | 0.597444 | 0.184895 | 0.856265 | 0.921714 | 0.927155 |
| 600 | 0.536778 | 0.162512 | 0.868510 | 0.927005 | 0.934093 |
| 650 | 0.479925 | 0.168978 | 0.871753 | 0.930636 | 0.935583 |
| 700 | 0.502618 | 0.164519 | 0.879012 | 0.937912 | 0.939080 |
| 750 | 0.467282 | 0.153028 | 0.881514 | 0.940468 | 0.940289 |
| 800 | 0.467932 | 0.144112 | 0.883306 | 0.937682 | 0.941710 |
| 850 | 0.431447 | 0.151438 | 0.886583 | 0.942071 | 0.943153 |
| 900 | 0.399628 | 0.167569 | 0.886700 | 0.944128 | 0.942990 |
| 950 | 0.381932 | 0.160242 | 0.893196 | 0.947131 | 0.946525 |
| 1000 | 0.369715 | 0.162735 | 0.884690 | 0.937434 | 0.942576 |
| 1050 | 0.345649 | 0.131802 | 0.892249 | 0.941647 | 0.946573 |
| 1100 | 0.332378 | 0.169347 | 0.891527 | 0.945024 | 0.945764 |
| 1150 | 0.340844 | 0.163626 | 0.893032 | 0.945505 | 0.946602 |
| 1200 | 0.322200 | 0.134006 | 0.897493 | 0.947045 | 0.949057 |
| 1250 | 0.346208 | 0.144183 | 0.897964 | 0.947958 | 0.949236 |
| 1300 | 0.269047 | 0.148853 | 0.898353 | 0.950265 | 0.949230 |
| 1350 | 0.307173 | 0.135006 | 0.899843 | 0.949810 | 0.950143 |
| 1400 | 0.271770 | 0.127039 | 0.904730 | 0.950220 | 0.952928 |
| 1450 | 0.291706 | 0.129262 | 0.905611 | 0.953041 | 0.953163 |
| 1500 | 0.250916 | 0.123366 | 0.908183 | 0.954049 | 0.954543 |
| 1550 | 0.231043 | 0.144542 | 0.903000 | 0.951219 | 0.951833 |
| 1600 | 0.294212 | 0.144375 | 0.901732 | 0.950847 | 0.951137 |
| 1650 | 0.251245 | 0.128137 | 0.909208 | 0.954412 | 0.955094 |
| 1700 | 0.271466 | 0.151096 | 0.905853 | 0.952751 | 0.953329 |
| 1750 | 0.266456 | 0.140041 | 0.908930 | 0.953813 | 0.954990 |
| 1800 | 0.228918 | 0.151888 | 0.909199 | 0.954953 | 0.955040 |
| 1850 | 0.228202 | 0.151153 | 0.908780 | 0.954197 | 0.954870 |
| 1900 | 0.208997 | 0.152194 | 0.907438 | 0.952721 | 0.954240 |
| 1950 | 0.228944 | 0.132538 | 0.910801 | 0.954675 | 0.955976 |
| 2000 | 0.227027 | 0.153123 | 0.910771 | 0.955110 | 0.955920 |
| 2050 | 0.219777 | 0.141278 | 0.910891 | 0.954667 | 0.956028 |
| 2100 | 0.198307 | 0.163344 | 0.911012 | 0.955124 | 0.956056 |
| 2150 | 0.243533 | 0.153936 | 0.910390 | 0.954723 | 0.955739 |
| 2200 | 0.236717 | 0.156609 | 0.911101 | 0.954744 | 0.956140 |
| 2250 | 0.207064 | 0.126566 | 0.916876 | 0.958004 | 0.959113 |
| 2300 | 0.190514 | 0.133154 | 0.914487 | 0.956703 | 0.957882 |
| 2350 | 0.178746 | 0.148450 | 0.913637 | 0.956660 | 0.957406 |
| 2400 | 0.166011 | 0.160268 | 0.915477 | 0.957933 | 0.958335 |
| 2450 | 0.196403 | 0.141673 | 0.914689 | 0.957156 | 0.957957 |
| 2500 | 0.215211 | 0.123658 | 0.917562 | 0.957976 | 0.959500 |
| 2550 | 0.208421 | 0.120984 | 0.918453 | 0.958758 | 0.959933 |
| 2600 | 0.197911 | 0.138357 | 0.914630 | 0.956862 | 0.957948 |
| 2650 | 0.215075 | 0.161515 | 0.915042 | 0.958480 | 0.958044 |
| 2700 | 0.174134 | 0.141127 | 0.916788 | 0.958814 | 0.958997 |
| 2750 | 0.173907 | 0.152304 | 0.914980 | 0.957854 | 0.958061 |
| 2800 | 0.181011 | 0.129302 | 0.918496 | 0.959378 | 0.959907 |
| 2850 | 0.162832 | 0.140981 | 0.916569 | 0.958853 | 0.958871 |
| 2900 | 0.184312 | 0.139942 | 0.917133 | 0.958889 | 0.959184 |
| 2950 | 0.165062 | 0.123137 | 0.920010 | 0.959770 | 0.960719 |
| 3000 | 0.163735 | 0.154118 | 0.917802 | 0.958905 | 0.959557 |
| 3050 | 0.182082 | 0.143552 | 0.917877 | 0.958849 | 0.959604 |
| 3100 | 0.194908 | 0.142205 | 0.915139 | 0.956879 | 0.958234 |
| 3150 | 0.162194 | 0.146037 | 0.917206 | 0.958679 | 0.959242 |
| 3200 | 0.170922 | 0.142756 | 0.919463 | 0.960042 | 0.960393 |
| 3250 | 0.169780 | 0.147471 | 0.918793 | 0.960061 | 0.960018 |
| 3300 | 0.157165 | 0.142714 | 0.918479 | 0.959735 | 0.959869 |
| 3350 | 0.173145 | 0.158115 | 0.917240 | 0.959317 | 0.959209 |
| 3400 | 0.159588 | 0.149568 | 0.918940 | 0.959729 | 0.960126 |
| 3450 | 0.175285 | 0.147751 | 0.918704 | 0.959962 | 0.959976 |
| 3500 | 0.194479 | 0.151453 | 0.918665 | 0.960078 | 0.959945 |
| 3550 | 0.156082 | 0.152605 | 0.919010 | 0.960017 | 0.960142 |
| 3600 | 0.159745 | 0.144882 | 0.920035 | 0.959982 | 0.960716 |
| 3650 | 0.160174 | 0.143822 | **0.920758** | 0.960786 | 0.961054 |
| 3700 | 0.149978 | 0.146524 | 0.920649 | 0.960526 | 0.961014 |
| 3750 | 0.157148 | 0.150848 | 0.920715 | 0.961122 | 0.961005 |
| 3800 | 0.160439 | 0.150216 | 0.920560 | 0.960921 | 0.960934 |
| 3850 | 0.167557 | 0.154219 | 0.920719 | 0.961162 | 0.961004 |
| 3900 | 0.153323 | 0.149251 | 0.920896 | 0.961107 | 0.961106 |
| 3950 | 0.150596 | 0.151870 | 0.920647 | 0.961069 | 0.960971 |
| 4000 | 0.161224 | 0.146984 | 0.920888 | 0.961001 | 0.961110 |
| 4050 | 0.153686 | 0.146767 | 0.920735 | 0.960727 | 0.961047 |
| 4100 | 0.161373 | 0.154603 | 0.920505 | 0.960678 | 0.960922 |
| 4150 | 0.152599 | 0.152027 | 0.920421 | 0.960839 | 0.960863 |
| 4200 | 0.134717 | 0.158856 | 0.919918 | 0.960594 | 0.960602 |
| 4250 | 0.142358 | 0.157902 | 0.920189 | 0.960298 | 0.960776 |
| 4300 | 0.163007 | 0.156406 | 0.920540 | 0.960719 | 0.960939 |
| 4350 | 0.146249 | 0.156187 | 0.920650 | 0.960669 | 0.961004 |
| 4400 | 0.151652 | 0.157657 | 0.920399 | 0.960877 | 0.960848 |
| 4450 | 0.143361 | 0.157167 | 0.920657 | 0.960954 | 0.960985 |
| 4500 | 0.155910 | 0.157435 | 0.920704 | 0.960936 | 0.961013 |

> Mejor Mean Iou del Trainer: **0.920758** en step 3650 (época ~81). A partir de ahí el modelo se estabiliza alrededor de 0.920 sin mejorar significativamente, pero el patience=50 impide que el early stopping se dispare.

## Métricas de validación (76 imgs, evaluación con evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8951** |
| Median IoU | 0.9303 |
| Std | 0.0958 |
| Min | 0.5375 |
| Max | 0.9810 |

## Métricas de test (78 imgs, nunca vistas, evaluación con evaluate_set a resolución original)

| Métrica | Valor |
|---|---|
| **Mean IoU (sugar_cane)** | **0.8903** |
| Median IoU | 0.9192 |
| Std | 0.0869 |
| Min | 0.6115 |
| Max | 0.9782 |

- ✅ Hipótesis superada: mIoU en test set = 0.8903 ≥ 0.85.

## Observaciones

- ✅ Hipótesis superada: mIoU en test set = 0.8903 ≥ 0.85.
- **Early stopping NO activado:** el modelo corrió las 100 épocas completas. Con patience=50, el early stopping necesita 50 evaluaciones consecutivas sin mejora en mIoU para dispararse. Como el modelo siguió mejorando marginalmente (o sin deteriorarse lo suficiente), nunca se agotó el patience.
- **Mejora respecto a ejecución 01:** mIoU test subió de 0.8535 a 0.8903 (+0.0368). mIoU val subió de 0.8727 a 0.8951 (+0.0224).
- **Menor variabilidad:** Std en test bajó de 0.1411 a 0.0869 (-0.0542). Std en val bajó de 0.1052 a 0.0958 (-0.0094). El modelo es más consistente entre distritos con más épocas y patience mayor.
- **Min IoU más alto:** en test subió de 0.1593 a 0.6115 (+0.4522). En val subió de 0.4752 a 0.5375 (+0.0623). El peor caso mejoró significativamente.
- **Diferencia val-test:** 0.8951 - 0.8903 = 0.0048 (muy pequeña; antes era 0.0192). El modelo generaliza mejor.
- **Training loss:** bajó de 1.966 (step 50) a 0.156 (step 4500). Descenso suave y estable, sin oscilaciones preocupantes.
- **Estabilización del mIoU del Trainer:** a partir del step ~3650 (época ~81), el mIoU del Trainer se estabiliza alrededor de 0.920 sin mejoras significativas. Esto sugiere que el modelo alcanzó su meseta de aprendizaje alrededor de la época 81.
- Modelo guardado en /content/drive/MyDrive/sugarcane/segformer-sugarcane-final el 22 jun a las 21:06.
- Runtime: ~90 min (1.5× más que ejec. 01 que fue ~56 min).

## Comparación con ejecución 01

| Métrica | Ejec. 01 (50 ep, p=10) | Ejec. 02 (100 ep, p=50) | Δ |
|---|---|---|---|
| Épocas máx | 50 | 100 | +50 |
| Patience | 10 | 50 | +40 |
| Épocas ejecutadas | 43.33 | 100 | +56.67 |
| Early stopping | Sí (época 43.33) | No (corrió todas) | — |
| Pasos totales | 1.950 | 4.500 | +2.550 |
| Training loss final | 0.5416 | 0.1559 | -0.3857 |
| Runtime | ~56 min | ~90 min | +34 min |
| **mIoU val (evaluate_set)** | 0.8727 | **0.8951** | **+0.0224** |
| mIoU val median | 0.9136 | 0.9303 | +0.0167 |
| mIoU val std | 0.1052 | 0.0958 | -0.0094 |
| mIoU val min | 0.4752 | 0.5375 | +0.0623 |
| mIoU val max | 0.9804 | 0.9810 | +0.0006 |
| **mIoU test (evaluate_set)** | 0.8535 | **0.8903** | **+0.0368** |
| mIoU test median | 0.9082 | 0.9192 | +0.0110 |
| mIoU test std | 0.1411 | 0.0869 | -0.0542 |
| mIoU test min | 0.1593 | 0.6115 | +0.4522 |
| mIoU test max | 0.9781 | 0.9782 | +0.0001 |
| Δ val-test | 0.0192 | 0.0048 | -0.0144 |
| Hipótesis (≥0.85) | ✅ | ✅ | — |

> **Conclusión:** aumentar épocas máx (50→100) y patience (10→50) mejoró el mIoU en +0.037 en test y +0.022 en val. La variabilidad bajó (std test -0.054), el peor caso mejoró drásticamente (min test +0.452) y la generalización mejoró (Δ val-test bajó de 0.019 a 0.005). El training loss bajó de 0.54 a 0.16. El modelo se estabilizó alrededor de la época 81 sin necesidad de early stopping. Costo: ~2.3× más pasos de entrenamiento (1950 → 4500).

## 🔗 Notas relacionadas
- [[ejecucion-01-mitb3-50epocas-patience10]] — ejecución anterior, para comparar
- [[ejecuciones/README]] — índice de todas las ejecuciones
- [[04 - Resultados y Validación]] — consolidación final para el manuscrito
- [[08 - Configuración Técnica]] — parámetros del script
