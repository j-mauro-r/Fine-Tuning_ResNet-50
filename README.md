# Microproyecto1 — Clasificación de MRI con ResNet50

## 📘 Descripción general

Este proyecto implementa un pipeline **reproducible**, **documentado (estilo Google)** y **modular** para el entrenamiento y análisis de una red **ResNet50** ajustada (fine-tuning) sobre imágenes **MRI** de cerebro.

El objetivo es clasificar cada imagen en una de las **4 categorías**:

- `glioma`
- `healthy`
- `meningioma`
- `pituitary`

El sistema realiza la **preparación, entrenamiento, validación y análisis** completo, consolidando resultados, métricas y visualizaciones en una estructura ordenada de salida.

---

## 🧭 Flujo de trabajo recomendado

El flujo del cuaderno `microproyecto1_resnet50_skeleton.ipynb` está dividido en **dos capítulos principales**:

### 🔹 Capítulo 1 — Datos

**Objetivo:** preparar y validar el pipeline de datos.

1. **Imports & Configuración:** activa el modo seguro (`num_workers=0`, `pin_memory=False`).
2. **Transformaciones baseline:** resize a 224×224, normalización tipo ImageNet, sin flips ni rotaciones.
3. **Carga del dataset:** usa `ImageFolder` y realiza split estratificado 80/10/10.
4. **Chequeos y ejemplos:** imprime tamaños, distribución por clase y muestra 5 imágenes transformadas del set de entrenamiento.
5. **LR Finder:** ejecuta un test sobre 500 imágenes y 100 iteraciones para encontrar el LR inicial adecuado.

> 💡 **Tip:** el LR sugerido por el Finder debe asignarse manualmente a `cfg.base_lr` antes del entrenamiento.

---

### 🔹 Capítulo 2 — Modelo

**Objetivo:** entrenar, evaluar y analizar el modelo ResNet50.

1. **ModelBuilder:** crea ResNet50 preentrenada (ImageNet) y reemplaza la capa final para 4 clases.
2. **Trainer:** ejecuta el ciclo de entrenamiento con `ReduceLROnPlateau` y `EarlyStopping`.
3. **ExperimentLogger:** registra automáticamente todos los artefactos relevantes.
4. **Entrenamiento (`main(cfg)`):** ejecuta el flujo completo y guarda checkpoints, curvas y métricas.
5. **Consolidación:** permite regenerar artefactos (curvas, métricas, matrices) sin reentrenar.
6. **Análisis avanzado:**
   - Matrices de confusión (conteos y normalizada).
   - Curvas Loss/Acc/F1.
   - Curvas ROC y Precision–Recall (OvR).
   - Muestra balanceada (1 por clase) con tabla comparativa `real vs predicho`.
   - Grad-CAM: 5 aleatorias + 1 por clase (layer4[-1].conv3).

---

## 📂 Estructura de salidas

Los resultados se guardan automáticamente en:

```
outputs/
├── resnet50_mri_ft_<timestamp>/
│   ├── config.json
│   ├── classes.json
│   ├── split_sizes.json
│   ├── split_indices.json
│   ├── history.npy / history.csv
│   ├── curves.png / val_metrics.png
│   ├── test_confusion_matrix.png
│   ├── test_metrics.json
│   ├── requirements.freeze.txt
│   ├── checkpoint.txt
│   └── analysis/
│       ├── confusion_matrices.png
│       ├── curves_loss.png
│       ├── roc_multiclass.png
│       ├── pr_multiclass.png
│       ├── sample_balanced_table.csv
│       ├── random_test_gradcam.png
│       └── classwise_gradcam.png
└── resnet50_mri_ft_latest/  → enlace simbólico a la última corrida
```

---

## ⚙️ Requisitos principales

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
```

> Todos los paquetes se registran automáticamente en `requirements.freeze.txt` al finalizar el experimento.

---

## 🚀 Cómo ejecutar

1. **Abrir `microproyecto1_resnet50_skeleton.ipynb` en Jupyter / Colab / Coursera.**
2. **Configurar `cfg.data_dir`** para apuntar a la carpeta raíz con subcarpetas por clase.
3. Ejecutar las celdas del **Capítulo 1** hasta el LR Finder.
4. Ajustar el valor de `cfg.base_lr` según la sugerencia del Finder.
5. Ejecutar el **Capítulo 2 → `main(cfg)`** para entrenar el modelo.
6. Ejecutar la sección de **Consolidación y Análisis** para generar métricas y gráficas.

---

## 🧠 Consideraciones técnicas

- Todas las imágenes se convierten a **RGB (3 canales)** para compatibilidad con los pesos preentrenados de ImageNet.
- El pipeline está diseñado para funcionar incluso en entornos con memoria limitada (Coursera, notebooks en línea).
- Se prioriza la reproducibilidad y consistencia del experimento (semillas fijas, splits estratificados).
- El Grad-CAM se aplica sobre la última capa convolucional (`layer4[-1].conv3`) para visualizar regiones discriminativas.

---

## 🏁 Créditos

**Desarrollo:** Mauricio  
**Asistencia técnica y documentación:** IA especializada en Vibe Coding (Deep Learning)

---

**Versión:** Consolidado (2025)
