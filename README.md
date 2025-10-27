# 🧠 Microproyecto1 - Fine-Tuning ResNet-50 en Imágenes de MRI Cerebrales

## 🎯 Objetivo
Este proyecto implementa un pipeline completo para **clasificar imágenes de resonancia magnética (MRI)** en **4 clases**:
- **glioma**
- **meningioma**
- **pituitary (hipófisis)**
- **healthy (sano)**

El modelo base es una **ResNet-50 preentrenada en ImageNet**, a la cual se le realiza **fine-tuning** utilizando un conjunto de imágenes en **escala de grises**.

---

## 🧩 Estructura del proyecto
microproyecto1/
│
├── microproyecto1_resnet50_skeleton.ipynb # Notebook principal con pipeline completo
├── microproyecto_CNN.pdf # Enunciado oficial del microproyecto
├── Matriz_General_Miniproyectos-2.pdf # Rúbrica de evaluación
├── Doc_ResNet.pdf # Documento de referencia teórica sobre ResNet
├── Template-Curso-Tecnicas-de-Deep-Learning.docx # Plantilla de informe
│
├── data_set/ # Carpeta raíz con las imágenes organizadas por clase
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── healthy/
│
├── outputs/ # Pesos, métricas y gráficos generados automáticamente
│ ├── resnet50_grayscale_ft_best.pth
│ ├── resnet50_grayscale_ft_history.npy
│ └── resnet50_grayscale_ft_test_metrics.json
│
├── README.md
└── requirements.txt

🚀 Ejecución paso a paso

1️⃣ Clonar el proyecto

git clone https://github.com/tu_usuario/microproyecto1.git
cd microproyecto1


2️⃣ Crear entorno virtual

python -m venv venv
source venv/bin/activate  # En Windows: venv\\Scripts\\activate


3️⃣ Instalar dependencias

pip install -r requirements.txt


4️⃣ Preparar dataset
Estructura requerida:

data_set/
 ├── glioma/
 ├── meningioma/
 ├── pituitary/
 └── healthy/


5️⃣ Ejecutar notebook
Abre microproyecto1_resnet50_skeleton.ipynb en Jupyter o VSCode y ejecuta secuencialmente.

6️⃣ Resultados
Se generarán automáticamente en la carpeta outputs/:

Pesos del mejor modelo.

Métricas finales.

Historia de entrenamiento para graficar.

🧠 Autores y créditos

Autor principal: Mauricio Rodríguez

Rol: Project Manager | Deep Learning Student | AI Developer

Objetivo académico: Maestría en Inteligencia Artificial - Universidad de los Andes

Licencia: Uso académico y de investigación (sin fines comerciales).

📈 Futuras mejoras

Integrar matriz de confusión y Grad-CAM para interpretación visual.

Extender a ResNet-101 y ViT para comparación.

Implementar early-freezing warm-up dinámico en capas convolucionales.

Exportar modelo a TorchScript para despliegue.

💬 Contacto

📧 mauricio.rodriguez@example.com

🌐 LinkedIn: linkedin.com/in/mauriciorodriguez

💡 “Structure brings clarity. Clarity brings performance.”

---

## ⚙️ Características técnicas

### 🔸 Preprocesamiento (pipeline integrado)
- El dataset se divide **automáticamente 80/10/10** (train/val/test) **sin mover archivos**, usando `StratifiedShuffleSplit` para garantizar balance entre clases.
- Se aplican transformaciones con `torchvision.transforms`:
  - Escala a 224×224
  - Conversión a `Grayscale`
  - Normalización
  - Aumentos moderados (flips y ligeros cambios de brillo/contraste)
- Compatible con datasets monocromáticos (MRI en escala de grises).

### 🔸 Modelo
- **Backbone:** ResNet-50 (`torchvision.models.resnet50`)
- **Fine-tuning completo o parcial**, configurable.
- Dos modos de entrada:
  - `replicate_to_3`: replica el canal gris a RGB (por defecto).
  - `single_channel_conv`: adapta `conv1` a 1 canal promedio de los pesos RGB.

### 🔸 Entrenamiento
- **Optimizers soportados:** `AdamW` y `SGD + momentum`.
- **Schedulers:** `OneCycleLR` o `ReduceLROnPlateau`.
- **Early Stopping:** monitorea `macro-F1`.
- **Métricas:** `accuracy`, `macro-F1`, `val_loss`.

### 🔸 Evaluación
- Al finalizar el entrenamiento se evalúa en el **set de test**.
- Se guardan:
  - `best.pth` (mejor modelo)
  - `history.npy` (histórico de entrenamiento)
  - `test_metrics.json` (métricas finales)

---

## 🧠 Requisitos previos

### Dependencias principales
- Python **3.10 o superior**
- PyTorch **>=2.0**
- torchvision
- scikit-learn
- numpy, matplotlib
- tqdm (opcional para barra de progreso)
- GPU compatible con CUDA (opcional pero recomendado)

Instalar con:
```bash
pip install -r requirements.txt
