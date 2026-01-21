# Optimización de Trayectorias en Robótica mediante Modelos Subrogados

> **Trabajo de Fin de Máster (TFM)** | Máster en Inteligencia Artificial Aplicada  
> **Autor:** Esteban Ruiz Hernández 
> **Tutor:** Carlos Cernuda

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Descripción del Proyecto

Este proyecto aborda uno de los desafíos críticos en la robótica moderna: la **planificación de movimiento en tiempo real** en entornos dinámicos de alta dimensionalidad.

Las simulaciones físicas de alta fidelidad (como PyBullet) son precisas pero computacionalmente costosas (~50ms por evaluación), lo que impide su uso para evaluar miles de trayectorias candidatas en tiempo real. Este TFM propone e implementa un **Gemelo Digital basado en IA** que utiliza **Modelos Subrogados (Surrogate Models)** para predecir la viabilidad, el coste energético y el riesgo de colisión de una trayectoria en microsegundos.

### Características Clave
*   **Simulación Física (Ground Truth):** Entorno robótico basado en **PyBullet** con un manipulador KUKA IIWA de 7-DoF.
*   **Alta Dimensionalidad:** Manejo de trayectorias complejas con **350 dimensiones** ($7 \text{ joints} \times 50 \text{ steps}$).
*   **Deep Learning:** Implementación de un **Autoencoder Profundo** para reducción de dimensionalidad no lineal.
*   **Comparativa de Modelos:** Benchmarking riguroso entre **Redes Neuronales (DNN)**, **Kriging (Gaussian Processes)**, **SVR** y **RBF**.
*   **Interfaz Gráfica:** Aplicación interactiva en **Streamlit** para visualización y validación en tiempo real.

---

## Arquitectura del Sistema

El flujo de trabajo (*pipeline*) se divide en tres etapas críticas:

1.  **Generación de Datos (DoE):**
    *   Generación estocástica de escenarios con obstáculos dinámicos.
    *   Cálculo de trayectorias mediante Cinemática Inversa (IK) con inyección de ruido y variabilidad.
    *   Evaluación física en PyBullet para obtener el coste real y etiquetas de colisión.
2.  **Entrenamiento:**
    *   Compresión del espacio de entrada (350D $\to$ 16D Latentes) mediante Autoencoder.
    *   Entrenamiento condicional de modelos subrogados: $f(\text{Latente}, \text{Contexto}) \to \text{Coste}$.
3.  **Inferencia (Aplicación):**
    *   Uso del modelo para filtrar miles de trayectorias candidatas en milisegundos.

---

## Estructura del Repositorio

```text
TFM_Surrogate_Robot/
│
├── src/                      # Código fuente del núcleo
│   ├── simulation.py         # Motor físico (PyBullet)
│   ├── data_generation.py    # Generador de escenarios y trayectorias
│   ├── preprocessing.py      # Autoencoder y reducción de dimensionalidad
│   └── surrogate_models.py   # Definición de Kriging, NN, RBF, SVR
│
├── notebooks/                # Análisis y justificación estadística
│   └── EDA.ipynb  # EDA detallado de los datos generados
│
├── app.py                    # Interfaz gráfica (Gemelo Digital)
├── main.py                   # Script maestro de entrenamiento
└── requirements.txt          # Dependencias del proyecto
```

Nota: Las carpetas data/ y models/ no se incluyen en el repositorio para mantenerlo ligero. Se generan automáticamente al ejecutar el código.

## Instalación y Reproducción
Este proyecto está diseñado para ser totalmente reproducible. Sigue estos pasos para generar los datos, entrenar los modelos desde cero y lanzar la aplicación.
1. Clonar y Configurar Entorno
Se recomienda usar un entorno virtual (venv o conda).

- git clone https://github.com/tu-usuario/TFM_Surrogate_Robot.git
- cd TFM_Surrogate_Robot

# Crear entorno virtual 
- python -m venv venv
# Activar entorno (Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate)

# Instalar dependencias
- pip install -r requirements.txt

2. Generación de Datos y Entrenamiento
Ejecuta el script maestro. Este proceso generará 20,000 muestras de simulación física, entrenará el Autoencoder en GPU y ajustará los 4 modelos subrogados.

- python main.py

Tiempo estimado: 20-30 minutos (dependiendo de la GPU/CPU).
Al finalizar, verás las métricas de rendimiento en la consola y los gráficos en la carpeta results/.

3. Lanzar el Gemelo Digital
Una vez finalizado el entrenamiento, inicia la interfaz web:

- streamlit run app.py

Esto abrirá una pestaña en tu navegador donde podrás interactuar con el robot, generar escenarios aleatorios y probar la optimización en tiempo real.

## Resultados Obtenidos

El sistema ha sido validado con un dataset de prueba de **4,000 muestras inéditas** (20% de un total de 20,000), obteniendo los siguientes resultados:

| Modelo | R² Score (Precisión) | Recall (Seguridad)* | Speedup (vs Física) |
| :--- | :---: | :---: | :---: |
| **Neural Network (GPU)** | **0.961** | **97.0%** | **~1400x** |
| Kriging (Standard) | 0.864 | 93.4% | ~40x |
| SVR (Sklearn) | 0.809 | 92.4% | ~800x |
| RBF (SMT) | 0.644 | 40.5% | ~10x |

*\*Recall de Seguridad: Capacidad del modelo para detectar una colisión real. Un 97% indica que el sistema identificó correctamente el 97% de los choques peligrosos.*

## Tecnologías Utilizadas
Lenguaje: Python 3.9+
Simulación: PyBullet
Deep Learning: PyTorch (CUDA support)
Machine Learning: Scikit-Learn, SMT (Surrogate Modeling Toolbox)
Visualización: Plotly, Matplotlib, Seaborn
Frontend: Streamlit

## Contacto
Esteban Ruiz Hernández - estebanruiz435@gmial.com
Enlace al Proyecto: [URL de tu repositorio]
