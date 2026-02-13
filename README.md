# Robotic Trajectory Optimization using Surrogate Models

> **(TFM)** | Master in Applied Artificial Intelligence
> - **Author:** Esteban Ruiz Hernández
> - **Supervisor:** Carlos Cernuda

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Description

This project addresses a critical challenge in modern robotics: **real-time motion planning** in high-dimensional dynamic environments.

High-fidelity physical simulations (such as PyBullet) are accurate but computationally expensive (~50ms per evaluation), preventing their use for evaluating thousands of candidate trajectories in real-time. This TFM proposes and implements an **AI-based Digital Twin** that utilizes **Surrogate Models** to predict the viability, energy cost, and collision risk of a trajectory in microseconds.

### Key Features
*   **Physical Simulation (Ground Truth):** Robotic environment based on **PyBullet** with a 7-DoF KUKA IIWA manipulator.
*   **High Dimensionality:** Handling of complex trajectories with **350 dimensions** ($7 \text{ joints} \times 50 \text{ steps}$).
*   **Deep Learning:** Implementation of a **Deep Autoencoder** for nonlinear dimensionality reduction.
*   **Model Comparison:** Rigorous benchmarking between **Neural Networks (DNN)**, **Physics-Guided Neural Networks (PINN)**, **Kriging (Gaussian Processes)**, **SVR**, and **RBF**.
*   **Automated Optimization (AutoML):** Integration of **Optuna** for hyperparameter tuning.
*   **Multi-Task Architecture:** Simultaneous prediction of regression (cost) and classification (collision) metrics.
*   **Graphical Interface:** Interactive application in **Streamlit** for real-time visualization and validation.

---

## System Architecture

The workflow (*pipeline*) is divided into four critical stages:

1.  **Data Generation (DoE):**
    *   Stochastic generation of scenarios with dynamic obstacles.
    *   Trajectory calculation using Inverse Kinematics (IK) with noise injection and variability.
    *   Physical evaluation in PyBullet to obtain real cost and collision labels.
2.  **Preprocessing & Dimensionality Reduction:**
    *   Compression of the input space (350D $\to$ 16D Latent) using a Deep Autoencoder.
    *   Data preparation for Multi-Task learning.
3.  **Training & Optimization:**
    *   Automated hyperparameter search using **Optuna**.
    *   Conditional training of surrogate models: $f(\text{Latent}, \text{Context}) \to (\text{Cost}, \text{Collision Prob})$.
    *   Implementation of Physics-Guided Loss functions.
4.  **Inference (Application):**
    *   Use of the model to filter thousands of candidate trajectories in milliseconds.

---

## Repository Structure

```text
TFM_Surrogate_Robot/
│
├── src/                      # Core source code
│   ├── simulation.py         # Physics engine (PyBullet)
│   ├── data_generation.py    # Scenario and trajectory generator
│   ├── preprocessing.py      # Autoencoder and dimensionality reduction
│   ├── surrogate_models.py   # Definition of Kriging, NN, PINN, RBF, SVR
│   └── hyperparameter_tuning.py # Optuna optimization script
│
├── notebooks/                # Analysis and statistical justification
│   └── EDA.ipynb             # Detailed EDA of generated data
│
├── app.py                    # Graphical Interface (Digital Twin)
├── main.py                   # Master training script
└── requirements.txt          # Project dependencies
```

Note: The data/ and models/ folders are not included in the repository to keep it lightweight. They are automatically generated when running the code.

---

## Installation and Reproduction
This project is designed to be fully reproducible. Follow these steps to generate data, train models from scratch, and launch the application.
1. Clone and Configure Environment
Using a virtual environment (venv or conda) is recommended.

- git clone https://github.com/esteb01/TFM_Surrogate_Robot.git
- cd TFM_Surrogate_Robot

---

## Create virtual environment 
- python -m venv venv
- Activate environment (Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate)

---

## Install dependencies
- pip install -r requirements.txt

2. Data Generation and Training
Run the master script. This process will generate 20,000 physical simulation samples, train the Autoencoder on GPU, optimize hyperparameters with Optuna, and fit the 5 surrogate models.

- python main.py

Estimated time: 20-30 minutes (depending on GPU/CPU).
Upon completion, you will see performance metrics in the console and graphs in the results/ folder.

3. Launch the Digital Twin
Once training is finished, start the web interface:

- streamlit run app.py

This will open a tab in your browser where you can interact with the robot, generate random scenarios, and test real-time optimization.

---

## Obtained Results

The system has been validated with a test dataset of 4,000 unseen samples (20% of a total of 20,000). The Neural Network achieved the best regression accuracy, while the PINN demonstrated superior safety recall.

| Model | R² Score (Precision) | Recall (Safety)* | Speedup (vs Physics) |
| :--- | :---: | :---: | :---: |
| **Neural Network (Multi-Task)** | **0.918** | 84.6% | **~1400x** |
| Kriging (Standard) | 0.895 | 93.6% | ~40x |
| PINN (Physics-Guided) | 0.852 | **96.2%** | ~1400x |
| SVR (Sklearn) | 0.856 | 94.4% | ~800x |
| RBF (SMT) | 0.851 | 92.0% | ~10x |

*\*Safety Recall: The model's ability to detect a real collision. A higher value indicates the system correctly identified more dangerous crashes (fewer false negatives).*

---

## Software 
- Language: Python 3.9+
- Simulation: PyBullet
- Deep Learning: PyTorch (CUDA support)
- AutoML: Optuna
- Machine Learning: Scikit-Learn, SMT (Surrogate Modeling Toolbox)
- Visualization: Plotly, Matplotlib, Seaborn
- Frontend: Streamlit

---

## Contact
- Esteban Ruiz Hernández - estebanruiz435@gmail.com
- Project Link: (https://github.com/esteb01/TFM_Surrogate_Robot)
