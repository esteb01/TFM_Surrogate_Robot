import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Importación de módulos propios
from src.simulation import RobotSimulator
from src.data_generation import DataGenerator
from src.preprocessing import DimensionalityReducer
from src.surrogate_models import KrigingSurrogate, NeuralSurrogate, RBFSurrogate, SVRSurrogate


def main():
    # ==============================================================================
    # 1. CONFIGURACIÓN DEL EXPERIMENTO
    # ==============================================================================
    # Definición de hiperparámetros globales para garantizar reproducibilidad.
    N_SAMPLES = 20000  # Tamaño del dataset para capturar la variabilidad estocástica
    N_DIMENSIONS = 350  # Dimensionalidad del input (7 joints * 50 steps)
    LATENT_DIM = 16  # Dimensión del espacio latente (Compresión ~95%)
    CONTEXT_DIM = 6  # Variables de contexto (Target + Obstáculo)

    # Estructura de directorios
    DATA_DIR = os.path.join('data', 'raw')
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'

    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    RAW_DATA_PATH = os.path.join(DATA_DIR, 'robot_final_data.csv')

    # ==============================================================================
    # 2. GENERACIÓN DE DATOS (High-Fidelity Simulation)
    # ==============================================================================
    print(f"--- 1. Generación de Datos ({N_SAMPLES} muestras) ---")

    if not os.path.exists(RAW_DATA_PATH):
        # Si no existe dataset previo, ejecutamos la simulación física masiva.
        sim = RobotSimulator(dimension=N_DIMENSIONS)
        gen = DataGenerator(simulator=sim)

        # Generación de trayectorias balanceadas (Seguras vs. Colisiones)
        df = gen.create_dataset(n_samples=N_SAMPLES, save_path=RAW_DATA_PATH)
    else:
        print("Cargando dataset existente desde disco...")
        df = pd.read_csv(RAW_DATA_PATH)

    # ==============================================================================
    # 3. PREPROCESAMIENTO Y TRANSFORMACIÓN
    # ==============================================================================
    # Separación de variables según su naturaleza (Trayectoria vs Contexto)
    traj_cols = [c for c in df.columns if c.startswith('dim_')]
    context_cols = ['target_x', 'target_y', 'target_z', 'obs_x', 'obs_y', 'obs_z']

    X_traj = df[traj_cols].values
    X_context = df[context_cols].values
    y_raw = df['cost'].values

    # Transformación Logarítmica de la Variable Objetivo ($J$)
    # Objetivo: Suavizar la distribución de costes y reducir el impacto de outliers (colisiones).
    y_log = np.log1p(y_raw)

    # División Train/Test (80/20) estratificada implícitamente por el orden aleatorio
    idx = np.arange(len(X_traj))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_traj_train, X_context_train, y_train = X_traj[idx_train], X_context[idx_train], y_log[idx_train]
    X_traj_test, X_context_test, y_test = X_traj[idx_test], X_context[idx_test], y_log[idx_test]

    # ==============================================================================
    # 4. REDUCCIÓN DE DIMENSIONALIDAD (Autoencoder Profundo)
    # ==============================================================================
    print(f"\n--- 2. Entrenamiento Autoencoder (GPU Accelerated) ---")
    dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)

    # Aprendizaje no supervisado de la variedad (manifold) de trayectorias
    Z_train = dr.fit_transform(X_traj_train)
    Z_test = dr.transform(X_traj_test)

    # ==============================================================================
    # 5. FUSIÓN DE CARACTERÍSTICAS (Feature Fusion)
    # ==============================================================================
    # Construcción del vector de entrada híbrido para los modelos subrogados.
    # Input = [Espacio Latente (Z) + Variables de Contexto (C)]
    X_model_train = np.hstack([Z_train, X_context_train])
    X_model_test = np.hstack([Z_test, X_context_test])

    # Normalización del Target (0-1) para facilitar convergencia de NN y SVR
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

    # Preprocesamiento específico para Kriging (Eliminación de duplicados)
    # Kriging requiere matrices de covarianza no singulares.
    df_k = pd.DataFrame(X_model_train)
    df_k['y'] = y_train_scaled
    df_k = df_k.drop_duplicates(subset=df_k.columns[:-1])
    X_kc = df_k.iloc[:, :-1].values
    y_kc = df_k['y'].values

    # ==============================================================================
    # 6. ENTRENAMIENTO DE MODELOS SUBROGADOS
    # ==============================================================================
    print("\n--- 3. Entrenamiento y Evaluación de Modelos ---")

    models = [
        KrigingSurrogate(),
        NeuralSurrogate(input_dim=LATENT_DIM + CONTEXT_DIM),
        RBFSurrogate(),
        SVRSurrogate()
    ]

    results = []
    # Mapeo para persistencia de archivos
    name_map = {
        "Kriging (Standard)": "kriging",
        "Neural Network (GPU)": "neural_net",
        "RBF (SMT)": "rbf",
        "SVR (Sklearn)": "svr"
    }

    for model in models:
        print(f"\nProcesando {model.name}...")

        # Estrategias de entrenamiento adaptadas a la complejidad computacional de cada algoritmo
        if "Kriging" in model.name:
            # Kriging escala cúbicamente O(N^3). Limitamos muestras para viabilidad.
            limit = min(len(X_kc), 350)
            print(f"[INFO] Kriging limitado a {limit} muestras (Restricción Computacional).")
            model.fit(X_kc[:limit], y_kc[:limit])

        elif "Neural" in model.name:
            # Deep Learning aprovecha grandes volúmenes de datos.
            model.fit(X_model_train, y_train_scaled, epochs=400, batch_size=256)

        elif "RBF" in model.name:
            # RBF es sensible a matrices densas grandes.
            limit = min(len(X_kc), 4000)
            model.fit(X_kc[:limit], y_kc[:limit])

        else:  # SVR
            limit = min(len(X_model_train), 5000)
            model.fit(X_model_train[:limit], y_train_scaled[:limit])

        # Evaluación en conjunto de test (datos no vistos)
        metrics = model.evaluate(X_model_test, y_test_scaled)
        metrics['Model'] = model.name
        results.append(metrics)

        # Persistencia del modelo entrenado
        filename = name_map.get(model.name, "m") + ".pkl"
        model.save(os.path.join(MODELS_DIR, filename))

    # Guardado de artefactos globales para inferencia
    dr.save(os.path.join(MODELS_DIR, 'autoencoder.pkl'))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, 'target_scaler.pkl'))

    # ==============================================================================
    # 7. ANÁLISIS DE RESULTADOS (Clasificación + Regresión)
    # ==============================================================================
    print("\n--- 4. Generando Análisis de Desempeño ---")

    # Definición del Umbral de Seguridad
    # Basado en el análisis EDA, J=1500 separa trayectorias seguras de colisiones.
    COLLISION_THRESHOLD_REAL = 1500
    thresh_log = np.log1p(COLLISION_THRESHOLD_REAL)
    thresh_scaled = target_scaler.transform([[thresh_log]])[0][0]

    final_results = []

    # Configuración de gráficos comparativos
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
    axes_cm = axes_cm.flatten()

    fig_reg, axes_reg = plt.subplots(2, 2, figsize=(12, 10))
    axes_reg = axes_reg.flatten()

    for i, res in enumerate(results):
        model_name = res['Model']
        y_true = res['y_true']
        y_pred = res['y_pred'].flatten()

        # Cálculo de métricas de clasificación (Seguridad)
        true_class = (y_true > thresh_scaled).astype(int)
        pred_class = (y_pred > thresh_scaled).astype(int)

        acc = np.mean(true_class == pred_class)
        prec = precision_score(true_class, pred_class, zero_division=0)
        rec = recall_score(true_class, pred_class, zero_division=0)
        f1 = f1_score(true_class, pred_class, zero_division=0)

        # Registro de métricas extendidas
        metrics_row = {
            'Model': model_name,
            'R2': res['r2'],
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }
        final_results.append(metrics_row)

        # Visualización: Matriz de Confusión
        cm = confusion_matrix(true_class, pred_class)
        ax = axes_cm[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Seguro', 'Choque'], yticklabels=['Seguro', 'Choque'])
        ax.set_title(f"{model_name}\nRecall: {rec:.1%} | F1: {f1:.1%}")
        ax.set_ylabel('Realidad')
        ax.set_xlabel('Predicción')

        # Visualización: Regresión (Predicho vs Real)
        axr = axes_reg[i]
        axr.scatter(y_true, y_pred, alpha=0.3, s=5, c='purple')
        axr.plot([0, 1], [0, 1], 'r--', lw=2)  # Línea ideal
        axr.set_title(f"{model_name} ($R^2={res['r2']:.3f}$)")
        axr.grid(True, alpha=0.3)

    # Exportación de gráficos
    plt.figure(fig_cm.number)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))

    plt.figure(fig_reg.number)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'regression_plots.png'))

    # Exportación de tabla final
    df_final = pd.DataFrame(final_results)
    print("\n--- TABLA FINAL DE RESULTADOS ---")
    print(df_final.to_string(index=False))
    df_final.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

    print("\n[ÉXITO] Pipeline completo finalizado.")


if __name__ == "__main__":
    main()