import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Project Module Imports
from src.simulation import RobotSimulator
from src.data_generation import DataGenerator
from src.preprocessing import DimensionalityReducer
from src.surrogate_models import KrigingSurrogate, NeuralSurrogate, RBFSurrogate, SVRSurrogate, PhysicsGuidedSurrogate
from src.hyperparameter_tuning import HyperparameterOptimizer


def main():
    # ==========================================
    # 1. EXPERIMENTAL SETUP & CONFIGURATION
    # ==========================================
    # Hyperparameters for the experiment
    N_SAMPLES = 20000  # Large dataset to ensure statistical significance
    N_DIMENSIONS = 350  # High-dimensional input (7 joints * 50 time steps)
    LATENT_DIM = 16  # Target dimension for the Autoencoder compression
    CONTEXT_DIM = 6  # Environmental context variables (Target XYZ + Obstacle XYZ)
    HPO_TRIALS = 15  # Number of Optuna trials per model for hyperparameter tuning

    # Directory management
    DATA_DIR = os.path.join('data', 'raw')
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    RAW_DATA_PATH = os.path.join(DATA_DIR, 'robot_final_data.csv')

    # ==========================================
    # 2. DATA ACQUISITION
    # ==========================================
    print(f"--- 1. Data Generation Pipeline ({N_SAMPLES} samples) ---")

    if not os.path.exists(RAW_DATA_PATH):
        # Initialize simulator and data generator
        sim = RobotSimulator(dimension=N_DIMENSIONS)
        gen = DataGenerator(simulator=sim)

        # Generate stochastic scenarios including edge cases (near-misses and collisions)
        df = gen.create_dataset(n_samples=N_SAMPLES, save_path=RAW_DATA_PATH)
    else:
        print("Loading existing dataset from disk...")
        df = pd.read_csv(RAW_DATA_PATH)

    # ==========================================
    # 3. PREPROCESSING & FEATURE ENGINEERING
    # ==========================================
    # Extract feature subsets
    traj_cols = [c for c in df.columns if c.startswith('dim_')]
    context_cols = ['target_x', 'target_y', 'target_z', 'obs_x', 'obs_y', 'obs_z']

    X_traj = df[traj_cols].values
    X_context = df[context_cols].values
    y_raw = df['cost'].values

    # Log-Transformation of the Target
    # Compresses the dynamic range of the cost function (e.g., [100, 3000] -> [4.6, 8.0])
    # improving convergence for gradient-based methods.
    y_log = np.log1p(y_raw)

    # Binary Classification Labeling
    # Threshold empirically determined via EDA (Separation of safe vs. collision modes)
    COLLISION_THRESHOLD = 1500
    y_cls = (y_raw > COLLISION_THRESHOLD).astype(int)

    # Stratified Split (80% Train / 20% Test)
    idx = np.arange(len(X_traj))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_traj_train = X_traj[idx_train]
    X_traj_test = X_traj[idx_test]

    # ==========================================
    # 4. DIMENSIONALITY REDUCTION (Deep Autoencoder)
    # ==========================================
    print(f"\n--- 2. Training Autoencoder (GPU Accelerated) ---")
    dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)

    # Train the encoder to compress trajectory data
    Z_train = dr.fit_transform(X_traj_train)
    # Project test data into the learned latent space
    Z_test = dr.transform(X_traj_test)

    # ==========================================
    # 5. FEATURE FUSION & SCALING
    # ==========================================
    # Construct final input vectors: [Latent Features + Environmental Context]
    X_model_train = np.hstack([Z_train, X_context[idx_train]])
    X_model_test = np.hstack([Z_test, X_context[idx_test]])

    # Targets for Training
    y_train = y_log[idx_train]
    y_c_train = y_cls[idx_train]

    # MinMax Scaling for Regression Targets (Crucial for Neural Net convergence)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    # Transform test targets using training statistics
    y_test_scaled = target_scaler.transform(y_log[idx_test].reshape(-1, 1))

    # Calculate scaled threshold for evaluation metrics
    thresh_log = np.log1p(COLLISION_THRESHOLD)
    thresh_scaled = target_scaler.transform([[thresh_log]])[0][0]

    # Data Cleaning for Kriging Stability
    # Kriging requires a non-singular covariance matrix; duplicate points must be removed.
    df_k = pd.DataFrame(X_model_train)
    df_k['y'] = y_train_scaled
    df_k = df_k.drop_duplicates(subset=df_k.columns[:-1])
    X_kc = df_k.iloc[:, :-1].values
    y_kc = df_k['y'].values

    # ==========================================
    # 6. AUTOMATED HYPERPARAMETER OPTIMIZATION
    # ==========================================
    # Initialize and run Bayesian Optimization via Optuna
    optimizer = HyperparameterOptimizer(X_model_train, y_train_scaled, y_c_train, LATENT_DIM + CONTEXT_DIM, MODELS_DIR)
    best_params = optimizer.optimize_all(n_trials=HPO_TRIALS)

    # ==========================================
    # 7. FINAL MODEL TRAINING
    # ==========================================
    print("\n--- 3. Training Final Surrogate Models ---")

    # Instantiate models using optimized hyperparameters
    models = [
        KrigingSurrogate(params=best_params.get("Kriging")),
        NeuralSurrogate(input_dim=LATENT_DIM + CONTEXT_DIM, params=best_params.get("Neural Network")),
        PhysicsGuidedSurrogate(input_dim=LATENT_DIM + CONTEXT_DIM, params=best_params.get("PINN")),
        RBFSurrogate(params=best_params.get("RBF")),
        SVRSurrogate(params=best_params.get("SVR"))
    ]

    results = []
    # Naming convention for serialization
    name_map = {
        "Kriging (Standard)": "kriging",
        "Neural Network (Multi-Task)": "neural_net",
        "PINN (Physics-Guided)": "pinn",
        "RBF (SMT)": "rbf",
        "SVR (Sklearn)": "svr"
    }

    for model in models:
        print(f"\n>> Processing {model.name}...")

        # Training logic adapted to algorithm complexity
        if "Kriging" in model.name:
            # Kriging scales as O(N^3). Subsampling is required for feasibility.
            limit = min(len(X_kc), 400)
            print(f"[INFO] Kriging subsampled to {limit} points (Computational constraint).")
            model.fit(X_kc[:limit], y_kc[:limit])
        elif "Neural" in model.name or "PINN" in model.name:
            # Deep Learning models utilize the full dataset and GPU acceleration.
            model.fit(X_model_train, y_train_scaled, collision_labels=y_c_train, epochs=400, batch_size=256)
        elif "RBF" in model.name:
            # RBF memory usage is high; limited subset used.
            model.fit(X_kc[:3000], y_kc[:3000])
        else:
            # SVR scales quadratically; limited subset used.
            model.fit(X_model_train[:5000], y_train_scaled[:5000])

        # Evaluate on the hold-out test set
        metrics = model.evaluate(X_model_test, y_test_scaled, collision_threshold=thresh_scaled)
        metrics['Model'] = model.name
        results.append(metrics)

        # Serialize model
        filename = name_map.get(model.name, "model") + ".pkl"
        model.save(os.path.join(MODELS_DIR, filename))

    # Serialize global artifacts (Autoencoder & Scaler)
    dr.save(os.path.join(MODELS_DIR, 'autoencoder.pkl'))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, 'target_scaler.pkl'))

    # ==========================================
    # 8. ANALYSIS & VISUALIZATION
    # ==========================================
    print("\n--- 4. Generating Performance Analysis ---")

    final_results = []

    # Configure subplots (2x3 grid to accommodate 5 models)
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
    axes_cm = axes_cm.flatten()

    fig_reg, axes_reg = plt.subplots(2, 3, figsize=(18, 10))
    axes_reg = axes_reg.flatten()

    for i, res in enumerate(results):
        model_name = res['Model']
        y_true = res['y_true']
        y_pred = res['y_pred'].flatten()

        # Binary Classification Analysis
        y_true_cls = (y_true > thresh_scaled).astype(int)
        y_pred_cls = (y_pred > thresh_scaled).astype(int)
        cm = confusion_matrix(y_true_cls, y_pred_cls)

        # Aggregating metrics for CSV report
        metrics_row = {
            'Model': model_name, 'r2': res['r2'], 'rmse': res['rmse'], 'mae': res['mae'],
            'accuracy': res['accuracy'], 'precision': res['precision'], 'recall': res['recall'], 'f1': res['f1']
        }
        final_results.append(metrics_row)

        # Visualization: Confusion Matrix
        ax = axes_cm[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Safe', 'Crash'], yticklabels=['Safe', 'Crash'])
        ax.set_title(f"{model_name}\nRecall: {res['recall']:.1%}")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # Visualization: Regression Scatter Plot
        axr = axes_reg[i]
        axr.scatter(y_true, y_pred, alpha=0.3, s=5, c='purple')
        axr.plot([0, 1], [0, 1], 'r--', lw=2)  # Reference identity line
        axr.set_title(f"{model_name} ($R^2={res['r2']:.3f}$)")
        axr.grid(True, alpha=0.3)

    # Hide unused subplots
    if len(results) < len(axes_cm):
        for idx in range(len(results), len(axes_cm)):
            axes_cm[idx].axis('off')
            axes_reg[idx].axis('off')

    # Save plots to disk
    plt.figure(fig_cm.number)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))

    plt.figure(fig_reg.number)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'regression_plots.png'))

    print("[INFO] Analysis plots saved to /results folder.")

    # Save Metrics CSV
    df_final = pd.DataFrame(final_results)
    print("\n--- FINAL RESULTS TABLE ---")
    print(df_final.to_string(index=False))
    df_final.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

    print("\n[SUCCESS] Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()