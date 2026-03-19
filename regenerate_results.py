"""
regenerate_results.py
=====================
Re-evaluates all saved surrogate models on the original test set and
regenerates model_comparison.csv WITH the SMAPE column.

NO RETRAINING. All models are loaded from the existing .pkl files.
The train/test split uses random_state=42 (same as main.py) so the
test set is identical to the one used during training.

Usage:
    python regenerate_results.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.preprocessing import DimensionalityReducer
from src.surrogate_models import (
    KrigingSurrogate, NeuralSurrogate, RBFSurrogate,
    SVRSurrogate, PhysicsGuidedSurrogate
)

# ── Config (must match main.py exactly) ──────────────────────────────────────
N_DIMENSIONS      = 350
LATENT_DIM        = 16
CONTEXT_DIM       = 6
COLLISION_THRESHOLD = 1500

DATA_DIR    = os.path.join('data', 'raw')
MODELS_DIR  = 'models'
RESULTS_DIR = 'results'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'robot_final_data.csv')
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  SMAPE CSV Regeneration — no retraining")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    traj_cols    = [c for c in df.columns if c.startswith('dim_')]
    context_cols = ['target_x', 'target_y', 'target_z', 'obs_x', 'obs_y', 'obs_z']

    X_traj   = df[traj_cols].values
    X_context = df[context_cols].values
    y_raw    = df['cost'].values
    y_log    = np.log1p(y_raw)
    y_cls    = (y_raw > COLLISION_THRESHOLD).astype(int)

    # Identical split to main.py (random_state=42 → same test indices)
    idx = np.arange(len(X_traj))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    # 2. Load autoencoder and target scaler
    print("[2/4] Loading autoencoder and target scaler...")
    dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)
    dr.load(os.path.join(MODELS_DIR, 'autoencoder.pkl'))

    target_scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))

    # Project test trajectories into latent space
    Z_test       = dr.transform(X_traj[idx_test])
    X_model_test = np.hstack([Z_test, X_context[idx_test]])

    # Scale test targets
    y_test_scaled = target_scaler.transform(y_log[idx_test].reshape(-1, 1))

    thresh_log    = np.log1p(COLLISION_THRESHOLD)
    thresh_scaled = target_scaler.transform([[thresh_log]])[0][0]

    # 3. Load each model and evaluate
    print("[3/4] Evaluating models...\n")

    configs = [
        ("Kriging (Standard)",        "kriging.pkl",    KrigingSurrogate),
        ("Neural Network (Multi-Task)","neural_net.pkl", lambda: NeuralSurrogate(LATENT_DIM + CONTEXT_DIM)),
        ("PINN (Physics-Guided)",      "pinn.pkl",       lambda: PhysicsGuidedSurrogate(LATENT_DIM + CONTEXT_DIM)),
        ("RBF (SMT)",                  "rbf.pkl",        RBFSurrogate),
        ("SVR (Sklearn)",              "svr.pkl",        SVRSurrogate),
    ]

    rows = []
    for name, filename, constructor in configs:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] {name} — file not found: {path}")
            continue

        model = constructor()
        model.load(path)

        res = model.evaluate(X_model_test, y_test_scaled, collision_threshold=thresh_scaled)

        rows.append({
            'Model':     name,
            'r2':        res['r2'],
            'rmse':      res['rmse'],
            'mae':       res['mae'],
            'smape':     res['smape'],
            'accuracy':  res['accuracy'],
            'precision': res['precision'],
            'recall':    res['recall'],
            'f1':        res['f1'],
        })

    # 4. Save updated CSV
    print("\n[4/4] Saving updated model_comparison.csv...")
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    df_out.to_csv(out_path, index=False)

    print("\n── RESULTS ──────────────────────────────────────────────")
    print(df_out[['Model', 'r2', 'smape', 'recall']].to_string(index=False))
    print(f"\n[OK] CSV saved to {out_path}")
    print("Restart the Streamlit app to see SMAPE in the dashboard.")


if __name__ == "__main__":
    main()