import optuna
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.surrogate_models import NeuralSurrogate, KrigingSurrogate, RBFSurrogate, SVRSurrogate, PhysicsGuidedSurrogate
from src.preprocessing import DimensionalityReducer

# Reduce verbosity for cleaner CLI output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Orchestrates Bayesian optimization for surrogate model hyperparameters using Optuna.
    Handles data splitting, objective definition, and serialization of optimal parameters.
    """

    def __init__(self, X_full, y_reg_scaled, y_cls, input_dim, models_dir):
        """
        Initialize optimizer with preprocessed feature vectors and targets.

        Args:
            X_full: Combined Latent Space + Context features.
            y_reg_scaled: Normalized regression targets (log-cost).
            y_cls: Binary classification targets (collision).
        """
        self.X = X_full
        self.y_reg = y_reg_scaled
        self.y_cls = y_cls
        self.input_dim = input_dim
        self.models_dir = models_dir
        self.params_file = os.path.join(models_dir, 'best_params.json')

        # Stratified split for tuning validation (25% of the training set provided by main)
        # Random state fixed for reproducibility during HPO
        self.X_train, self.X_val, self.y_train, self.y_val, self.y_c_train, self.y_c_val = train_test_split(
            X_full, y_reg_scaled, y_cls, test_size=0.25, random_state=42
        )

    def _obj_kriging(self, trial):
        # Search space for Gaussian Process Regression
        params = {
            'poly': trial.suggest_categorical('poly', ['constant', 'linear']),
            'corr': trial.suggest_categorical('corr', ['squar_exp', 'abs_exp']),
            'nugget': trial.suggest_float('nugget', 1e-6, 1e-2, log=True),
            'theta0': trial.suggest_float('theta0', 1e-3, 1e-1, log=True)
        }

        # Subsample training data to speed up O(N^3) matrix inversion during search
        limit = 200
        model = KrigingSurrogate(params=params)
        model.fit(self.X_train[:limit], self.y_train[:limit])

        # Evaluate regression performance
        res = model.evaluate(self.X_val, self.y_val, collision_threshold=0.5)
        return res['r2']

    def _obj_neural(self, trial):
        # Search space for Standard DNN
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'epochs': 30  # Reduced epochs for rapid iteration
        }

        model = NeuralSurrogate(input_dim=self.input_dim, params=params)
        model.fit(self.X_train, self.y_train, collision_labels=self.y_c_train)

        res = model.evaluate(self.X_val, self.y_val, collision_threshold=0.5)
        return res['r2']

    def _obj_pinn(self, trial):
        # Search space for Physics-Guided NN
        # PINNs often require finer LR tuning due to the composite loss function
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
            'epochs': 30
        }

        model = PhysicsGuidedSurrogate(input_dim=self.input_dim, params=params)
        model.fit(self.X_train, self.y_train, collision_labels=self.y_c_train)

        res = model.evaluate(self.X_val, self.y_val, collision_threshold=0.5)
        return res['r2']

    def _obj_svr(self, trial):
        # Search space for Support Vector Regression
        params = {
            'C': trial.suggest_float('C', 1, 1000, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.2),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }

        limit = 2000
        model = SVRSurrogate(params=params)
        model.fit(self.X_train[:limit], self.y_train[:limit])

        res = model.evaluate(self.X_val, self.y_val, collision_threshold=0.5)
        return res['r2']

    def _obj_rbf(self, trial):
        # Search space for Radial Basis Functions
        params = {
            'd0': trial.suggest_float('d0', 0.1, 3.0),
            'poly_degree': trial.suggest_categorical('poly_degree', [0, 1]),
            'reg': trial.suggest_float('reg', 1e-5, 0.1, log=True)
        }

        limit = 2000
        model = RBFSurrogate(params=params)
        model.fit(self.X_train[:limit], self.y_train[:limit])

        res = model.evaluate(self.X_val, self.y_val, collision_threshold=0.5)
        return res['r2']

    def optimize_all(self, n_trials=15):
        """
        Executes studies for all registered models and saves results.
        """
        print(f"\n[HPO] Starting Hyperparameter Optimization ({n_trials} trials/model)...")
        best_params = {}

        studies = [
            ("Neural Network", self._obj_neural),
            ("PINN", self._obj_pinn),
            ("Kriging", self._obj_kriging),
            ("SVR", self._obj_svr),
            ("RBF", self._obj_rbf)
        ]

        for name, objective in studies:
            print(f"  > Tuning {name}...")
            # Maximize R2 Score
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            best_params[name] = study.best_params
            print(f"    Best Validation R2: {study.best_value:.4f}")

        # Serialize configuration
        with open(self.params_file, 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"[HPO] Optimization complete. Config saved to {self.params_file}")
        return best_params