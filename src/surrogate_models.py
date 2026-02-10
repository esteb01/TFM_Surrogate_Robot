import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG, RBF
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, accuracy_score, recall_score, \
    f1_score, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


class SurrogateModel:
    """
    Abstract base class for surrogate models.
    Standardizes interface for training, inference, and evaluation metrics.
    """

    def __init__(self, name, params=None):
        self.name = name
        self.model = None
        self.params = params if params else {}
        self.is_trained = False

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X_test, y_test, collision_threshold):
        """
        Computes regression (efficiency) and classification (safety) metrics.
        """
        if not self.is_trained:
            return {
                'rmse': 999.0, 'r2': -999.0, 'mae': 999.0, 'max_error': 999.0,
                'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
                'y_pred': np.zeros_like(y_test), 'y_true': y_test
            }

        y_pred = self.predict(X_test.astype(np.float64))

        # Handle Multi-Task output vs Single Output
        if isinstance(y_pred, tuple):
            y_pred_reg = y_pred[0]
            y_pred_class_prob = y_pred[1]
        else:
            y_pred_reg = y_pred
            y_pred_class_prob = None

        # Regression Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
        r2 = r2_score(y_test, y_pred_reg)
        mae = mean_absolute_error(y_test, y_pred_reg)
        max_err = max_error(y_test, y_pred_reg)

        # Classification Logic
        y_test_class = (y_test > collision_threshold).astype(int)

        if y_pred_class_prob is not None:
            # Explicit probability from NN head
            y_pred_class = (y_pred_class_prob > 0.5).astype(int)
        else:
            # Thresholding on regression output
            y_pred_class = (y_pred_reg > collision_threshold).astype(int)

        acc = accuracy_score(y_test_class, y_pred_class)
        rec = recall_score(y_test_class, y_pred_class, zero_division=0)
        prec = precision_score(y_test_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

        print(f"[{self.name}] Reg(R2={r2:.4f}) | Class(Recall={rec:.4f})")

        return {
            'rmse': rmse, 'r2': r2, 'mae': mae, 'max_error': max_err,
            'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1,
            'y_pred': y_pred_reg, 'y_true': y_test
        }

    def save(self, path):
        if self.is_trained:
            joblib.dump({'model': self.model, 'params': self.params}, path)
            print(f"[{self.name}] Model saved at {path}")

    def load(self, path):
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.params = data.get('params', {})
            self.is_trained = True
        except Exception as e:
            print(f"[{self.name}] Error loading model: {e}")


# ==============================================================================
# 1. KRIGING (Gaussian Process)
# ==============================================================================
class KrigingSurrogate(SurrogateModel):
    def __init__(self, params=None):
        super().__init__("Kriging (Standard)", params)
        theta0 = self.params.get('theta0', 1e-2)
        poly = self.params.get('poly', 'constant')
        corr = self.params.get('corr', 'squar_exp')
        nugget = self.params.get('nugget', 1e-4)

        self.model = KRG(theta0=[theta0], print_global=False, poly=poly, corr=corr, nugget=nugget)

    def fit(self, X, y, **kwargs):
        try:
            self.model.set_training_values(X.astype(np.float64), y.astype(np.float64))
            self.model.train()
            self.is_trained = True
        except Exception as e:
            print(f"[ERROR] Kriging training failed (Singular Matrix): {e}")
            self.is_trained = False

    def predict(self, X):
        if not self.is_trained: return np.zeros((len(X), 1))
        return self.model.predict_values(X.astype(np.float64))


# ==============================================================================
# 2. MULTI-TASK NEURAL NETWORK
# ==============================================================================
class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(MultiTaskNet, self).__init__()
        # Shared Feature Extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # Classification Head (Sigmoid for probability)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features), self.classification_head(features)


class NeuralSurrogate(SurrogateModel):
    def __init__(self, input_dim, params=None):
        super().__init__("Neural Network (Multi-Task)", params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.params.get('hidden_dim', 256)
        dropout = self.params.get('dropout', 0.1)
        self.lr = self.params.get('lr', 1e-3)
        self.epochs = self.params.get('epochs', 300)
        self.batch_size = self.params.get('batch_size', 256)

        self.model = MultiTaskNet(input_dim, hidden_dim, dropout).to(self.device)

    def fit(self, X, y, **kwargs):
        try:
            collision_labels = kwargs.get('collision_labels')
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_reg = torch.tensor(y, dtype=torch.float32).to(self.device)
            y_cls = torch.tensor(collision_labels, dtype=torch.float32).to(self.device)

            dataset = TensorDataset(X_t, y_reg, y_cls)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion_reg = nn.MSELoss()
            criterion_cls = nn.BCELoss()

            self.model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y_reg, batch_y_cls in loader:
                    optimizer.zero_grad()
                    pred_cost, pred_prob = self.model(batch_x)
                    loss = criterion_reg(pred_cost, batch_y_reg.view(-1, 1)) + 0.5 * criterion_cls(pred_prob,
                                                                                                   batch_y_cls.view(-1,
                                                                                                                    1))
                    loss.backward()
                    optimizer.step()
            self.is_trained = True
        except Exception as e:
            print(f"[ERROR] NN Training failed: {e}")
            self.is_trained = False

    def predict(self, X):
        if not self.is_trained: return np.zeros((len(X), 1)), np.zeros((len(X), 1))
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            cost, prob = self.model(X_t)
        return cost.cpu().numpy(), prob.cpu().numpy()

    def save(self, path):
        if self.is_trained:
            torch.save({'state_dict': self.model.state_dict(), 'params': self.params}, path)
            print(f"[{self.name}] Weights saved at {path}")

    def load(self, path):
        try:
            state = torch.load(path, weights_only=False)
            self.params = state['params']
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()
            self.is_trained = True
        except Exception as e:
            print(f"[{self.name}] Error loading weights: {e}")

# ==============================================================================
# 3. RBF & 4. SVR (Robust)
# ==============================================================================
class RBFSurrogate(SurrogateModel):
    def __init__(self, params=None):
        super().__init__("RBF (SMT)", params)
        self.scaler = StandardScaler()
        d0 = self.params.get('d0', 1.0)
        reg = self.params.get('reg', 0.1)
        self.smt_model = RBF(d0=d0, poly_degree=0, print_global=False, reg=reg)

    def fit(self, X, y, **kwargs):
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.smt_model.set_training_values(X_scaled.astype(np.float64), y.astype(np.float64))
            self.smt_model.train()
            self.model = {'smt': self.smt_model, 'scaler': self.scaler}
            self.is_trained = True
        except Exception:
            self.model = None

    def predict(self, X):
        if not self.is_trained or self.model is None: return np.zeros((len(X), 1))
        scaler = self.model['scaler']
        smt = self.model['smt']
        X_scaled = scaler.transform(X)
        preds = smt.predict_values(X_scaled.astype(np.float64))
        return np.nan_to_num(preds, nan=0.0)

    def save(self, path):
        if self.is_trained: joblib.dump({'model': self.model, 'params': self.params}, path)

    def load(self, path):
        try:
            data = joblib.load(path);
            self.model = data['model'];
            self.params = data.get('params', {});
            self.is_trained = True
        except:
            pass


class SVRSurrogate(SurrogateModel):
    def __init__(self, params=None):
        super().__init__("SVR (Sklearn)", params)
        C = self.params.get('C', 50)
        epsilon = self.params.get('epsilon', 0.05)
        gamma = self.params.get('gamma', 'scale')
        self.model = Pipeline(
            [('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))])

    def fit(self, X, y, **kwargs):
        try:
            self.model.fit(X, y.ravel())
            self.is_trained = True
        except Exception:
            self.is_trained = False

    def predict(self, X):
        if not self.is_trained: return np.zeros((len(X), 1))
        return self.model.predict(X).reshape(-1, 1)

# ==============================================================================
# 5. PHYSICS-GUIDED NEURAL NETWORK (PINN) - V2 (Balanced & Deeper)
# ==============================================================================
class PhysicsGuidedSurrogate(NeuralSurrogate):
    """
    Extends NeuralSurrogate with a re-balanced Physics-Informed Loss function.
    Focuses on improving classification Recall by up-weighting the safety-critical loss components.
    """

    def __init__(self, input_dim, params=None):
        super().__init__(input_dim, params)
        self.name = "PINN (Physics-Guided)"
        # Use a scaled threshold for calculations
        self.phys_threshold = 0.5  # A mid-point in the scaled [0, 1] range

    def fit(self, X, y, **kwargs):
        """
        Custom training loop with weighted multi-task and physics loss.
        """
        try:
            collision_labels = kwargs.get('collision_labels')
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_reg = torch.tensor(y, dtype=torch.float32).to(self.device)
            y_cls = torch.tensor(collision_labels, dtype=torch.float32).to(self.device)

            dataset = TensorDataset(X_t, y_reg, y_cls)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

            # --- CRITICAL CHANGE: WEIGHTED LOSS ---
            # Define loss functions without reduction to apply weights manually
            criterion_reg = nn.MSELoss()
            criterion_cls = nn.BCELoss(reduction='none')  # No reduction

            self.model.train()
            print(f"[{self.name}] Training with Balanced Physics Loss...")

            for epoch in range(self.epochs):
                for batch_x, batch_y_reg, batch_y_cls in loader:
                    optimizer.zero_grad()

                    pred_cost, pred_prob = self.model(batch_x)

                    # 1. Regression Loss (Standard)
                    loss_reg = criterion_reg(pred_cost, batch_y_reg.view(-1, 1))

                    # 2. Classification Loss (Manually Weighted)
                    # We give much more importance to getting crashes right.
                    bce_loss = criterion_cls(pred_prob, batch_y_cls.view(-1, 1))

                    # If label is 1 (crash), weight is high. If 0 (safe), weight is low.
                    # This tells the model: "It's much worse to miss a crash".
                    weights = torch.where(batch_y_cls.view(-1, 1) == 1, 10.0, 1.0)  # 10x weight for crashes
                    loss_cls = torch.mean(bce_loss * weights)

                    # 3. Physics Loss (Consistency)
                    # Penalty if (Cost < Threshold) AND (Prob > 0.5)
                    # This term is now less dominant, acting as a fine-tuner
                    violation = torch.nn.functional.relu(self.phys_threshold - pred_cost)
                    loss_phy = torch.mean(violation * pred_prob)

                    # --- FINAL LOSS COMPOSITION ---
                    # Give classification MUCH more weight than before
                    total_loss = loss_reg + 5.0 * loss_cls + 0.1 * loss_phy

                    total_loss.backward()
                    optimizer.step()

            self.is_trained = True

        except Exception as e:
            print(f"[ERROR] PINN Training failed: {e}")
            self.is_trained = False


