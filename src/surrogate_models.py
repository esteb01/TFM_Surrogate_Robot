import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG, RBF
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


class SurrogateModel:
    """
    Clase base abstracta para modelos subrogados.
    Define la interfaz estándar para entrenamiento, predicción y evaluación.
    """

    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en un conjunto de prueba y calcula métricas clave.
        """
        y_pred = self.predict(X_test.astype(np.float64))

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        print(f"[{self.name}] R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MaxError: {max_err:.4f}")

        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'max_error': max_err,
            'y_pred': y_pred,
            'y_true': y_test
        }

    def save(self, path):
        """Persistencia estándar con Joblib."""
        joblib.dump(self.model, path)
        print(f"[{self.name}] Modelo guardado en {path}")

    def load(self, path):
        self.model = joblib.load(path)


# ==============================================================================
# 1. KRIGING (Gaussian Process Regression)
# ==============================================================================
class KrigingSurrogate(SurrogateModel):
    def __init__(self):
        super().__init__("Kriging (Standard)")
        # Configuración robusta con kernel exponencial cuadrado y nugget para estabilidad numérica.
        self.model = KRG(theta0=[1e-2], print_global=False, poly='constant', corr='squar_exp', nugget=1e-4)

    def fit(self, X, y):
        print(f"[{self.name}] Iniciando entrenamiento (KRG)...")
        try:
            self.model.set_training_values(X.astype(np.float64), y.astype(np.float64))
            self.model.train()
        except Exception as e:
            print(f"[ERROR] Fallo en convergencia de Kriging: {e}")

    def predict(self, X):
        return self.model.predict_values(X.astype(np.float64))


# ==============================================================================
# 2. DEEP NEURAL NETWORK (GPU Accelerated)
# ==============================================================================
class DNNNet(nn.Module):
    def __init__(self, input_dim):
        super(DNNNet, self).__init__()
        # Arquitectura profunda con normalización por capas y activaciones GELU.
        # Diseñada para capturar no-linealidades complejas (colisiones).
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),  # Regularización

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 1)  # Salida escalar (Coste)
        )

    def forward(self, x):
        return self.net(x)


class NeuralSurrogate(SurrogateModel):
    def __init__(self, input_dim):
        super().__init__("Neural Network (GPU)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DNNNet(input_dim).to(self.device)

    def fit(self, X, y, epochs=400, batch_size=256):
        print(f"[{self.name}] Entrenando en backend: {self.device}")

        # Conversión a tensores GPU
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizador AdamW con ciclo de Learning Rate
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0008, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(loader), epochs=epochs)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                scheduler.step()

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t)
        return preds.cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[{self.name}] Pesos de red guardados en {path}")

    def load(self, path):
        # Carga segura de pesos
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()


# ==============================================================================
# 3. RADIAL BASIS FUNCTION (Robust)
# ==============================================================================
class RBFSurrogate(SurrogateModel):
    def __init__(self):
        super().__init__("RBF (SMT)")
        # Pipeline manual: SMT RBF no incluye escalado interno.
        self.scaler = StandardScaler()
        # Configuración ajustada para evitar matrices singulares en alta dimensión.
        # reg=0.1 sacrifica interpolación exacta por estabilidad numérica.
        self.smt_model = RBF(d0=1.0, poly_degree=0, print_global=False, reg=0.1)

    def fit(self, X, y):
        print(f"[{self.name}] Entrenando con pre-escalado y regularización...")
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.smt_model.set_training_values(X_scaled.astype(np.float64), y.astype(np.float64))
            self.smt_model.train()
            # Guardamos como diccionario para persistir scaler + modelo
            self.model = {'smt': self.smt_model, 'scaler': self.scaler}
        except Exception as e:
            print(f"[ERROR] Fallo crítico en RBF: {e}")
            self.model = None

    def predict(self, X):
        if self.model is None: return np.zeros((len(X), 1))

        try:
            scaler = self.model['scaler']
            smt = self.model['smt']
            X_scaled = scaler.transform(X)
            preds = smt.predict_values(X_scaled.astype(np.float64))

            # Sanitización de NaNs (Protección contra inestabilidad numérica)
            if np.isnan(preds).any():
                # print(f"[{self.name}] Advertencia: NaNs detectados. Reemplazando por 0.")
                preds = np.nan_to_num(preds, nan=0.0)

            return preds
        except Exception as e:
            print(f"[{self.name}] Error en predicción: {e}")
            return np.zeros((len(X), 1))

    def save(self, path):
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f"[{self.name}] Modelo guardado en {path}")

    def load(self, path):
        self.model = joblib.load(path)


# ==============================================================================
# 4. SUPPORT VECTOR REGRESSION (SVR)
# ==============================================================================
class SVRSurrogate(SurrogateModel):
    def __init__(self):
        super().__init__("SVR (Sklearn)")
        # Pipeline automático de Sklearn
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.05))
        ])

    def fit(self, X, y):
        print(f"[{self.name}] Entrenando...")
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)