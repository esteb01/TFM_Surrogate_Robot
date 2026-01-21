import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderNet(nn.Module):
    """
    Arquitectura de Autoencoder Profundo para reducción de dimensionalidad no lineal.
    Diseñada para capturar la variedad topológica de trayectorias robóticas de alta dimensión.
    """

    def __init__(self, input_dim, latent_dim):
        super(AutoencoderNet, self).__init__()

        # Encoder: Compresión progresiva (Input -> 512 -> 256 -> 128 -> Latent)
        # Se utiliza BatchNorm y LeakyReLU para estabilizar el entrenamiento y evitar el desvanecimiento del gradiente.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, latent_dim)
        )

        # Decoder: Reconstrucción simétrica (Latent -> 128 -> 256 -> 512 -> Input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


class DimensionalityReducer:
    """
    Controlador para técnicas de reducción de dimensionalidad.
    Soporta PCA (lineal) y Autoencoders (no lineal) con aceleración GPU.
    """

    def __init__(self, method='pca', n_components=2):
        self.method = method
        self.n_components = n_components

        # Normalizadores específicos según el método
        self.scaler = StandardScaler()  # PCA requiere media 0 y varianza 1
        self.minmax = MinMaxScaler()  # NN convergen mejor en rango [0, 1]

        self.model = None
        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.method == 'autoencoder':
            print(f"[INFO] Backend de computación configurado: {self.device}")

    def fit_transform(self, X):
        """
        Ajusta el modelo de reducción y transforma los datos de entrada.
        """
        self.input_dim = X.shape[1]

        if self.method == 'pca':
            X_scaled = self.scaler.fit_transform(X)
            self.model = PCA(n_components=self.n_components)
            return self.model.fit_transform(X_scaled)

        elif self.method == 'autoencoder':
            # Preprocesamiento y carga en GPU
            X_scaled = self.minmax.fit_transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

            # Dataloader para entrenamiento por lotes (Batch Training)
            dataset = TensorDataset(X_tensor, X_tensor)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            # Inicialización del modelo
            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.n_components).to(self.device)

            # Optimizador AdamW con Scheduler para ajuste dinámico del Learning Rate
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            criterion = nn.MSELoss()

            epochs = 200
            self.model.train()
            print(f"[INFO] Iniciando entrenamiento del Autoencoder ({epochs} épocas)...")

            for epoch in range(epochs):
                total_loss = 0
                for batch_x, _ in loader:
                    optimizer.zero_grad()
                    rec, _ = self.model(batch_x)
                    loss = criterion(rec, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Monitoreo y ajuste de LR
                avg_loss = total_loss / len(loader)
                scheduler.step(avg_loss)

                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

            # Inferencia final para obtener el espacio latente
            self.model.eval()
            with torch.no_grad():
                _, X_latent = self.model(X_tensor)

            return X_latent.cpu().numpy()

    def transform(self, X):
        """
        Aplica la reducción de dimensionalidad a nuevos datos utilizando el modelo ajustado.
        """
        if self.method == 'pca':
            X_scaled = self.scaler.transform(X)
            return self.model.transform(X_scaled)

        elif self.method == 'autoencoder':
            X_scaled = self.minmax.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

            self.model.eval()
            with torch.no_grad():
                _, z = self.model(X_tensor)
            return z.cpu().numpy()

    def save(self, path):
        """Persistencia del modelo y sus preprocesadores."""
        state = {
            'method': self.method,
            'scaler': self.scaler,
            'minmax': self.minmax,
            'n_components': self.n_components,
            'input_dim': self.input_dim
        }
        if self.method == 'autoencoder':
            state['model_state_dict'] = self.model.state_dict()
        else:
            state['model'] = self.model
        joblib.dump(state, path)

    def load(self, path):
        """Carga del modelo desde disco."""
        state = joblib.load(path)
        self.method = state['method']
        self.scaler = state['scaler']
        self.minmax = state['minmax']
        self.n_components = state['n_components']
        self.input_dim = state.get('input_dim', 350)  # Fallback seguro

        if self.method == 'autoencoder':
            # Reconstrucción de la arquitectura PyTorch
            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.n_components).to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.model.eval()
        else:
            self.model = state['model']