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
    Deep Autoencoder architecture for nonlinear dimensionality reduction.
    Designed to capture the topological manifold of high-dimensional robotic trajectories.
    """

    def __init__(self, input_dim, latent_dim):
        super(AutoencoderNet, self).__init__()

        # Encoder: Progressive compression
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

        # Decoder: Symmetric reconstruction
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
    Wrapper for dimensionality reduction techniques.
    Supports PCA (linear) and Autoencoders (nonlinear) with GPU acceleration.
    """

    def __init__(self, method='pca', n_components=2):
        self.method = method
        self.n_components = n_components

        self.scaler = StandardScaler()  # For PCA
        self.minmax = MinMaxScaler()  # For NN

        self.model = None
        self.input_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.method == 'autoencoder':
            print(f"[INFO] Compute backend: {self.device}")

    def fit_transform(self, X):
        self.input_dim = X.shape[1]

        if self.method == 'pca':
            X_scaled = self.scaler.fit_transform(X)
            self.model = PCA(n_components=self.n_components)
            return self.model.fit_transform(X_scaled)

        elif self.method == 'autoencoder':
            # Preprocessing and GPU transfer
            X_scaled = self.minmax.fit_transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

            dataset = TensorDataset(X_tensor, X_tensor)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.n_components).to(self.device)
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            criterion = nn.MSELoss()

            epochs = 200
            self.model.train()
            print(f"[INFO] Starting Autoencoder training ({epochs} epochs)...")

            for epoch in range(epochs):
                total_loss = 0
                for batch_x, _ in loader:
                    optimizer.zero_grad()
                    rec, _ = self.model(batch_x)
                    loss = criterion(rec, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                scheduler.step(avg_loss)

                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

            # Inference
            self.model.eval()
            with torch.no_grad():
                _, X_latent = self.model(X_tensor)

            return X_latent.cpu().numpy()

    def transform(self, X):
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
        state = joblib.load(path)
        self.method = state['method']
        self.scaler = state['scaler']
        self.minmax = state['minmax']
        self.n_components = state['n_components']
        self.input_dim = state.get('input_dim', 350)

        if self.method == 'autoencoder':
            self.model = AutoencoderNet(input_dim=self.input_dim, latent_dim=self.n_components).to(self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.model.eval()
        else:
            self.model = state['model']