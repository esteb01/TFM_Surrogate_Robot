import numpy as np
import pandas as pd
from src.simulation import RobotSimulator


class DataGenerator:
    """
    Gestor de generación de datasets sintéticos para entrenamiento de modelos subrogados.
    Implementa estrategias de muestreo balanceado para cubrir casos de éxito y fallo.
    """

    def __init__(self, simulator: RobotSimulator):
        self.simulator = simulator

    def generate_complex_dataset(self, n_samples=1000):
        """
        Genera un conjunto de trayectorias, objetivos y obstáculos con distribución controlada.
        Busca un balance entre trayectorias viables y colisiones para enriquecer el entrenamiento.
        """
        print(f"[INFO] Generando {n_samples} escenarios estocásticos con balanceo de clases...")

        X_traj = []
        X_targets = []
        X_obstacles = []

        for i in range(n_samples):
            # 1. Definición del Objetivo (Target)
            # Muestreo uniforme en el espacio de trabajo operativo del robot.
            tx = np.random.uniform(0.4, 0.7)
            ty = np.random.uniform(-0.4, 0.4)
            target = np.array([tx, ty, 0.05])

            # 2. Posicionamiento del Obstáculo
            # Se interpola entre el origen y el target para maximizar la probabilidad de interferencia.
            alpha = np.random.uniform(0.3, 0.7)
            ox = 0 + (tx - 0) * alpha
            oy = 0 + (ty - 0) * alpha

            # Pequeña perturbación estocástica para evitar alineación perfecta.
            ox += np.random.uniform(-0.05, 0.05)
            oy += np.random.uniform(-0.05, 0.05)

            obstacle = np.array([ox, oy, 0.25])

            # 3. Generación de Trayectoria (Estrategia de Muestreo Mixto)
            # Se fuerza una distribución de alturas para garantizar ejemplos de ambas clases (seguro/colisión).
            r = np.random.rand()

            if r < 0.50:
                # Clase A: Trayectorias Seguras (50%)
                # Offset de altura positivo para esquivar el obstáculo por arriba.
                h_offset = np.random.uniform(0.2, 0.6)
            elif r < 0.80:
                # Clase B: Colisiones Directas (30%)
                # Altura insuficiente, forzando interacción con el obstáculo.
                h_offset = np.random.uniform(-0.2, 0.0)
            else:
                # Clase C: Casos Límite / Frontera (20%)
                # Altura crítica donde la colisión depende de la geometría fina.
                h_offset = np.random.uniform(0.0, 0.2)

            # Cálculo de la cinemática inversa base con el offset de altura.
            base_traj = self.simulator.get_ik_trajectory_advanced(target, mid_point_height_offset=h_offset)

            # Inyección de ruido gaussiano para simular incertidumbre en el control o sensores.
            noise = np.random.normal(0, 0.01, base_traj.shape)
            sample_traj = base_traj + noise

            # Almacenamiento de vectores aplanados.
            X_traj.append(sample_traj)
            X_targets.append(target)
            X_obstacles.append(obstacle)

            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{n_samples} muestras generadas")

        return np.array(X_traj), np.array(X_targets), np.array(X_obstacles)

    def create_dataset(self, n_samples=1000, save_path=None):
        """
        Orquesta la generación de datos y su evaluación física.
        Retorna un DataFrame estructurado con inputs (trayectoria, contexto) y output (coste).
        """
        # Generación de la geometría del movimiento.
        X_traj, X_targets, X_obstacles = self.generate_complex_dataset(n_samples)

        print("[INFO] Ejecutando simulación física en PyBullet (Evaluación de Costes)...")
        # El simulador actúa como 'oráculo', calculando el coste real (ground truth).
        y = self.simulator.evaluate(X_traj, X_targets, X_obstacles)

        # Estructuración de datos en DataFrame.
        cols_traj = [f'dim_{i}' for i in range(self.simulator.dimension)]
        df = pd.DataFrame(X_traj, columns=cols_traj)

        # Añadimos variables de contexto explícitas.
        df['target_x'] = X_targets[:, 0]
        df['target_y'] = X_targets[:, 1]
        df['target_z'] = X_targets[:, 2]
        df['obs_x'] = X_obstacles[:, 0]
        df['obs_y'] = X_obstacles[:, 1]
        df['obs_z'] = X_obstacles[:, 2]

        # Variable objetivo.
        df['cost'] = y

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"[INFO] Dataset persistido en: {save_path}")

        return df