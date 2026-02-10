import numpy as np
import pandas as pd
from src.simulation import RobotSimulator


class DataGenerator:
    """
    Manages synthetic data generation for surrogate model training.
    Implements balanced sampling strategies to cover both successful and failure cases.
    """

    def __init__(self, simulator: RobotSimulator):
        self.simulator = simulator

    def generate_complex_dataset(self, n_samples=1000):
        """
        Generates stochastic scenarios with targets and obstacles.
        Enforces a distribution of trajectory heights to ensure class balance (Safe/Collision).
        """
        print(f"[INFO] Generating {n_samples} stochastic scenarios...")

        X_traj = []
        X_targets = []
        X_obstacles = []

        for i in range(n_samples):
            # 1. Target Sampling (Workspace coverage)
            tx = np.random.uniform(0.4, 0.7)
            ty = np.random.uniform(-0.4, 0.4)
            target = np.array([tx, ty, 0.05])

            # 2. Obstacle Interposition
            # Interpolate between origin and target to maximize conflict probability
            alpha = np.random.uniform(0.3, 0.7)
            ox = 0 + (tx - 0) * alpha
            oy = 0 + (ty - 0) * alpha

            # Stochastic perturbation
            ox += np.random.uniform(-0.05, 0.05)
            oy += np.random.uniform(-0.05, 0.05)

            obstacle = np.array([ox, oy, 0.25])

            # 3. Trajectory Generation Strategy
            # We force specific height offsets to ensure we have data for both classes
            r = np.random.rand()

            if r < 0.50:
                # Class A: Safe Trajectories (50%)
                # High positive offset ensures obstacle avoidance
                h_offset = np.random.uniform(0.2, 0.6)
            elif r < 0.80:
                # Class B: Direct Collisions (30%)
                # Low/Negative offset forces interaction with the obstacle
                h_offset = np.random.uniform(-0.2, 0.0)
            else:
                # Class C: Boundary/Edge Cases (20%)
                # Critical height where collision depends on fine geometry
                h_offset = np.random.uniform(0.0, 0.2)

            base_traj = self.simulator.get_ik_trajectory_advanced(target, mid_point_height_offset=h_offset)

            # Gaussian noise injection for robustness
            noise = np.random.normal(0, 0.01, base_traj.shape)
            sample_traj = base_traj + noise

            X_traj.append(sample_traj)
            X_targets.append(target)
            X_obstacles.append(obstacle)

            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{n_samples} samples generated")

        return np.array(X_traj), np.array(X_targets), np.array(X_obstacles)

    def create_dataset(self, n_samples=1000, save_path=None):
        """
        Orchestrates generation and physical evaluation.
        Returns a structured DataFrame.
        """
        X_traj, X_targets, X_obstacles = self.generate_complex_dataset(n_samples)

        print("[INFO] Running Physics Engine (Ground Truth Evaluation)...")
        y = self.simulator.evaluate(X_traj, X_targets, X_obstacles)

        # Structure data
        cols_traj = [f'dim_{i}' for i in range(self.simulator.dimension)]
        df = pd.DataFrame(X_traj, columns=cols_traj)

        # Explicit context variables
        df['target_x'] = X_targets[:, 0]
        df['target_y'] = X_targets[:, 1]
        df['target_z'] = X_targets[:, 2]
        df['obs_x'] = X_obstacles[:, 0]
        df['obs_y'] = X_obstacles[:, 1]
        df['obs_z'] = X_obstacles[:, 2]

        df['cost'] = y

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"[INFO] Dataset saved to: {save_path}")

        return df