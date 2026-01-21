import pybullet as p
import pybullet_data
import numpy as np
import imageio
import os


class RobotSimulator:
    """
    Entorno de simulación física de alta fidelidad basado en PyBullet.
    Modela un manipulador KUKA IIWA de 7 grados de libertad en tareas de Pick & Place con obstáculos.
    """

    def __init__(self, dimension=350, gui_mode=False):
        # Configuración dimensional del problema
        self.dimension = dimension
        self.gui_mode = gui_mode
        self.n_joints = 7
        self.n_steps = dimension // self.n_joints

        # Estado inicial seguro (Home Pose) para garantizar repetibilidad cinemática
        self.home_pose = [0, 0, 0, -1.57, 0, 1.57, 0]
        self.robot_id = None
        self.obstacle_id = None
        self.joint_indices = []

    def setup_bullet(self, obstacle_pos=None):
        """
        Inicializa el motor de física, carga assets (URDFs) y configura la escena.
        """
        # Selección del modo de renderizado (GUI para debug visual, DIRECT para velocidad en headless)
        connection_mode = p.GUI if self.gui_mode else p.DIRECT
        if p.isConnected(): p.disconnect()
        p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Carga de elementos estáticos del entorno
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.65], globalScaling=0.5)

        # Carga del robot manipulador
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Configuración del obstáculo dinámico
        # Posición por defecto lejana para evitar interferencias si no se especifica
        obs_pos = obstacle_pos if obstacle_pos is not None else [10, 10, 10]

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.15],
                                            rgbaColor=[0, 0, 1, 1])
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.15])
        self.obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collisionShapeId,
                                             baseVisualShapeIndex=visualShapeId, basePosition=obs_pos)

        # Mapeo de articulaciones controlables
        self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id)) if
                              p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED][:7]

    def get_ik_trajectory_advanced(self, target_pos, mid_point_height_offset=0.0):
        """
        Generador de trayectorias base mediante Cinemática Inversa (IK).
        Permite inyectar un offset de altura en el punto medio para modular la estrategia de evasión.
        """
        self.setup_bullet()

        # Reset del estado del robot
        for idx, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, self.home_pose[idx])

        trajectory = []

        # Definición de waypoints estratégicos
        # El punto medio (mid_z) es crítico: determina si la trayectoria intenta pasar por encima (offset positivo)
        # o colisionar (offset negativo/cero).
        mid_x = target_pos[0] / 2
        mid_y = target_pos[1] / 2
        mid_z = 0.5 + mid_point_height_offset

        waypoints = [
            [mid_x, mid_y, mid_z],  # Evasión/Tránsito
            [target_pos[0], target_pos[1], 0.35],  # Pre-agarre
            target_pos,  # Agarre (Target)
            target_pos,  # Estabilización
            [target_pos[0], target_pos[1], 0.35]  # Retirada
        ]

        current_joints = self.home_pose
        steps_per_segment = self.n_steps // len(waypoints)

        # Interpolación en el espacio articular
        for wp in waypoints:
            target_joint_pos = list(p.calculateInverseKinematics(self.robot_id, 6, wp)[:7])
            for i in range(steps_per_segment):
                alpha = (i + 1) / steps_per_segment
                interp = np.array(current_joints) * (1 - alpha) + np.array(target_joint_pos) * alpha
                trajectory.append(interp)
            current_joints = target_joint_pos

        p.disconnect()

        # Padding para consistencia dimensional
        trajectory = np.array(trajectory)
        if len(trajectory) < self.n_steps:
            padding = np.tile(trajectory[-1], (self.n_steps - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])

        return trajectory[:self.n_steps].T.flatten()

    def evaluate(self, X, targets, obstacles):
        """
        Motor de evaluación física (Ground Truth).
        Calcula la función de coste J(x) basada en precisión, suavidad y penalizaciones por colisión.
        """
        X = np.atleast_2d(X)
        targets = np.atleast_2d(targets)
        obstacles = np.atleast_2d(obstacles)

        costs = []
        self.setup_bullet()

        for i in range(len(X)):
            # Configuración del escenario i-ésimo
            traj_flat = X[i]
            target = targets[i]
            obstacle = obstacles[i]

            p.resetBasePositionAndOrientation(self.obstacle_id, obstacle, [0, 0, 0, 1])
            for idx, j_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, j_idx, self.home_pose[idx])

            traj = traj_flat.reshape(self.n_joints, self.n_steps)

            total_jerk = 0
            prev_vel = 0
            collision_penalty = 0
            min_dist_to_target = 100.0

            # Ejecución temporal de la trayectoria
            for t in range(self.n_steps):
                target_angles = traj[:, t]
                p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL,
                                            targetPositions=target_angles, forces=[500] * 7)
                p.stepSimulation()

                # 1. Métrica de Suavidad (Jerk acumulado)
                if t > 0:
                    vel = np.linalg.norm(traj[:, t] - traj[:, t - 1])
                    acc = abs(vel - prev_vel)
                    total_jerk += acc
                    prev_vel = vel

                # 2. Detección de Colisión (Obstáculo)
                # Penalización acumulativa por cada paso de tiempo en contacto
                contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obstacle_id)
                if len(contacts) > 0:
                    collision_penalty += 50

                    # 3. Detección de Colisión (Mesa/Suelo)
                ee_pos = p.getLinkState(self.robot_id, 6)[0]
                if ee_pos[2] < 0.0:
                    collision_penalty += 100

                    # Rastreo de mínima distancia al objetivo
                dist = np.linalg.norm(np.array(ee_pos) - target)
                if dist < min_dist_to_target: min_dist_to_target = dist

            # Función de Coste Agregada
            # Coste = Precisión + Suavidad + Penalizaciones
            final_cost = (min_dist_to_target * 500) + (total_jerk * 1.0) + collision_penalty
            costs.append(final_cost)

        p.disconnect()
        return np.array(costs).reshape(-1, 1)

    def generate_gif(self, trajectory_flat, target_pos, obstacle_pos, filename="simulation.gif"):
        """
        Renderiza la ejecución física para validación visual.
        """
        self.setup_bullet(obstacle_pos)

        # Elemento visual para el objetivo
        vid_cup = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05], rgbaColor=[1, 0, 0, 1])
        cup_body = p.createMultiBody(baseVisualShapeIndex=vid_cup, basePosition=target_pos)

        frames = []
        traj = trajectory_flat.reshape(self.n_joints, self.n_steps)

        # Configuración de cámara
        viewMatrix = p.computeViewMatrix([1.5, -1.0, 1.5], [0.5, 0, 0.2], [0, 0, 1])
        projectionMatrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)

        for idx, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, self.home_pose[idx])

        for t in range(self.n_steps):
            target_angles = traj[:, t]
            p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL,
                                        targetPositions=target_angles, forces=[500] * 7)

            # Simulación visual de agarre (Magnetic Grasp)
            ee_pos = p.getLinkState(self.robot_id, 6)[0]
            cup_curr_pos, _ = p.getBasePositionAndOrientation(cup_body)

            if np.linalg.norm(np.array(ee_pos) - np.array(cup_curr_pos)) < 0.12 and t > (self.n_steps * 0.4):
                p.resetBasePositionAndOrientation(cup_body, ee_pos, [0, 0, 0, 1])

            for _ in range(3): p.stepSimulation()

            # Captura de frame
            w, h, rgb, _, _ = p.getCameraImage(320, 240, viewMatrix, projectionMatrix, renderer=p.ER_TINY_RENDERER)
            img = np.array(rgb, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            frames.append(img)

        p.disconnect()
        imageio.mimsave(filename, frames, fps=20, loop=0)
        return filename