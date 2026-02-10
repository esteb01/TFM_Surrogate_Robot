import pybullet as p
import pybullet_data
import numpy as np
import imageio
import os


class RobotSimulator:
    """
    High-fidelity physical simulation environment based on PyBullet.
    Models a KUKA IIWA manipulator in a dynamic pick-and-place task with obstacles.
    """

    def __init__(self, dimension=350, gui_mode=False):
        self.dimension = dimension
        self.gui_mode = gui_mode
        self.n_joints = 7
        # Calculate time steps based on input dimension (e.g., 350 / 7 = 50 steps)
        self.n_steps = dimension // self.n_joints

        # Home pose (upright) to avoid initial singularities
        self.home_pose = [0, 0, 0, -1.57, 0, 1.57, 0]
        self.robot_id = None
        self.obstacle_id = None
        self.joint_indices = []

    def setup_bullet(self, obstacle_pos=None):
        """Initializes the physics engine, loads URDF assets, and configures the scene."""
        connection_mode = p.GUI if self.gui_mode else p.DIRECT
        if p.isConnected(): p.disconnect()
        p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load static environment
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.65], globalScaling=0.5)

        # Load Manipulator
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Configure Dynamic Obstacle
        # Default position is far away if not specified
        obs_pos = obstacle_pos if obstacle_pos is not None else [10, 10, 10]

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.15],
                                            rgbaColor=[0, 0, 1, 1])
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.15])
        self.obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collisionShapeId,
                                             baseVisualShapeIndex=visualShapeId, basePosition=obs_pos)

        # Identify controllable joints
        self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id)) if
                              p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED][:7]

    def get_ik_trajectory_advanced(self, target_pos, mid_point_height_offset=0.0):
        """
        Generates a baseline trajectory using Inverse Kinematics (IK).
        Introduces a variable mid-point height to simulate avoidance (high offset) or collision (low offset).
        """
        self.setup_bullet()

        # Reset to home
        for idx, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, self.home_pose[idx])

        trajectory = []

        # Define waypoints: Mid-point controls the arc height
        mid_x = target_pos[0] / 2
        mid_y = target_pos[1] / 2
        mid_z = 0.5 + mid_point_height_offset

        waypoints = [
            [mid_x, mid_y, mid_z],  # 1. Evasion/Transit point
            [target_pos[0], target_pos[1], 0.35],  # 2. Pre-grasp approach
            target_pos,  # 3. Grasp position
            target_pos,  # 4. Hold
            [target_pos[0], target_pos[1], 0.35]  # 5. Retract
        ]

        current_joints = self.home_pose
        steps_per_segment = self.n_steps // len(waypoints)

        # Interpolate in joint space
        for wp in waypoints:
            target_joint_pos = list(p.calculateInverseKinematics(self.robot_id, 6, wp)[:7])
            for i in range(steps_per_segment):
                alpha = (i + 1) / steps_per_segment
                interp = np.array(current_joints) * (1 - alpha) + np.array(target_joint_pos) * alpha
                trajectory.append(interp)
            current_joints = target_joint_pos

        p.disconnect()

        # Padding to ensure exact input dimension
        trajectory = np.array(trajectory)
        if len(trajectory) < self.n_steps:
            padding = np.tile(trajectory[-1], (self.n_steps - len(trajectory), 1))
            trajectory = np.vstack([trajectory, padding])

        return trajectory[:self.n_steps].T.flatten()

    def evaluate(self, X, targets, obstacles):
        """
        Ground Truth Evaluation.
        Computes the physical cost J(x) based on precision, energy (jerk), and collision penalties.
        """
        X = np.atleast_2d(X)
        targets = np.atleast_2d(targets)
        obstacles = np.atleast_2d(obstacles)

        costs = []
        self.setup_bullet()

        for i in range(len(X)):
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

            # Step-by-step physics simulation
            for t in range(self.n_steps):
                target_angles = traj[:, t]
                p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL,
                                            targetPositions=target_angles, forces=[500] * 7)
                p.stepSimulation()

                # 1. Smoothness Metric (Accumulated Jerk)
                if t > 0:
                    vel = np.linalg.norm(traj[:, t] - traj[:, t - 1])
                    acc = abs(vel - prev_vel)
                    total_jerk += acc
                    prev_vel = vel

                # 2. Collision Detection (Obstacle)
                contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obstacle_id)
                if len(contacts) > 0:
                    collision_penalty += 50  # Accumulative penalty per frame

                # 3. Collision Detection (Table/Floor)
                ee_pos = p.getLinkState(self.robot_id, 6)[0]
                if ee_pos[2] < 0.0:
                    collision_penalty += 100

                    # Track minimum distance to target
                dist = np.linalg.norm(np.array(ee_pos) - target)
                if dist < min_dist_to_target: min_dist_to_target = dist

            # Aggregated Cost Function
            final_cost = (min_dist_to_target * 500) + (total_jerk * 1.0) + collision_penalty
            costs.append(final_cost)

        p.disconnect()
        return np.array(costs).reshape(-1, 1)

    def generate_gif(self, trajectory_flat, target_pos, obstacle_pos, filename="simulation.gif"):
        """Renders the physical execution for visual validation."""
        self.setup_bullet(obstacle_pos)

        # Visual marker for Target
        vid_cup = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05], rgbaColor=[1, 0, 0, 1])
        cup_body = p.createMultiBody(baseVisualShapeIndex=vid_cup, basePosition=target_pos)

        frames = []
        traj = trajectory_flat.reshape(self.n_joints, self.n_steps)

        # Camera setup
        viewMatrix = p.computeViewMatrix([1.5, -1.0, 1.5], [0.5, 0, 0.2], [0, 0, 1])
        projectionMatrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)

        for idx, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, self.home_pose[idx])

        for t in range(self.n_steps):
            target_angles = traj[:, t]
            p.setJointMotorControlArray(self.robot_id, self.joint_indices, p.POSITION_CONTROL,
                                        targetPositions=target_angles, forces=[500] * 7)

            # Visual "Magnetic Grasp" simulation
            ee_pos = p.getLinkState(self.robot_id, 6)[0]
            cup_curr_pos, _ = p.getBasePositionAndOrientation(cup_body)

            if np.linalg.norm(np.array(ee_pos) - np.array(cup_curr_pos)) < 0.12 and t > (self.n_steps * 0.4):
                p.resetBasePositionAndOrientation(cup_body, ee_pos, [0, 0, 0, 1])

            # Physics substeps for smoother video
            for _ in range(3): p.stepSimulation()

            # Frame capture
            w, h, rgb, _, _ = p.getCameraImage(320, 240, viewMatrix, projectionMatrix, renderer=p.ER_TINY_RENDERER)
            img = np.array(rgb, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
            frames.append(img)

        p.disconnect()
        imageio.mimsave(filename, frames, fps=20, loop=0)
        return filename