import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import gymnasium as gym
from typing import Optional

# Constants
DIST_WEIGHT = 100000
MAX_REWARD = 1000
ACTION_SCALER = np.array([7, 7, 7, 1])  # Scaling for x, y, z velocities and gripper
EPISODE_LEN = 6000
URDF_PATH = "ur5_robot/ur5.urdf"  # Adjust the path to your URDF file

class UR5Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_len=EPISODE_LEN):
        super().__init__()
        
        # PyBullet setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load robot
        self.robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)

        # Observation and action spaces
        self.num_robot_joints = p.getNumJoints(self.robot_id)
        obs_dim = 2 * self.num_robot_joints + 3  # qpos, qvel, and sensor data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.episode_len = episode_len
        self.step_number = 0
        self.done = False

        # Robot initial position
        self.init_joint_positions = [-1.82, -1.82, 1.57, -2.95, -1.57, 1.45]
        self.reset()

    def reset(self):
        self.step_number = 0
        for i, pos in enumerate(self.init_joint_positions):
            p.resetJointState(self.robot_id, i, pos)
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, range(self.num_robot_joints))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        sensor_data = [0.0, 0.0, 0.0]  # Placeholder for sensor data
        return np.concatenate([joint_positions, joint_velocities, sensor_data])

    def step(self, action):
        # Scale action
        scaled_action = action * ACTION_SCALER

        # Apply action
        for i in range(self.num_robot_joints):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=scaled_action[i])
        
        p.stepSimulation()
        self.step_number += 1

        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self.step_number >= self.episode_len
        return observation, reward, terminated, False, {}

    def _calculate_reward(self):
        # Simplified reward logic
        return MAX_REWARD

    def close(self):
        p.disconnect()

# Create an instance of the environment
env = UR5Env()
