import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.robot_gripper_sim import UR5Sim

DIST_WEIGHT = 100000
MAX_REWARD = 1000
ACTION_SCALER = np.array([7, 7, 7, 1])  # Max velocities for x, y, z (gripper scaling is done in step function)
EPISODE_LEN = 6000

MAX_DISTANCE = 2.0  # Maximum allowable distance from target before termination

class ur5e_2f85_pybulletEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, target=np.array([0.5, 0.0, 0.3]), max_steps=500, render_mode=None):
        super().__init__()

        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: joint positions only (6D)
        self.observation_space = spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(6,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps)
        self.current_step = 0

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        # action is a 3D vector (vx, vy, vz)
        # Apply a scaling factor to translate [-1,1] action space to a suitable velocity range:
        velocity_scale = 0.06
        end_effector_velocity = action * velocity_scale

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_cmd=-1.0)

        obs = self._get_obs()

        reward = self._calculate_reward()
        
        self.current_step += 1
        self.done = self._check_done()
        terminated = self.done
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        # Compute reward: reward is high when close to the target, penalized with distance
        
        # Apply constant negative reward per step to encourage efficient behavior
        step_penalty = -10

        # Determine the current tile and whether it's on the edge
        if self.done:
            terminated_penalty = -1000
            reward = 0
        else:
            terminated_penalty = 0
            ee_pos, _ = self.sim.get_current_pose()
            dist = np.linalg.norm(ee_pos - self.target)
            reward = MAX_REWARD / (0.001 + dist)  # Reward function

        total_reward = step_penalty + reward + terminated_penalty
        return total_reward

    def _check_done(self):
        """Terminate the episode if the end-effector is too far from the target."""
        ee_pos, _ = self.sim.get_current_pose()
        dist_to_target = np.linalg.norm(ee_pos - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        # Joint angles as observation
        joints = self.sim.get_joint_angles()
        return np.array(joints, dtype=np.float32)

    def render(self):
        # If render_mode == "human", PyBullet GUI is already open.
        pass

    def close(self):
        self.sim.close()


# #Example usage:
# env = UR5e2f85BulletEnv(render_mode="human")
# obs, info = env.reset()
# for _ in range(10000):
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         break
# env.close()