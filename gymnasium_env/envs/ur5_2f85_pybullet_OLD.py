import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from simulation import RopeRobotSimulation  # The PyBullet-based simulation class we created

DIST_WEIGTH = 100000
MAX_REWARD = 1000
EPISODE_LEN = 6000  # Same as original

class UR5e2f85BulletEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(self, episode_len=EPISODE_LEN, gui=True):
        super().__init__()

        self.gui = gui
        self.episode_len = episode_len
        self.num_robot_joints = 6
        self.num_sensor_readings = 3  # for example, the EE pos
        obs_dim = 2*self.num_robot_joints + self.num_sensor_readings
        
        # Observation space: (joint_positions, joint_velocities, ee_pos)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space: (x, y, z, gripper fraction) in [-1,1]
        # We'll interpret these as incremental commands for the EE position and a fraction for the gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.sim = None
        self.current_step = 0

    def _init_simulation(self):
        if self.sim is not None:
            self.sim.close()
        self.sim = RopeRobotSimulation(gui=self.gui)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_simulation()
        obs_dict = self.sim.reset_simulation()
        self.current_step = 0
        obs = self._dict_to_obs(obs_dict)
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Interpret action
        # action[0:3]: desired ee position offsets (relative)
        # action[3]: gripper fraction in [-1,1], mapped to [0,1]
        gripper_fraction = (action[3] + 1.0) / 2.0  # maps [-1,1] to [0,1]

        # Get current observation to know current EE pos
        obs_dict = self.sim.get_observation()
        ee_pos = obs_dict["ee_pos"]  # current ee position (x,y,z)
        
        # Apply action: move EE by small increments, roll/pitch/yaw static for simplicity
        # We'll add scaled increments to the EE position
        delta_pos = action[:3] * 0.05  # scale increments
        new_ee_pos = np.array(ee_pos) + delta_pos
        # keep orientation fixed for simplicity
        ee_orientation = (0, 0, 0)

        # Move the robot
        # (x, y, z, roll, pitch, yaw) in 'end' control method
        # We keep orientation at 0 for simplicity
        self.sim.apply_action((new_ee_pos[0], new_ee_pos[1], new_ee_pos[2], ee_orientation[0], ee_orientation[1], ee_orientation[2]))
        self.sim.move_gripper(gripper_fraction)

        # Step simulation a few times
        # The original code steps simulation internally at a high frequency already
        # We'll just step a fixed number of times
        for _ in range(5):
            self.sim._step_simulation()
            time.sleep(1/240.0 if self.gui else 0.0)

        obs_dict = self.sim.get_observation()
        obs = self._dict_to_obs(obs_dict)

        reward = self._calculate_reward(obs_dict)
        done = self._check_done()
        terminated = done
        truncated = self.current_step >= self.episode_len

        return obs, reward, terminated, truncated, {}

    def _dict_to_obs(self, obs_dict):
        # obs_dict contains 'positions', 'velocities', 'ee_pos'
        positions = np.array(obs_dict["positions"], dtype=np.float32)
        velocities = np.array(obs_dict["velocities"], dtype=np.float32)
        ee_pos = np.array(obs_dict["ee_pos"], dtype=np.float32)
        return np.concatenate([positions, velocities, ee_pos], axis=0)

    def _calculate_reward(self, obs_dict):
        # Simple placeholder reward:
        # Reward based on EE closeness to a target point (e.g. (0.5,0.0,1.0))
        target = np.array([0.5, 0.0, 1.0])
        ee_pos = np.array(obs_dict["ee_pos"])
        dist = np.linalg.norm(ee_pos - target)
        # Inverse distance reward
        reward = 1/(dist+1e-3) - 0.1  # small step penalty
        return reward

    def _check_done(self):
        # Placeholder done condition: if EE goes too far from origin
        # or if simulation steps exceed a limit
        # You can refine this based on rope states if needed.
        return False

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
