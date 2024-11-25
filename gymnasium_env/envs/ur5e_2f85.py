import numpy as np
import gymnasium as gym
import gymnasium_env
from gymnasium import spaces
import cv2
from typing import Optional
import random

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

mujocosim_path = "mujoco_ur5/scene_ur5_2f85.xml"

class ur5e_2f85Env(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath(mujocosim_path),
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        self.done = False
        self.reward = 0


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_number = 0

        # for example, noise is added to positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        self.done = False
        self.current_step = 0
        self.obs = self._get_observation()
        return self.obs

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        self.current_step += 1

        self.obs = self._get_observation()
        self.done = self._check_done()
        truncated = self.current_step > self.episode_len
        return self.obs, reward, self.done, truncated, {}
    
    def _get_observation(self):
        # Observation includes the joint positions and vels of robot + gripper and sensor
        obs = np.concatenate((np.array(self.data.qpos[0:13]), #All joints ur5 + gripper
                              np.array(self.data.qvel[0:13]),
                              np.array(self.data.sensordata)), axis=0)
        return obs
    
    def _compute_dist_rope_COMs(self,rope_pos, x0, L, z0):
        # Reshape rope_pos to extract positions of the 4 capsules
        self.rope_pos = self.rope_pos.reshape(-1, 4)  # Each capsule has 4 values, assuming COM is in first 3
        
        # Extract the actual COM positions (first 3 values for each capsule)
        actual_positions = self.rope_pos[:, :3]  # Shape: (4, 3)

        # Calculate the expected positions for the capsules
        expected_positions = np.array([
            [x0, L / 2 + L * n, z0] for n in range(4)
        ])  # Shape: (4, 3)

        # Compute Euclidean distances
        distances = np.linalg.norm(actual_positions - expected_positions, axis=1)  # Shape: (4,)

        return distances
        
    def _calculate_reward(self):
        # Apply constant negative reward per step to encourage efficient behavior
        step_penalty = -0.1

        #Reward based on how close the rope is to being horizontal
        self.rope_pos = self.data.qpos[13:30]
        com_dists = self._compute_dist_rope_COMs(rope_pos,0.5,0.185,0.8)
        #rope initial pos= 0.5 0 0.8
        #rope horizontal -> centers of mass of capsules in 3d line 0.5 n*x 0.8
        # 0.5 0.0925 0.8, 0.5 0.2775
        # link_n_end_pos = (x0, L/2+L*n, z0) (first is link n=0)

        #IMPORTANT capsule positions are expressed in quaternions


        # Reward for moving over unvisited edge tiles
        if not hasattr(self, 'visited_tiles'):
            self.visited_tiles = set()

        # Determine the current tile and whether it's on the edge (Positions could be divided to created larger tiles -> less memory)
        if self.done:
            terminated_penalty = -100
            reward = 0
        else:
            terminated_penalty = 0
            # Compute the reward
            # Reward is inversely proportional to the distances; closer means higher reward
            # Sum of inverses of distances for each capsule (avoid division by zero with a small epsilon)
            epsilon = 1e-6
            reward = np.sum(1 / (distances + epsilon))                
            
        # Calculate total reward per step
        total_reward = step_penalty + reward + terminated_penalty
        return total_reward
    
    def _check_done(self):
        # End the episode if the maximum number of steps is reached or the robot is too far from the edge
        distance_to_edge = self._calculate_distance_to_edge()
        return self.current_step >= self.max_steps or distance_to_edge > MAX_DIST
    

