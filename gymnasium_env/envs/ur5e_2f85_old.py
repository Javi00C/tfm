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

DIST_WEIGTH = 100000
MAX_REWARD = 1000
mujocosim_path = "scene_ur5_2f85.xml"

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

        self.init_qpos = [-1.82, -1.82, 1.57, -2.95, -1.57, 1.45, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.init_qvel = [0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_number = 0

        # for example, noise is added to positions and velocities
        qpos = np.array(self.init_qpos)
        # + self.np_random.uniform(
        #     size=self.model.nq, low=-0.01, high=0.01
        # )
        qvel = np.array(self.init_qvel) 
        # + self.np_random.uniform(
        #     size=self.model.nv, low=-0.01, high=0.01
        # )
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
    
    #Computes the distance from each COM of each capsule to the 
    # desired final pose of each COM
    def _compute_dist_rope_COM(self, x0, L, z0):
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
        com_dists = self._compute_dist_rope_COM(0.5,0.185,0.8)
        #rope initial pos= 0.5 0 0.8
        #rope horizontal -> centers of mass of capsules in 3d line 0.5 n*x 0.8
        # 0.5 0.0925 0.8, 0.5 0.2775
        # link_n_end_pos = (x0, L/2+L*n, z0) (first is link n=0)

        #IMPORTANT capsule positions are expressed in quaternions

        # Determine the current tile and whether it's on the edge (Positions could be divided to created larger tiles -> less memory)
        if self.done:
            terminated_penalty = -100
            reward = 0
        else:
            terminated_penalty = 0
            # Compute the reward
            # Reward is inversely proportional to the distances; closer means higher reward
            # Sum of inverses of distances for each capsule (avoid division by zero with a small epsilon)
            # Min distance to COM = 0.01
            epsilon = 1e-6
            reward = DIST_WEIGTH*(np.sum(1 / (com_dists + epsilon)))                
            if reward > MAX_REWARD:
                reward = MAX_REWARD
        # Calculate total reward per step
        total_reward = step_penalty + reward + terminated_penalty
        return total_reward

    def _check_done(self):
        """
        Checks whether the episode is done based on:
        1. The distance of the robot's TCP (attachment site) from the desired final position of the rope.
        2. Whether the rope is grasped (force sensor reading indicates no grasp).

        Returns:
        - done: bool, True if the episode should terminate, False otherwise.
        """
        # Constants
        MAX_TCP_DIST = 0.6  # Maximum allowable distance from TCP to final rope position
        MIN_FORCE_THRESHOLD = 0.1  # Minimum force threshold to consider the rope grasped

        # Get the TCP position from the site in the XML (attachment_site)
        tcp_position = self.data.site_xpos[self.model.site_name2id("attachment_site")]

        # Compute the final desired position of the rope's COM
        rope_com_desired = np.array([0.5, 0.185, 0.8])  # Adjust based on your target position

        # Compute the distance from TCP to the rope's desired final position
        tcp_to_rope_dist = np.linalg.norm(tcp_position - rope_com_desired)

        # Check if TCP is too far
        too_far = tcp_to_rope_dist > MAX_TCP_DIST

        # Check if the rope is grasped
        # Assuming sensor data includes forces, and the rope is not grasped if force < MIN_FORCE_THRESHOLD
        gripper_force = np.linalg.norm(self.data.sensordata[:3])  # Adjust sensor indices as necessary
        rope_not_grasped = gripper_force < MIN_FORCE_THRESHOLD

        # Episode is done if either condition is true
        return self.current_step >= self.max_steps or too_far or rope_not_grasped