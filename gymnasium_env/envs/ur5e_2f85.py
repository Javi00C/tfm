import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import mujoco

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import os

DIST_WEIGTH = 100000
MAX_REWARD = 1000
ACTION_SCALER = np.array([2, 2, 2, 255])  # Max velocities for x, y, z, and gripper actuation

#MAX_ACTION = np.array([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831, 255])

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

        # Observation space
        self.num_robot_joints = 6
        self.num_sensor_readings = 3
        obs_dim = 2*self.num_robot_joints + self.num_sensor_readings # multiplication by 2 because of qvel of robot
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath(mujocosim_path),
            5, # 5 frames between actions (frame_skip)
            observation_space=self.observation_space,
            **kwargs
        )

        # Define the action space (3d TCP velocities and gripper actuator velocity) ACTION SPACE IS DEFINED AFTER THE INIT BECAUSE IF NOT IT IS CHANGED BY IT
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.step_number = 0
        self.episode_len = episode_len
        self.done = False
        self.reward = 0

        # Initialize qpos and qvel
        robot_ini_pos = [-1.82, -1.82, 1.57, -2.95, -1.57, 1.45]
        gripper_ini_pos = [0.51020483, -0.15356462,  0.39897643, -0.17712295,  0.51018538, -0.25556113, 0.31270676,  0.04160611]
        roope_ini_pos = [0.0] * (self.model.nq - 14)
        self.init_qpos = robot_ini_pos + gripper_ini_pos + roope_ini_pos
        self.init_qvel = [0.0] * self.model.nv  # All velocities are initialized to zero

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_number = 0

        qpos = np.array(self.init_qpos, dtype=np.float32)
        qvel = np.array(self.init_qvel, dtype=np.float32)
        self.set_state(qpos, qvel)
        self.done = False
        self.current_step = 0
        self.obs = self._get_observation()
        return self.obs, {}

    def step(self, action):
        # Scale the normalized action to the desired velocity range
        scaled_action = action * ACTION_SCALER  # scaled_action.shape = (4,)

        # Extract the TCP velocities from the scaled action
        tcp_velocity = scaled_action[:3]  # Shape: (3,)

        # Compute the Jacobian at the current configuration
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        # Jacobian concerning only the six robot joints
        jacp_robot = jacp[:, :self.num_robot_joints]  # Shape: (3,6)

        # Compute joint velocities for the arm
        q_dot = np.linalg.pinv(jacp_robot) @ tcp_velocity  # q_dot shape: (6,)

        # Compute the time step
        dt = self.model.opt.timestep * self.frame_skip

        # Get current joint positions
        current_qpos = self.data.qpos[:self.num_robot_joints]

        # Compute desired joint positions by integrating velocities
        qpos_desired = current_qpos + q_dot * dt

        # Apply the desired joint positions as control inputs
        ctrl = np.zeros(self.model.nu)
        ctrl[:self.num_robot_joints] = qpos_desired  # Set desired positions for robot joints

        # Set the gripper joint control
        gripper_control = abs(scaled_action[3])  # The fourth element of the action vector THE ABSOLUTE VALUE HAS BEEN ADDED SINCE THE RANGE IS [0 255]
        # Assuming the gripper actuator is a position actuator
        ctrl[self.num_robot_joints] = gripper_control  # ctrl[6] = gripper_control

        # Do simulation
        self.do_simulation(ctrl, self.frame_skip)
        self.current_step += 1

        self.obs = self._get_observation()
        observation = self.obs
        self.done = self._check_done()
        terminated = self.done
        truncated = self.current_step >= self.episode_len

        reward = self._calculate_reward()
        return observation, reward, terminated, truncated, {}


    def _get_observation(self):
        obs = np.concatenate((
            np.array(self.data.qpos[:self.num_robot_joints], dtype=np.float32),
            np.array(self.data.qvel[:self.num_robot_joints], dtype=np.float32),
            np.array(self.data.sensordata, dtype=np.float32)
        ), axis=0)
        return obs

    
    # Computes the distance from each COM of each capsule to the desired final pose of each COM
    def _compute_dist_rope_COM(self, x0, L, z0):
        # Reshape rope_pos to extract positions of the 4 capsules
        self.rope_pos = self.data.qpos[14:].reshape(-1, 4)  # Each capsule has 4 values, assuming COM is in first 3
        
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

        # Reward based on how close the rope is to being horizontal
        self.rope_pos = self.data.qpos[13:]
        com_dists = self._compute_dist_rope_COM(0.5, 0.185, 0.8)
        # Rope initial pos= 0.5 0 0.8
        # Rope horizontal -> centers of mass of capsules in 3D line 0.5 n*x 0.8

        # IMPORTANT: Capsule positions are expressed in quaternions

        # Determine the current tile and whether it's on the edge
        if self.done:
            terminated_penalty = -100
            reward = 0
        else:
            terminated_penalty = 0
            # Compute the reward
            epsilon = 1e-3
            reward = DIST_WEIGTH * (np.sum(1 / (com_dists + epsilon)))                
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
        MAX_TCP_DIST = 10  # Maximum allowable distance from TCP to final rope position
        MIN_FORCE_THRESHOLD = 0.001  # Minimum force threshold to consider the rope grasped

        # Get the TCP position from the site in the XML (attachment_site)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
        tcp_position = self.data.site_xpos[site_id]

        # Compute the final desired position of the rope's COM
        rope_com_desired = np.array([0.5, 0.185, 0.8]) 

        # Compute the distance from TCP to the rope's desired final position
        tcp_to_rope_dist = np.linalg.norm(tcp_position - rope_com_desired)

        # Check if TCP is too far
        too_far = tcp_to_rope_dist > MAX_TCP_DIST

        # Check if the rope is grasped
        # Assuming sensor data includes forces, and the rope is not grasped if force < MIN_FORCE_THRESHOLD
        gripper_force = np.linalg.norm(self.data.sensordata[:3]) 
        rope_not_grasped = gripper_force < MIN_FORCE_THRESHOLD

        # Episode is terminated if either condition is true
        done = too_far or rope_not_grasped
        return bool(done) 

