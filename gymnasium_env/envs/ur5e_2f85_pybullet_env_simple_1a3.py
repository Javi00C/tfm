import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.ur5e_gripper_sim_simple import UR5Sim

MAX_REWARD = 1000
MAX_DISTANCE = 1.0  # Maximum allowable distance from target before termination
MAX_DIST_REW = 2.0
MAX_STEPS_SIM = 4000
VELOCITY_SCALE = 1.0 #Originally at 0.3
CLOSE_REWARD_DIST = 0.1

class ur5e_2f85_pybulletEnv_Simple_1a3(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.5, 0.5, 0.5]), max_steps=MAX_STEPS_SIM, render_mode=None):
        super().__init__()

        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation space
        self.num_robot_joints = 3
        self.num_sensor_readings = 160*120
        self.rope_link_pose = 3
        self.target_size = 3
        obs_dim = 2*self.num_robot_joints + self.target_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps, goal_position=self.target)
        self.current_step = 0
        self.time_near_target = 0

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.time_near_target = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        #Only 3D cartesian action is being used
        cartesian_action = np.array(action[:3])
        angular_action = np.array([0,0,0])
        velocity_action = np.concatenate((cartesian_action,angular_action),axis=None)

        #gripper_action = action[6]
        gripper_action = 1.0

        velocity_scale = VELOCITY_SCALE #Maximum velocity that is stable in the simulation
        end_effector_velocity = velocity_action * velocity_scale

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_action)

        obs = self._get_obs()

        self.done = self._check_done()
        reward = self._calculate_reward()
        
        self.current_step += 1

        terminated = self.done
        truncated = self.current_step >= self.max_steps
        #print(f"tcp angles: {self.sim.get_ee_angles()}")
        #print(f"robot tcp pose: {self.sim.get_end_eff_pose()}")
        distance_dict = {}  # Create a dictionary if it doesn't exist yet
        cart_dist_to_goal = np.linalg.norm(self.sim.get_end_eff_pose()[:3] - self.target[:3])  
        # Add the distance to the dictionary with an appropriate key
        distance_dict["distance_to_goal"] = cart_dist_to_goal
        return obs, reward, terminated, truncated, distance_dict

    def _calculate_reward(self):
        ee_pose = self.sim.get_end_eff_pose()
        position_error = np.linalg.norm(ee_pose[:3] - self.target)
        
        if self.done:
            reward = -2
        else:
        
            reward = -position_error  # Weighted penalty for position and orientation errors
            
            # Bonus for being close to the target
            if position_error < CLOSE_REWARD_DIST:
                reward += 2.0
                self.time_near_target += 1
                reward += 0.1 * self.time_near_target
            else:
                self.time_near_target = 0
            # Small penalty for every time step
            reward -= 0.01  # Step penalty

        return reward


    def _check_done(self):
        """Terminate the episode if the end-effector is too far from the target."""
        ee_pose = self.sim.get_end_eff_pose()
        dist_to_target = np.linalg.norm(ee_pose[:3] - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        
        tcp_pos = self.sim.get_end_eff_pose()
        tcp_vel = self.sim.get_end_eff_vel()

        obs = np.concatenate((
            self.target,
            np.array(tcp_pos[:3], dtype=np.float32),
            np.array(tcp_vel[:3], dtype=np.float32)
        ), axis=None)

        obs = obs.flatten()
        obs = np.squeeze(obs)
        return obs

    def render(self):
        pass

    def close(self):
        self.sim.close()
