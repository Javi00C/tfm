import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time
import random
import math

from gymnasium_env.envs.pybullet_ur5e_sim.ur5e_sim import UR5Sim

MAX_DISTANCE = 1.0  # Maximum allowable distance from target before termination
MAX_STEPS_SIM = 10000
#VELOCITY_SCALE = 0.02 
VELOCITY_SCALE = 0.06 
CLOSE_REWARD_DIST = 0.01

GOAL_SPAWN_RADIUS = 0.2


class ur5e_pybulletEnv_1a5(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.2, 0.2, 0.5]), max_steps=MAX_STEPS_SIM, render_mode=None):
        super().__init__()

        #self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode


        # Observation space
        self.num_robot_joints = 6
        self.ee_position_size = 3
        self.target_size = 3
        obs_dim = self.num_robot_joints + self.target_size + self.ee_position_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps)
        self.current_step = 0
        self.reward = 0
        self.distance = 0

        #Create random goal
        self.radius = GOAL_SPAWN_RADIUS
        self.center = self.sim.get_end_eff_position()#[0.49, 0.13, 0.69]
        self.target = np.array(self.random_point_in_sphere(self.radius, self.center),dtype=np.float32)

        self.done = False
        self.goal = False
        self.flag = False

    def random_point_in_sphere(self,radius, center):

        while True:
            # Generate a random point in the cube that bounds the sphere
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            z = random.uniform(-radius, radius)

            # Check if the point is inside the sphere
            if x**2 + y**2 + z**2 <= radius**2:
                return (center[0] + x, center[1] + y, center[2] + z)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.target = np.array(self.random_point_in_sphere(self.radius, self.center),dtype=np.float32)
        if self.flag == False:
            self.flag = True
            self.sim.reset()

        self.sim.add_visual_goal(self.target)

        self.current_step = 0
        self.done = False
        self.goal = False
        obs = self._get_obs()


        return obs, {}

    def step(self, action):
        #execute a step given the action
        end_effector_velocity = np.concatenate((action, np.array([0.0, 0.0, 0.0]))) * VELOCITY_SCALE
        self.sim.step(end_effector_velocity)
        #update the observation
        obs = self._get_obs()
        #compute the policy reward
        reward = self._calculate_reward()
        #increment current_step 
        self.current_step += 1
        #check if no more steps are needed
        self.done = self._check_done()
        self.goal = self._check_goal()
        #terminated flag True -> if _check_done() True
        terminated = self.done or self.goal
        #truncated flag True -> if maximum steps exectuted
        truncated = self.current_step >= self.max_steps
        #print(f"tcp angles: {self.sim.get_ee_angles()}")
        #print(f"robot tcp pose: {self.sim.get_end_eff_pose()}")
        #print(f"Distance ee to goal: {np.linalg.norm(self.sim.get_end_eff_position()-self.target[:3])}")
        distance_dict = {}  # Create a dictionary if it doesn't exist yet
        cart_dist_to_goal = np.linalg.norm(self.sim.get_end_eff_pose()[:3] - self.target[:3])  
        # Add the distance to the dictionary with an appropriate key
        distance_dict["distance_to_goal"] = cart_dist_to_goal
        return obs, reward, terminated, truncated, distance_dict

    def _calculate_reward(self):
        ee_pose = self.sim.get_end_eff_pose()
        position_error = np.linalg.norm(ee_pose[:3] - self.target[:3])
       
        if self.current_step == 0:
           self.distance = position_error
           self.reward = 0
        else:
           self.reward = (self.distance - position_error)*10
           self.distance = position_error
           
        if self.goal:
            reward += 10 
        #print(f"Reward: {self.reward}")
        return self.reward

    def _check_goal(self):
        ee_position = self.sim.get_end_eff_position()
        dist_to_target = np.linalg.norm(ee_position - self.target)
        if dist_to_target < CLOSE_REWARD_DIST :
            return True
        return False


    def _check_done(self):
        #ee_pose = self.sim.get_end_eff_pose()
        #dist_to_target = np.linalg.norm(ee_pose - self.target)
        ee_position = self.sim.get_end_eff_position()
        dist_to_target = np.linalg.norm(ee_position - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        
        tcp_pos = self.sim.get_end_eff_position()
        tcp_vel = self.sim.get_end_eff_vel()

        obs = np.concatenate((
            self.target,
            np.array(tcp_pos, dtype=np.float32),
            np.array(tcp_vel, dtype=np.float32)
        ), axis=0)

        obs = obs.flatten()
        obs = np.squeeze(obs)
        return obs

    def render(self):
        pass

    def close(self):
        self.sim.close()
