import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time
import random
import math
import csv

from gymnasium_env.envs.pybullet_ur5e_sim.ur5e_sim_orient import UR5Sim

MAX_DISTANCE = 2.0  # Maximum allowable distance from target before termination
MAX_STEPS_SIM = 10000
#VELOCITY_SCALE = 0.02 
CARTESIAN_VEL_SCALE = 0.1 
ANGULAR_VEL_SCALE = 0.5
CLOSE_REWARD_DIST = 0.01

GOAL_SPAWN_RADIUS = 0.05


class ur5e_pybulletEnv_orient(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.2, 0.2, 0.5]), max_steps=MAX_STEPS_SIM, render_mode=None):
        super().__init__()

        self.goal = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation space
        self.tcp_coordinates = 6
        self.ee_pose_size = 6
        self.goal_size = 6
        obs_dim = self.tcp_coordinates + self.goal_size + self.ee_pose_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps)
        self.current_step = 0
        self.reward = 0
        self.distance_cart = 0
        self.distance_orient = 0


        self.done = False
        self.goal_reached = False

    # def create_goal(self):
    #     dist_pose_rchd = 1
    #     while dist_pose_rchd > 0.1:
    #         self.orientation_goal = self.random_orient_in_sphere()
    #         self.position_goal = self.random_point_in_sphere(self.radius,self.center)
    #         self.goal = np.array(self.orientation_goal + self.position_goal,dtype=np.float32)
    #         joint_angles = self.sim.calculate_ik(self.position_goal,self.orientation_goal)
    #         self.sim.set_joint_angles(joint_angles)
            
    #         self.sim.goal_step_sim() # steps the simulation so that it tries to get to goal point

    #         pose_rchd = self.sim.get_end_eff_pose()
    #         dist_pose_rchd = np.linalg.norm(pose_rchd - self.goal)
    #     print("Goal found")
    #     self.sim.add_visual_goal_orient(self.goal)

    def create_goal(self):
        reachable_goals_file = "reachable_goals.csv"

        # Read all reachable goals from the file
        with open(reachable_goals_file, mode='r') as file:
            reader = csv.reader(file)
            goals = [list(map(float, row)) for row in reader]

        if not goals:
            raise ValueError("No goals found in the reachable goals file.")

        # Select a random goal from the list
        random_goal = random.choice(goals)
        self.goal = np.array(random_goal, dtype=np.float32)

        self.sim.add_visual_goal_orient(self.goal)

        return self.goal

    def random_orient_in_sphere(self): # returns tuple
        roll = random.uniform(-math.pi, math.pi)
        pitch = random.uniform(-math.pi, math.pi)
        yaw = random.uniform(-math.pi, math.pi)
        return (roll,pitch,yaw)

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
        
        self.sim.reset()

        self.current_step = 0
        self.done = False
        self.goal_reached = False
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        #execute a step given the action
        #velocity_action = action[:6]
        cartesian_action = action[:3] * CARTESIAN_VEL_SCALE
        angular_action = action[3:] * ANGULAR_VEL_SCALE
        end_effector_velocity = np.concatenate((cartesian_action, angular_action))
 
        self.sim.step(end_effector_velocity)
        #update the observation
        obs = self._get_obs()
        #compute the policy reward
        reward = self._calculate_reward()
        #increment current_step 
        self.current_step += 1
        #check if no more steps are needed
        self.done = self._check_done()
        self.goal_reached = self._check_goal()
        #terminated flag True -> if _check_done() True
        terminated = self.done or self.goal_reached
        #truncated flag True -> if maximum steps exectuted
        truncated = self.current_step >= self.max_steps
        #print(f"tcp angles: {self.sim.get_ee_angles()}")
        #print(f"robot tcp pose: {self.sim.get_end_eff_pose()}")
        #print(f"Distance ee to goal: {np.linalg.norm(self.sim.get_end_eff_position()-self.goal[:3])}")
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        ee_pose = self.sim.get_end_eff_pose()
        cart_error = np.linalg.norm(ee_pose[:3] - self.goal[:3])
        orient_error = np.linalg.norm(ee_pose[3:] - self.goal[3:])
       
        #sfoix reward
        if self.current_step == 0:
           self.distance_cart = cart_error
           self.distance_orient = orient_error
           self.reward = 0
        else:
           self.reward = (self.distance_cart - cart_error)*10 + (self.distance_orient - orient_error)*10
           self.distance_cart = cart_error
           self.distance_orient = orient_error
           
        if self.goal_reached:
            reward += 10 
        #print(f"Reward: {self.reward}")
        return self.reward

    def _check_goal(self):
        ee_pose = self.sim.get_end_eff_pose()

        cart_dist_to_target = np.linalg.norm(ee_pose[:3] - self.goal[:3])
        orient_dist_to_target = np.linalg.norm(ee_pose[3:] - self.goal[3:])

        if cart_dist_to_target < CLOSE_REWARD_DIST and orient_dist_to_target < CLOSE_REWARD_DIST:
            return True
        return False


    def _check_done(self):
        ee_pose = self.sim.get_end_eff_pose()
        cart_dist_to_target = np.linalg.norm(ee_pose[:3] - self.goal[:3])
        if cart_dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        
        tcp_pos = self.sim.get_end_eff_pose()
        tcp_vel = self.sim.get_end_eff_vel()

        obs = np.concatenate((
            self.goal,
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
