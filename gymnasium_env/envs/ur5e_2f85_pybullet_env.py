##############################################
#ENVIRONMENT WITH FIXED CLOSED GRIPPER AND ROPE
##############################################
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.ur5e_gripper_sim import UR5Sim

MAX_REWARD = 1000
MAX_DISTANCE = 1.0  # Maximum allowable distance from target before termination
MAX_DIST_REW = 2.0
MAX_STEPS_SIM = 10000
CLOSE_REWARD_DIST = 0.1
VELOCITY_SCALE = 0.1 #Originally at 0.3

class ur5e_2f85_pybulletEnv(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.5, 0.4, 0.6]), max_steps=MAX_STEPS_SIM, render_mode=None):
        super().__init__()

        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.target = np.array(target, dtype=np.float32)

        # Observation space
        self.num_robot_joints = 6
        self.num_sensor_readings = 160*120
        self.target_size = 3
        self.last_link_size = 3
        obs_dim = 2*self.num_robot_joints + self.target_size + self.last_link_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

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
        
        velocity_action = action[:6]
        #gripper_action = action[6]
        gripper_action = 1.0

        end_effector_velocity = velocity_action * VELOCITY_SCALE

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_action)
        
        obs = self._get_obs()

        self.done = self._check_done()
        reward = self._calculate_reward()
        
        self.current_step += 1
        
        terminated = self.done
        truncated = self.current_step >= self.max_steps

        #print(f"Last rope link position: {self.sim.get_last_rope_link_position()}")
        return obs, reward, terminated, truncated, {}


    # def _calculate_reward(self):

    #     if self.done:
    #         reward = -5
    #     else:
    #         link_rope_pos = self.sim.get_last_rope_link_position()
    #         dist = np.linalg.norm(link_rope_pos - self.target)
            
    #         reward = -dist # Weighted penalty for position and orientation errors
            
    #         # Bonus for being close to the target
    #         if dist < CLOSE_REWARD_DIST:
    #             #Minimum reward = 2.0 (sum of both contributions)
    #             reward += np.clip(0.1/dist,1,5)
    #             reward += 1.0
    #             self.time_near_target += 1
    #             reward += 0.1 * self.time_near_target
    #         else:
    #             self.time_near_target = 0
    #         # Small penalty for every time step
    #         reward -= 0.01  # Step penalty

    #     return reward

    def _calculate_reward(self):
        ll_pos = self.sim.get_last_rope_link_position()
        position_error = np.linalg.norm(ll_pos - self.target)
                
        if self.current_step == 0:
            self.distance = position_error
            self.reward = 0
        else:
            self.reward = (self.distance - position_error)*10
            self.distance = position_error

        #print(f"Reward: {self.reward}")
        return self.reward

    def _check_done(self):
        """Terminate the episode if the end-effector is too far from the target."""
        link_rope_pos = self.sim.get_last_rope_link_position()
        dist_to_target = np.linalg.norm(link_rope_pos - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        
        tcp_pos = self.sim.get_end_eff_pose()
        tcp_vel = self.sim.get_end_eff_vel()

        #sensor_reading = self.sim.get_sensor_reading()
        #sensor_reading = sensor_reading.ravel()
        
        
        last_link_rope_pos = self.sim.get_last_rope_link_position()

        obs = np.concatenate((
            self.target,
            np.array(last_link_rope_pos, dtype=np.float32),
            np.array(tcp_pos, dtype=np.float32),
            np.array(tcp_vel, dtype=np.float32)
            #np.array(sensor_reading, dtype=np.float32)
        ), axis=None)

        obs = obs.flatten()
        obs = np.squeeze(obs)
        return obs

    def render(self):
        pass

    def close(self):
        self.sim.close()
