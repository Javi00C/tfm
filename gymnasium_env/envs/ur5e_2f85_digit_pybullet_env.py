##############################################
#ENVIRONMENT WITH FIXED CLOSED GRIPPER AND ROPE
##############################################
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.ur5e_gripper_digit_sim import UR5Sim

MAX_DISTANCE = 1.0  # Maximum allowable distance from target before termination
MAX_STEPS_SIM = 20000
#VELOCITY_SCALE = 0.8 #First training was using 0.1

CARTESIAN_VEL_SCALE = 0.1 
ANGULAR_VEL_SCALE = 0.4
# CARTESIAN_VEL_SCALE = 1.0 
# ANGULAR_VEL_SCALE = 1.0

DIST_TCP_LL_THRESH = 0.02 # length of a segment is 0.06 in this case

CLOSE_REWARD_DIST = 0.01

class ur5e_2f85_pybulletEnv_digit(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.5,0.4,0.6]), max_steps=MAX_STEPS_SIM, render_mode=None): #[0.5,0.4,0.6]
        super().__init__()

        
        self.max_steps = max_steps
        self.render_mode = render_mode
        

        # Observation space
        self.num_robot_joints = 6
        #self.num_sensor_readings = 160*120
        self.num_sensor_readings = 2
        self.target_size = 3
        self.last_link_size = 3
        obs_dim = 2*self.num_robot_joints + self.target_size + self.num_sensor_readings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps, goal_position=self.target)
        self.current_step = 0
        self.goal_flag1 = False

        #Goal position  
        self.target = self.sim.get_last_rope_link_position()
        visual_target = np.concatenate((np.array(self.target,dtype=np.float32),
                                                 np.array([0,0,0],dtype=np.float32)),axis=None)
        self.sim.add_visual_goal(visual_target)

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.goal_flag1 = False
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        
        #velocity_action = action[:6]
        #gripper_action = action[6]
        gripper_action = 1.0

        cartesian_action = action[:3] * CARTESIAN_VEL_SCALE
        angular_action = action[3:] * ANGULAR_VEL_SCALE
        end_effector_velocity = np.concatenate((np.array(cartesian_action,dtype=np.float32),
                                                 np.array(angular_action,dtype=np.float32)),axis=None)

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_action)
        
        obs = self._get_obs()

        self.done = self._check_done()
        reward = self._calculate_reward()
        
        self.current_step += 1
        
        terminated = self.done
        truncated = self.current_step >= self.max_steps

        #print(f"Last rope link position: {self.sim.get_last_rope_link_position()}")
        distance_dict = {}  # Create a dictionary if it doesn't exist yet
        # Compute the distance
        dist_ll_goal = np.linalg.norm(self.sim.get_last_rope_link_position() - self.target)
        # Add the distance to the dictionary with an appropriate key
        distance_dict["distance_to_goal"] = dist_ll_goal
        return obs, reward, terminated, truncated, distance_dict


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
        tcp_pos = self.sim.get_end_eff_position()
        position_error = np.linalg.norm(tcp_pos - self.target)
                
        if self.current_step == 0:
            self.distance = position_error
            self.reward = 0
        else:
            self.reward = (self.distance - position_error)*10
            self.distance = position_error
            if self.sensor_touch == 0 and self.sensor_certainty > 0.5: #(is touching)
                self.reward += 0.1
        #print(f"Reward: {self.reward}")
        return self.reward
    

    # def _calculate_reward(self):
    #     ll_pos = self.sim.get_last_rope_link_position()
    #     tcp_pose = self.sim.get_end_eff_pose()
    #     dist_tcp_ll = np.linalg.norm(ll_pos - tcp_pose[:3])

    #     if dist_tcp_ll >= DIST_TCP_LL_THRESH and self.goal_flag1 == False:            
                
    #         if self.current_step == 0:
    #             self.last_dist_tcp_ll = dist_tcp_ll
    #             self.reward = 0
    #         else:
    #             self.reward = (self.last_dist_tcp_ll - dist_tcp_ll)*10
    #             self.last_dist_tcp_ll = dist_tcp_ll
    #     else:
            
    #         dist_ll_goal = np.linalg.norm(ll_pos - self.target)
                
    #         if self.self.goal_flag1 == False:
    #             self.goal_flag1 = True
    #             self.last_dist_ll_goal = dist_ll_goal
    #             self.reward = 0
    #         else:
    #             self.reward = (self.last_dist_ll_goal - dist_ll_goal)*10
    #             self.last_dist_ll_goal = dist_ll_goal


    #     #print(f"Reward: {self.reward}")
    #     return self.reward

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
        
        
        #last_link_rope_pos = self.sim.get_last_rope_link_position()

        self.sensor_touch, self.sensor_certainty = self.sim.get_sensor_reading()

        obs = np.concatenate((
            np.array(self.sensor_touch, dtype=np.float32),
            np.array(self.sensor_certainty, dtype=np.float32),
            np.array(self.target, dtype=np.float32),
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
