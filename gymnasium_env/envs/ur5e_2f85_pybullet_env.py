import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.ur5e_gripper_sim import UR5Sim

MAX_REWARD = 1000
MAX_DISTANCE = 10.0  # Maximum allowable distance from target before termination
MAX_DIST_REW = 2.0
MAX_STEPS_SIM = 4000

class ur5e_2f85_pybulletEnv(gym.Env):
    metadata = {"render_modes": ["human","training"], "render_fps": 100}

    def __init__(self, target=np.array([0.5, 0.5, 0.5]), max_steps=MAX_STEPS_SIM, render_mode=None):
        super().__init__()

        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation space
        self.num_robot_joints = 6
        self.num_sensor_readings = 160*120
        self.rope_link_pose = 3
        obs_dim = 2*self.num_robot_joints + self.num_sensor_readings + self.rope_link_pose
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps)
        self.current_step = 0

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # action is a 3D vector (vx, vy, vz)
        # Apply a scaling factor to translate [-1,1] action space to a suitable velocity range:
        velocity_action = action[:6]
        gripper_action = action[6]

        velocity_scale = 0.3 #Maximum velocity that is stable in the simulation
        end_effector_velocity = velocity_action * velocity_scale

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_action)

        obs = self._get_obs()

        reward = self._calculate_reward()
        
        self.current_step += 1
        self.done = self._check_done()
        terminated = self.done
        truncated = self.current_step >= self.max_steps

        #print(f"Last rope link position: {self.sim.get_last_rope_link_position()}")
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        # Compute reward: reward is high when close to the target, penalized with distance
        
        # Apply constant negative reward per step to encourage efficient behavior
        step_penalty = -10

        # Determine the current tile and whether it's on the edge

        max_rew = MAX_REWARD
        max_dist = MAX_DIST_REW
        a = -max_rew/max_dist
        b = max_rew

        if self.done:
            terminated_penalty = -MAX_REWARD
            reward = 0
        else:
            terminated_penalty = 0
            ee_pos, _ = self.sim.get_end_eff_pose()
            dist = np.linalg.norm(ee_pos - self.target)
            #reward = MAX_REWARD / (0.001 + dist)  # Reward function
            reward = a*dist+b

        total_reward = step_penalty + reward + terminated_penalty
        return total_reward

    def _check_done(self):
        """Terminate the episode if the end-effector is too far from the target."""
        ee_pos, _ = self.sim.get_end_eff_pose()
        dist_to_target = np.linalg.norm(ee_pos - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        # Joint angles as observation
        #joint_positions = self.sim.get_joint_angles()
        #joint_velocities = self.sim.get_joint_velocities()
        
        tcp_pos = self.sim.get_end_eff_pose()
        tcp_vel = self.sim.get_end_eff_vel()

        sensor_reading = self.sim.get_sensor_reading()
        sensor_reading = sensor_reading.ravel()
        
        last_link_rope_pos = self.sim.get_last_rope_link_position()

        obs = np.concatenate((
            #np.array(joint_positions, dtype=np.float32),
            #np.array(joint_velocities, dtype=np.float32),
            np.array(tcp_pos, dtype=np.float32),
            np.array(tcp_vel, dtype=np.float32),
            np.array(last_link_rope_pos, dtype=np.float32),
            np.array(sensor_reading, dtype=np.float32)
        ), axis=0)

        obs = obs.flatten()
        obs = np.squeeze(obs)
        return obs

    def render(self):
        pass

    def close(self):
        self.sim.close()
