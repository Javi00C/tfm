# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from typing import Optional
# import time

# from pybullet_ur5_gripper.robot_gripper_sim import UR5Sim

# DIST_WEIGTH = 100000
# MAX_REWARD = 1000
# EPISODE_LEN = 6000  # Same as original

# class UR5e2f85BulletEnv(gym.Env):
#     metadata = {
#         "render_modes": ["human"],
#         "render_fps": 100,
#     }

#     def __init__(self, episode_len=EPISODE_LEN, gui=True):
#         super().__init__()

#         self.gui = gui
#         self.episode_len = episode_len
#         #self.num_robot_joints = 6
#         #self.num_sensor_readings = 3  # for example, the EE pos
#         #obs_dim = 2*self.num_robot_joints + self.num_sensor_readings
        
#         # Observation space: (joint_positions, joint_velocities, ee_pos)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

#         # Action space: (x, y, z, gripper fraction) in [-1,1]
#         # We'll interpret these as incremental commands for the EE position and a fraction for the gripper
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

#         self.sim = None
#         self.current_step = 0

#     def _init_simulation(self):
#         if self.sim is not None:
#             self.sim.close()
#         self.sim = RopeRobotSimulation(gui=self.gui)

#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
#         super().reset(seed=seed)
#         self._init_simulation()
#         obs_dict = self.sim.reset_simulation()
#         self.current_step = 0
#         obs = self._dict_to_obs(obs_dict)
#         return obs, {}

#     def step(self, action):
#         self.current_step += 1

#         # Interpret action
#         # action[0:3]: desired ee position offsets (relative)
#         # action[3]: gripper fraction in [-1,1], mapped to [0,1]
#         gripper_fraction = (action[3] + 1.0) / 2.0  # maps [-1,1] to [0,1]

#         # Get current observation to know current EE pos
#         obs_dict = self.sim.get_observation()
#         ee_pos = obs_dict["ee_pos"]  # current ee position (x,y,z)
        
#         # Apply action: move EE by small increments, roll/pitch/yaw static for simplicity
#         # We'll add scaled increments to the EE position
#         delta_pos = action[:3] * 0.05  # scale increments
#         new_ee_pos = np.array(ee_pos) + delta_pos
#         # keep orientation fixed for simplicity
#         ee_orientation = (0, 0, 0)

#         # Move the robot
#         # (x, y, z, roll, pitch, yaw) in 'end' control method
#         # We keep orientation at 0 for simplicity
#         self.sim.apply_action((new_ee_pos[0], new_ee_pos[1], new_ee_pos[2], ee_orientation[0], ee_orientation[1], ee_orientation[2]))
#         self.sim.move_gripper(gripper_fraction)

#         # Step simulation a few times
#         # The original code steps simulation internally at a high frequency already
#         # We'll just step a fixed number of times
#         for _ in range(5):
#             self.sim._step_simulation()
#             time.sleep(1/240.0 if self.gui else 0.0)

#         obs = self._get_observation()


#         reward = self._calculate_reward(obs_dict)
#         done = self._check_done()
#         terminated = done
#         truncated = self.current_step >= self.episode_len

#         return obs, reward, terminated, truncated, {}

#     def _get_observation(self):
#         obs_dict = self.sim.get_observation()
#         # obs_dict contains 'positions', 'velocities', 'ee_pos'
#         positions = np.array(obs_dict["positions"], dtype=np.float32)
#         velocities = np.array(obs_dict["velocities"], dtype=np.float32)
#         ee_pos = np.array(obs_dict["ee_pos"], dtype=np.float32)
#         return np.concatenate([positions, velocities, ee_pos], axis=0)

#     def _calculate_reward(self, obs_dict):
#         # Simple placeholder reward:
#         # Reward based on EE closeness to a target point (e.g. (0.5,0.0,1.0))
#         target = np.array([0.5, 0.0, 1.0])
#         ee_pos = np.array(obs_dict["ee_pos"])
#         dist = np.linalg.norm(ee_pos - target)
#         # Inverse distance reward
#         reward = 1/(dist+1e-3) - 0.1  # small step penalty
#         return reward

#     def _check_done(self):
#         # Placeholder done condition: if EE goes too far from origin
#         # or if simulation steps exceed a limit
#         # You can refine this based on rope states if needed.
#         return False

#     def close(self):
#         if self.sim is not None:
#             self.sim.close()
#             self.sim = None
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import time

from gymnasium_env.envs.pybullet_ur5_gripper.robot_gripper_sim import UR5Sim

DIST_WEIGHT = 100000
MAX_REWARD = 1000
ACTION_SCALER = np.array([7, 7, 7, 1])  # Max velocities for x, y, z (gripper scaling is done in step function)
EPISODE_LEN = 6000

MAX_DISTANCE = 2.0  # Maximum allowable distance from target before termination

class ur5e_2f85_pybulletEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, target=np.array([0.5, 0.0, 0.3]), max_steps=500, render_mode=None):
        super().__init__()

        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: joint positions only (6D)
        self.observation_space = spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(6,), dtype=np.float32)

        # Action: 3D end-effector velocity in world coordinates
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Initialize simulation
        self.sim = UR5Sim(useIK=True, renders=(self.render_mode == "human"), maxSteps=self.max_steps)
        self.current_step = 0

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        # action is a 3D vector (vx, vy, vz)
        # Apply a scaling factor to translate [-1,1] action space to a suitable velocity range:
        velocity_scale = 0.06
        end_effector_velocity = action * velocity_scale

        # Fix gripper open
        self.sim.step(end_effector_velocity, gripper_cmd=-1.0)

        obs = self._get_obs()

        reward = self._calculate_reward()
        
        self.current_step += 1
        self.done = self._check_done()
        terminated = self.done
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self):
        # Compute reward: reward is high when close to the target, penalized with distance
        
        # Apply constant negative reward per step to encourage efficient behavior
        step_penalty = -10

        # Determine the current tile and whether it's on the edge
        if self.done:
            terminated_penalty = -1000
            reward = 0
        else:
            terminated_penalty = 0
            ee_pos, _ = self.sim.get_current_pose()
            dist = np.linalg.norm(ee_pos - self.target)
            reward = MAX_REWARD / (0.001 + dist)  # Reward function

        total_reward = step_penalty + reward + terminated_penalty
        return total_reward

    def _check_done(self):
        """Terminate the episode if the end-effector is too far from the target."""
        ee_pos, _ = self.sim.get_current_pose()
        dist_to_target = np.linalg.norm(ee_pos - self.target)
        if dist_to_target > MAX_DISTANCE:
            return True
        return False

    def _get_obs(self):
        # Joint angles as observation
        joints = self.sim.get_joint_angles()
        return np.array(joints, dtype=np.float32)

    def render(self):
        # If render_mode == "human", PyBullet GUI is already open.
        pass

    def close(self):
        self.sim.close()


# #Example usage:
# env = UR5e2f85BulletEnv(render_mode="human")
# obs, info = env.reset()
# for _ in range(10000):
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         break
# env.close()