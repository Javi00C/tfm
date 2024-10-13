import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Optional
import random

MAX_REWARD = 100
WINDOW_SIZE = 512
MAX_DIST = WINDOW_SIZE / 2

class SimpleRobotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None):
        super(SimpleRobotEnv, self).__init__()
        
        # Define action and observation space
        # Action space now represents velocities in x and y directions (continuous values)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)  # Velocity in x and y
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)  # 64x64 RGB image around the robot
        
        # Initial state setup
        self.full_state = None  # Full environment state (larger image)
        self.state = None  # Observation window around the robot
        self.done = False
        self.size = 512
        self.reward = 0
        self.index = 1
        self.avgd = 0
        self.time_near_edge = 0  # Track time spent near the edge
        # Define robot's initial position with a wider range for better generalization
        x0 = random.randint(0, 511)
        y0 = random.randint(0, 511)
        self.robot_pos = [x0, y0]  # Randomized initial position within a wider range
        
        # Define maximum steps per episode
        self.max_steps = 1000  # Set a maximum number of steps for an episode
        self.current_step = 0
        
        self.render_mode = render_mode
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Set the seed if provided (to maintain reproducibility)
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the state of the environment to an initial state
        self.full_state = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # Start with a blank RGB image
        
        # Draw a straight white line in the middle of the environment
        cv2.line(self.full_state, (0, 0), (0, self.size), (255, 255, 255), self.size)  # Draw a vertical white line on the left side so that there is only one edge
        
        # Set the initial observation window around the robot
        self.state = self._get_observation()
        self.done = False
        self.current_step = 0  # Reset step count
        self.time_near_edge = 0  # Reset time spent near the edge
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        # Update the robot's position based on the action
        velocity_x, velocity_y = action
        self.robot_pos[0] = int(np.clip(self.robot_pos[0] + velocity_x, 0, 511))
        self.robot_pos[1] = int(np.clip(self.robot_pos[1] + velocity_y, 0, 511))
        
        # Update the observation window
        self.state = self._get_observation()
        reward = self._calculate_reward()
        self.index += 1
        
        # Update environment state and check if episode is done
        self.current_step += 1
        self.done = self._check_done()  # Define a termination condition
        
        # Return observation, reward, done, and optional info dictionary
        return self.state, reward, self.done, False, {}

    def _get_observation(self):
        # Get a 64x64 window around the robot's current position
        x, y = self.robot_pos
        half_window = 32
        x1 = max(0, x - half_window)
        x2 = min(512, x + half_window)
        y1 = max(0, y - half_window)
        y2 = min(512, y + half_window)
        
        # Extract the window without resizing to maintain consistent observation size
        cropped_obs = np.zeros((64, 64, 3), dtype=np.uint8)
        cropped_obs[:y2 - y1, :x2 - x1] = self.full_state[y1:y2, x1:x2]
        return cropped_obs

    def _calculate_distance_to_edge(self):
        # Calculate the distance from the robot to the white line (edge)
        car_x, _ = self.robot_pos
        return abs(car_x - 0)  # The distance is simply the x-coordinate difference from the edge at x=0
    
    def _calculate_reward(self):
        # Calculate the lateral distance to the edge
        distance_to_edge = self._calculate_distance_to_edge()
        
        # Reward for getting close to the edge (higher reward when close)
        edge_proximity_reward = MAX_REWARD * (1 - distance_to_edge / MAX_DIST)
        
        # Reward for time spent near the edge
        if distance_to_edge <= 50:  # Consider being "near" the edge if within 50 pixels
            self.time_near_edge += 1
            time_near_edge_reward = 10 * self.time_near_edge  # Reward increases with time spent near the edge
        else:
            self.time_near_edge = 0  # Reset if the robot moves away from the edge
            time_near_edge_reward = 0
        
        # Penalize moving away from the edge
        movement_penalty = -20 if distance_to_edge > 50 else 0
        
        # Total reward
        self.reward = edge_proximity_reward + time_near_edge_reward + movement_penalty
        return self.reward

    def _calculate_distance_to_point(self):
        # Calculate the distance from the robot to the specific point (self.size, 0)
        car_x, car_y = self.robot_pos
        target_x, target_y = self.size // 2, 0
        return np.sqrt((car_x - target_x) ** 2 + (car_y - target_y) ** 2)

    def render(self, mode='human'):
        # Render the environment to the screen
        if mode == 'human':
            # Draw the robot as a red dot in the full state image
            temp_image = self.full_state.copy()
            cv2.circle(temp_image, (self.robot_pos[0], self.robot_pos[1]), 5, (0, 0, 255), -1)
            cv2.imshow('SimpleRobotEnv', temp_image)
            cv2.waitKey(1)  # Display the frame for a short period to create animation

    def close(self):
        # Perform any necessary cleanup
        cv2.destroyAllWindows()

    def _check_done(self):
        # End the episode if the maximum number of steps is reached or the robot is too far from the edge
        distance_to_edge = self._calculate_distance_to_edge()
        return self.current_step >= self.max_steps or distance_to_edge > 255

'''

# Example usage:
if __name__ == '__main__':
    env = SimpleRobotEnv()
    obs, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()  # Take random action
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break
    env.close()
'''