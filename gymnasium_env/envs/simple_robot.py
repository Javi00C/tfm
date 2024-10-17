import numpy as np
import gymnasium as gym
import gymnasium_env
from gymnasium import spaces
import cv2
from typing import Optional
import random

#Parameters
MAX_REWARD = 1000
WINDOW_SIZE = 512
MAX_DIST = 50
OBS_WINDOW = 128
NUM_EDGE_TILES = WINDOW_SIZE*2

class SimpleRobotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None):
        super(SimpleRobotEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)  # 2D velocity control: x and y velocities
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_WINDOW, OBS_WINDOW, 3), dtype=np.uint8)  # 64x64 RGB image around the robot
        
        # Initial state setup
        self.full_state = None  # Full environment state (larger image)
        self.state = None  # Observation window around the robot
        self.done = False
        self.size = 512
        self.reward = 0
        self.avgd = 0
        # Define robot's initial position
        x0 = random.randint(240, 260)
        y0 = random.randint(0, 511)
        self.robot_pos = [x0, y0]  # Start in the center of the full state image
        
        # Define maximum steps per episode
        self.max_steps = 1000  # Set a maximum number of steps for an episode
        self.current_step = 0
        
        self.render_mode = render_mode
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset visited tiles for new episode
        self.visited_tiles = set()
        #self.edge_tiles = set()
        #for i in range(0,WINDOW_SIZE-1):
        #    self.edge_tiles.add((WINDOW_SIZE//2,i))

        # Set the seed if provided (to maintain reproducibility)
        if seed is not None:
            np.random.seed(seed)

        # Randomize the robot's initial position on reset
        x0 = random.randint(240, 260)
        y0 = random.randint(0, 511)
        self.robot_pos = [x0, y0]
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the state of the environment to an initial state
        self.full_state = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # Start with a blank black-white image
        
        # Draw a straight white line in the middle of the environment
        cv2.line(self.full_state, (0, 0), (0, self.size), (255, 255, 255), self.size)  # Draw a vertical white line on the left side so that there is only one edge
        
        # Set the initial observation window around the robot
        self.state = self._get_observation()
        self.done = False
        self.current_step = 0  # Reset step count
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        # Update the robot's position based on the action
        velocity_x, velocity_y = action
        
        # Update position based on velocity (new pos bounded [0,511])
        self.robot_pos[0] = np.clip(self.robot_pos[0] + velocity_x, 0, self.size - 1)
        self.robot_pos[1] = np.clip(self.robot_pos[1] + velocity_y, 0, self.size - 1)
        
        # Update environment state and check if episode is done
        self.current_step += 1
        self.done = self._check_done()  # Define a termination condition
       
        # Update the observation window
        self.state = self._get_observation()
        reward = self._calculate_reward()
        
        # Return observation, reward, done, and optional info dictionary
        return self.state, reward, self.done, False, {}

    def _get_observation(self):
        # Get a 128x128 window around the robot's current position
        x, y = self.robot_pos
        half_window = OBS_WINDOW//2
        x1 = max(0, int(x - half_window))
        x2 = min(self.size, int(x + half_window))
        y1 = max(0, int(y - half_window))
        y2 = min(self.size, int(y + half_window))
        
        # Extract the window and resize if necessary to maintain consistent observation size
        cropped_obs = self.full_state[y1:y2, x1:x2]
        resized_obs = cv2.resize(cropped_obs, (OBS_WINDOW, OBS_WINDOW), interpolation=cv2.INTER_AREA)
        return resized_obs

    def _calculate_distance_to_edge(self):
        # Calculate the distance from the robot to the white line (edge)
        car_x, _ = self.robot_pos
        return abs(car_x-WINDOW_SIZE//2)  # The distance is simply the x-coordinate difference from the edge at x=0

    #Average distance reward -> poor performance
    #def _calculate_average_dist(self):
    #    self.avgd = ((self.index - 1) * self.avgd + self._calculate_distance_to_edge()) / self.index
    
    def _calculate_reward(self):
        # Apply constant negative reward per step to encourage efficient behavior
        step_penalty = -0.1

        # Reward for moving over unvisited edge tiles
        if not hasattr(self, 'visited_tiles'):
            self.visited_tiles = set()

        # Determine the current tile and whether it's on the edge (Positions could be divided to created larger tiles -> less memory)
        if self.done:
            terminated_penalty = -100
            reward_for_tile = 0
        else:
            terminated_penalty = 0
            current_tile = (int(self.robot_pos[0]), int(self.robot_pos[1]))
            if self._check_on_edge():  # Check if on the edge (white line)
                if current_tile not in self.visited_tiles:
                    self.visited_tiles.add(current_tile)
                    reward_for_tile = MAX_REWARD / max(1, len(self.visited_tiles))
                else:
                    reward_for_tile = 0
            else:
                reward_for_tile = 0
            
        # Calculate total reward per step
        total_reward = step_penalty + reward_for_tile + terminated_penalty
        return total_reward

    def _check_on_edge(self):
        x = int(self.robot_pos[0])
        y = int(self.robot_pos[1])
        if x + 1 < self.size - 1 and x - 1 > 0:
            return (self.full_state[y, x, 0] == 255 and self.full_state[y, x+1, 0] == 0) or (self.full_state[y, x, 0] == 0 and self.full_state[y, x-1, 0] == 255)
        else: 
            return False

    def render(self, mode='human'):
        # Render the environment to the screen
        if mode == 'human':
            # Draw the robot as a red dot in the full state image
            temp_image = self.full_state.copy()
            cv2.circle(temp_image, (int(self.robot_pos[0]), int(self.robot_pos[1])), 5, (0, 0, 255), -1)
            cv2.imshow('SimpleRobotEnv', temp_image)
            cv2.waitKey(100)  # Display the frame for a short period to create animation

    def close(self):
        # Perform any necessary cleanup
        cv2.destroyAllWindows()

    def _check_done(self):
        # End the episode if the maximum number of steps is reached or the robot is too far from the edge
        distance_to_edge = self._calculate_distance_to_edge()
        return self.current_step >= self.max_steps or distance_to_edge > MAX_DIST

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
    env.close()))))
'''
