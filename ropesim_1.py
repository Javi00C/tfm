import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import random
import pygame

# Parameters
MAX_REWARD = 1000
WINDOW_SIZE = 512
OBS_WINDOW = 128
ROPE_SEGMENTS = 60  # Number of points in the rope
ROPE_LENGTH = WINDOW_SIZE  # Length of the rope to span the entire window height
GRAVITY = 0.1  # Gravity force applied to rope points
ROPE_SPRING_STIFFNESS = 0.1  # Stiffness of the rope's spring connections
MAX_SEGMENT_LENGTH = ROPE_LENGTH // ROPE_SEGMENTS * 1.5  # Maximum allowable distance between rope points

class RopeSimulation:
    def __init__(self, segments, length):
        self.segments = segments
        self.length = length
        self.points = self._initialize_rope()

    def _initialize_rope(self):
        points = []
        spacing = self.length // self.segments
        for i in range(self.segments):
            y_position = i * spacing  # Rope goes from one side of the window to the other 
            points.append([WINDOW_SIZE // 2, y_position])  # Distribute horizontally across the window
        return points

    def apply_gravity(self):
        for i in range(1, len(self.points) - 1):  # Skip applying gravity to the first and last point
            self.points[i][0] += GRAVITY # Gravity on the vertical axis 

    def apply_spring_forces(self):
        for i in range(1, len(self.points)):
            prev_point = np.array(self.points[i - 1])
            curr_point = np.array(self.points[i])
            dist_vector = curr_point - prev_point
            dist = np.linalg.norm(dist_vector)
            rest_length = ROPE_LENGTH // self.segments
            force = ROPE_SPRING_STIFFNESS * (dist - rest_length)
            normalized_vector = dist_vector / (dist + 1e-6)
            if i < len(self.points) - 1:  # Skip applying forces to the last point
                self.points[i - 1] += force * normalized_vector
                self.points[i] -= force * normalized_vector

            # Ensure the rope does not stretch beyond the maximum allowed segment length
            if dist > MAX_SEGMENT_LENGTH:
                correction_vector = normalized_vector * (dist - MAX_SEGMENT_LENGTH)
                if i < len(self.points) - 1:
                    self.points[i] -= correction_vector / 2
                    self.points[i - 1] += correction_vector / 2
                else:
                    self.points[i - 1] += correction_vector

        # Keep the first and last points of the rope fixed
        self.points[0] = [WINDOW_SIZE//2, 0]  # Fixed at the top left corner of the window
        self.points[-1] = [WINDOW_SIZE//2, self.length-1]  # Fixed at the bottom right corner of the window

    def get_rope_image(self):
        # Start with a black background
        img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

        # Build polygon for the area over the rope
        over_rope_pts = []

        # Add the top-left and top-right corners
        over_rope_pts.append([WINDOW_SIZE - 1, 0])  # Top-right corner
        over_rope_pts.append([WINDOW_SIZE - 1, WINDOW_SIZE - 1])  # Top-left corner
        

        # Append the rope points from bottom to top
        for pt in reversed(self.points):
            over_rope_pts.append([int(pt[0]), int(pt[1])])

        # Convert points to a NumPy array
        over_rope_pts = np.array([over_rope_pts], dtype=np.int32)

        # Fill the polygon with white color
        cv2.fillPoly(img, over_rope_pts, (255, 255, 255))

        return img


"""
    def get_rope_image(self):
        img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        for i in range(1, len(self.points)):
            cv2.line(img, tuple(map(int, self.points[i - 1])), tuple(map(int, self.points[i])), (255, 255, 255), 2)
        return img

"""

class SimpleRobotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(SimpleRobotEnv, self).__init__()
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_WINDOW, OBS_WINDOW, 3), dtype=np.uint8)
        self.render_mode = render_mode

        self.rope_simulation = RopeSimulation(ROPE_SEGMENTS, ROPE_LENGTH)
        self.robot_pos = [WINDOW_SIZE // 2, WINDOW_SIZE // 2]  # Center start
        self.done = False
        self.current_step = 0

    def reset(self):
        self.rope_simulation = RopeSimulation(ROPE_SEGMENTS, ROPE_LENGTH)
        self.robot_pos = [WINDOW_SIZE // 2, WINDOW_SIZE // 2]
        self.done = False
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        velocity_x, velocity_y = action
        self.robot_pos[0] = np.clip(self.robot_pos[0] + velocity_x, 0, WINDOW_SIZE - 1)
        self.robot_pos[1] = np.clip(self.robot_pos[1] + velocity_y, 0, WINDOW_SIZE - 1)

        self.rope_simulation.apply_gravity()
        self.rope_simulation.apply_spring_forces()
        self._robot_interact_with_rope()

        self.current_step += 1
        self.done = self.current_step >= 1000
        return self._get_observation(), self._calculate_reward(), self.done, False, {}

    def _robot_interact_with_rope(self):
        for point in self.rope_simulation.points:
            distance = np.linalg.norm(np.array(point) - np.array(self.robot_pos))
            if distance < 10:
                direction = np.array(self.robot_pos) - np.array(point)
                point += 1.5 * direction

    def _get_observation(self):
        rope_img = self.rope_simulation.get_rope_image()
        cv2.circle(rope_img, tuple(map(int, self.robot_pos)), 5, (0, 0, 255), -1)
        x, y = self.robot_pos
        half_window = OBS_WINDOW // 2
        x1 = max(0, int(x - half_window))
        x2 = min(WINDOW_SIZE, int(x + half_window))
        y1 = max(0, int(y - half_window))
        y2 = min(WINDOW_SIZE, int(y + half_window))
        cropped_obs = rope_img[y1:y2, x1:x2]
        resized_obs = cv2.resize(cropped_obs, (OBS_WINDOW, OBS_WINDOW), interpolation=cv2.INTER_AREA)
        return resized_obs

    def _calculate_reward(self):
        reward = -0.1
        return reward

    def render(self, mode='human'):
        if mode == 'human':
            img = self.rope_simulation.get_rope_image()
            cv2.circle(img, tuple(map(int, self.robot_pos)), 5, (0, 0, 255), -1)
            cv2.imshow('Rope Simulation', img)
            cv2.waitKey(100)

    def close(self):
        cv2.destroyAllWindows()

# Pygame Integration for Manual Control
def run_with_manual_control():
    env = SimpleRobotEnv()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    
    obs, _ = env.reset()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Manual Control Logic
        keys = pygame.key.get_pressed()
        action = [0, 0]
        if keys[pygame.K_LEFT]:
            action[1] = -5
        if keys[pygame.K_RIGHT]:
            action[1] = 5
        if keys[pygame.K_UP]:
            action[0] = -5
        if keys[pygame.K_DOWN]:
            action[0] = 5

        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

        # Update the Pygame screen
        img = env.rope_simulation.get_rope_image()
        cv2.circle(img, tuple(map(int, env.robot_pos)), 5, (0, 0, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(screen, img)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    env.close()

# Example usage
if __name__ == '__main__':
    run_with_manual_control()
