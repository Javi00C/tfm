import gymnasium
from gymnasium.wrappers import RecordVideo

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

import os
import numpy
import platform
import scipy
import matplotlib
import matplotlib.pyplot
import torch
import time
import cv2

from importlib.metadata import version

print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {version('gymnasium')}")
print(f"Numpy Version: {version('numpy')}")
print(f"Stable Baselines3 Version: {version('stable_baselines3')}")
print(f"Scipy Version: {version('scipy')}")

backend = torch.backends.quantized.engine
print(f"Currently using backend: {backend}")

# Custom observation wrapper to create a focused observation window
def create_focused_observation_wrapper(env, window_size):
    class FocusedObservationWrapper(gymnasium.ObservationWrapper):
        def __init__(self, env, window_size):
            super(FocusedObservationWrapper, self).__init__(env)
            self.window_size = window_size  # (height, width)
            # Define a new observation space that matches the cropped window size
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(window_size[0], window_size[1], 3), dtype=numpy.uint8
            )

        def observation(self, observation):
            # Placeholder for car's current position in observation, replace with actual method to get car position
            car_x, car_y = self._get_car_position(observation)
            
            # Calculate crop box to focus on the area around the car
            half_h, half_w = self.window_size[0] // 2, self.window_size[1] // 2
            x1, x2 = max(0, car_x - half_w), min(observation.shape[1], car_x + half_w)
            y1, y2 = max(0, car_y - half_h), min(observation.shape[0], car_y + half_h)

            # Crop the observation to focus around the car
            cropped_obs = observation[y1:y2, x1:x2]

            # Resize the cropped observation to maintain consistent input size
            resized_obs = cv2.resize(cropped_obs, self.window_size, interpolation=cv2.INTER_AREA)
            return resized_obs

        def _get_car_position(self, observation):
            # Placeholder method to determine the car's position, replace with proper logic
            car_x, car_y = observation.shape[1] // 2, observation.shape[0] // 2
            return car_x, car_y

    return FocusedObservationWrapper(env, window_size)

# Custom reward wrapper to reward the agent for being close to the edge of the track
def create_edge_following_reward_wrapper(env):
    class EdgeFollowingRewardWrapper(gymnasium.RewardWrapper):
        def __init__(self, env):
            super(EdgeFollowingRewardWrapper, self).__init__(env)

        def reward(self, reward):
            # Placeholder: Calculate distance to edge of track
            # The closer the car is to the edge, the higher the reward (up to a limit)
            distance_to_edge = self._calculate_distance_to_edge()
            optimal_distance = 5.0  # Desired distance from the edge (arbitrary value)
            distance_penalty = abs(distance_to_edge - optimal_distance)

            # Reward adjustment
            reward += max(0, 10 - distance_penalty)  # Reward decreases as distance increases

            return reward

        def _calculate_distance_to_edge(self):
            # Placeholder method to calculate distance to track edge, replace with actual calculation
            return 5.0

    return EdgeFollowingRewardWrapper(env)

env_str = "CarRacing-v2"
log_dir = "Models"

# Create Training CarRacing environment
env = make_vec_env(env_str, n_envs=1)
# Apply custom observation wrapper
env = create_focused_observation_wrapper(env, window_size=(64, 64))
# Apply custom reward wrapper
env = create_edge_following_reward_wrapper(env)
# Parameterize n_stack to allow flexible configuration
n_stack = 4  # Set default value for n_stack, can be adjusted for experimentation
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Create Evaluation CarRacing environment
env_val = make_vec_env(env_str, n_envs=1)
# Apply custom observation wrapper
env_val = create_focused_observation_wrapper(env_val, window_size=(64, 64))
# Apply custom reward wrapper
env_val = create_edge_following_reward_wrapper(env_val)
env_val = VecFrameStack(env_val, n_stack=n_stack)
env_val = VecTransposeImage(env_val)

# Create Evaluation Callback
# eval_freq - increased to reduce potential learning instability due to frequent evaluations
eval_callback = EvalCallback(env_val,
                             best_model_save_path=log_dir,
                             log_path=log_dir,
                             eval_freq=50000,
                             render=False,
                             n_eval_episodes=20)

# Initialize PPO
# buffer_size is not required for PPO as it is an on-policy method
model = PPO('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=200000, progress_bar=True, callback=eval_callback)

# Save the model
model.save(os.path.join(log_dir, "ppo_car_racing"))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()

# Create Evaluation CarRacing environment
env = make_vec_env(env_str, n_envs=1)
# Apply custom observation wrapper
env = create_focused_observation_wrapper(env, window_size=(64, 64))
# Apply custom reward wrapper
env = create_edge_following_reward_wrapper(env)
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Load the best model
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = PPO.load(best_model_path, env=env)

mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record video of the best model playing CarRacing
env = VecVideoRecorder(env, "./videos/",
                       video_length=5000,  # Reduced video length to prevent large file sizes
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="best_model_car_racing_ppo")

obs = env.reset()
for _ in range(5000):  # Adjusted to match video length
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()  # Correctly reset without passing any parameters

env.close()

# Load the evaluations.npz file
data = numpy.load(os.path.join(log_dir, "evaluations.npz"))

# Extract the relevant data
timesteps = data['timesteps']
results = data['results']

# Calculate the mean and standard deviation of the results
mean_results = numpy.mean(results, axis=1)
std_results = numpy.std(results, axis=1)

# Plot the results
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(timesteps, mean_results)
matplotlib.pyplot.fill_between(timesteps,
                               mean_results - std_results,
                               mean_results + std_results,
                               alpha=0.3)

matplotlib.pyplot.xlabel("Timesteps")
matplotlib.pyplot.ylabel("Mean Reward")
matplotlib.pyplot.title(f"PPO Performance on {env_str}")
matplotlib.pyplot.show()
