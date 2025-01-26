import gymnasium
from gymnasium.wrappers import RecordVideo, ResizeObservation

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

import os
import numpy
import platform
import matplotlib.pyplot as plt
import torch

from importlib.metadata import version

print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {version('gymnasium')}")
print(f"Numpy Version: {version('numpy')}")
print(f"Stable Baselines3 Version: {version('stable_baselines3')}")

env_str = "CarRacing-v2"
log_dir = "Models/sac_models"

# Create Training CarRacing environment
env = make_vec_env(env_str, n_envs=1)
env = ResizeObservation(env, shape=(48, 48))  # Resize to 48x48 to reduce memory requirements
n_stack = 2  # Reduce stacking to lower memory usage
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Create Evaluation CarRacing environment
env_val = make_vec_env(env_str, n_envs=1)
env_val = ResizeObservation(env_val, shape=(48, 48))
env_val = VecFrameStack(env_val, n_stack=n_stack)
env_val = VecTransposeImage(env_val)

# Create Evaluation Callback
eval_callback = EvalCallback(env_val,
                             best_model_save_path=log_dir,
                             log_path=log_dir,
                             eval_freq=50000,
                             render=False,
                             n_eval_episodes=20)

# Initialize SAC with a reduced buffer size
model = SAC('CnnPolicy', env, verbose=1, buffer_size=50000)  # Reduced buffer size

# Train the model
model.learn(total_timesteps=200000, progress_bar=True, callback=eval_callback)

# Save the model
model.save(os.path.join(log_dir, "sac_car_racing"))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()

# Create Evaluation CarRacing environment
env = make_vec_env(env_str, n_envs=1)
env = ResizeObservation(env, shape=(48, 48))
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Load the best model
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = SAC.load(best_model_path, env=env)

mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record video of the best model playing CarRacing
env = VecVideoRecorder(env, "./videos/",
                       video_length=5000,
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="best_model_car_racing_sac")

obs = env.reset()
for _ in range(5000):  # Adjusted to match video length
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

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
plt.figure()
plt.plot(timesteps, mean_results)
plt.fill_between(timesteps,
                 mean_results - std_results,
                 mean_results + std_results,
                 alpha=0.3)

plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title(f"SAC Performance on {env_str}")
plt.show()