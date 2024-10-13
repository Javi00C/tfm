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

from importlib.metadata import version

print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {version('torch')}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {version('gymnasium')}")
print(f"Numpy Version: {version('numpy')}")
print(f"Stable Baselines3 Version: {version('stable_baselines3')}")
print(f"Scipy Version: {version('scipy')}")

TIMESTEPS = 300000
#env_str = "CarRacing-v2"
env_str = "SimpleRobotEnv-v0"
log_dir = "/home/javi/tfm/models"

backend = torch.backends.quantized.engine
print(f"Currently using backend: {backend}")

env = gymnasium.make(env_str)
print("Observation Space Size: ", env.observation_space.shape)
print("Action Space Size: ", env.action_space.shape)
env.close()



# Create Training CarRacing environment
env = make_vec_env(env_str, n_envs=1)
# Parameterize n_stack to allow flexible configuration
n_stack = 4  # Set default value for n_stack, can be adjusted for experimentation
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Create Evaluation CarRacing environment
env_val = make_vec_env(env_str, n_envs=1)
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
model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=eval_callback)

# Save the model
model.save(os.path.join(log_dir, "ppo_robot"))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()

# Create Evaluation CarRacing environment
env = make_vec_env(env_str, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Load the best model
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = PPO.load(best_model_path, env=env)

mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record video of the best model playing CarRacing
env = VecVideoRecorder(env, "/home/tfm/videos/",
                       video_length=5000,  # Reduced video length to prevent large file sizes
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="best_model_car_racing_ppo")

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