import gymnasium
import gymnasium_env
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

#Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

TIMESTEPS = 200000
DEVICE_USED = 'cpu'
#env_str = "CarRacing-v2"
env_str = "gymnasium_env/SimpleRobotEnv-v0" #Straight line edge
#env_str = "gymnasium_env/SimpleRobotEnv-v1" #Sine wave edge


backend = torch.backends.quantized.engine
print(f"Currently using backend: {backend}")

env = gymnasium.make(env_str, render_mode="human")
print("Observation Space Size: ", env.observation_space.shape)
print("Action Space Size: ", env.action_space.shape)
env.close()



# Create Training environment
env = make_vec_env(env_str, n_envs=1, env_kwargs={"render_mode": "human"})

n_stack = 4
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)


# Create Evaluation environment
env_val = make_vec_env(env_str, n_envs=1, env_kwargs={"render_mode": "human"})
env_val = VecFrameStack(env_val, n_stack=n_stack)
env_val = VecTransposeImage(env_val)

# Create Evaluation Callback
eval_callback = EvalCallback(env_val,
                             best_model_save_path=model_dir,
                             log_path=log_dir,
                             eval_freq=50000,
                             render=False,
                             n_eval_episodes=20)

# Initialize PPO
model = PPO('CnnPolicy', env, verbose=1, ent_coef=0.01, device=DEVICE_USED)


# Train the model
model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=eval_callback)

# Save the model
model.save(os.path.join(model_dir, "ppo_robot"))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
env_val.close()

# Create Evaluation CarRacing environment
env = make_vec_env(env_str, n_envs=1, seed=0, env_kwargs={"render_mode": "rgb_array"})
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Load the best model
best_model_path = os.path.join(model_dir, "best_model")
best_model = PPO.load(best_model_path, env=env)

mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")