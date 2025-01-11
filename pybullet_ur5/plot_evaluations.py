import os

import gymnasium
import gymnasium_env  # Import custom environment package

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import torch
import matplotlib.pyplot as plt
import platform    

log_dir = "logs_pybullet"
env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0"
# Plot evaluation results
data = np.load(os.path.join(log_dir, "evaluations_exp_1c.npz"))
timesteps = data['timesteps']
results = data['results']

mean_results = np.mean(results, axis=1)
std_results = np.std(results, axis=1)

plt.figure()
plt.plot(timesteps, mean_results)
plt.fill_between(timesteps,
                    mean_results - std_results,
                    mean_results + std_results,
                    alpha=0.3)
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title(f"PPO Performance on {env_str}")
plt.show()