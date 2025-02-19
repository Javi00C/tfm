import os

#For server execution without screen
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium
import gymnasium_env  # Import custom environment package

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback

import numpy as np
import torch
import matplotlib.pyplot as plt
import platform

# Optional: Print package versions for debugging
print(f"Python Version: {platform.python_version()}")
print(f"Torch Version: {torch.__version__}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Cuda Version: {torch.version.cuda}")
print(f"Gymnasium Version: {gymnasium.__version__}")
print(f"Numpy Version: {np.__version__}")
# print(f"Stable Baselines3 Version: {stable_baselines3.__version__}")

# Define the number of environments
NUM_ENVS = 25  # Adjust based on system's capacity
#NUM_ENVS = 10  # Adjust based on system's capacity
TIMESTEPS = 2048000
#TIMESTEPS = 1024000
#TIMESTEPS = 81920
DEVICE_USED = 'cpu'
env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0"
#env_str = "gymnasium_env/ur5e_pybulletEnv-v1"
#env_str = "gymnasium_env/ur5e_pybulletEnv-v2"
#env_str = "gymnasium_env/ur5e_pybulletEnv-v3"
#env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v4"
#env_str = "gymnasium_env/ur5e_pybulletEnv-v4"

# Function to create environments (needed for SubprocVecEnv)
def make_env():
   return gymnasium.make(env_str, render_mode='training')
   #return gymnasium.make(env_str, render_mode='human')

if __name__ == '__main__':
    # Create directories to hold models and logs
    print(f"[INFO]: Creating directories to hold models and logs")
    model_dir = "models_pybullet"
    log_dir = "logs_pybullet"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Verify observation and action spaces
    print(f"[INFO]: Verifying observation and action spaces")
    sample_env = gymnasium.make(env_str, render_mode='training')
    obs_space_shape = sample_env.observation_space.shape[0]
    action_space_shape = sample_env.action_space.shape[0]
    print("Observation Space Shape: ", obs_space_shape)
    print("Action Space Shape: ", action_space_shape)
    sample_env.close()

    # Create a list of NUM_ENVS training environments
    print(f"[INFO]: Creating vectorized training environment")
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # Create a list of a single environment for evaluation
    print(f"[INFO]: Creating vectorized evaluation environment with a single environment")
    eval_env = SubprocVecEnv([make_env for _ in range(1)])

    # Create Evaluation Callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_dir,
                                 log_path=log_dir,
                                 eval_freq=5120,  # Adjust evaluation frequency
                                 n_eval_episodes=5)

    # Create a PPO agent
    print(f"[INFO]: PPO model Initialization")
    model = PPO('MlpPolicy', env, verbose=1, device=DEVICE_USED)

    # Train the model
    print(f"[INFO]: Model Training ...")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=eval_callback)

    # Save the model
    print(f"[INFO]: Saving the model ...")
    model_filename = f"ppo_ur5_simple_demo_{obs_space_shape}_{action_space_shape}_{NUM_ENVS}_{TIMESTEPS}"
    model.save(os.path.join(model_dir, model_filename))

    # Evaluate the trained model
    print(f"[INFO]: Evaluating the model ...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close environments
    print(f"[INFO]: Closing environments ...")
    env.close()
    eval_env.close()
