import os

#For server execution without screen
os.environ["PYOPENGL_PLATFORM"] = "egl"

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
TIMESTEPS = 1000000
DEVICE_USED = 'cuda'
env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0"
# Function to create environments (needed for SubprocVecEnv)
def make_env():
    return gymnasium.make(env_str, render_mode='training')

if __name__ == '__main__':
    # Create directories to hold models and logs
    model_dir = "models_pybullet"
    log_dir = "logs_pybullet"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    

    # Verify observation and action spaces
    sample_env = gymnasium.make(env_str, render_mode='training')
    print("Observation Space Shape: ", sample_env.observation_space.shape)
    print("Action Space Shape: ", sample_env.action_space.shape)
    sample_env.close()

    # Create vectorized training environment
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # Create vectorized evaluation environment with a single environment
    eval_env = SubprocVecEnv([make_env for _ in range(1)])

    # Create Evaluation Callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_dir,
                                 log_path=log_dir,
                                 eval_freq=10000,  # Adjust evaluation frequency
                                 n_eval_episodes=5)

    # Initialize PPO model
    model = PPO('MlpPolicy', env, verbose=1, device=DEVICE_USED)

    # Train the model
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=eval_callback)

    # Save the model
    model.save(os.path.join(model_dir, "ppo_robot_pyB"))

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Close environments
    env.close()
    eval_env.close()

    # Load and evaluate the best model
    best_model_path = os.path.join(model_dir, "best_model")
    best_model = PPO.load(best_model_path)

    # Create a new evaluation environment
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=5)
    print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    # Plot evaluation results
    data = np.load(os.path.join(log_dir, "evaluations.npz"))
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
