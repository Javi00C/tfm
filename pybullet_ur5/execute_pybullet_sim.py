import gymnasium as gym
import gymnasium_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the saved PPO model
model = PPO.load("models_pybullet/best_model")
#model = PPO.load("models_pybullet/best_model")

env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0"

# Create the environment with rendering enabled
env = gym.make(env_str, render_mode='human')

# Wrap the environment to make it compatible with Stable Baselines3
env = DummyVecEnv([lambda: env])

# Reset the environment
obs = env.reset()
#obs = np.squeeze(obs)
# print(f"Type of obs: {type(obs)}")
# print(f"Shape of obs: {obs.shape}")
# print(f"Contents of obs: {obs[:10]}")  # Print a sample


# Execute the policy for a longer duration
episode_length = 10000000  # Adjust as needed
for _ in range(episode_length):
    #print(f"Shape of observation passed to model: {obs.shape}")
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Render the environment
    env.envs[0].render()
        
    #if done[0]:  # Since 'done' is an array in DummyVecEnv
    #    obs = env.reset()

env.close()
