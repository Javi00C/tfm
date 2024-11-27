import gymnasium as gym
import gymnasium_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import mujoco

print("Gymnasium version:", gym.__version__)
print("Mujoco version:", mujoco.__version__)

# Load the saved PPO model
model = PPO.load("/home/javi/tfm/src/mujoco_ur5/models/ppo_robot.zip")

env_str = "gymnasium_env/ur5e_2f85Env-v0"

# Create the environment with rendering enabled
env = gym.make(env_str, render_mode='human')

# Wrap the environment to make it compatible with Stable Baselines3
env = DummyVecEnv([lambda: env])

# Reset the environment
obs = env.reset()

# Execute the policy for a longer duration
episode_length = 5000  # Adjust as needed
for _ in range(episode_length):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    # Render the environment
    env.envs[0].render()

    if done[0]:  # Since 'done' is an array in DummyVecEnv
        obs = env.reset()

env.close()
