import gymnasium as gym
import gymnasium_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the saved PPO model
model = PPO.load("models_pybullet/best_model")
#model = PPO.load("models_pybullet/ppo_robot_py_25_2048000")
#model = PPO.load("models_pybullet/ppo_ur5_simple_demo_25_2048000")
#model = PPO.load("models_pybullet/ppo_ur5_simple_demo_10_81920")
#model = PPO.load("models_pybullet/ppo_ur5_simple_demo_25_81920")
#model = PPO.load("models_pybullet/ppo_ur5_simple_demo_12_3_25_1024000")

#env_str = "gymnasium_env/ur5e_pybulletEnv-v1"
#env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0"
env_str = "gymnasium_env/ur5e_pybulletEnv-v2"
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
episode_length = 100000  # Adjust as needed
for _ in range(episode_length):
    #print(f"Shape of observation passed to model: {obs.shape}")
    action, _states = model.predict(obs)
    # getting rid of orientation actions
    #action = np.array([np.concatenate((action[0,:3], np.array([0.0, 0.0, 0.0])))])
    #action = np.array([[0.0, 0.0, 0.01, 0.0, 0.0, 0.0]])
    obs, reward, done, info = env.step(action)
    print(f"action: {action}")
    print(f"obs: {obs}")
    print(f"Reward: {reward}")
    # Render the environment
    env.envs[0].render()
        
    #if done[0]:  # Since 'done' is an array in DummyVecEnv
    #    obs = env.reset()

env.close()
