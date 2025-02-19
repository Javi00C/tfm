import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

env_str = "CarRacing-v2"
#env_str = 'SimpleRobotEnv-v0'

# Load the saved PPO model
model = PPO.load("/home/javi/tfm/models/ppo_models_car_racing/ppo1_200000ts/ppo_car_racing")

# Create the CarRacing environment with rendering enabled
env = gym.make(env_str, render_mode='human')

# Wrap the environment to ensure consistency with training
env = DummyVecEnv([lambda: env])  # DummyVecEnv to make the environment compatible with Stable Baselines3
n_stack = 4 
env = VecFrameStack(env, n_stack=n_stack)
env = VecTransposeImage(env)

# Reset the environment
obs = env.reset()

# Execute the policy for a longer duration
episode_length = 5000 

for _ in range(episode_length):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()