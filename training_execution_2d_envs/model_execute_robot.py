import gymnasium as gym
import gymnasium_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Load the saved PPO model
model = PPO.load("/home/javi/tfm/best_model_simple_sims/best_model_sine")
#model = PPO.load("/home/javi/tfm/src/training_execution_2d_envs/models/best_model")

#env_str = "CarRacing-v2"
#env_str = "SimpleRobotEnv-v0" #Straight line edge
env_str = "gymnasium_env/SimpleRobotEnv-v1" #Sine wave edge

# Create the SimpleRobotEnv environment with rendering enabled
env = gym.make(env_str, render_mode='human')

# Wrap the environment to stack frames, ensuring observation shape matches what the model expects
env = DummyVecEnv([lambda: env])  # DummyVecEnv to make the environment compatible with Stable Baselines3
n_stack = 4  # Set n_stack to match the one used in training
env = VecFrameStack(env, n_stack=n_stack)

# Reset the environment
obs = env.reset()

# Execute the policy for a longer duration
episode_length = 5000  # Adjusted episode length for better visualization
r = 0
for _ in range(episode_length):
    action, _states = model.predict(obs)
    obs, step_reward, done, info = env.step(action)
    #r += step_reward

    # Render the environment after each action
    env.envs[0].render()  # Use the underlying environment to render, bypassing DummyVecEnv

    if done[0]:  # Adjusted to work with DummyVecEnv which returns list-like `done`
        obs = env.reset()

env.close()