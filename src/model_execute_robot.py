import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Load the saved PPO model
model = PPO.load("/home/javi/tfm/models/ppo_robot")

# Create the SimpleRobotEnv environment with rendering enabled
env = gym.make('SimpleRobotEnv-v0', render_mode='human')

# Wrap the environment to stack frames, ensuring observation shape matches what the model expects
env = DummyVecEnv([lambda: env])  # DummyVecEnv to make the environment compatible with Stable Baselines3
n_stack = 4  # Set n_stack to match the one used in training
env = VecFrameStack(env, n_stack=n_stack)

# Reset the environment
obs = env.reset()

# Execute the policy for a longer duration
episode_length = 5000  # Adjusted episode length for better visualization

for _ in range(episode_length):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Render the environment after each action
    env.envs[0].render()  # Use the underlying environment to render, bypassing DummyVecEnv

    if done[0]:  # Adjusted to work with DummyVecEnv which returns list-like `done`
        obs = env.reset()

env.close()