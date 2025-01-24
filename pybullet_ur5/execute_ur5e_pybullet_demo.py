import gymnasium as gym
import gymnasium_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt
import pandas as pd

# Load the saved PPO model
model = PPO.load("models_pybullet/best_model_exp_1A_2")


env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v0" # Rope no gripper control

#env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v1" # env simple
#env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v2" # env simple 3d


#env_str = "gymnasium_env/ur5e_pybulletEnv-v0" # Specific cartesian goal
#env_str = "gymnasium_env/ur5e_pybulletEnv-v1" # Random cartesian goal
#env_str = "gymnasium_env/ur5e_pybulletEnv-v2" # Specific cartesian and orientation goal
#env_str = "gymnasium_env/ur5e_pybulletEnv-v3" # Random cartesian and orientation goal
#env_str = "gymnasium_env/ur5e_2f85_pybulletEnv-v4"

#env_str = "gymnasium_env/ur5e_pybulletEnv-v4"
# Create the environment with rendering enabled
env = gym.make(env_str, render_mode='human')

# Wrap the environment to make it compatible with Stable Baselines3
env = DummyVecEnv([lambda: env])

# Reset the environment
obs = env.reset()

# Initialize recording variables
record_interval = 1  # Record every n steps
distance_data = []  # To store distance values
reward_data = []  # To store reward values
cumulative_rewards = []  # To store cumulative rewards
cumulative_reward = 0
time_steps = []  # To store corresponding time steps

# Execute the policy for a longer duration
episode_length = 5000  # Adjust as needed
for i in range(episode_length):

    action, _states = model.predict(obs)

    obs, reward, done, info = env.step(action)
    distance_value = info[0]["distance_to_goal"]
    print(distance_value)

    # Update cumulative reward
    cumulative_reward += reward[0]  # Accumulate the reward

    # Record data every `record_interval` steps
    if i % record_interval == 0:
        distance_data.append(distance_value)
        cumulative_rewards.append(cumulative_reward)
        time_steps.append(i)

    #print(f"action: {action}")
    #print(f"obs: {obs}")
    #print(f"Reward: {reward}")
    # Render the environment
    env.envs[0].render()
        
    #if done[0]:  # Since 'done' is an array in DummyVecEnv
    #    obs = env.reset()

env.close()


# Save data as a CSV file
data = pd.DataFrame({"Time Step": time_steps, "Distance": distance_data, "Cumulative Reward": cumulative_rewards})
data.to_csv("distance_cumulative_reward_vs_time.csv", index=False)

# Plot 1: Distance vs Time
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, distance_data,
    linestyle='-', linewidth=0.7, color='blue', label="Distance to Goal"
)
plt.xlabel("Time Step")
plt.ylabel("Distance to Goal")
plt.title("Distance vs Time")
plt.grid(True)
plt.legend()
plt.savefig("distance_vs_time.png")
plt.show()

# Plot 2: Cumulative Reward vs Time
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, cumulative_rewards,
    linestyle='-', linewidth=0.7, color='green', label="Cumulative Reward"
)
plt.xlabel("Time Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward vs Time")
plt.grid(True)
plt.legend()
plt.savefig("cumulative_reward_vs_time.png")
plt.show()

# Plot 3: Distance and Cumulative Reward vs Time with Dual Axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot distance on the left axis
ax1.plot(time_steps, distance_data, 'b-', label="Distance to Goal")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Distance to Goal", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create the right axis for cumulative reward
ax2 = ax1.twinx()
ax2.plot(time_steps, cumulative_rewards, 'g-', label="Cumulative Reward")
ax2.set_ylabel("Cumulative Reward", color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add grid, title, and legends
fig.suptitle("Distance and Cumulative Reward vs Time")
ax1.grid(True)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig("distance_cumulative_reward_vs_time.png")
plt.show()