import gymnasium as gym
import gymnasium_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------
# 1) Define possible environments and models via dictionaries
# ----------------------------------------------------
env_mapping = {
    "1A-1": "gymnasium_env/ur5e_2f85_pybulletEnv-v0",
    "1A-2": "gymnasium_env/ur5e_2f85_pybulletEnv-v1",
    "1A-3": "gymnasium_env/ur5e_2f85_pybulletEnv-v2",
    "1A-4": "gymnasium_env/ur5e_pybulletEnv-v0",
    "1A-5": "gymnasium_env/ur5e_pybulletEnv-v1",
    "2A":   "gymnasium_env/ur5e_2f85_pybulletEnv-v3",
    "2B":   "gymnasium_env/ur5e_2f85_pybulletEnv-v4",
    "2C":   "gymnasium_env/ur5e_2f85_pybulletEnv-v5",
    "2D":   "gymnasium_env/ur5e_2f85_pybulletEnv-v6",
}

model_mapping = {
    "1A-1": "models_pybullet/model_1A_1",
    "1A-2": "models_pybullet/model_1A_2",
    "1A-3": "models_pybullet/model_1A_3",
    "1A-4": "models_pybullet/model_1A_4",
    "1A-5": "models_pybullet/model_1A_5",
    "2A":   "models_pybullet/model_2A",
    "2B":   "models_pybullet/model_2B",
    "2C":   "models_pybullet/model_2C",
    "2D":   "models_pybullet/model_2D",
}

# ----------------------------------------------------
# 2) Prompt user for environment and model selections
# ----------------------------------------------------
print("Available experiment IDs:", ", ".join(env_mapping.keys()))
env_key = input("Please select the experiment ID (e.g. '1A-1', '2A', etc.): ").strip()

model_key = env_key
# Validate user input and fetch the environment/model
if env_key not in env_mapping:
    raise ValueError(f"Invalid experiment key: {env_key}")
if model_key not in model_mapping:
    raise ValueError(f"Invalid model key: {model_key}")

env_str = env_mapping[env_key]
model_path = model_mapping[model_key]

# ----------------------------------------------------
# 3) Create environment and load model
# ----------------------------------------------------
env = gym.make(env_str, render_mode='human')
env = DummyVecEnv([lambda: env])

model = PPO.load(model_path)

# ----------------------------------------------------
# 4) Run the simulation, record data, and plot
# ----------------------------------------------------
# Reset the environment
obs = env.reset()

# Initialize recording variables
record_interval = 1  # Record every n steps
distance_data = []   # To store distance values
cumulative_rewards = []
cumulative_reward = 0
time_steps = []

episode_length = 5000  # Adjust as needed

for i in range(episode_length):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    distance_value = info[0]["distance_to_goal"]
    print(distance_value)

    # Update cumulative reward
    cumulative_reward += reward[0]

    # Record data periodically
    if i % record_interval == 0:
        distance_data.append(distance_value)
        cumulative_rewards.append(cumulative_reward)
        time_steps.append(i)

    # Render the environment
    env.envs[0].render()

env.close()

# ----------------------------------------------------
# 5) Save recorded data as CSV
# ----------------------------------------------------
data = pd.DataFrame({
    "Time Step": time_steps,
    "Distance": distance_data,
    "Cumulative Reward": cumulative_rewards
})
data.to_csv("distance_cumulative_reward_vs_time.csv", index=False)

# ----------------------------------------------------
# 6) Plot Distance vs Time
# ----------------------------------------------------
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

# ----------------------------------------------------
# 7) Plot Cumulative Reward vs Time
# ----------------------------------------------------
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

# ----------------------------------------------------
# 8) Combined Plot with Dual Axes
# ----------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot distance on the left axis
ax1.plot(time_steps, distance_data, 'b-', label="Distance to Goal")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Distance to Goal", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create right axis for cumulative reward
ax2 = ax1.twinx()
ax2.plot(time_steps, cumulative_rewards, 'g-', label="Cumulative Reward")
ax2.set_ylabel("Cumulative Reward", color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.suptitle("Distance and Cumulative Reward vs Time")
ax1.grid(True)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig("distance_cumulative_reward_vs_time.png")
plt.show()
