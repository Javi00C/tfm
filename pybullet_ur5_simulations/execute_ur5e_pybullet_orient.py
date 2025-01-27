import gymnasium as gym
import gymnasium_env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
env_mapping = {
    "1B-1": "gymnasium_env/ur5e_pybulletEnv-v2",
    "1B-2": "gymnasium_env/ur5e_pybulletEnv-v3",
}
model_mapping = {
    "1B-1": "models_pybullet/model_1B_1",
    "1B-2": "models_pybullet/model_1B_2",
}


# ----------------------------------------------------
# 2) Prompt user for environment and model selections
# ----------------------------------------------------
print("Available experiment IDs:", ", ".join(env_mapping.keys()))
env_key = input("Please select the experiment ID ('1B-1', '1B-2'): ").strip()

model_key = env_key
# Validate user input and fetch the environment/model
if env_key not in env_mapping:
    raise ValueError(f"Invalid experiment key: {env_key}")
if model_key not in model_mapping:
    raise ValueError(f"Invalid model key: {model_key}")

env_str = env_mapping[env_key]
model_path = model_mapping[model_key]

env = gym.make(env_str, render_mode='human')
env = DummyVecEnv([lambda: env])

# Load the saved PPO model
model = PPO.load(model_path)

# Reset the environment
obs = env.reset()

# Initialize recording variables
record_interval = 1  # Record every n steps
cart_distance_data = []  # To store cartesian distance values
orient_distance_data = []  # To store orientation distance values
cumulative_rewards = []  # To store cumulative rewards
cumulative_reward = 0  # Initialize cumulative reward
time_steps = []  # To store corresponding time steps

# Simulation parameters
episode_length = 2500  # Adjust as needed

# Execute the policy
for i in range(episode_length):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    # Extract distances from `info`
    cart_dist_to_goal = info[0]["cart_dist_to_goal"]
    orient_dist_to_goal = info[0]["orient_dist_to_goal"]
    
    # Update cumulative reward
    cumulative_reward += reward[0]  # Accumulate the reward

    # Record data every `record_interval` steps
    if i % record_interval == 0:
        cart_distance_data.append(cart_dist_to_goal)
        orient_distance_data.append(orient_dist_to_goal)
        cumulative_rewards.append(cumulative_reward)
        time_steps.append(i)

    # Render the environment
    env.envs[0].render()

    # Reset environment if done
    if done[0]:
        obs = env.reset()

env.close()

# Save data as a CSV file
data = pd.DataFrame({
    "Time Step": time_steps,
    "Cart Distance": cart_distance_data,
    "Orient Distance": orient_distance_data,
    "Cumulative Reward": cumulative_rewards
})
data.to_csv("distances_cumulative_reward_vs_time.csv", index=False)

# Plot 1: Cartesian Distance vs Time
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, cart_distance_data,
    linestyle='-', linewidth=0.7, color='blue', label="Cartesian Distance to Goal"
)
plt.xlabel("Time Step")
plt.ylabel("Cartesian Distance")
plt.title("Cartesian Distance vs Time")
plt.grid(True)
plt.legend()
plt.savefig("cartesian_distance_vs_time.png")
plt.show()

# Plot 2: Orientation Distance vs Time
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, orient_distance_data,
    linestyle='-', linewidth=0.7, color='orange', label="Orientation Distance to Goal"
)
plt.xlabel("Time Step")
plt.ylabel("Orientation Distance")
plt.title("Orientation Distance vs Time")
plt.grid(True)
plt.legend()
plt.savefig("orientation_distance_vs_time.png")
plt.show()

# Plot 3: Cumulative Reward vs Time
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

# Plot 4: Distances and Cumulative Reward vs Time with Dual Axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot cartesian and orientation distances on the left axis
ax1.plot(time_steps, cart_distance_data, 'b-', label="Cartesian Distance to Goal")
ax1.plot(time_steps, orient_distance_data, 'orange', label="Orientation Distance to Goal")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Distance to Goal", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create the right axis for cumulative reward
ax2 = ax1.twinx()
ax2.plot(time_steps, cumulative_rewards, 'g-', label="Cumulative Reward")
ax2.set_ylabel("Cumulative Reward", color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add grid, title, and legends
fig.suptitle("Distances and Cumulative Reward vs Time")
ax1.grid(True)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig("distances_cumulative_reward_vs_time.png")
plt.show()
