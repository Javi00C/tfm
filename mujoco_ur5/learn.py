from stable_baselines3.common.env_checker import check_env
import gymnasium_env
from stable_baselines3 import PPO
import gymnasium


# initialize your enviroment
#env = SimpleMujocoEnv(render_mode="rgb_array")
env_str = "gymnasium_env/ur5e_2f85Env-v0"
env = gymnasium.make(env_str, render_mode="rgb_array")
# it will check your custom environment and output additional warnings if needed
check_env(env)
# learning with tensorboard logging and saving model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ur5_tensorboard/")
model.learn(total_timesteps=50000, log_interval=4)
model.save("ppo_ur5")

