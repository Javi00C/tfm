from stable_baselines3.common.env_checker import check_env
import gymnasium_env
from stable_baselines3 import SAC
import gymnasium


# initialize your enviroment
#env = SimpleMujocoEnv(render_mode="rgb_array")
env_str = "gymnasium_env/SimpleMujocoEnv-v0"
env = gymnasium.make(env_str, render_mode="rgb_array")
# it will check your custom environment and output additional warnings if needed
check_env(env)
# learning with tensorboard logging and saving model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_ball_balance_tensorboard/")
model.learn(total_timesteps=50000, log_interval=4)
model.save("sac_ball_balance")

