from gymnasium.envs.registration import register

#MY SIMULATIONS
register(
    id="SimpleRobotEnv-v0",
    entry_point="gymnasium_env.envs:SimpleRobotEnv",
    max_episode_steps=1000,
    reward_threshold=900,
)

register(
    id="SimpleRobotEnv-v1",
    entry_point="gymnasium_env.envs:SimpleRobotEnvSine",
    max_episode_steps=1000,
    reward_threshold=900,
)