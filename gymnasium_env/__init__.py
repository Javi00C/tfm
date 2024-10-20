from gymnasium.envs.registration import register

#MY SIMULATIONS
register(
    id="SimpleRobotEnv-v0",
    entry_point="gymnasium_env.envs:SimpleRobotEnv",
)

register(
    id="SimpleRobotEnv-v1",
    entry_point="gymnasium_env.envs:SimpleRobotEnvSine",
)