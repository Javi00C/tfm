from gymnasium.envs.registration import register

#MY SIMULATIONS
#register(
#    id="SimpleRobotEnv-v0",
#    entry_point="gymnasium_env.envs.simple_robot:SimpleRobotEnv",
#)

register(
    id="gymnasium_env/SimpleRobotEnv-v1",
    entry_point="gymnasium_env.envs:SimpleRobotEnvSine",
)
