from gymnasium.envs.registration import register

# #Frist 2d simulations
# register(
#    id="gymnasium_env/SimpleRobotEnv-v0",
#    entry_point="gymnasium_env.envs:SimpleRobotEnv",
# )

# register(
#    id="gymnasium_env/SimpleRobotEnv-v1",
#    entry_point="gymnasium_env.envs:SimpleRobotEnvSine",
# )

# #Mujoco ball control example env
# register(
#    id="gymnasium_env/SimpleMujocoEnv-v0",
#    entry_point="gymnasium_env.envs:SimpleMujocoEnv",
# )

# #Mujoco ur5e and 2f85 gripper env
# register(
#    id="gymnasium_env/ur5e_2f85Env-v0",
#    entry_point="gymnasium_env.envs:ur5e_2f85Env",
# )

#PyBullet robot simulation
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v0",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv",
)

#PyBullet robot simulation
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v1",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple",
)

#PyBullet robot simulation
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v2",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple_3d",
)

#PyBullet robot simulation
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v3",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple_6d",
)

#PyBullet UR5e robot simulation
register(
    id="gymnasium_env/ur5e_pybulletEnv-v0",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv",
)

