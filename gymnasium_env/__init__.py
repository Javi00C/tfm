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



#PyBullet robot simulation 1A-1
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v0",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple_1a1",
)

#PyBullet robot simulation 1A-2
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v1",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple_1a2",
)

#PyBullet robot simulation 1A-3
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v2",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_Simple_1a3",
)

#PyBullet robot simulation 1A-4 (GOAL: cartesian)
register(
    id="gymnasium_env/ur5e_pybulletEnv-v0",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_1a4",
)

#PyBullet UR5e robot simulation 1A-5 (GOAL: random cartesian)
register(
    id="gymnasium_env/ur5e_pybulletEnv-v1",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_1a5",
)


#PyBullet UR5e robot simulation 1B-1 (GOAL: cartesian and orientation)
register(
    id="gymnasium_env/ur5e_pybulletEnv-v2",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_1b1",
)

#PyBullet UR5e robot simulation 1B-2 (GOAL: cartesian and orientation)
register(
    id="gymnasium_env/ur5e_pybulletEnv-v3",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_1b2",
)


#PyBullet robot simulation 2A (Rope and no gripper control)
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v3",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_2a",
)

#PyBullet robot simulation 2B (Rope and no gripper control)
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v4",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_2b",
)

#PyBullet robot simulation 2C (Rope and no gripper control)
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v5",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_2c",
)

#PyBullet robot simulation 2D (Rope and no gripper control)
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v6",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_2d",
)

#PyBullet simulation with DIGIT sensor (Rigid rope)
register(
    id="gymnasium_env/ur5e_2f85_pybulletEnv-v7",
    entry_point="gymnasium_env.envs:ur5e_2f85_pybulletEnv_digit",
)


#PyBullet UR5e robot simulation GOAL: random cartesian and random orientation
register(
    id="gymnasium_env/ur5e_pybulletEnv-v4",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_random_orient",
)

#PyBullet simulation with lower frequency in simulation
register(
    id="gymnasium_env/ur5e_pybulletEnv-v5",
    entry_point="gymnasium_env.envs:ur5e_pybulletEnv_lowfreq",
)