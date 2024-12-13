# Connect to PyBullet and setup environment
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import gymnasium as gym
from typing import Optional

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load an MJCF file
mjcf_path = "path_to_your_mjcf_file.xml"  # Replace with your MJCF file path
mjcf_objects = p.loadMJCF(mjcf_path)

# Print all loaded objects
print("Loaded MJCF Objects:", mjcf_objects)

# For example, set initial joint states
for obj_id in mjcf_objects:
    num_joints = p.getNumJoints(obj_id)
    for joint_index in range(num_joints):
        p.resetJointState(obj_id, joint_index, targetValue=0)

# Step the simulation
for _ in range(1000):
    p.stepSimulation()
