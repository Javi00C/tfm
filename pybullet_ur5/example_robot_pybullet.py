import pybullet as p
import pybullet_data
import time
import os

# Connect to PyBullet and set up environment
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a URDF file
urdf_path = "/home/javi/tfm/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf"  # Replace with your URDF file path
if not os.path.exists(urdf_path):
    print("Error: URDF file not found at", urdf_path)
    exit()

# Load URDF model
robot_id = p.loadURDF(urdf_path, useFixedBase=True) 
print(f"Loaded URDF Object with ID: {robot_id}")

# Debugging: Check loaded robot properties
num_joints = p.getNumJoints(robot_id)
print(f"Number of joints for the loaded robot: {num_joints}")
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    print(f"Joint {joint_index} info: {joint_info}")

# Set initial vertical position for the robot
# Adjust joint positions as needed for your robot
initial_positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Example for 6-DOF UR5 arm
if len(initial_positions) != num_joints:
    print("Warning: Initial positions do not match the number of joints. Adjust as needed.")

for joint_index in range(min(num_joints, len(initial_positions))):
    p.resetJointState(robot_id, joint_index, targetValue=initial_positions[joint_index])

# Adjust camera to focus on the robot
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

# Simulation duration in seconds
simulation_time = 100
simulation_steps = simulation_time * 240  # Assuming 240 Hz simulation frequency

# Run simulation
for step in range(simulation_steps):
    p.stepSimulation()
    time.sleep(1 / 240)  # Sleep to maintain real-time simulation
