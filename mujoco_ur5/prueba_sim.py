import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

# Load the MuJoCo model
model_path = "scene_ur5_2f85.xml"  # Replace with your MuJoCo model file path
model = mujoco.MjModel.from_xml_path(model_path)

# Create a data object to hold simulation states
data = mujoco.MjData(model)

# Initialize joints to keyframe values (from <keyframe> qpos)
# Replace these values with the `qpos` values from your keyframe
initial_qpos = [
    -1.82, -1.82, 1.57, -2.95, -1.57, 1.45, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]
data.qpos[:len(initial_qpos)] = initial_qpos  # Set qpos
data.qvel[:] = 0  # Set velocities to zero
data.ctrl[:] = [-1.82, -1.82, 1.57, -2.95, -1.57, 1.45, 163]  # Set control inputs to zero (if applicable)

# Function to display key mjData attributes
def display_data(data):
    #print(f"Time: {data.time:.3f}")
    print(f"Position_ur5: {data.qpos[0:6]}")
    print(f"Position_gripper: {data.qpos[6:13]}")
    print(f"Position_rope: {data.qpos[13:30]}")
    #print(f"Velocity: {data.qvel}")
    #print(f"Control Inputs: {data.ctrl}")
    #print(f"Sensor Data: {data.sensordata}")
    obs = np.concatenate((np.array(data.qpos),
                        np.array(data.qvel),
                        np.array(data.sensordata)), axis=0)
    #print(f"CONCATENATED: {obs}")
    print("-" * 40)

# Initialize a viewer for visualization
with viewer.launch_passive(model, data) as sim_viewer:
    print("Simulation started. Close the window to exit.")

    # Track the last time data was displayed
    last_display_time = time.time()

    while sim_viewer.is_running():
        # Step through the simulation
        mujoco.mj_step(model, data)

        # Check if 0.5 seconds have passed since the last display
        current_time = time.time()
        if current_time - last_display_time >= 0.5:
            display_data(data)
            last_display_time = current_time

        # Update the viewer to show the simulation
        sim_viewer.sync()

print("Simulation ended.")
