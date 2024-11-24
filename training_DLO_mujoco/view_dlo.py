#import mujoco
import mujoco.viewer
import time

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('assets/DLO_Simulation.xml')
data = mujoco.MjData(model)

# Create a viewer to render the simulation
viewer = mujoco.viewer.MujocoViewer(model, data)

# Run the simulation
while True:
    mujoco.mj_step(model, data)
    viewer.render()

    # Slow down the simulation for viewing
    time.sleep(0.01)

    # Exit on pressing the Escape key
    if viewer.is_terminated():
        break

viewer.close()
