import time
import numpy as np
from ur5e_gripper_sim import UR5Sim

if __name__ == "__main__":
    # Example usage of the simulation:
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500)

    # Run a simple loop: move the end-effector around and try to open/close gripper
    for i in range(60000):
        # Try moving along the x-axis slowly and keep the gripper open
        end_effector_velocity = np.array([0.005, 0.000, 0.000])
        gripper_cmd = -1.0  # open
        sim.step(end_effector_velocity, gripper_cmd)

    print("Simulation ended.")
    pybullet.disconnect()
