import time
import numpy as np
import pybullet

from ur5e_gripper_sim import UR5Sim

if __name__ == "__main__":
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500)

    # Move the arm a bit, keep the gripper open
    for i in range(1000000):
        end_effector_velocity = np.array([0.001, 0.0, 0.0, 0.00, 0.00, 0.00])
        gripper_cmd = 0.7  # open
        sim.step(end_effector_velocity, gripper_cmd)
        #print(f"Sensor reading{np.shape(sim.get_sensor_reading())}")
        sim.get_sensor_reading()
    print("Simulation ended.")
    sim.close()
