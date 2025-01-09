import time
import numpy as np
import pybullet

from ur5e_gripper_sim_simple import UR5Sim

if __name__ == "__main__":
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500,goal_position=[0.5,0.5,0.5])

    # Move the arm a bit, keep the gripper open
    for i in range(1000000):
        end_effector_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
        gripper_cmd = 1.0  # open
        sim.step(end_effector_velocity, gripper_cmd)
        pose = sim.get_end_eff_pose()
        print(f"Orientation: {pose[3:6]}")
        #print(f"Pose: {sim.get_end_eff_pose()}")
        #sim.get_sensor_reading()
    print("Simulation ended.")
    sim.close()
