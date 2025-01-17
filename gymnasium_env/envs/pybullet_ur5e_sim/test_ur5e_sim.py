import time
import numpy as np
import pybullet

from ur5e_sim import UR5Sim

if __name__ == "__main__":
    goal = [0.2,0.2,0.5]
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500)
    # Move the arm a bit, keep the gripper open
    for i in range(800):
        end_effector_velocity = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        sim.step(end_effector_velocity)
        print(f"Distance ee to goal: {np.linalg.norm(sim.get_end_eff_position()-goal)}")
        #print(f"Distance link to goal: {np.linalg.norm(sim.get_last_rope_link_position()-goal)}")
        #pose = sim.get_end_eff_pose()
        #print(f"Orientation: {pose[3:6]}")
    print("Simulation ended.")
    sim.close()
