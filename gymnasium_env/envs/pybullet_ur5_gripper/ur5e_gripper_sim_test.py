import time
import numpy as np
import pybullet

#from ur5e_gripper_sim_simple import UR5Sim
from ur5e_gripper_sim_simple import UR5Sim

if __name__ == "__main__":
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500,goal_position=[0.5,0.4,0.6])
    goal = [0.5,0.4,0.6]
    # Move the arm a bit, keep the gripper open
    for i in range(1000000):
        end_effector_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gripper_cmd = 0.6  # open
        sim.step(end_effector_velocity, gripper_cmd)
        #print(f"Distance link to goal: {np.linalg.norm(sim.get_last_rope_link_position()-goal)}")
        #pose = sim.get_end_eff_pose()
        #print(f"Distance EE to goal: {np.linalg.norm(pose[:3]-goal)}")
        
        pose = sim.get_end_eff_pose()
        print(f"Position: {pose[:3]}")
        #sim.get_sensor_reading()
    print("Simulation ended.")
    sim.close()
