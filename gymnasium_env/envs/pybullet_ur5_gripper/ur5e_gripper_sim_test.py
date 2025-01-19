import time
import numpy as np
import pybullet

#from ur5e_gripper_sim_simple import UR5Sim
from ur5e_gripper_sim import UR5Sim
#from gymnasium_env.envs.pybullet_ur5e_sim.ur5e_sim_orient import UR5Sim
#from gymnasium_env.envs.pybullet_ur5e_sim.ur5e_sim_orient_generator import UR5Sim
#goal = [0.4,0.15,0.4,0,1.57,0]

goal = [0.5,0.3,0.4]
if __name__ == "__main__":
    sim = UR5Sim(useIK=True, renders=True, maxSteps=500, goal_position=goal)
    # Move the arm a bit, keep the gripper open
    sim.add_visual_goal(goal)
    # necessary_angles = sim.calculate_ik(goal[:3],goal[3:])
    # #orientation_euler = pybullet.getEulerFromQuaternion(orientation)
    # #sim.pybullet.get
    # sim.set_joint_angles(necessary_angles)

    for i in range(1000000):
        end_effector_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gripper_cmd = 0.6  # open
        sim.step(end_effector_velocity, gripper_cmd)
        #sim.step(end_effector_velocity)
        #print(f"Distance link to goal: {np.linalg.norm(sim.get_last_rope_link_position()-goal)}")
        #pose = sim.get_last_rope_link_position()
        #print(f"Distance EE to goal: {np.linalg.norm(pose[:3]-goal)}")
        
        pose = sim.get_end_eff_pose()
        #print(f"Orient: {pose[3:]}")
        print(f"Pose: {pose}")
        #sim.get_sensor_reading()
    print("Simulation ended.")
    sim.close()
