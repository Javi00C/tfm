import pybullet as p
import pybullet_data
import time
import numpy as np
import os
from robot import UR5Robotiq85  # Import the robot class

class RopeRobotSimulation:
    def __init__(self, gui=True):
        self.gui = gui
        self.num_segments = 30
        self.segment_length = 0.03
        self.segment_radius = 0.01
        self.mass = 0.1
        self.friction = 0.5
        self.start_position = [0.4, 0, 1]  # Adjusted to be above and away from the robot
        self.rope_segments = []
        self.constraints = []

        self._connect_to_pybullet()
        self._load_environment()
        self._load_robot()
        self._load_rope()
        self._adjust_camera()

    def _connect_to_pybullet(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def _load_robot(self):
        self.robot = UR5Robotiq85(pos=[0, 0, 0], ori=[0, 0, 0])
        self.robot.step_simulation = self._step_simulation
        self.robot.load()
        self.robot.reset()

    def _load_rope(self):
        self.rope_segments = []
        self.constraints = []
        for i in range(self.num_segments):
            segment_position = [self.start_position[0],
                                self.start_position[1],
                                self.start_position[2] - i * self.segment_length]

            # Create a segment as a capsule
            segment_id = p.createCollisionShape(p.GEOM_CAPSULE, 
                                                radius=self.segment_radius, 
                                                height=self.segment_length)
            visual_id = p.createVisualShape(p.GEOM_CAPSULE, 
                                            radius=self.segment_radius, 
                                            length=self.segment_length, 
                                            rgbaColor=[0, 1, 0, 1])
            body_id = p.createMultiBody(baseMass=self.mass, 
                                        baseCollisionShapeIndex=segment_id,
                                        baseVisualShapeIndex=visual_id, 
                                        basePosition=segment_position)

            # Set friction for each segment
            p.changeDynamics(body_id, -1, lateralFriction=self.friction)
            self.rope_segments.append(body_id)

            # Disable collisions between adjacent segments
            if i > 0:
                p.setCollisionFilterPair(self.rope_segments[i - 1], self.rope_segments[i], -1, -1, enableCollision=0)

            # Connect this segment to the previous one with a constraint
            if i > 0:
                c = p.createConstraint(
                    parentBodyUniqueId=self.rope_segments[i - 1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=self.rope_segments[i],
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, -self.segment_length / 2],
                    childFramePosition=[0, 0, self.segment_length / 2],
                )
                self.constraints.append(c)

        # Create a small, static "anchor" body to hold the first rope segment
        anchor_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
        anchor_body = p.createMultiBody(baseMass=0, 
                                        baseCollisionShapeIndex=anchor_shape, 
                                        basePosition=self.start_position)

        # Fix the first rope segment to the anchor with a fixed joint
        p.createConstraint(
            parentBodyUniqueId=anchor_body,
            parentLinkIndex=-1,
            childBodyUniqueId=self.rope_segments[0],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

    def _step_simulation(self):
        p.stepSimulation()

    def _adjust_camera(self):
        # Adjust camera to show both robot and rope
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])

    def apply_stiffness_and_damping(self):
        stiffness = 0.2  # Stiffness factor
        damping = 0.1    # Damping factor
        for i in range(1, self.num_segments):
            parent_id = self.rope_segments[i - 1]
            child_id = self.rope_segments[i]

            parent_pos, _ = p.getBasePositionAndOrientation(parent_id)
            child_pos, _ = p.getBasePositionAndOrientation(child_id)
            displacement = np.array(child_pos) - np.array(parent_pos)
            restoring_force = -stiffness * displacement

            parent_vel, _ = p.getBaseVelocity(parent_id)
            child_vel, _ = p.getBaseVelocity(child_id)
            relative_velocity = np.array(child_vel) - np.array(parent_vel)
            damping_force = -damping * relative_velocity
  
            total_force = restoring_force + damping_force
            p.applyExternalForce(child_id, -1, forceObj=total_force.tolist(), posObj=child_pos, flags=p.WORLD_FRAME)
            p.applyExternalForce(parent_id, -1, forceObj=(-total_force).tolist(), posObj=parent_pos, flags=p.WORLD_FRAME)

    def apply_action(self, action):
        # action could be a tuple: (x, y, z, roll, pitch, yaw) for end-effector
        # or joint targets if you prefer. Adjust as needed.
        if len(action) == 6:
            # Treat as end-effector pose command
            self.robot.move_ee(action, control_method='end')
        else:
            # Otherwise treat as joint space action
            self.robot.move_ee(action, control_method='joint')

    def move_gripper(self, fraction):
        # fraction in [0,1], interpolates between closed and open gripper positions
        gripper_range = self.robot.gripper_range[1] - self.robot.gripper_range[0]
        gripper_pos = self.robot.gripper_range[0] + fraction * gripper_range
        self.robot.move_gripper(gripper_pos)

    def get_observation(self):
        # Example: Return joint positions and EE position
        obs = self.robot.get_joint_obs()
        # You can add rope state or other info as needed.
        return obs

    def reset_simulation(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_environment()
        self._load_robot()
        self._load_rope()
        self._adjust_camera()
        return self.get_observation()

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    # Example usage of the simulation class
    sim = RopeRobotSimulation(gui=True)
    
    # Move the robot's End-Effector slightly as a demonstration
    # Using 'end' control method: x, y, z, roll, pitch, yaw
    sim.apply_action((0.4, 0.0, 0.5, 0, 0, 0))
    for _ in range(100):
        sim._step_simulation()
        time.sleep(1/240)

    # Open gripper slightly
    sim.move_gripper(0.2)
    print("Gripper max value:", sim.robot.gripper_range[1])
    for _ in range(100):
        sim._step_simulation()
        time.sleep(1 / 240)

    # Run rope simulation with stiffness and damping for a while
    simulation_time = 10
    simulation_steps = int(simulation_time * 240)
    for step in range(simulation_steps):
        sim.apply_stiffness_and_damping()
        sim._step_simulation()
        time.sleep(1.0 / 240.0)

    sim.close()
