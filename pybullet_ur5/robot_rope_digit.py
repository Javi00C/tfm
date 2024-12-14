import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import cv2
import pybulletX as px
import tacto
import hydra
from omegaconf import OmegaConf
from pybullet_env_classes.robot import UR5Robotiq85  # Import the robot class

class RopeRobotSimulation:
    def __init__(self, gui=True, cfg=None):
        self.gui = gui
        self.num_segments = 30
        self.segment_length = 0.03
        self.segment_radius = 0.01
        self.mass = 0.1
        self.friction = 0.5
        self.start_position = [0.4, 0, 1]

        self.rope_segments = []
        self.constraints = []
        self.digits = None
        self.digit_body = None
        self.obj = None
        self.panel = None

        # If cfg not provided, load via Hydra for demonstration
        if cfg is None:
            # Manually compose a config using Hydra
            # Normally you'd call @hydra.main in a separate script
            # But here we load it manually:
            hydra.initialize(config_path="conf")
            cfg = hydra.compose(config_name="digit")

        self.cfg = cfg

        self._connect_to_pybullet()

        self._load_environment()
        self._load_robot()
        self._load_rope()
        self._load_digit_sensor()  # Load the DIGIT sensor after other parts are ready

        self._adjust_camera()

    def _connect_to_pybullet(self):
        # Directly initialize pybulletX, which also sets up PyBullet
        px.init()  # This will connect to PyBullet GUI by default
        
        # Now configure PyBullet after px.init() is done
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


    def _load_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def _load_robot(self):
        self.robot = UR5Robotiq85(pos=[0,0,0], ori=[0,0,0])
        self.robot.step_simulation = self._step_simulation
        self.robot.load()
        self.robot.reset()

    def _load_rope(self):
        for i in range(self.num_segments):
            segment_position = [
                self.start_position[0],
                self.start_position[1],
                self.start_position[2] - i * self.segment_length
            ]

            segment_id = p.createCollisionShape(p.GEOM_CAPSULE, 
                                                radius=self.segment_radius, 
                                                height=self.segment_length)
            visual_id = p.createVisualShape(p.GEOM_CAPSULE, 
                                            radius=self.segment_radius, 
                                            length=self.segment_length,
                                            rgbaColor=[0,1,0,1])
            body_id = p.createMultiBody(baseMass=self.mass, 
                                        baseCollisionShapeIndex=segment_id,
                                        baseVisualShapeIndex=visual_id, 
                                        basePosition=segment_position)

            p.changeDynamics(body_id, -1, lateralFriction=self.friction)
            self.rope_segments.append(body_id)

            if i > 0:
                p.setCollisionFilterPair(self.rope_segments[i - 1], self.rope_segments[i], -1, -1, enableCollision=0)
                c = p.createConstraint(
                    parentBodyUniqueId=self.rope_segments[i - 1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=self.rope_segments[i],
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=[0,0,0],
                    parentFramePosition=[0,0,-self.segment_length/2],
                    childFramePosition=[0,0,self.segment_length/2],
                )
                self.constraints.append(c)

        # Create anchor
        anchor_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
        anchor_body = p.createMultiBody(baseMass=0, 
                                        baseCollisionShapeIndex=anchor_shape, 
                                        basePosition=self.start_position)
        p.createConstraint(
            parentBodyUniqueId=anchor_body,
            parentLinkIndex=-1,
            childBodyUniqueId=self.rope_segments[0],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0],
            childFramePosition=[0,0,0]
        )

    def _load_digit_sensor(self):
        # Load background image for DIGIT
        bg = cv2.imread("conf/bg_digit_240_320.jpg")

        # Initialize DIGIT sensor from cfg
        self.digits = tacto.Sensor(**self.cfg.tacto, background=bg)

        # Create and initialize DIGIT
        self.digit_body = px.Body(**self.cfg.digit)
        self.digits.add_camera(self.digit_body.id, [-1])

        # # Add object to pybullet and tacto
        # self.obj = px.Body(**self.cfg.object)
        # self.digits.add_body(self.obj)

        # # Create control panel for object
        # self.panel = px.gui.PoseControlPanel(self.obj, **self.cfg.object_control_panel)
        # self.panel.start()

    def _step_simulation(self):
        p.stepSimulation()
        if self.digits is not None:
            color, depth = self.digits.render()
            self.digits.updateGUI(color, depth)

    def _adjust_camera(self):
        p.resetDebugVisualizerCamera(**self.cfg.pybullet_camera)

    def apply_stiffness_and_damping(self):
        stiffness = 0.2
        damping = 0.1
        for i in range(1, self.num_segments):
            parent_id = self.rope_segments[i - 1]
            child_id = self.rope_segments[i]

            parent_pos,_ = p.getBasePositionAndOrientation(parent_id)
            child_pos,_ = p.getBasePositionAndOrientation(child_id)
            displacement = np.array(child_pos) - np.array(parent_pos)
            restoring_force = -stiffness * displacement

            parent_vel,_ = p.getBaseVelocity(parent_id)
            child_vel,_ = p.getBaseVelocity(child_id)
            relative_velocity = np.array(child_vel) - np.array(parent_vel)
            damping_force = -damping * relative_velocity

            total_force = restoring_force + damping_force
            p.applyExternalForce(child_id, -1, forceObj=total_force.tolist(), posObj=child_pos, flags=p.WORLD_FRAME)
            p.applyExternalForce(parent_id, -1, forceObj=(-total_force).tolist(), posObj=parent_pos, flags=p.WORLD_FRAME)

    def apply_action(self, action):
        # (x,y,z,roll,pitch,yaw) or joint actions
        if len(action) == 6:
            self.robot.move_ee(action, control_method='end')
        else:
            self.robot.move_ee(action, control_method='joint')

    def move_gripper(self, fraction):
        gripper_range = self.robot.gripper_range[1] - self.robot.gripper_range[0]
        gripper_pos = self.robot.gripper_range[0] + fraction * gripper_range
        self.robot.move_gripper(gripper_pos)

    def get_observation(self):
        obs = self.robot.get_joint_obs()
        return obs

    def reset_simulation(self):
        p.resetSimulation()
        px.reset()
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_environment()
        self._load_robot()
        self._load_rope()
        self._load_digit_sensor()
        self._adjust_camera()
        return self.get_observation()

    def close(self):
        if self.panel is not None:
            self.panel.stop()
        p.disconnect()


if __name__ == "__main__":
    # Just run the simulation as a demonstration
    # Hydra config is automatically loaded by `@hydra.main` in the original snippet,
    # here we are loading it manually in __init__.
    
    # Initialize sim
    sim = RopeRobotSimulation(gui=True)

    # Move the robot's EE slightly
    sim.apply_action((0.4, 0.0, 0.5, 0, 0, 0))
    for _ in range(100):
        sim._step_simulation()
        time.sleep(1/240)

    # Open gripper slightly
    sim.move_gripper(0.2)
    for _ in range(100):
        sim._step_simulation()
        time.sleep(1/240)

    # Run rope simulation with stiffness and damping
    for step in range(2400):  # ~10 seconds at 240Hz
        sim.apply_stiffness_and_damping()
        sim._step_simulation()
        time.sleep(1.0/240.0)

    sim.close()
