##################################
#SENSOR IS NOT BEING LOADED AND URDF BODIES ARE NOT GIVEN TO THE SENSOR IN THE LOAD ROPE!!!
##################################
import random
import time
import numpy as np
import sys
import os
import math
import pybullet
import pybullet_data
import pybulletX as px
import matplotlib.pyplot as plt

import tacto
import hydra
import cv2

import pytouch
from pytouch.handlers import ImageHandler
from pytouch.sensors import DigitSensor
from pytouch.tasks import ContactArea
from pytouch.tasks import TouchDetect
from PIL import Image

from datetime import datetime
from collections import namedtuple
from attrdict import AttrDict

import gymnasium as gym
from gymnasium import spaces

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the URDF file
ROBOT_URDF_PATH = os.path.join(script_dir, "robots", "urdf", "ur5e_with_gripper_digit.urdf")


class UR5Sim:
    def __init__(self,
                 useIK=True,
                 renders=True,
                 maxSteps=1000,
                 cfg=None):

        self.renders = renders
        self.useIK = useIK
        self.maxSteps = maxSteps

        # ---------------------------
        # 1) Connect to PyBullet
        # ---------------------------
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-9.81)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5,
                                            cameraYaw=60,
                                            cameraPitch=-30,
                                            cameraTargetPosition=[0.3,0,0])
        # Initialize pybulletX
        px.init(mode=pybullet.DIRECT)

        # ---------------------------
        # 2) Handle Hydra Config
        # ---------------------------
        if cfg is None:
            # If no config is passed, load an example config
            hydra.initialize(config_path="conf")
            cfg = hydra.compose(config_name="digit")
        self.cfg = cfg

        # ---------------------------
        # 3) Load the Robot
        # ---------------------------
        self.gripper_main_control_joint_name = "robotiq_85_left_knuckle_joint"
        self.mimic_joint_name = [
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint"
        ]
        self.mimic_multiplier = [1, 1, 1, -1, -1]

        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(
            ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags
        )
        self.num_joints = pybullet.getNumJoints(self.ur5)

        self.control_joints = ["shoulder_pan_joint",
                               "shoulder_lift_joint",
                               "elbow_joint",
                               "wrist_1_joint",
                               "wrist_2_joint",
                               "wrist_3_joint"]
        self.joint_type_list = [
            "REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"
        ]
        self.joint_info = namedtuple("jointInfo",
                                     ["id",
                                      "name",
                                      "type",
                                      "lowerLimit",
                                      "upperLimit",
                                      "maxForce",
                                      "maxVelocity",
                                      "controllable"])

        self.joints = AttrDict()
        for i in range(1, self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (
                True if (jointName in self.control_joints or
                         jointName in self.mimic_joint_name)
                else False
            )
            info = self.joint_info(jointID,
                                   jointName,
                                   jointType,
                                   jointLowerLimit,
                                   jointUpperLimit,
                                   jointMaxForce,
                                   jointMaxVelocity,
                                   controllable)
            if info.type == "REVOLUTE" and info.controllable:
                pybullet.setJointMotorControl2(
                    self.ur5, info.id,
                    pybullet.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=info.maxForce
                )
            self.joints[info.name] = info

        self.end_effector_index = 7
        self.ur5_or = [0.0, math.pi/2, 0.0]
        self.stepCounter = 0
        self.ur5_joint_ids = [1,2,3,4,5,6]
        
        #Load DIGIT sensor
        self.digits = None
        self.digit_body = None
        self._initialize_tacto_sensor_in_urdf()

        if self.digits is not None:
            self.color, self.depth = self.digits.render()
            self.digits.updateGUI(self.color, self.depth)
        #Create visual goal in simulation
        #self.add_visual_goal(goal_position)

        # Reset Robot to Default Pose and Load Rope
        self.reset()

        # Optional: Print joints for debugging
        # for i in range(pybullet.getNumJoints(self.ur5)):
        #     joint_info = pybullet.getJointInfo(self.ur5, i)
        #     print("Joint index:", i,
        #           "Name:", joint_info[1].decode(),
        #           "Type:", joint_info[2])


        

    def _initialize_tacto_sensor_in_urdf(self):
        # 1) Initialize TACTO with your config
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the URDF file
        conf_dir = os.path.join(script_dir, "conf", "bg_digit_240_320.jpg")
        bg = cv2.imread(conf_dir)  # If you have a background image
        self.digits = tacto.Sensor(**self.cfg.tacto, background=bg, visualize_gui=False)

        # 2) Identify the link index in your newly updated robot URDF
        #    Suppose you named the sensor link "digit_link" in your URDF <link name="digit_link">
        digit_link_name = "sensor_gripper_joint"  # or whatever you used
        digit_link_index = None

        # If you stored your link info in self.joints, find it:
        for joint_info in self.joints.values():
            if joint_info.name == digit_link_name:
                digit_link_index = joint_info.id
                break

        if digit_link_index is None:
            raise ValueError(f"Could not find link named {digit_link_name} in the robot's URDF")

        # 3) Let TACTO treat that link as the camera link
        #    The first argument is the “bodyUniqueId” for the robot,
        #    second argument is a list of link indices that have digit sensors.
        self.digits.add_camera(self.ur5, [digit_link_index])

        #print("DIGIT sensor is integrated into the robot URDF and recognized by TACTO!")


    def _load_digit_sensor(self):
        # 1) Load background for DIGIT
        bg = cv2.imread("conf/bg_digit_240_320.jpg")

        # 2) Initialize Tacto sensor with config
        self.digits = tacto.Sensor(**self.cfg.tacto, background=bg)

        # 3) Create the DIGIT sensor body (via pybulletX)
        self.digit_body = px.Body(**self.cfg.digit)
        print(f"Digit body ID: {self.digit_body.id}")

        # 4) Link the camera to the new body
        self.digits.add_camera(self.digit_body.id, [-1])

        # 5) Attach the DIGIT to a chosen link
        attach_link_name = "robotiq_85_left_finger_tip_joint"
        attach_link_id = self.joints[attach_link_name].id
        print(f"Attaching DIGIT to link ID: {attach_link_id}")

        # Example using JOINT_POINT2POINT with an offset:
        parent_pos, parent_orn = pybullet.getLinkState(self.ur5, attach_link_id)[:2]
        child_pos, child_orn = pybullet.getBasePositionAndOrientation(self.digit_body.id)
        print(f"Parent pos: {parent_pos}")
        print(f"Child pos: {child_pos}")
        pybullet.createConstraint(
            parentBodyUniqueId=self.ur5,
            parentLinkIndex=attach_link_id,
            childBodyUniqueId=self.digit_body.id,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0,0,-0.1],
            parentFrameOrientation=parent_orn,
            childFramePosition=[0,0,0],
            childFrameOrientation=[0,0,0,1]
        )

        # 6) Disable collisions between the robot and DIGIT
        robot_num_joints = pybullet.getNumJoints(self.ur5)
        digit_num_joints = pybullet.getNumJoints(self.digit_body.id)
        for r_link in range(-1, robot_num_joints):
            for d_link in range(-1, digit_num_joints):
                pybullet.setCollisionFilterPair(self.ur5,
                                                self.digit_body.id,
                                                r_link,
                                                d_link,
                                                enableCollision=0)
        print("DIGIT sensor attached and collisions disabled.")

    def _load_rope_mujoco(self):
        mjcf_path = os.path.join(os.getcwd(), "objects/rope.xml")
        body_ids = pybullet.loadMJCF(mjcf_path)  
        print(f"Body:{body_ids}")
        #Remove joints
        for rope_id in body_ids:
            num_joints = pybullet.getNumJoints(rope_id)
            print(f"Num_joints rope:{num_joints}")
            print(f"Body id:{rope_id}")
            for joint_index in range(num_joints):
                # Get the joint info
                joint_info = pybullet.getJointInfo(rope_id, joint_index)
                print(f"Removing joint: {joint_info[1]} at index {joint_index}")
                # You can programmatically "remove" joints by disabling them
                pybullet.resetJointState(rope_id, joint_index, targetValue=0)

        segment_length = 0.0575625  # Total length of a segment
        half_length = segment_length / 2
        
        for i in range(len(body_ids) - 1):
            parent_id = body_ids[i]
            child_id = body_ids[i + 1]
            
            # Create a spherical joint between parent and child
            pybullet.createConstraint(
                parentBodyUniqueId=parent_id,
                parentLinkIndex=-1,  # Base of the parent
                childBodyUniqueId=child_id,
                childLinkIndex=-1,  # Base of the child
                jointType=pybullet.JOINT_SPHERICAL,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, -half_length],  # Bottom of the parent
                childFramePosition=[0, 0, half_length]     # Top of the child
            )


        #for rid in body_ids:
        #    self.digits.add_object("objects/rope_dummy.urdf", rid, globalScaling=1.0)    

    def _load_rope_urdf(self):
        self.rope = px.Body(**self.cfg.rope)
        print(f"Rope body ID: {self.rope.id}")
        print(f"Rope attributes: {dir(self.rope)}")
        
        
        #print(f"Rope configuration: {self.cfg.rope}")

        # 2) Add sphere to the Tacto sensor to detect collisions
        if self.digits is not None:
            self.digits.add_body(self.rope)
        else:
            print("Digits is NONE")

        for joint_index in range(pybullet.getNumJoints(self.rope.id)):
            pybullet.changeDynamics(self.rope.id, joint_index, jointDamping=0)
            pybullet.changeDynamics(self.rope.id, joint_index, lateralFriction=0.1)

            pybullet.setJointMotorControl2(
                bodyUniqueId=self.rope.id,
                jointIndex=joint_index,
                controlMode=pybullet.VELOCITY_CONTROL,
                force=0
            )


    # def _load_rope(self):
    #     # Parameters MUST BE CHANGED IN THE DUMMY URDF FOR TACTO
    #     self.num_segments = 10
    #     self.total_length = 0.6   

    #     # Segment properties based on the number of segments
    #     self.segment_radius = 0.02
    #     self.segment_length = self.total_length / self.num_segments
    #     self.mass = 0.1
    #     self.friction = 0.5
    #     self.start_position = [0.60, 0.135, 0.9]
    #     self.rope_segments = []
    #     self.constraints = []

    #     for i in range(self.num_segments):
    #         segment_position = [
    #             self.start_position[0],
    #             self.start_position[1],
    #             self.start_position[2] - i * self.segment_length
    #         ]

    #         segment_id = pybullet.createCollisionShape(pybullet.GEOM_CAPSULE, 
    #                                             radius=self.segment_radius, 
    #                                             height=self.segment_length)
    #         visual_id = pybullet.createVisualShape(pybullet.GEOM_CAPSULE, 
    #                                         radius=self.segment_radius, 
    #                                         length=self.segment_length,
    #                                         rgbaColor=[0,1,0,1])
    #         body_id = pybullet.createMultiBody(baseMass=self.mass, 
    #                                     baseCollisionShapeIndex=segment_id,
    #                                     baseVisualShapeIndex=visual_id, 
    #                                     basePosition=segment_position)

    #         pybullet.changeDynamics(body_id, -1, lateralFriction=self.friction)
    #         self.rope_segments.append(body_id)

    #         # Add link to DIGIT sensor 
    #         script_dir = os.path.dirname(os.path.abspath(__file__))
    #         dummy_dir = os.path.join(script_dir, "objects", "rope_dummy.urdf")
    #         self.digits.add_object(dummy_dir, body_id, globalScaling=1.0)    

    #         if i > 0:
    #             pybullet.setCollisionFilterPair(self.rope_segments[i - 1], self.rope_segments[i], -1, -1, enableCollision=0)
    #             c = pybullet.createConstraint(
    #                 parentBodyUniqueId=self.rope_segments[i - 1],
    #                 parentLinkIndex=-1,
    #                 childBodyUniqueId=self.rope_segments[i],
    #                 childLinkIndex=-1,
    #                 jointType=pybullet.JOINT_POINT2POINT,
    #                 jointAxis=[0,0,0],
    #                 parentFramePosition=[0,0,-self.segment_length/2],
    #                 childFramePosition=[0,0,self.segment_length/2],
    #             )
    #             self.constraints.append(c)

    #     # Create anchor
    #     anchor_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.001)
    #     anchor_body = pybullet.createMultiBody(baseMass=0, 
    #                                     baseCollisionShapeIndex=anchor_shape, 
    #                                     basePosition=self.start_position)
    #     pybullet.setCollisionFilterPair(anchor_body, self.rope_segments[0], -1, -1, enableCollision=0)
    #     pybullet.createConstraint(
    #         parentBodyUniqueId=anchor_body,
    #         parentLinkIndex=-1,
    #         childBodyUniqueId=self.rope_segments[0],
    #         childLinkIndex=-1,
    #         jointType=pybullet.JOINT_FIXED,
    #         jointAxis=[0,0,0],
    #         parentFramePosition=[0,0,0],
    #         childFramePosition=[0,0,0]
    #     )

    def _load_rope(self):
        # Parameters
        self.num_segments = 10
        self.total_length = 0.6
        self.segment_radius = 0.02
        self.segment_length = self.total_length / self.num_segments
        
        # Zero mass ensures no movement due to physics
        self.mass = 0.0
        self.friction = 0.5
        self.start_position = [0.60, 0.135, 0.9]
        self.rope_segments = []
        self.constraints = []

        for i in range(self.num_segments):
            segment_position = [
                self.start_position[0],
                self.start_position[1],
                self.start_position[2] - i * self.segment_length
            ]

            segment_id = pybullet.createCollisionShape(pybullet.GEOM_CAPSULE, 
                                                    radius=self.segment_radius, 
                                                    height=self.segment_length)
            visual_id = pybullet.createVisualShape(pybullet.GEOM_CAPSULE, 
                                                radius=self.segment_radius, 
                                                length=self.segment_length,
                                                rgbaColor=[0,1,0,1])
            # All segments have zero mass
            body_id = pybullet.createMultiBody(baseMass=self.mass, 
                                            baseCollisionShapeIndex=segment_id,
                                            baseVisualShapeIndex=visual_id, 
                                            basePosition=segment_position)

            pybullet.changeDynamics(body_id, -1, lateralFriction=self.friction)
            self.rope_segments.append(body_id)

            # Add link to DIGIT sensor
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dummy_dir = os.path.join(script_dir, "objects", "rope_dummy.urdf")
            self.digits.add_object(dummy_dir, body_id, globalScaling=1.0)    

            if i > 0:
                # Disable collisions between adjacent segments if desired
                pybullet.setCollisionFilterPair(self.rope_segments[i - 1],
                                                self.rope_segments[i],
                                                -1, -1,
                                                enableCollision=0)

                # Use a fixed joint instead of a point-to-point joint
                c = pybullet.createConstraint(
                    parentBodyUniqueId=self.rope_segments[i - 1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=self.rope_segments[i],
                    childLinkIndex=-1,
                    jointType=pybullet.JOINT_FIXED,  # Fixes them relative to each other
                    jointAxis=[0,0,0],
                    parentFramePosition=[0,0,-self.segment_length/2],
                    childFramePosition=[0,0,self.segment_length/2],
                )
                self.constraints.append(c)

        # Create anchor with zero mass
        anchor_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.001)
        anchor_body = pybullet.createMultiBody(baseMass=0, 
                                            baseCollisionShapeIndex=anchor_shape, 
                                            basePosition=self.start_position)
        pybullet.setCollisionFilterPair(anchor_body, self.rope_segments[0], -1, -1, enableCollision=0)
        
        # Attach the first segment to the anchor with a fixed joint
        pybullet.createConstraint(
            parentBodyUniqueId=anchor_body,
            parentLinkIndex=-1,
            childBodyUniqueId=self.rope_segments[0],
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0],
            childFramePosition=[0,0,0]
        )


    def _load_sphere(self):
        self.sphere = px.Body(**self.cfg.object)
        print(f"Sphere body ID: {self.sphere.id}")

        if hasattr(self.sphere, 'geometry'):
            print(f"Sphere geometry: {self.sphere.geometry}")
        else:
            print("Sphere has no geometry")

        # 2) Add sphere to the Tacto sensor to detect collisions
        if self.digits is not None:
            self.digits.add_body(self.sphere)

        # 3) (Optional) If you want a UI slider panel to move the sphere
        #    around for easy contact testing:
        if "object_control_panel" in self.cfg:
            self.panel_sphere = px.gui.PoseControlPanel(self.sphere, **self.cfg.object_control_panel)
            self.panel_sphere.start()

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5,
            indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=poses,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )

    def control_gripper(self, gripper_opening_angle):
        pybullet.setJointMotorControl2(
            self.ur5,
            self.joints[self.gripper_main_control_joint_name].id,
            pybullet.POSITION_CONTROL,
            targetPosition=gripper_opening_angle,
            force=self.joints[self.gripper_main_control_joint_name].maxForce,
            maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity
        )
        for i, mimic_name in enumerate(self.mimic_joint_name):
            joint = self.joints[mimic_name]
            factor = self.mimic_multiplier[i]
            pybullet.setJointMotorControl2(
                self.ur5,
                joint.id,
                pybullet.POSITION_CONTROL,
                targetPosition=gripper_opening_angle * factor,
                force=joint.maxForce,
                maxVelocity=joint.maxVelocity
            )

    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [ math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [-0.34, -1.57, 1.80, -1.57, -1.57, 0.00]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5,
            self.end_effector_index,
            position,
            quaternion,
            jointDamping=[0.01]*self.num_joints,
            upperLimits=upper_limits,
            lowerLimits=lower_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        return joint_angles

    def add_visual_goal(self, pose):
        """
        Adds a visual-only sphere to the simulation as a goal marker.

        :param position: List or tuple with 3 elements [x, y, z]
        :param radius: Radius of the sphere
        :param color: RGBA color of the sphere [R, G, B, A]
        """
        position = pose[:3]
        radius=0.01
        color=[0, 1, 0, 1]
        visual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        body_id = pybullet.createMultiBody(
            baseMass=0,  # No mass, purely visual
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        return body_id

    def get_end_eff_pose(self): #whole pose including orientation (euler)
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        orientation_euler = pybullet.getEulerFromQuaternion(orientation)
        ee_pose = np.concatenate((
            np.array(position),
            np.array(orientation_euler),
            ),axis=0)
        return ee_pose
    
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return np.array(position)
    
    def get_end_eff_vel(self):
        return self.end_effector_vel    

    def get_joint_angles(self):
        joint_states = pybullet.getJointStates(self.ur5, self.ur5_joint_ids)
        joint_positions = [state[0] for state in joint_states]
        return joint_positions
    
    def get_joint_velocities(self):
        joint_states = pybullet.getJointStates(self.ur5, self.ur5_joint_ids)
        joint_velocities = [state[1] for state in joint_states]
        return joint_velocities
    
    def get_last_rope_link_position(self):
        
        # Get the ID of the last rope segment
        last_segment_id = self.rope_segments[-1]
        
        # Get the position of the last rope segment
        position, _ = pybullet.getBasePositionAndOrientation(last_segment_id)
        return np.array(position, dtype=np.float32)
    
    # def get_sensor_reading(self): 
        
    #     # Elipse calculation
    #     color_imgs, depth_imgs = self.digits.render()
    #     sample_img = color_imgs[0]  
    #     #ImageHandler.save("digit_base_image.png", color_imgs[0])
    #             # Visualize the color image
    #     # plt.imshow(color_imgs[0])
    #     # plt.title("Sensor Output")
    #     # plt.show()
    #     # plt.close()  # Close the plot to free resources

    #     #Upload base image
    #     # script_dir = os.path.dirname(os.path.abspath(__file__))
    #     # base_img_path = os.path.join(script_dir, "objects", "digit_base_image.png")
    #     # base_img = ImageHandler(base_img_path).nparray
        
    #     # #Sensor touch inference 
    #     # color_img_pil = Image.fromarray((color_imgs[0] * 255).astype(np.uint8))
    #     # print(color_img_pil)
    #     # pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
    #     # is_touching, certainty = pt.TouchDetect(color_img_pil)
    #     # print(f"Sensor Touching:{is_touching}")

        
    #     # diff_img = sample_img - base_img
    #     # print(f"Shape diff-img:{diff_img.shape}")

    #     # #Elipse computation
    #     #pt = pytouch.PyTouch(DigitSensor, tasks=[ContactArea])
    #     # contact_area = ContactArea(base=base_img,contour_threshold=100,draw_poly=False)
    #     # major, minor = contact_area(sample_img)
    #     # print("Major Axis: {0}, minor axis: {1}".format(*major, *minor))

        
    #     # 2) Assume a single DIGIT sensor or pick sensor index 0
    #     depth_img = depth_imgs[0]
        
    #     # 3) Convert depth to 'pressure' map
    #     #    The constant offset (0.022) is an example from the original TACTO DIGIT renderer,
    #     #    but you can calibrate or adjust to taste.
    #     pressure_img = 0.022 - depth_img
        
    #     # 4) Scale and convert to 8-bit
    #     min_val, max_val = pressure_img.min(), pressure_img.max()
    #     if max_val > min_val:
    #         pressure_img = (pressure_img - min_val) / (max_val - min_val)
    #     else:
    #         pressure_img[:] = 0  # Degenerate case if there's no variation
        
    #     # Now pressure_img lies in [0, 1]. Scale it to [0, 255].
    #     #pressure_img = (pressure_img * 255.0).astype(np.uint8)
        
    #     # 5) Return or save this grayscale image
    #     return pressure_img


    def get_sensor_reading(self): 
            
            # Elipse calculation
            #color_imgs, depth_imgs = self.digits.render()
            sample_img = self.color[0]  
            #ImageHandler.save("digit_base_image.png", color_imgs[0])
                    # Visualize the color image
            # plt.imshow(color_imgs[0])
            # plt.title("Sensor Output")
            # plt.show()
            # plt.close()  # Close the plot to free resources

            #Upload base image
            # script_dir = os.path.dirname(os.path.abspath(__file__))
            # base_img_path = os.path.join(script_dir, "objects", "digit_base_image.png")
            # base_img = ImageHandler(base_img_path).nparray
            
            #Sensor touch inference 
            color_img_pil = Image.fromarray((self.color[0] * 255).astype(np.uint8))
            #print(color_img_pil)
            pt = pytouch.PyTouch(DigitSensor, tasks=[TouchDetect])
            #print("Digit sensor object:")
            #print(DigitSensor)
            is_touching, certainty = pt.TouchDetect(color_img_pil)
            print(f"Sensor Touching:{is_touching}")
            print(f"Sensor Certainty:{certainty}")
            return is_touching,certainty

    def reset(self):
        self.stepCounter = 0
        joint_angles = (0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, 0)#(0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
        self.set_joint_angles(joint_angles)
        self.end_effector_vel = np.zeros(6)
        for i in range(100):
            pybullet.stepSimulation()

        # Reset rope: remove old segments and constraints
        if hasattr(self, 'rope_segments'):
            for seg_id in self.rope_segments:
                pybullet.removeBody(seg_id)
            self.rope_segments = []
        if hasattr(self, 'constraints'):
            self.constraints = []

        # 3) Reload rope from scratch
        self._load_rope()
        
        #Close gripper
        self.control_gripper(0.7)
        for i in range(100):
            pybullet.stepSimulation()

    def step(self, end_effector_velocity, gripper_cmd):
        self.end_effector_vel = end_effector_velocity
        active_joint_indices = []
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            joint_type = info[2]
            if joint_type in [pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC]:
                active_joint_indices.append(i)

        joint_states = pybullet.getJointStates(self.ur5, active_joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_accelerations = [0.0]*len(joint_positions)

        linear_jacobian, angular_jacobian = pybullet.calculateJacobian(
            self.ur5,
            self.end_effector_index,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=joint_accelerations
        )
        J_lin_full = np.array(linear_jacobian)
        J_ang_full = np.array(angular_jacobian)
        J_full = np.vstack((J_lin_full, J_ang_full))
        q_dot = np.linalg.pinv(J_full) @ end_effector_velocity

        for i, name in enumerate(self.control_joints):
            joint_info = self.joints[name]
            max_vel = joint_info.maxVelocity
            q_dot[i] = np.clip(q_dot[i], -max_vel, max_vel)

        for i, name in enumerate(self.control_joints):
            joint_info = self.joints[name]
            pybullet.setJointMotorControl2(
                bodyUniqueId=self.ur5,
                jointIndex=joint_info.id,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocity=q_dot[i],
                force=joint_info.maxForce
            )

        #gripper_action = np.clip(gripper_cmd * 0.4, -0.4, 0.4)
        #self.control_gripper(gripper_action)
        self.control_gripper(gripper_cmd)

        pybullet.stepSimulation()
        self.stepCounter += 1
        time.sleep(1./240.)

        if self.digits is not None:
            self.color, self.depth = self.digits.render()
            self.digits.updateGUI(self.color, self.depth)

    def close(self):
        pybullet.disconnect()
