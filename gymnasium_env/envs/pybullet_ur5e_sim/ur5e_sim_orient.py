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
import random
import math

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
ROBOT_URDF_PATH = os.path.join(script_dir, "robots", "urdf", "ur5e.urdf")


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
        pybullet.setRealTimeSimulation(False) # We need stepSimulation command
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

        print(f"[INFO]: URDF file path: {ROBOT_URDF_PATH}")
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
        

        # Reset Robot to Default Pose and Load Rope
        self.reset()

        # Optional: Print joints for debugging
        # for i in range(pybullet.getNumJoints(self.ur5)):
        #     joint_info = pybullet.getJointInfo(self.ur5, i)
        #     print("Joint index:", i,
        #           "Name:", joint_info[1].decode(),
        #           "Type:", joint_info[2])


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


    def _load_rope_urdf(self):
        self.rope = px.Body(**self.cfg.rope)
        print(f"Rope body ID: {self.rope.id}")
        print(f"Rope attributes: {dir(self.rope)}")
        
        
        #print(f"Rope configuration: {self.cfg.rope}")

        for joint_index in range(pybullet.getNumJoints(self.rope.id)):
            pybullet.changeDynamics(self.rope.id, joint_index, jointDamping=0)
            pybullet.changeDynamics(self.rope.id, joint_index, lateralFriction=0.1)

            pybullet.setJointMotorControl2(
                bodyUniqueId=self.rope.id,
                jointIndex=joint_index,
                controlMode=pybullet.VELOCITY_CONTROL,
                force=0
            )


    def _load_rope(self):
        # Parameters MUST BE CHANGED IN THE DUMMY URDF FOR TACTO
        self.num_segments = 10
        self.total_length = 0.6   

        # Segment properties based on the number of segments
        self.segment_radius = 0.02
        self.segment_length = self.total_length / self.num_segments
        self.mass = 0.1
        self.friction = 0.5
        self.start_position = [0.60, 0.135, 1]
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
            body_id = pybullet.createMultiBody(baseMass=self.mass, 
                                        baseCollisionShapeIndex=segment_id,
                                        baseVisualShapeIndex=visual_id, 
                                        basePosition=segment_position)

            pybullet.changeDynamics(body_id, -1, lateralFriction=self.friction)
            self.rope_segments.append(body_id)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the absolute path to the URDF file
            dummy_dir = os.path.join(script_dir, "objects", "rope_dummy.urdf")

            if i > 0:
                pybullet.setCollisionFilterPair(self.rope_segments[i - 1], self.rope_segments[i], -1, -1, enableCollision=0)
                c = pybullet.createConstraint(
                    parentBodyUniqueId=self.rope_segments[i - 1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=self.rope_segments[i],
                    childLinkIndex=-1,
                    jointType=pybullet.JOINT_POINT2POINT,
                    jointAxis=[0,0,0],
                    parentFramePosition=[0,0,-self.segment_length/2],
                    childFramePosition=[0,0,self.segment_length/2],
                )
                self.constraints.append(c)

        # Create anchor
        anchor_shape = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=0.001)
        anchor_body = pybullet.createMultiBody(baseMass=0, 
                                        baseCollisionShapeIndex=anchor_shape, 
                                        basePosition=self.start_position)
        pybullet.setCollisionFilterPair(anchor_body, self.rope_segments[0], -1, -1, enableCollision=0)
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
        orientation_euler = pose[3:]
        orientation_quat = pybullet.getQuaternionFromEuler(orientation_euler)
        print(f"Orientation_quat= {orientation_quat}")

        color=[0, 1, 0, 1]
        visual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=[0.01, 0.01, 0.01],
            rgbaColor=color
        )
        body_id = pybullet.createMultiBody(
            baseMass=0,  # No mass, purely visual
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation_quat
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
    
    def get_end_eff_position(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return np.array(position)
    
    def get_end_eff_vel(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True, computeLinkVelocity=1)
        cartesian_vel, angular_vel_euler = linkstate[6], linkstate[7] # no transformation from quaternion is needed as it already gives cartesian euler angles 
        ee_vel = np.concatenate((
            np.array(cartesian_vel),
            np.array(angular_vel_euler),
            ),axis=0)
        return ee_vel

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

    def random_point_in_sphere(self,radius, center):
        """
        Generates a random 3D point inside a sphere of a given radius centered at a given point.

        Args:
            radius (float): Radius of the sphere.
            center (tuple): Coordinates of the sphere's center (x, y, z).

        Returns:
            tuple: Random point (x, y, z) inside the sphere.
        """
        while True:
            # Generate a random point in the cube that bounds the sphere
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            z = random.uniform(-radius, radius)

            # Check if the point is inside the sphere
            if x**2 + y**2 + z**2 <= radius**2:
                return (center[0] + x, center[1] + y, center[2] + z)

    def goal_step_sim(self):
        for i in range(200):
                pybullet.stepSimulation()

    def reset(self):

        self.stepCounter = 0
        joint_angles = (0, -math.pi/2, math.pi/2, math.pi, -math.pi/2, 0)#(0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
        # dist_pose_rchd = 1
        # while dist_pose_rchd > 0.02:
        #     roll = random.uniform(-math.pi, math.pi)
        #     pitch = random.uniform(-math.pi, math.pi)
        #     yaw = random.uniform(-math.pi, math.pi)
        #     radius = 0.1
        #     center = [0.4,0,0.4]
        #     rand_or = (roll,pitch,yaw)
        #     rand_pos = self.random_point_in_sphere(radius,center)
        #     target = np.array(rand_pos + rand_or,dtype=np.float32)
        #     joint_angles = self.calculate_ik(rand_pos,rand_or)
        #     self.set_joint_angles(joint_angles)
            
        #     for i in range(200):
        #         pybullet.stepSimulation()

        #     pose_rchd = self.get_end_eff_pose()
        #     dist_pose_rchd = np.linalg.norm(pose_rchd - target)
        self.set_joint_angles(joint_angles)
        for i in range(200):
            pybullet.stepSimulation()
        # Reset rope: remove old segments and constraints
        # if hasattr(self, 'rope_segments'):
        #     for seg_id in self.rope_segments:
        #         pybullet.removeBody(seg_id)
        #     self.rope_segments = []
        # if hasattr(self, 'constraints'):
        #     self.constraints = []

        # 3) Reload rope from scratch
        #self._load_rope()
        
        for i in range(100):
            pybullet.stepSimulation()


    def visualize_tcp(self, tcp_pose):
        """
        Visualize the TCP position and orientation in the simulation.

        Args:
            tcp_pose (list): A list containing position [x, y, z] and orientation [roll, pitch, yaw].
        """
        # Remove previous visualization if it exists
        if hasattr(self, 'tcp_visuals'):
            for visual_id in self.tcp_visuals:
                pybullet.removeUserDebugItem(visual_id)
        self.tcp_visuals = []

        position = tcp_pose[:3]
        orientation_euler = tcp_pose[3:]
        orientation_quat = pybullet.getQuaternionFromEuler(orientation_euler)

        # Add a sphere for the TCP position
        tcp_sphere_id = pybullet.addUserDebugText(
            text="TCP",
            textPosition=position,
            textColorRGB=[1, 0, 0],  # Red text for position label
            textSize=1.5
        )
        self.tcp_visuals.append(tcp_sphere_id)

        # Orientation visualization (arrows for axes)
        arrow_length = 0.1  # Length of the arrows
        arrow_colors = {
            "x": [1, 0, 0],  # Red for X-axis
            "y": [0, 1, 0],  # Green for Y-axis
            "z": [0, 0, 1]   # Blue for Z-axis
        }
        rotation_matrix = pybullet.getMatrixFromQuaternion(orientation_quat)

        # Extract the direction vectors for the axes
        x_axis = [rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]]
        y_axis = [rotation_matrix[3], rotation_matrix[4], rotation_matrix[5]]
        z_axis = [rotation_matrix[6], rotation_matrix[7], rotation_matrix[8]]

        # Add arrows for orientation
        for axis, direction in zip(["x", "y", "z"], [x_axis, y_axis, z_axis]):
            arrow_id = pybullet.addUserDebugLine(
                position,
                [position[0] + arrow_length * direction[0],
                 position[1] + arrow_length * direction[1],
                 position[2] + arrow_length * direction[2]],
                lineColorRGB=arrow_colors[axis],
                lineWidth=2
            )
            self.tcp_visuals.append(arrow_id)

    def update_tcp_visualization(self):
        """
        Updates the TCP visualization in the simulation.
        Call this method at each iteration to refresh the TCP visualization.
        """
        tcp_pose = self.get_end_eff_pose()
        self.visualize_tcp(tcp_pose)


    def step(self, end_effector_velocity):
        #self.end_effector_vel = end_effector_velocity
        #self.update_tcp_visualization()
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

        pybullet.stepSimulation()
        self.stepCounter += 1
        time.sleep(1./240.)

    def close(self):
        pybullet.disconnect()
