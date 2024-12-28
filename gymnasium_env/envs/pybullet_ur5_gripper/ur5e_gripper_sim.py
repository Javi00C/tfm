import random
import time
import numpy as np
import sys
import os
import math
import pybullet
import pybullet_data
import pybulletX as px

import tacto
import hydra
import cv2
from datetime import datetime
from collections import namedtuple
from attrdict import AttrDict

import gymnasium as gym
from gymnasium import spaces

ROBOT_URDF_PATH = "/robots/urdf/ur5e_with_gripper.urdf"

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
        pybullet.setGravity(0,0,-10)
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

        # ---------------------------
        # 4) Reset Robot to Default Pose
        # ---------------------------
        self.reset()

        # ---------------------------
        # 5) Load the DIGIT sensor
        # ---------------------------
        self.digits = None
        self.digit_body = None
        self._load_digit_sensor()

        # Optional: Print joints for debugging
        for i in range(pybullet.getNumJoints(self.ur5)):
            joint_info = pybullet.getJointInfo(self.ur5, i)
            print("Joint index:", i,
                  "Name:", joint_info[1].decode(),
                  "Type:", joint_info[2])

        # ---------------------------
        # 6) Load the sphere from config
        # ---------------------------
        #Loads a plane to act as floor
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = pybullet.loadURDF("plane.urdf", [0, 0, 0])


        #self._load_sphere()

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
        pybullet.createConstraint(
            parentBodyUniqueId=self.ur5,
            parentLinkIndex=attach_link_id,
            childBodyUniqueId=self.digit_body.id,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.01, -0.02, 0.0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=[0.0, 0.0, 0.0],
            childFrameOrientation=[0, 0, 0, 1]
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

    def _load_sphere(self):
        """
        Load the small sphere from the Hydra config and register it with Tacto.
        The config typically has:
            object:
              urdf_path: "objects/sphere_small.urdf"
              base_position: [...]
              global_scaling: ...
        """
        # 1) Create the sphere via px.Body
        #    This picks up base_position, global_scaling, etc. from cfg.object
        self.sphere = px.Body(**self.cfg.object)
        print(f"Sphere body ID: {self.sphere.id}")

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

    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (np.array(position), orientation)

    def reset(self):
        self.stepCounter = 0
        joint_angles = (0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
        self.set_joint_angles(joint_angles)
        self.control_gripper(-0.4)
        for i in range(100):
            pybullet.stepSimulation()

    def step(self, end_effector_velocity, gripper_cmd=-1.0):
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

        gripper_action = np.clip(gripper_cmd * 0.4, -0.4, 0.4)
        self.control_gripper(gripper_action)

        pybullet.stepSimulation()
        self.stepCounter += 1
        time.sleep(1./240.)

        if self.digits is not None:
            color, depth = self.digits.render()
            self.digits.updateGUI(color, depth)

    def close(self):
        pybullet.disconnect()
