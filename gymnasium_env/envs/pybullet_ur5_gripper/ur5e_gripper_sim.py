import random
import time
import numpy as np
import sys
import os
import math
import pybullet
import pybullet_data
from datetime import datetime
from collections import namedtuple
from attrdict import AttrDict

import gymnasium as gym
from gymnasium import spaces

ROBOT_URDF_PATH = os.path.join(os.environ['HOME'], "tfm/gymnasium_env/envs/pybullet_ur5_gripper/robots/urdf/ur5e_with_gripper.urdf")

class UR5Sim:
    def __init__(self,
                 useIK=True,
                 renders=True,
                 maxSteps=1000):

        self.renders = renders
        self.useIK = useIK
        self.maxSteps = maxSteps

        # Connect to PyBullet
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)
        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0.3,0,0])

        # Gripper info
        self.gripper_main_control_joint_name = "robotiq_85_left_knuckle_joint"
        self.mimic_joint_name = [
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint"
        ]
        self.mimic_multiplier = [1, 1, 1, -1, -1]

        # Load robot only
        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = pybullet.getNumJoints(self.ur5)

        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if (jointName in self.control_joints or jointName in self.mimic_joint_name) else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info

        self.end_effector_index = 7
        self.ur5_or = [0.0, math.pi/2, 0.0]

        self._action_bound = 1.0
        self.stepCounter = 0
        self.reset()

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
            self.ur5, indexes,
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
            maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity)
        
        for i in range(len(self.mimic_joint_name)):
            joint = self.joints[self.mimic_joint_name[i]]
            pybullet.setJointMotorControl2(
                self.ur5, joint.id, pybullet.POSITION_CONTROL,
                targetPosition=gripper_opening_angle * self.mimic_multiplier[i],
                force=joint.maxForce,
                maxVelocity=joint.maxVelocity)

    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints

    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [ math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [-0.34, -1.57, 1.80, -1.57, -1.57, 0.00]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*self.num_joints, upperLimits=upper_limits,  #PREVIOUSLY instead of using self.num_joints there was a *6 multiplying
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (np.array(position), orientation)

    def reset(self):
        self.stepCounter = 0
        joint_angles = (1.5, -1.57, 1.80, -1.57, -1.57, 0)
        self.set_joint_angles(joint_angles)
        self.control_gripper(-0.4)  # Open gripper
        for i in range(100):
            pybullet.stepSimulation()

    def step(self, end_effector_velocity, gripper_cmd=-1.0):
        """
        end_effector_velocity: np.array([vx, vy, vz]) in world coordinates
        gripper_cmd: fixed here for simplicity
        """
        cur_p, _ = self.get_current_pose()
        new_p = cur_p + end_effector_velocity

        if self.useIK:
            joint_angles = self.calculate_ik(new_p, self.ur5_or)
        else:
            joint_angles = self.get_joint_angles()

        self.set_joint_angles(joint_angles)

        gripper_action = np.clip(gripper_cmd * 0.4, -0.4, 0.4)
        self.control_gripper(gripper_action)

        pybullet.stepSimulation()
        self.stepCounter += 1
        time.sleep(1./240.)

    def close(self):
        pybullet.disconnect()

