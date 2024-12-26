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

# ROBOT_URDF_PATH = os.path.join(
#     os.environ['HOME'], 
#     "tfm/gymnasium_env/envs/pybullet_ur5_gripper/robots/urdf/ur5e_with_gripper.urdf"
# )
ROBOT_URDF_PATH = "/robots/urdf/ur5e_with_gripper.urdf"

class UR5Sim:
    def __init__(self,
                 useIK=True,           # You can keep or remove this flag.
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
        pybullet.resetDebugVisualizerCamera(cameraDistance=1.5,
                                            cameraYaw=60,
                                            cameraPitch=-30,
                                            cameraTargetPosition=[0.3,0,0])

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

        # Load robot
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
        for i in range(1,self.num_joints):
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
            print(info)
            # Initialize REVOLUTE joints in velocity control mode at 0 velocity
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

        self._action_bound = 1.0
        self.stepCounter = 0
        # for i in range(pybullet.getNumJoints(self.ur5)):
        #     joint_info = pybullet.getJointInfo(self.ur5, i)
        #     print("Joint index:", i,
        #     "Name:", joint_info[1].decode(),
        #     "Type:", joint_info[2])
        # for i in range(pybullet.getNumJoints(self.ur5)):
        #     joint_info = pybullet.getJointInfo(self.ur5, i)
        #     parent_idx = joint_info[16]
        #     child_idx  = joint_info[17]
        #     print(f"Joint {i} -> Parent link: {parent_idx}, Child link: {child_idx}, Type: {joint_info[2]}")
        self.reset()

    def set_joint_angles(self, joint_angles):
        """
        Retained for convenience if you occasionally want
        to do position control (e.g., a reset posture).
        """
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
        # Mimic other joints
        for i in range(len(self.mimic_joint_name)):
            joint = self.joints[self.mimic_joint_name[i]]
            pybullet.setJointMotorControl2(
                self.ur5,
                joint.id,
                pybullet.POSITION_CONTROL,
                targetPosition=gripper_opening_angle * self.mimic_multiplier[i],
                force=joint.maxForce,
                maxVelocity=joint.maxVelocity
            )

    def calculate_ik(self, position, orientation):
        """
        Standard PyBullet IK. We may not use it if fully switching to velocity control.
        """
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
        """
        End-effector world position, orientation (quaternion).
        """
        linkstate = pybullet.getLinkState(self.ur5,
                                          self.end_effector_index,
                                          computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (np.array(position), orientation)

    def reset(self):
        """
        Move robot to a known pose (via position control here).
        """
        self.stepCounter = 0
        np.pi
        joint_angles = (0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0)
        #(1.5, -1.57, 1.80, -1.57, -1.57, 0)
        self.set_joint_angles(joint_angles)
        self.control_gripper(-0.4)  # Open gripper
        for i in range(100):
            pybullet.stepSimulation()

    def step(self, end_effector_velocity, gripper_cmd=-1.0):

        active_joint_indices = []
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            joint_type = info[2]
            # 0 => REVOLUTE, 1 => PRISMATIC in PyBullet.
            if joint_type in [pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC]:
                active_joint_indices.append(i)

        #print("Active joints (i.e., DOF):", active_joint_indices)

        joint_states = pybullet.getJointStates(self.ur5, active_joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_accelerations = [0.0]*len(joint_positions)

        # print(f"End eff index: {self.end_effector_index}")
        # print(f"Joint angles: {joint_positions}")
        # print(f"Joint vels: {joint_velocities}")
        # print(f"Joint accel: {joint_accelerations}")
        
        linear_jacobian, angular_jacobian = pybullet.calculateJacobian(
            bodyUniqueId=self.ur5,
            linkIndex=self.end_effector_index,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=joint_accelerations
        )

        # Convert them to NumPy arrays
        J_lin_full = np.array(linear_jacobian)  # Shape => (3, 12)ç
        J_ang_full = np.array(angular_jacobian)
        #J_lin_arm = J_lin_full[:, :6]  # Now shape => (3, 6)
        #print(f"Linear Jacobian: {np.shape(J_lin_arm)}")
        J_full = np.vstack((J_lin_full, J_ang_full))
        q_dot = np.linalg.pinv(J_full) @ end_effector_velocity

        # Optionally, clamp or scale q_dot to respect joint velocity limits
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

        # Step the simulation
        pybullet.stepSimulation()
        self.stepCounter += 1
        time.sleep(1./240.)

    def close(self):
        pybullet.disconnect()


