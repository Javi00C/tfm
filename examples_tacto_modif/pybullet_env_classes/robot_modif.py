import pybullet as p
import pybulletX as px
import math
from collections import namedtuple
import numpy as np

class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        self.all_joints_ids = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            self.all_joints_ids.append(jointID)
            if controllable:
                self.controllable_joints.append(jointID)
                
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_ee_vel(self, action, control_method):
        assert control_method in ('joint', 'end'), "control_method must be 'joint' or 'end'"
        if control_method == 'end':
            x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot = action
            ee_velocity = np.array([x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot])
            
            # Use all controllable joints
            num_dofs = len(self.controllable_joints)
            assert num_dofs > 0, "Number of controllable joints (DoF) must be positive"
            
            joint_states = p.getJointStates(self.id,self.controllable_joints)#, self.controllable_joints)
            joint_positions = [state[0] for state in joint_states]
            joint_velocities = [state[1] for state in joint_states]
            joint_accelerations = [0.0] * num_dofs

            # assert len(joint_positions) == num_dofs, "Mismatch in joint_positions length"
            # assert len(joint_velocities) == num_dofs, "Mismatch in joint_velocities length"
            # assert len(joint_accelerations) == num_dofs, "Mismatch in joint_accelerations length"

            # Get the current position and orientation of the end-effector
            ee_link_state = p.getLinkState(self.id, self.eef_id, computeForwardKinematics=True)
            ee_position = ee_link_state[0]  # Position of the end-effector
            ee_orientation = ee_link_state[1]  # Orientation of the end-effector (quaternion)

            # Assume the local position of the end-effector is at its frame origin
            local_position = [0, 0, 0]  # Ensure this is size-3

            # Compute the Jacobian for the end-effector with respect to all controllable joints
            linear_jacobian, angular_jacobian = p.calculateJacobian(
                self.id,
                self.eef_id,
                local_position,
                joint_positions,
                joint_velocities,
                joint_accelerations
            )

            # # Example controllable list
            # controllable_list = [
            #     False, True, True, True, True, True, True, False, False, True,
            #     False, True, False, True, True, False, True, False, True
            # ]

            # Jacobian matrix (linear_jacobian and angular_jacobian combined)
            # Assuming linear_jacobian and angular_jacobian are numpy arrays of appropriate dimensions
            jacobian = np.vstack((linear_jacobian, angular_jacobian))

            # Create a mask to zero out the columns corresponding to `False` in controllable_list
            #mask = np.array(controllable_list, dtype=bool)

            # Set columns to zero for uncontrollable joints
            #jacobian[:, ~mask] = 0

            # Compute the required joint velocities using the pseudoinverse of the Jacobian
            joint_velocities_target = np.linalg.pinv(jacobian).dot(ee_velocity)

            # Set joint motor controls for the computed velocities
            for i, joint_id in enumerate(self.controllable_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=self.id,
                    jointIndex=joint_id,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=joint_velocities_target[i],
                    force=self.joints[joint_id].maxForce
                )

        elif control_method == 'joint':
            assert len(action) == len(self.controllable_joints), \
                "Action must match the number of controllable joints"
            # Directly set joint positions
            for i, joint_id in enumerate(self.controllable_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=self.id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=action[i],
                    force=self.joints[joint_id].maxForce
                )



    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            state = p.getJointState(self.id, joint_id)
            #print(f"Start state {state['joint_position']}")
            #print("END state")
            # if state:  # Check if state is valid
            #     pos, vel, _, _ = state
            #     positions.append(pos)
            #     velocities.append(vel)
            # else:
            #     positions.append(None)
            #     velocities.append(None)
            #pos, vel, _, _ = p.getJointState(self.id, joint_id)
            pos = state['joint_position']
            vel = state['joint_velocity']
            #_,pos, _, vel = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)



class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./pybullet_env_classes/urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
            # Setup mimic joints
            mimic_parent_name = 'finger_joint'
            mimic_children_names = {'right_outer_knuckle_joint': 1,
                                    'left_inner_knuckle_joint': 1,
                                    'right_inner_knuckle_joint': 1,
                                    'left_inner_finger_joint': -1,
                                    'right_inner_finger_joint': -1}
            self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

            # Once the robot is fully loaded, we can attach the DIGIT sensor if cfg and digits are provided
            # This method will be called after we set `self.cfg` and `self.digits` externally.
            if hasattr(self, 'cfg') and hasattr(self, 'digits'):
                self._attach_digit_sensor()

    def _attach_digit_sensor(self):
        # Load the DIGIT sensor body using the config
        # Note: `**self.cfg.digit` expands to urdf_path, base_position, base_orientation, use_fixed_base, etc.
        digit_body = px.Body(**self.cfg.digit)

        # Add the camera to the DIGIT sensor
        self.digits.add_camera(digit_body.id, [-1])

        # Find the gripper link to attach the DIGIT to. For example, attach to 'left_inner_finger_joint'
        # Print or inspect self.joints to find the correct link name.
        #attach_link_name = 'left_inner_finger_joint'  # Adjust this to the correct link
        attach_link_name = 'left_inner_finger_pad_joint'
        attach_link_id = [j.id for j in self.joints if j.name == attach_link_name][0]
        print(f"ATTACH LINK ID DIGIT: {attach_link_id}")
        # Create a fixed joint (constraint) between the chosen link and the DIGIT sensor
        # Adjust parentFramePosition and childFramePosition to place the sensor correctly.
        
        p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=attach_link_id,
            childBodyUniqueId=digit_body.id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0], # Adjust offsets as needed
            #parentFrameOrientation=[0.5,0,0],
            childFramePosition=[0,0,0]
        )
        #         # Disable collision between the robot and the digit sensor
        robot_id = self.id
        digit_id = digit_body.id
        # p.setCollisionFilterPair(robot_id, digit_id, -1, -1, enableCollision=0)
        robot_num_joints = p.getNumJoints(robot_id)
        digit_num_joints = p.getNumJoints(digit_id)

        # Disable collisions between every link of the robot and every link of the digit sensor
        for r_link in range(-1, robot_num_joints):
            for d_link in range(-1, digit_num_joints):
                p.setCollisionFilterPair(robot_id, digit_id, r_link, d_link, enableCollision=0)



    def setup_digit(self, cfg, digits):
        # A helper method called from the simulation class after robot.load()
        self.cfg = cfg
        self.digits = digits
        # Now that cfg and digits are set, call __post_load__ again or directly attach
        self._attach_digit_sensor()

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                    self.id, joint_id,
                                    jointType=p.JOINT_GEAR,
                                    jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)


# class UR5Robotiq140(UR5Robotiq85):
#     def __init_robot__(self):
#         self.eef_id = 7
#         self.arm_num_dofs = 6
#         self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
#                                -1.5707970583733368, 0.0009377758247187636]
#         self.id = p.loadURDF('./pybullet_env_classes/urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
#                              useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
#         self.gripper_range = [0, 0.085]
#         # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.

#     def __post_load__(self):
#         mimic_parent_name = 'finger_joint'
#         mimic_children_names = {'right_outer_knuckle_joint': -1,
#                                 'left_inner_knuckle_joint': -1,
#                                 'right_inner_knuckle_joint': -1,
#                                 'left_inner_finger_joint': 1,
#                                 'right_inner_finger_joint': 1}
#         self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
