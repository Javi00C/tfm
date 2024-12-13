# import pybullet as p
# import pybullet_data
# import time

# # Connect to PyBullet
# p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add default PyBullet data path

# # Create ground plane
# plane_id = p.loadURDF("plane.urdf")

# # Rope parameters
# num_segments = 10
# segment_length = 0.1
# segment_radius = 0.02
# mass = 0.1

# # Starting position of the rope
# start_position = [0, 0, 1]

# # Create the rope segments as capsules
# rope_segments = []
# constraints = []

# for i in range(num_segments):
#     # Calculate the position of each segment
#     segment_position = [start_position[0], start_position[1], start_position[2] - i * segment_length]

#     # Create a segment as a capsule
#     segment_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=segment_radius, height=segment_length)
#     visual_id = p.createVisualShape(p.GEOM_CAPSULE, radius=segment_radius, length=segment_length, rgbaColor=[0, 1, 0, 1])
#     body_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=segment_id,
#                                 baseVisualShapeIndex=visual_id, basePosition=segment_position)

#     rope_segments.append(body_id)

#     # Disable collisions between adjacent segments
#     if i > 0:
#         p.setCollisionFilterPair(rope_segments[i - 1], rope_segments[i], -1, -1, enableCollision=0)

#     # Connect this segment to the previous one with a constraint
#     if i > 0:
#         constraint = p.createConstraint(
#             parentBodyUniqueId=rope_segments[i - 1],
#             parentLinkIndex=-1,
#             childBodyUniqueId=rope_segments[i],
#             childLinkIndex=-1,
#             jointType=p.JOINT_POINT2POINT,
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[0, 0, -segment_length / 2],
#             childFramePosition=[0, 0, segment_length / 2],
#         )
#         constraints.append(constraint)

# # Simulation loop
# simulation_time = 10  # Run the simulation for 10 seconds
# simulation_steps = int(simulation_time * 240)  # Assuming 240 Hz simulation frequency

# for step in range(simulation_steps):
#     p.stepSimulation()
#     time.sleep(1 / 240)  # Sleep to maintain real-time simulation

# # Disconnect from PyBullet
# p.disconnect()

import pybullet as p
import pybullet_data
import time
import numpy as np  # Use numpy for vector operations

# Connect to PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Create ground plane
plane_id = p.loadURDF("plane.urdf")

# Rope parameters
num_segments = 30
segment_length = 0.03
segment_radius = 0.01
mass = 0.1
friction = 0.5  # Friction coefficient for the rope segments

# Starting position of the rope
start_position = [0, 0, 1]

# Create the rope segments as capsules
rope_segments = []
constraints = []

for i in range(num_segments):
    # Calculate the position of each segment
    segment_position = [start_position[0], start_position[1], start_position[2] - i * segment_length]

    # Create a segment as a capsule
    segment_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=segment_radius, height=segment_length)
    visual_id = p.createVisualShape(p.GEOM_CAPSULE, radius=segment_radius, length=segment_length, rgbaColor=[0, 1, 0, 1])
    body_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=segment_id,
                                baseVisualShapeIndex=visual_id, basePosition=segment_position)

    # Set friction for each segment
    p.changeDynamics(body_id, -1, lateralFriction=friction)

    rope_segments.append(body_id)

    # Disable collisions between adjacent segments
    if i > 0:
        p.setCollisionFilterPair(rope_segments[i - 1], rope_segments[i], -1, -1, enableCollision=0)

    # Connect this segment to the previous one with a constraint
    if i > 0:
        constraint = p.createConstraint(
            parentBodyUniqueId=rope_segments[i - 1],
            parentLinkIndex=-1,
            childBodyUniqueId=rope_segments[i],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -segment_length / 2],
            childFramePosition=[0, 0, segment_length / 2],
        )
        constraints.append(constraint)

# Add stiffness and damping to joints
def apply_stiffness_and_damping():
    stiffness = 0.2  # Stiffness factor
    damping = 0.1     # Damping factor

    for i in range(1, num_segments):  # Start from the second segment
        parent_id = rope_segments[i - 1]
        child_id = rope_segments[i]

        # Get the current positions of the parent and child
        parent_pos, _ = p.getBasePositionAndOrientation(parent_id)
        child_pos, _ = p.getBasePositionAndOrientation(child_id)

        # Compute the displacement vector
        displacement = np.array(child_pos) - np.array(parent_pos)
        displacement_magnitude = np.linalg.norm(displacement)

        # Compute the restoring force (Hooke's law)
        restoring_force = -stiffness * displacement

        # Get the current relative velocity of the child w.r.t the parent
        parent_vel, _ = p.getBaseVelocity(parent_id)
        child_vel, _ = p.getBaseVelocity(child_id)
        relative_velocity = np.array(child_vel) - np.array(parent_vel)

        # Compute the damping force
        damping_force = -damping * relative_velocity

        # Apply the total force to the child and counter-force to the parent
        total_force = restoring_force + damping_force
        p.applyExternalForce(child_id, -1, forceObj=total_force.tolist(), posObj=child_pos, flags=p.WORLD_FRAME)
        p.applyExternalForce(parent_id, -1, forceObj=(-total_force).tolist(), posObj=parent_pos, flags=p.WORLD_FRAME)

# Simulation loop
simulation_time = 100  # Run the simulation for 10 seconds
simulation_steps = int(simulation_time * 240)  # Assuming 240 Hz simulation frequency

for step in range(simulation_steps):
    apply_stiffness_and_damping()  # Apply stiffness and damping forces
    p.stepSimulation()
    time.sleep(1 / 240)  # Sleep to maintain real-time simulation

# Disconnect from PyBullet
p.disconnect()

