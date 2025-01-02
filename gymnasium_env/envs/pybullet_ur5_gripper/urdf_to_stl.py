# import trimesh

# # Create a cylinder with the desired radius and length
# radius = 0.01  # Example radius
# length = 0.06  # Example length
# cylinder = trimesh.creation.cylinder(radius=radius, height=length)

# # Save as an STL file
# cylinder.export("segment_cylinder.stl")
# print("Cylinder STL file saved as 'segment_cylinder.stl'")
import trimesh

# Parameters for the capsule
radius = 0.02 # Radius of the capsule
length = 0.06  # Length of the cylindrical part (excluding hemispherical ends)
 
# Create the capsule
capsule = trimesh.creation.capsule(radius=radius, height=length)

# Save the capsule as an STL file
stl_filename = "segment_capsule.stl"
capsule.export(stl_filename)

print(f"Capsule STL file saved as '{stl_filename}'")
