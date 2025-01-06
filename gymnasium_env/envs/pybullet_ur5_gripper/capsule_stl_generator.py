import trimesh

# Parameters for the capsule
length = 0.06  # Length of the cylindrical part (excluding hemispherical ends)
radius = 0.02 # Radius of the capsule
 
# Create the capsule
capsule = trimesh.creation.capsule(radius=radius, height=length)

# Save the capsule as an STL file
stl_filename = "segment_capsule.stl"
capsule.export(stl_filename)

print(f"Capsule STL file saved as '{stl_filename}'")
