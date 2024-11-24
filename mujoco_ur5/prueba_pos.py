import mujoco
import sys

xml_path = sys.argv[1]
model = mujoco.MjModel.from_xml_path(xml_path)
print(f"Total DOF (nq): {model.nq}")