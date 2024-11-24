import time
import mujoco.viewer
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type = str,
        help = "Path towards your xml file",
        required = True
    )
    
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    xml_path = args.path

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_steps = 5

    # viewer shows frame of environment every n_steps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while True:
            step_start = time.time()
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            viewer.sync()
            # Control the frame rate more directly, e.g., 30 FPS
            time.sleep(1 / 30)
