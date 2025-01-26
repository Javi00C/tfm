# 1.PYBULLET Env Installation
```bash
conda create -n pybullet_env python=3.9.21
```
```bash
conda activate pybullet_env
```
```bash
pip install --upgrade pip
```
```bash
pip3 install pybullet --upgrade --user
```
```bash
pip install numpy==1.23.0
```
```bash
pip install attrdict
```
```bash
pip install tacto
```
```bash
pip install torch
```
```bash
pip install stable-baselines3[extra]
```
```bash
pip install pybulletx
pip install pytouch
pip install networkx==3.2.1
cd 
git clone git@github.com:Javi00C/tfm.git
cd tfm
pip install -e .
# Important for headless server (and make use of egl in the learning script)
pip install PyOpenGL PyOpenGL-accelerate pyrender trimesh
```

This guide explains how to install PyBullet from a `.tar.gz` source file and resolve any dependency issues encountered during the installation process.

---
## Miniconda

Please refer to [Miniconda installation](https://docs.anaconda.com/miniconda/ "Miniconda official site")

## Prerequisites

1. **Python Environment**:
   - Use a virtual environment to avoid conflicts:
     ```bash
     conda create -n pybullet_env python=3.9.21
     ```

2. **Required Tools**:
   - Ensure `pip` is updated:
     ```bash
     conda activate pybullet_env
     pip install --upgrade pip
     ```

---

## Step 1: Extract the Source Code

1. Download the PyBullet `.tar.gz` file. and extract it using the following command:
   ```bash
   wget https://github.com/bulletphysics/bullet3/archive/refs/tags/3.25.tar.gz -o bullet3-3.25.tar.gz
   tar -xvzf bullet3-3.25.tar.gz
   cd bullet3-3.25
   ```

---

## Step 2: Install Dependencies

Before installing PyBullet, resolve its dependencies. Install all required packages:

```bash
pip install "pillow>=8.3.2" "scipy>=1.10" "decorator<5.0,>=4.0.2" "requests<3.0,>=2.8.1" "pytz>=2020.1" "packaging" "protobuf!=4.24.0,>=3.19.6" "six>1.9" "click" "pyyaml>=5.3.1" "prompt-toolkit<=3.0.36,>=2.0" "opencv-python" "pybulletX"
```

If additional conflicts arise, resolve them as prompted by `pip`.

---

## Step 3: Install PyBullet

After resolving dependencies, install PyBullet:

1. Using `setup.py`:
   ```bash
   cd bullet3-3.25
   python setup.py install
   ```

2. Alternatively, using `pip`:
   ```bash
   pip install .
   ```
   
## Step 4: Install Tacto
```bash
git clone https://github.com/facebookresearch/tacto.git
cd tacto
pip install .
pip install -r requirements/examples.txt
```
---
## Step 5: Modify Tacto Folder (NOT USING CURRENT ROBOT CONTROL CODE)
```bash
mv <path_to_repo>/examples_tacto_modif/ /<path_to_tacto_folder>/tacto/
```
---
## Step 6: Run simulation (NOT USING CURRENT ROBOT CONTROL CODE)
The simulation file is called examples_tacto_modif/robot_rope_digit.py, the robot description class is examples_tacto_modif/pybullet_env_classes/robot_modif.py
```bash
python3 robot_rope_digit.py
```
---

## Step 7: New robot control code 
The learn file is called <path_to_repo>pybullet_ur5/learn_paralel_pybullet.py, to execute the simulation controlled by agent <path_to_repo>pybullet_ur5/execute_pybullet_sim.py
```bash
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]
```
---

## Troubleshooting

1. **Dependency Issues**:
   If you encounter errors about missing or conflicting dependencies, manually resolve them by installing the specified versions.

2. **Check for Dependency Conflicts**:
   ```bash
   pip check
   ```
---

# 2.MUJOCO Env Installation

## Miniconda

Please refer to [Miniconda installation](https://docs.anaconda.com/miniconda/ "Miniconda official site")

## Create Conda Environment with dependencies
```bash
conda create -n mujoco_env python=3.10.12 swig gymnasium=1.0.0 pytz scipy sympy gymnasium[mujoco]

conda activate mujoco_env

pip install stable-baselines3[extra]==2.4.0
```
## Mujoco_py
```bash
pip install mujoco_py --no-cache-dir
```
## Patchelf
```bash
conda install -c conda-forge patchelf
```
## Install Mujoco (just download and extract)
```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```
## .bashrc file Modification
```bash
# Mujoco (adjusted for proper path ordering)
export MUJOCO_PY_MUJOCO_PATH=/home/javi/.mujoco/mujoco210
export CPATH="/home/javi/.mujoco/mujoco210/include:$CPATH"
export LD_LIBRARY_PATH="/home/javi/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export PATH="/home/javi/.mujoco/mujoco210/bin:$PATH"
```

## Getting into the environment
```bash
conda activate mujoco_env
```

# 3.DIGIT SENSOR (TACTO MUJOCO library installation)

```bash
pip install dm-control==1.0.14

pip install pyopengl==3.1.4

sudo apt-get install libosmesa6-dev

Clone the Repository: Pull the TACTO-MuJoCo repository into a directory of your choice:
git clone https://github.com/L3S/TACTO-MuJoCo.git
cd TACTO-MuJoCo

Install Python Dependencies: Within the repository directory, run:
I have commented the line that mentions numpy version 1.20.3 seems to create an error
pip install -r requirements.txt

Verify Your Setup: Run a demo script to ensure the installation was successful:
python demo_mujoco_digit.py
```
## PROBLEM WITH TACTO MUJOCO library installation

```bash
The problem is this: pyrender 0.1.45 requires PyOpenGL==3.1.0, dm-control 1.0.14 requires pyopengl>=3.1.4
seems like both pyrender and dm-control are needed to run the demo_mujoco_digit.py
The pyrender version 0.1.45 is the highest version -> needs pyopengl 3.1.0
The lowest version of dm-control  0.0.28658793 is compatible with pyopengl 3.1.0 or at least lower than 3.1.4
seems like if version of dm-control 0.0.28658793 is used then it looks for document names of a lower version of mujoco (2.0.0)
which is not available for download (https://github.com/google-deepmind/mujoco/releases)
```
