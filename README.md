# 1.PYBULLET Env Installation
```bash
conda create -n pybullet_env python=3.9.21
conda activate pybullet_env
pip install --upgrade pip
pip3 install pybullet --upgrade --user
pip install numpy==1.23.0
pip install attrdict
pip install tacto
pip install torch
pip install stable-baselines3[extra]
pip install pybulletx
pip install pytouch
pip install networkx==3.2.1
cd 
git clone git@github.com:Javi00C/tfm.git
cd tfm
pip install -e .
# Important only for training (headless server must be used)
pip install PyOpenGL PyOpenGL-accelerate pyrender trimesh
```

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
