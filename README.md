# Installation

## Miniconda

Please refer to [Miniconda installation](https://docs.anaconda.com/miniconda/ "Miniconda official site")

## Create Conda Environment with dependencies

```bash
conda create -n gym python=3.10.12 swig gymnasium gymnasium[box2d] pytz scipy sympy gymnasium[mujoco]

conda activate gymenv

pip install stable-baselines3[extra]==2.4.0
```
## Install gymnasium
```bash
pip install gymnasium==1.0.0
```
## Install Mujoco (just download and extract)
```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
```
## Install mujoco_py
```bash
pip install mujoco_py --no-cache-dir
```
## Patchelf
```bash
conda install -c conda-forge patchelf
```
## .bashrc Modification
```bash
# Mujoco (adjusted for proper path ordering)
export MUJOCO_PY_MUJOCO_PATH=/home/javi/.mujoco/mujoco210
export CPATH="/home/javi/.mujoco/mujoco210/include:$CPATH"
export LD_LIBRARY_PATH="/home/javi/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export PATH="/home/javi/.mujoco/mujoco210/bin:$PATH"
```

## Getting into the environment

```bash
conda activate gym
```

# DIGIT SENSOR (TACTO MUJOCO library installation)

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
# PROBLEM WITH TACTO MUJOCO library installation

```bash
The problem is this: pyrender 0.1.45 requires PyOpenGL==3.1.0, dm-control 1.0.14 requires pyopengl>=3.1.4
seems like both pyrender and dm-control are needed to run the demo_mujoco_digit.py
The pyrender version 0.1.45 is the highest version -> needs pyopengl 3.1.0
The lowest version of dm-control  0.0.28658793 is compatible with pyopengl 3.1.0 or at least lower than 3.1.4
seems like if version of dm-control 0.0.28658793 is used then it looks for document names of a lower version of mujoco (2.0.0) which is not available for download (https://github.com/google-deepmind/mujoco/releases)
```
