conda create -n pybullet_env python=3.9.21
conda activate pybullet_env
pip install --upgrade pip
pip3 install pybullet --upgrade --user
pip install attrdict
pip install tacto
pip install torch
pip install stable-baselines3[extra]
pip install pybulletx
pip install pytouch
cd 
git clone git@github.com:Javi00C/tfm.git
cd tfm
pip install -e .
# Important for headless server (and make use of egl in the learning script)
pip install PyOpenGL PyOpenGL-accelerate pyrender trimesh

