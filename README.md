# Installation

## Miniconda

Please refer to [Miniconda installation](https://docs.anaconda.com/miniconda/ "Miniconda official site")

## Create Conda Environment with dependencies

```bash
conda create -n gymenv python=3.10.12 swig gymnasium gymnasium[box2d] stable-baselines3[extra] pytz scipy sympy gymnasium[mujoco]


```
## Stable baselines needs a version of Gymnasium < 0.30
```bash
pip install gymnasium==0.29.0
```
## Install Mujoco (just download and extract)
```bash
https://github.com/google-deepmind/mujoco/releases
```

```bash
conda install -c conda-forge patchelf
```

## Getting into the environment

```bash
conda activate gymenv
```

# Execution

```bash
python3 obs_wrapper_ppo.py
```

