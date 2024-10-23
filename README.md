# Installation

## Miniconda

Please refer to [Miniconda installation](https://docs.anaconda.com/miniconda/ "Miniconda official site")

## Create Conda Environment with dependencies

```bash
conda create -n gymenv python=3.10.12 swig gymnasium gymnasium[box2d]
```

## Getting into the environment

```bash
conda activate gymenv
```

# Execution

```bash
python3 obs_wrapper_ppo.py
```

