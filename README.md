# Optimization of digital material distribution

## Installation

This repo is built upon Python 3.8 and FEniCS 2019.1

```shell
conda create -n fenics-adjoint-19 python=3.8
conda activate fenics-adjoint-19
conda install -c conda-forge fenics==2019.1 matplotlib==3.0.3 superlu_dist==6.2.0 mpi4py==3.0.3 mshr
pip install scipy==1.7.3
```

If using VSCode to debug (with the extension `Python Debugger`), install a previous `Python Debugger` version `2024.10.0`.

## Usage

`hyperelasticity_adjoint.py` implements a simple implementation of adjoint-state-method-based SIMP topology optimization for a neo-Hookean model, given default parameters in `default_config.py`. Modify parameters in another config file (e.g., `config.py`) that is copied from `default_config.py`, and run the following command:

```shell
python hyperelasticity_adjoint.py -c config.py
```
