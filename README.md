# Optimization of digital material distribution

## Installation

This repo is built upon Python 3.8 and FEniCS 2019.1

```shell
conda create -n fenics-adjoint-19 python=3.8
conda activate fenics-adjoint-19
conda install -c conda-forge fenics==2019.1 matplotlib==3.0.3 superlu_dist==6.2.0 mpi4py==3.0.3 mshr meshio==4.4.0
pip install scipy==1.7.3
```

If using VSCode to debug (with the extension `Python Debugger`), install a previous `Python Debugger` version `2024.10.0`.

## Usage

### Implemented methods

* `src/neoHookean.py` implements a neo-Hookean model and a basic optimization framework as the foundation of all other implementations. 
* `src/neoHookean_topo_opt.py` implements a simple implementation of adjoint-state-method-based SIMP topology optimization for a neo-Hookean model. 

### Configuration and execution

Default parameters are in `default_config.py`. Copy `default_config.py` to another config file (e.g., `config.py`) and modify parameters there.

#### Example: neo-Hookean

Here is an example of solving a neo-Hookean body with dirichlet boundaries and neumann boundaries:

Generate a 2D triangular mesh using `gmsh`, with boundaries defined in "Physical groups" as "Curve", namely "upper" and "right", and the domain defined as "Surface", namely "main". The mesh is `mesh/square.geo` and `mesh/square.msh`.

In `config.py`:

```python
import dolfin as fe
from src.utils import dotdict
from src.msh2xdmf import msh2xdmf, import_mesh

params = dotdict(
    # Project settings
    project_name="square",
    # FEM settings
    dim=2,
    q_deg=2,
    # Elasticity parameters
    E=10.0,
    nu=0.3,
    # Body force
    B=fe.Constant((0.0, -0.0)),  # Body force per unit volume # Modify for 3D
)

mesh, boundaries, association_table = import_mesh("square", directory="mesh")

params.update(
    dotdict(
        mesh=mesh,
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && on_boundary", side=0.0
            ): fe.Constant(
                (0.0, 0.0)
            ),  # Modify for 3D
        },
        neumann_bcs_dict=None,
        boundaries=boundaries,  # boundaries is a MeshFunction with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=[
            (association_table["upper"], fe.Constant((0.0, -0.001))),
            (association_table["right"], fe.Constant((-0.001, 0.0))),
        ],  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], fe.Constant((0.0, 0.0)))
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
```

The main script

```python
from src.neoHookean import NeoHookean

problem = NeoHookean(config_path="config.py")
problem.forward()
problem.save()
```

#### Example: Topology optimization for neo-Hookean

This example generates mesh and define Dirichlet and Neumann boundaries using FEniCS itself instead of using `gmsh`.

In `config.py`:

```python
import dolfin as fe
from src.utils import dotdict

params = dotdict(
    # Project settings
    project_name="beam_rightlowercorner_load",
    # FEM settings
    dim=2,
    q_deg=2,
    elem_order=1,
    # Geometry settings
    width=4,
    height=1,
    mesh_size=0.04,
    # Optimization settings
    target_volume=0.5,
    min_density=0.001,
    init_penalty=1,
    max_penalty=4.0,
    r_min=0.12,
    output_per_step=1,
    maxiter=10000,
    lr=1,
    # Solver settings
    relative_tol=0.001,
    # Elasticity parameters
    E=10.0,
    nu=0.3,
    # Body force
    B=fe.Constant((0.0, -0.0)),  # Body force per unit volume # Modify for 3D
)

params.update(
    dotdict(
        mesh=fe.RectangleMesh.create(
            [fe.Point(0.0, 0.0), fe.Point(params.width, params.height)],
            [
                int(params.width // params.mesh_size),
                int(params.height // params.mesh_size),
            ],
            fe.CellType.Type.quadrilateral,
        ),  # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && on_boundary", side=0.0
            ): fe.Constant(
                (0.0, 0.0)
            ),  # Modify for 3D
        },
        neumann_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && near(x[1], pos, length) && on_boundary",
                side=params.width,
                pos=0,
                length=params.mesh_size * 1.1,
            ): fe.Constant(
                (0.000, -0.0001)
            ),  # Modify for 3D
        },
    )
)
```

The main script

```python
from src.neoHookean_topo_opt import NeoHookeanTopoOpt

problem = NeoHookeanTopoOpt(config_path="config.py")
problem.optimize()
```

