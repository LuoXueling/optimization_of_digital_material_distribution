# Optimization of digital material distribution

## Installation

This repo is built upon Python 3.12 and FEniCSx (DOLFINx)

```shell
conda create -n fenicsx python=3.12
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista 
conda install -c conda-forge gmsh python-gmsh matplotlib pandas scipy 
conda install -c conda-forge adios2 adios4dolfinx=0.9.5
```

## Usage

### Implemented methods

* `src/neoHookean.py` implements a neo-Hookean model and a basic optimization framework as the foundation of all other implementations. 
* `src/neoHookean_topo_opt.py` implements a simple implementation of adjoint-state-method-based SIMP topology optimization for a neo-Hookean model. 
* `src/neoHookean_incompressible.py` implements the incompressible neo-Hookean model using coupled u-p scheme and staggered (augmented Lagrange) scheme.

### Configuration and execution

Default parameters are in `default_config.py`. Copy `default_config.py` to another config file (e.g., `config.py`) and modify parameters there.

#### Example: neo-Hookean

Here is an example of solving a neo-Hookean body with dirichlet boundaries and neumann boundaries:

Generate a 2D triangular mesh using `gmsh`, with boundaries defined in "Physical groups" as "Curve", namely "upper" and "right", and the domain defined as "Surface", namely "main". The mesh is `mesh/square.geo` and `mesh/square.msh`.

In `config.py`:

```python
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="square",
    # FEM settings
    dim=2,
    q_deg=2,
    # Elasticity parameters
    E=10.0,
    nu=0.3,
)

mesh, cell_tags, boundaries, association_table = import_mesh("square", directory="mesh")

params.update(
    dotdict(
        mesh=mesh,
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): (0.0, 0.0),  # Modify for 3D
        },
        neumann_bcs_dict=None,
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=[
            (association_table["upper"], (0.0, -0.001)),
            (association_table["right"], (-0.001, 0.0)),
        ],  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], (0.0, 0.0))
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
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import numpy as np
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
    init_penalty=6.0,
    max_penalty=6.0,
    density_filter=0.01,
    clip_gradient=0.1,
    output_per_step=1,
    maxiter=10000,
    lr=100,
    fix_density_h=0.1,
    # Solver settings
    relative_tol=0.0001,
    # Elasticity parameters
    E=10.0,
    nu=0.3,
)

_mesh = dmesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (params.width, params.height)),
    n=(
        int(params.width // params.mesh_size),
        int(params.height // params.mesh_size),
    ),
    cell_type=dmesh.CellType.quadrilateral,
)  # Modify for 3D

params.update(
    dotdict(
        mesh=_mesh,
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): (0.0, 0.0),  # Modify for 3D
        },
        neumann_bcs_dict={
            lambda x: np.isclose(x[0], params.width)
            & np.isclose(x[1], 0, atol=params.mesh_size * 1.1): (0.0, -0.001),  # Modify for 3D
        },
    )
)
```

The main script

```python
from src.neoHookean_topo_opt import NeoHookeanPDEFilteredTopoOpt

problem = NeoHookeanPDEFilteredTopoOpt(config_path="config.py")
problem.optimize()
```

