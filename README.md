# Optimization of digital material distribution

This repository contains the code for the paper "Towards end-to-end optimization in multimaterial 3D printing" [LINK TO BE UPDATED], authored by Xue-Ling Luo, Steven Yang, Jingye Tan, Robert F. Shepherd, Noy Cohen, and Nikolaos Bouklas.

## Installation

This repo is built upon Python 3.12 and FEniCSx (DOLFINx), in an Anaconda environment. We also use gmsh GUI for mesh generation, and Paraview for visualization.

```shell
conda create -n fenicsx python=3.12
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista 
conda install -c conda-forge gmsh python-gmsh matplotlib pandas scipy 
conda install -c conda-forge adios2 adios4dolfinx=0.9.5
```

## File structure

```
.
├── src/
│   ├── default_config.py             # Default parameters, overridden by each problem's config.py
│   ├── msh2xdmf.py                   # Imports gmsh .msh files with their physical groups
│   ├── utils.py                      # Projection, VTU/PVD output, dotdict, logging
│   ├── solvers.py                    # Nonlinear/linear forward solvers and optimizers (MMA, CCSA-MMA, Barzilai-Borwein)
│   ├── neoHookean.py                 # Neo-Hookean model and basic optimization framework with Helmholtz filtering; base of all other implementations
│   ├── neoHookean_topo_opt.py        # Adjoint-state-method-based SIMP topology optimization
│   ├── neoHookean_tmc.py             # Third medium contact (TMC) for contact problems
│   ├── neoHookean_incompressible.py  # Incompressible neo-Hookean variants (mixed and staggered)
│   └── steven.py                     # pICNN-based constitutive model for digital materials and its optimization variants
├── mesh/                             # Configuration files and gmsh meshes for each example
├── main_*.py                         # Entry scripts for each example
└── command_lines.sh                  # Commands used to generate the results in the paper
```

The pICNN-based constitutive model in `src/steven.py` is derived from our previous work (Steven Yang et al., [10.1016/j.ijengsci.2025.104381](https://doi.org/10.1016/j.ijengsci.2025.104381)).

## Examples in the paper

All commands we used to generate results in the paper are in `command_lines.sh`, which will guide you to the corresponding `main_*.py` files that define the optimization problem and logic using interfaces defined in parent classes, and also the mesh and configuration files in `mesh/` folder.

* **Trapezoid**: `main_trapezoid_dist_opt.py` optimizes the material distribution of a trapezoid based on the traction on its top boundary, and can reload an optimized design to simulate its compression. Mesh: `mesh/trapezoid.geo` / `mesh/trapezoid.msh`. Configuration: `mesh/config_trapezoid.py`.
* **Semisphere**: `main_semisphere_dist_opt_tmc.py` compresses a semisphere into contact using third medium contact, then optimizes its material distribution based on the traction on the bottom boundary. Mesh: `mesh/semisphere.geo` / `mesh/semisphere.msh`. Configuration: `mesh/config_semisphere_tmc.py`.
* **Clamp**: `main_clamp.py` optimizes topology and material distribution of a clamp, together or separately, to maximize stored energy under displacement control, with an optional failure constraint. Mesh: none, as the rectangular mesh is generated in the configuration file. Configuration: `mesh/config_clamp.py`.

## Usage


### Configuration and execution

Default parameters are in `src/default_config.py`. Write your own `config.py` to override the default parameters. Surfaces where boundary conditions are applied and domains to be designed or frozen can be either defined by lambda functions or predefined in gmsh mesh files by "Physical groups". Finally, run a main script. 

Here are two examples of solving or optimizing a neo-Hookean body with dirichlet boundaries and neumann boundaries, from which you can see how easily it is to set up a problem with our implementation:

<details>
<summary>
Example: Neo-Hookean with Dirichlet and Neumann boundaries
</summary>

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

</details>

<details>
<summary>
Example: Topology optimization for neo-Hookean
</summary>

This example generates mesh and define Dirichlet and Neumann boundaries using FEniCSx itself instead of using `gmsh`.

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
    mesh_size=0.01,
    # Optimization settings
    target_volume=0.5,
    min_density=0.001,
    init_penalty=6.0,
    max_penalty=6.0,
    gradient_filter=None,
    filter_radius=0.01,
    clip_gradient=0.02,
    output_per_step=1,
    maxiter=100,
    fix_density_h=None,
    optimizer="MMA",
    # fix_design_domain="left_half",
    # Solver settings
    relative_tol=0.0001,
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 100,
        "snes_linesearch_max_it": 100,
        # "snes_linesearch_alpha": 1e-2,
        "snes_linesearch_order": 1,
    },
    # Elasticity parameters
    c=1.0,
    E=10.0,
    nu=0.3,
    kr_coef=1e-4,
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
        domain_dict={
            lambda x: x[0] < 0.5: "left_half",
        },
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): (0.0, 0.0),  # Modify for 3D
        },
        neumann_bcs_dict={
            lambda x: np.isclose(x[0], params.width)
            & np.isclose(x[1], 0, atol=params.mesh_size * 1.1): (
                0.0,
                -0.01,
            ),  # Modify for 3D
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

</details>

