from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import gmsh
from dolfinx.io import gmshio
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="clamp",
    # FEM settings
    dim=2,
    q_deg=2,
    elem_order=1,
    # Geometry settings
    width=1,
    height=1,
    mesh_size=0.005,
    # Optimization settings
    target_volume=0.3,
    volume_cons_weight=10,
    min_density=0.001,
    init_penalty=2.0,
    max_penalty=6.0,
    gradient_filter=None,
    density_filter=0.01,
    clip_gradient=0.1,
    output_per_step=1,
    maxiter=10000,
    lr=50,
    lr_1=0,
    # lower_bound_lr=1.0,
    fix_density_h=None,
    fix_design_domain="load",
    # Solver settings
    relative_tol=0.0005,
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 500,
        "snes_linesearch_max_it": 100,
        # "snes_linesearch_alpha": 1e-2,
        "snes_linesearch_order": 1,
    },
    # Elasticity parameters
    c=1.0,
    E=10.0,
    nu=0.49,
    kr_coef=1e-3,
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
            lambda x: ((x[0] >= params.width - 0.02) & (x[1] <= 0.4 * params.height))
            | ((x[0] <= 0.4 * params.width) & (x[1] >= params.height - 0.02)): "load"
        },
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[1], params.height)
            & (x[0] <= 0.4 * params.width): (0.0, 0.0),  # Modify for 3D
            lambda x: np.isclose(x[0], params.width)
            & (x[1] <= 0.4 * params.height): {0: -0.4},
        },
        neumann_bcs_dict={},
    )
)
