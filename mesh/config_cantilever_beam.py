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
    density_filter=0.01,
    clip_gradient=0.02,
    output_per_step=1,
    maxiter=10000,
    fix_density_h=None,
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
    nu=0.49,
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
                -0.001,
            ),  # Modify for 3D
        },
    )
)
