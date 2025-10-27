from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="c-shape",
    # FEM settings
    dim=2,
    q_deg=4,
    elem_order=2,
    # Geometry
    width=10.25,
    height=5,
    # Elasticity parameters
    E=0.6,
    nu=0.4,
    # Solver settings
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-5,
        "snes_atol": 1e-15,
        "snes_max_it": 200,
        "snes_linesearch_max_it": 100,
    },
    # TMC settings
    tmc_regularization="rotation_angle",
    kr_coef=1e-6 * 30,
    psi_coef=2e-6,
    main_medium="unmarked_cells",
    third_medium="third_medium",
)

_mesh = dmesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (params.width, params.height)),
    n=(41, 20),
    cell_type=dmesh.CellType.quadrilateral,
)  # Modify for 3D

params.update(
    dotdict(
        mesh=_mesh,
        # Body force
        B=fem.Constant(
            _mesh, ScalarType((0.0, 0.0))
        ),  # Body force per unit volume # Modify for 3D
        cell_tags=None,
        domain_dict={
            lambda x: np.logical_or(
                np.logical_and(
                    x[0] >= 1,
                    np.logical_and(x[1] >= 1, x[1] <= 4),
                ),
                x[0] >= 10,
            ): "third_medium",
        },
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): fem.Constant(
                _mesh, ScalarType((0.0, 0.0))
            ),  # Modify for 3D
            lambda x: np.logical_and(x[0] >= 10, np.isclose(x[1], params.height)): {
                1: -0.25
            },
        },
        neumann_bcs_dict=None,
    )
)
