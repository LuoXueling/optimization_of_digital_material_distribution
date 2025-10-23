from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="box",
    # FEM settings
    dim=2,
    q_deg=4,
    elem_order=2,
    # Geometry
    width=2,
    height=0.5,
    # Elasticity parameters
    E=25.714,
    nu=0.2857,
    # Solver settings
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-6,
        "snes_atol": 1e-15,
        "snes_max_it": 100,
    },
    # TMC settings
    kr_coef=1e-6 * 2**2,
    psi_coef=1e-7,
    main_medium="unmarked_cells",
    third_medium="third_medium",
)

_mesh = dmesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (params.width, params.height)),
    n=(40, 10),
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
            lambda x: np.logical_and(
                np.logical_and(x[0] >= 0.1, x[0] <= 1.9),
                np.logical_and(x[1] >= 0.1, x[1] <= 0.4),
            ): "third_medium",
        },
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): fem.Constant(
                _mesh, ScalarType((0.0, 0.0))
            ),  # Modify for 3D
            lambda x: np.isclose(x[0], 2.0): fem.Constant(
                _mesh, ScalarType((0.0, 0.0))
            ),
            # lambda x: np.logical_and(
            #     np.logical_and(
            #         x[0] > params.width / 1.025 * 0.8, x[0] < params.width / 1.025
            #     ),
            #     np.isclose(x[1], params.height),
            # ): {
            #     1: -0.25
            # },  # Modify for 3D
        },
        neumann_bcs_dict={
            lambda x: np.logical_and(
                np.logical_and(x[0] > 0.8, x[0] < 1.2),
                np.isclose(x[1], params.height),
            ): fem.Constant(
                _mesh, ScalarType((0.0, 0.0))
            ),  # Modify for 3D
        },
    )
)
