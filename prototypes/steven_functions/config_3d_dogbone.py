from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import numpy as np
from src.utils import dotdict

params = dotdict(
    # Project settings
    project_name="dogbone",
    # FEM settings
    dim=3,
    q_deg=4,
    elem_order=2,
    # Geometry settings
    width=15,
    length=60,
    thickness=2,
    mesh_size=1,
    # Elasticity parameters
    E=1,
    nu=0.49,
    c=0.0,
)

_mesh = dmesh.create_box(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0, 0.0), (params.width, params.length, params.thickness)),
    n=(
        int(params.width // params.mesh_size),
        int(params.length // params.mesh_size),
        int(params.thickness // params.mesh_size),
    ),
    cell_type=dmesh.CellType.hexahedron,
)

params.update(
    dotdict(
        mesh=_mesh,
        # Body force
        B=fem.Constant(_mesh, ScalarType((0.0, 0.0, 0.0))),
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[1], 0.0): fem.Constant(
                _mesh, ScalarType((0.0, 0.0, 0.0))
            ),
            lambda x: np.isclose(
                x[1], params.length, atol=params.mesh_size
            ): fem.Constant(_mesh, ScalarType((0.0, 0.1, 0.0))),
        },
        neumann_bcs_dict=None,
    )
)
