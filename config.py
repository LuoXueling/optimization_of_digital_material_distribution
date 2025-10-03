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

mesh, boundaries, association_table = import_mesh("square", directory="mesh")

params.update(
    dotdict(
        mesh=mesh,
        # Body force
        B=fem.Constant(
            mesh, ScalarType((0.0, 0.0))
        ),  # Body force per unit volume # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): fem.Constant(
                mesh, ScalarType((0.0, 0.0))
            ),  # Modify for 3D
        },
        neumann_bcs_dict=None,
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=[
            (association_table["upper"], fem.Constant(mesh, ScalarType((0.0, -0.001)))),
            (association_table["right"], fem.Constant(mesh, ScalarType((-0.001, 0.0)))),
        ],  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], fem.Constant(mesh, ScalarType((0.0, 0.0))))
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
