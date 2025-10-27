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
    # Optimization settings
    target_volume=0.95,
    min_density=0.001,
    init_penalty=1.0,
    max_penalty=4.0,
    fix_density_layers=None,
    r_min=None,
    output_per_step=1,
    maxiter=10000,
    lr=10,
    # clip_gradient=0.05,  # clipping acts on lr-scaled gradients
    # Elasticity parameters
    E=10.0,
    nu=0.3,
    c=0.5,
)

mesh, cell_tags, boundaries, association_table = import_mesh("square", directory="mesh")

params.update(
    dotdict(
        mesh=mesh,
        # Body force
        B=fem.Constant(
            mesh, ScalarType((0.0, 0.0))
        ),  # Body force per unit volume # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            # lambda x: np.isclose(x[1], 1.0): {1: lambda x: -0.01 * np.ones_like(x[0])}
        },
        neumann_bcs_dict={
            # lambda x: np.isclose(x[1], 1.0): lambda x: np.vstack(
            #     [np.zeros_like(x[0]), -0.001 * np.ones_like(x[0])]
            # )
        },
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=[
            # (association_table["upper"], fem.Constant(mesh, ScalarType((0.0, -0.001)))),
            # (association_table["right"], fem.Constant(mesh, ScalarType((-0.01, 0.0)))),
            (
                association_table["upper"],
                lambda x: np.vstack([np.zeros_like(x[0]), 0.001 * (x[0] - 0.75) * 4]),
            ),
            (
                association_table["right"],
                lambda x: np.vstack([0.001 * (x[1] - 0.75) * 4, np.zeros_like(x[0])]),
            ),
        ],  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], fem.Constant(mesh, ScalarType((0.0, 0.0)))),
            # (association_table["upper"], {1: lambda x: -0.001 * np.ones_like(x[0])}),
            # (
            #     association_table["right"],
            #     {0: lambda x: -0.001 * np.ones_like(x[1])},
            # ),
            # (association_table["upper"], {1: lambda x: -0.001 * (x[0] - 0.75) * 4}),
            # (
            #     association_table["right"],
            #     {0: lambda x: -0.001 * (x[1] - 0.75) * 4, 1: 0.0},
            # ),
            # (association_table["right"], {0: -0.001}),
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
