from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="trapezoid",
    # FEM settings
    dim=2,
    q_deg=2,
    elem_order=1,
    # Optimization settings
    clip_gradient=0.2,
    output_per_step=1,
    maxiter=10000,
    lr=1,
    fix_design_domain="fix_design",
    # Solver settings
    relative_tol=1e-6,
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
    E=1.3373,
    nu=0.49,
    c=0.5,
)

mesh, cell_tags, boundaries, association_table = import_mesh(
    "trapezoid", directory="mesh"
)

params.update(
    dotdict(
        mesh=mesh,
        domain_dict={lambda x: x[1] < 1 - 0.05: "grad_integral_domain"},
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict=None,
        neumann_bcs_dict=None,
        cell_tags=cell_tags,
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=None,  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], (0.0, 0.0)),
            (
                association_table["top"],
                (-0.001, -0.01),
            ),
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
