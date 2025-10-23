from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="ball",
    # FEM settings
    dim=2,
    q_deg=4,
    elem_order=2,
    # Geometry
    width=1,
    height=1.2,
    # Elasticity parameters
    E=1,
    nu=0.1,
    # Solver settings
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-6,
        "snes_atol": 1e-15,
        "snes_max_it": 100,
    },
    # TMC settings
    kr_coef=1e-4 * 1.2**2,
    psi_coef=1e-3,
    main_medium="main",
    third_medium="third_medium",
)

mesh, cell_tags, boundaries, association_table = import_mesh(
    "ball_indent", directory="mesh"
)

params.update(
    dotdict(
        mesh=mesh,
        # Body force
        B=fem.Constant(
            mesh, ScalarType((0.0, 0.0))
        ),  # Body force per unit volume # Modify for 3D
        cell_tags=cell_tags,
        boundaries=boundaries,
        association_table=association_table,
        domain_dict=None,
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs=[
            (
                association_table["bottom"],
                fem.Constant(mesh, ScalarType((0.0, 0.0))),
            ),
            (association_table["ball"], fem.Constant(mesh, ScalarType((0.0, 0.0)))),
        ],
        dirichlet_bcs_dict=None,
        neumann_bcs_dict=None,
    )
)
