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
    q_deg=4,
    elem_order=2,
    # Optimization settings
    target_volume=0.5,
    min_density=0.001,
    init_penalty=1.0,
    max_penalty=4.0,
    gradient_filter=None,
    fix_density_layers=None,
    fictitious_interpolation=False,  # For finite strain, 10.1016/j.cma.2014.03.021
    clip_gradient=0.2,
    output_per_step=1,
    maxiter=10000,
    lr=1,
    # clip_gradient=0.05,  # clipping acts on lr-scaled gradients
    # Solver settings
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-5,
        "snes_atol": 1e-5,
        "snes_max_it": 200,
        "snes_linesearch_max_it": 100,
        "snes_linesearch_order": 1,
    },
    # Elasticity parameters
    E=1.3373,
    nu=0.49,
    c=1.0,
    # TMC settings
    tmc_regularization="rotation_angle",
    kr_coef=4e-2,
    # Increase it for better convergence when seeing that elements are overlapping.
    # Convergence is much different when psi_coef is too high
    psi_coef=5e-5,
    # Suddenly diverges at contact? Increase it.
    # If the third medium is too thick and they deforms far before contact,
    # use thinner third medium, quad elements, and decrease this value
    main_medium="main",
    third_medium="third_medium",
)

mesh, cell_tags, boundaries, association_table = import_mesh(
    "trapezoid_tmc", directory="mesh"
)

params.update(
    dotdict(
        mesh=mesh,
        # Body force
        B=fem.Constant(
            mesh, ScalarType((0.0, 0.0))
        ),  # Body force per unit volume # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict=None,
        neumann_bcs_dict=None,
        cell_tags=cell_tags,
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        domain_dict=None,
        neumann_bcs=None,  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], fem.Constant(mesh, ScalarType((0.0, 0.0)))),
            (
                association_table["top"],
                fem.Constant(mesh, ScalarType((0.0, -0.25))),
            ),
            (
                association_table["third_medium_bottom"],
                {1: 0},
            ),
            # (association_table["left"], {0: 0}),
            # (association_table["right"], {0: 0}),
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
