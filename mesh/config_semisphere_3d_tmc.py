from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.fem as fem
import numpy as np
from src.utils import dotdict
from src.msh2xdmf import import_mesh

params = dotdict(
    # Project settings
    project_name="semisphere_3d",
    # FEM settings
    dim=3,
    q_deg=4,
    elem_order=2,
    # Optimization settings
    clip_gradient=0.2,
    output_per_step=10,
    maxiter=100000,
    lr=10,
    fix_design_domain="third_medium",
    fix_density_h=0.04,
    # clip_gradient=0.01,  # clipping acts on lr-scaled gradients
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
    nu=0.49,
    c=0.5,
    # TMC settings
    tmc_regularization="HuHu",
    kr_coef=1e-2,
    # Increase it for better convergence when seeing that elements are overlapping.
    # Convergence is much different when psi_coef is too high
    psi_coef=1e-5,
    # Suddenly diverges at contact? Increase it.
    # If the third medium is too thick and they deforms far before contact,
    # use thinner third medium, larger elements in third medium, quad elements, and decrease this value
    main_medium="main",
    third_medium="third_medium",
)

mesh, cell_tags, boundaries, association_table = import_mesh(
    "semisphere_3d", directory="mesh", dim=params.dim
)

params.update(
    dotdict(
        mesh=mesh,
        domain_dict=None,
        # domain_dict={
        #     lambda x: np.logical_and(
        #         (x[0] - 0.75) ** 2 + x[1] ** 2 <= (0.5 - 0.04) ** 2, x[1] > 0.04
        #     ): "grad_integral_domain"
        # },
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict=None,
        neumann_bcs_dict=None,
        cell_tags=cell_tags,
        boundaries=boundaries,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=association_table,
        neumann_bcs=None,  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], (0.0, 0.0, 0.0)),
            (
                association_table["top"],
                {0: 0.0, 1: 0.0, 2: (lambda x: (-0.1 / 1.5 * x[0] - 0.36))},
                # (-0.1 / 1.5 * x[0] - 0.36)
                # (-0.3 / 1.5 * x[0] - 0.24)
                # (-0.5 / 1.5 * x[0] - 0.12)
                # (-0.7 / 1.5 * x[0])
                # (-0.9 / 1.5 * x[0] + 0.12)
            ),
            (association_table["third_medium_bottom"], {2: 0}),
        ],
    )
)
