from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import numpy as np
from src.utils import dotdict


params = dotdict(
    # Project settings
    project_name="beam_rightcenter_load",
    output_root="outputs",
    # FEM settings
    dim=2,
    q_deg=2,
    elem_order=1,  # Only setting order of displacement field. The pressure field and optimized variables are kept first order
    # Geometry settings
    width=4,
    height=1,
    mesh_size=0.02,
    # Optimization settings
    output_per_step=10,
    maxiter=10000,
    lr=1,
    clip_gradient=0.2,  # clipping acts on lr-scaled gradients
    # Solver settings
    relative_tol=0.001,
    snes_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-10,
        "snes_max_it": 100,
    },
    num_retries=None,  # Number of retries after the solver fails. Add random pertubation to the initial value after failure. Useful when numerical error is essential.
    # Elasticity parameters
    E=10.0,
    nu=0.3,
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
        # Body force
        B=None,  # Body force per unit volume # Modify for 3D
        # Domain definitions (for subdomain integration)
        # domain_dict: Define domains using geometric marker functions
        #   Format: {lambda x: condition: "domain_name"}
        #   Examples:
        #     lambda x: x[0] < 0.5: "left_half"
        #     lambda x: (x[0] - 0.5)**2 + (x[1] - 0.5)**2 < 0.25: "circle"
        domain_dict=None,
        # Mark Dirichlet boundary subdomains and values
        # dirichlet_bcs_dict: Define Dirichlet BCs using geometric marker functions
        #   - Key: lambda function that marks boundary (e.g., lambda x: np.isclose(x[0], 0.0))
        #   - Value: tuple, callable function, or dict for component-wise BCs
        #
        #   Constant example:
        #     lambda x: np.isclose(x[0], 0.0): (0.0, 0.0)
        #   Function example (spatially-varying):
        #     lambda x: np.isclose(x[0], width): lambda x: np.vstack([np.zeros_like(x[0]), -0.01*(x[1]-0.5)**2])
        #   Component-wise example:
        #     lambda x: np.isclose(x[0], 0.0): {0: 0.0, 1: lambda x: 0.01*x[0]}  # u_x=0, u_y varies
        dirichlet_bcs_dict=None,
        # Mark Neumann boundary subdomains and values
        # It has a syntax similar to dirichlet_bcs_dict, except that only the Constant and Function is supported
        neumann_bcs_dict=None,
        # An alternative to define Dirichlet/Neumann boundaries by using msh2xdmf.py
        # This approach can be used together with dirichlet_bcs_dict and neumann_bcs_dict
        boundaries=None,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=None,  # A dict that maps boundary names to boundary id
        # dirichlet_bcs: Define Dirichlet BCs using boundary marker IDs (from MeshTags)
        #   Format: list of tuples (marker_id, value) where value follows same format as dirichlet_bcs_dict
        #   Examples: [(1, (0.0, 0.0))] or [(1, {0: 0.0}), (2, {1: 0.0})]
        dirichlet_bcs=None,
        # neumann_bcs has a syntax similar to dirichlet_bcs, except that only the Constant and Function is supported
        neumann_bcs=None,
        # dirichlet_bcs_dict and neumann_bcs_dict will be merged into dirichlet_bcs and neumann_bcs
    )
)
