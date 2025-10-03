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
    plane_state="stress",
    q_deg=2,
    elem_order=1,
    # Geometry settings
    width=4,
    height=1,
    mesh_size=0.02,
    # Optimization settings
    output_per_step=10,
    maxiter=10000,
    lr=1,
    # Solver settings
    relative_tol=0.001,
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
        B=fem.Constant(_mesh, ScalarType((0.0, 0.0))),  # Body force per unit volume # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        # dirichlet_bcs_dict: Define Dirichlet BCs using geometric marker functions
        #   - Key: lambda function that marks boundary (e.g., lambda x: np.isclose(x[0], 0.0))
        #   - Value: Either a full-vector Constant (all components) OR a dict for component-wise BCs
        #
        #   Full-vector example (constrains all displacement components):
        #     lambda x: np.isclose(x[0], 0.0): fem.Constant(_mesh, ScalarType((0.0, 0.0)))
        #
        #   Component-wise example (constrains only specific components):
        #     lambda x: np.isclose(x[0], 0.0): {0: 0.0}  # Only constrains u_x = 0.0
        #     lambda x: np.isclose(x[1], 0.0): {1: fem.Constant(_mesh, ScalarType(0.0))}  # Only constrains u_y
        #     lambda x: np.isclose(x[0], width): {0: 0.1, 1: 0.0}  # Constrains both u_x and u_y
        dirichlet_bcs_dict={
            lambda x: np.isclose(x[0], 0.0): fem.Constant(_mesh, ScalarType((0.0, 0.0))),  # Modify for 3D
        },
        # Mark Neumann boundary subdomains and values
        neumann_bcs_dict={
            lambda x: np.isclose(x[0], params.width) & np.isclose(x[1], params.height / 2, atol=params.height / 8 / 2): fem.Constant(_mesh, ScalarType((0.0, -0.0001))),  # Modify for 3D
        },
        # An alternative to define Dirichlet/Neumann boundaries by using msh2xdmf.py
        # This approach can be used together with dirichlet_bcs_dict and neumann_bcs_dict
        boundaries=None,  # boundaries is a MeshTags with some boundaries marked for neumann bcs
        association_table=None,  # A dict that maps boundary names to boundary id
        neumann_bcs=None,  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        # dirichlet_bcs: Define Dirichlet BCs using boundary marker IDs (from MeshTags)
        #   Format: list of tuples (marker_id, value) where value follows same format as dirichlet_bcs_dict
        #   Examples: [(1, fem.Constant(_mesh, ScalarType((0.0, 0.0))))] or [(1, {0: 0.0}), (2, {1: 0.0})]
        dirichlet_bcs=None,
    )
)
