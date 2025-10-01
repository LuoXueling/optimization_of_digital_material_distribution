import dolfin as fe
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
    # Body force
    B=fe.Constant((0.0, -0.0)),  # Body force per unit volume # Modify for 3D
)

params.update(
    dotdict(
        mesh=fe.RectangleMesh.create(
            [fe.Point(0.0, 0.0), fe.Point(params.width, params.height)],
            [
                int(params.width // params.mesh_size),
                int(params.height // params.mesh_size),
            ],
            fe.CellType.Type.quadrilateral,
        ),  # Modify for 3D
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && on_boundary", side=0.0
            ): fe.Constant(
                (0.0, 0.0)
            ),  # Modify for 3D
        },
        # Mark Neumann boundary subdomains and values
        neumann_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && near(x[1], pos, length/2) && on_boundary",
                side=params.width,
                pos=params.height / 2,
                length=params.height / 8,
            ): fe.Constant(
                (0.000, -0.0001)
            ),  # Modify for 3D
            # fe.CompiledSubDomain(
            #     "near(x[0], side) && on_boundary",
            #     side=width,
            # ): fe.Constant(
            #     (0.0001, 0.000)
            # ),  # Modify for 3D
        },
        # An alternative to define Dirichlet/Neumann boundaries by using msh2xdmf.py
        # This approach can be used together with dirichlet_bcs_dict and neumann_bcs_dict
        boundaries=None,  # boundaries is a MeshFunction with some boundaries marked for neumann bcs
        association_table=None,  # A dict that maps boundary names to boundary id
        neumann_bcs=None,  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=None,  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
