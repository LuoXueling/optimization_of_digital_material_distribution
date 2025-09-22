import dolfin as fe
from utils import dotdict

params = dotdict(
    # Project settings
    project_name="beam_rightcenter_load",
    # FEM settings
    dim=2,
    q_deg=2,
    elem_order=1,
    # Geometry settings
    width=4,
    height=1,
    mesh_size=0.04,
    # Optimization settings
    target_volume=0.5,
    min_density=0.001,
    init_penalty=1,
    max_penalty=4.0,
    r_min=0.16,
    output_per_step=1,
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
        neumann_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && near(x[1], pos, length) && on_boundary",
                side=params.width,
                pos=params.height / 2,
                length=params.mesh_size * 1.1,
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
    )
)
