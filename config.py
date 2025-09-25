import dolfin as fe
from src.utils import dotdict
from src.msh2xdmf import msh2xdmf, import_mesh

params = dotdict(
    # Project settings
    project_name="square",
    # FEM settings
    dim=2,
    q_deg=2,
    # Elasticity parameters
    E=10.0,
    nu=0.3,
    # Body force
    B=fe.Constant((0.0, -0.0)),  # Body force per unit volume # Modify for 3D
)

mesh, boundaries, association_table = import_mesh("square", directory="mesh")

params.update(
    dotdict(
        mesh=mesh,
        # Mark Dirichlet boundary subdomains and values
        dirichlet_bcs_dict={
            fe.CompiledSubDomain(
                "near(x[0], side) && on_boundary", side=0.0
            ): fe.Constant(
                (0.0, 0.0)
            ),  # Modify for 3D
        },
        neumann_bcs_dict=None,
        boundaries=boundaries,  # boundaries is a MeshFunction with some boundaries marked for neumann bcs
        neumann_bcs=[
            (association_table["upper"], fe.Constant((0.0, -0.001))),
            (association_table["right"], fe.Constant((-0.001, 0.0))),
        ],  # a list of tuples (boundary_id, (f_x, f_y, f_z)).
        dirichlet_bcs=[
            (association_table["bottom"], fe.Constant((0.0, 0.0)))
        ],  # a list of tuples (boundary_id, (u_x, u_y, u_z))
    )
)
