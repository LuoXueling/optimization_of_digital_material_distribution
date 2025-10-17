import dolfinx.mesh as dmesh
from mpi4py import MPI
import dolfinx.fem as fem
import ufl

_mesh = dmesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2, 1)),
    n=(10, 5),
    cell_type=dmesh.CellType.quadrilateral,
)

V_e = fem.functionspace(_mesh, ("Lagrange", 1))

e = fem.Function(V_e)
e_test = ufl.TestFunction(V_e)

form = e * ufl.dx
ufl.derivative(form, e, e_test)
