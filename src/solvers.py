"""
Helper functions for solving problems in FEniCSx/DOLFINx
"""

from mpi4py import MPI
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import dolfinx.fem as fem


def solve_nonlinear_problem(
    F,
    u,
    bcs=None,
    J=None,
    rtol=1e-9,
    max_it=50,
    linear_solver="mumps",
    convergence_criterion="residual",
    comm=None,
    log_file_path=None,
):
    """
    Solve a nonlinear variational problem using Newton's method.

    This is a helper function to simplify the common pattern of setting up
    and solving nonlinear problems in DOLFINx.

    Arguments:
        F: The residual form (UFL form)
        u: The solution function (fem.Function)
        bcs: List of Dirichlet boundary conditions (optional)
        J: The Jacobian form (optional, will be computed automatically if not provided)
        rtol: Relative tolerance for convergence (default: 1e-6)
        max_it: Maximum number of Newton iterations (default: 50)
        linear_solver: Linear solver type ("mumps", "superlu_dist", etc.) (default: "mumps")
        convergence_criterion: Convergence criterion ("incremental" or "residual") (default: "residual")
        comm: MPI communicator (default: MPI.COMM_WORLD)

    Returns:
        tuple: (num_iterations, converged) where num_iterations is the number of iterations
               and converged is a boolean indicating if the solver converged

    Example:
        >>> F = ufl.derivative(Pi, u, v)
        >>> num_its, converged = solve_nonlinear_problem(F, u, bcs=bcs)
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if bcs is None:
        bcs = []

    # Create the nonlinear problem
    problem = NonlinearProblem(F, u, bcs=bcs, J=J)

    # Create Newton solver
    solver = NewtonSolver(comm, problem)

    # Configure solver
    solver.convergence_criterion = convergence_criterion
    solver.rtol = rtol
    solver.max_it = max_it

    # Set up logging
    from dolfinx import log as dlog
    from src.utils import redirect_stdout_to_log, restore_stdout_from_log

    # Temporarily set log level to INFO to capture Newton iterations
    original_log_level = dlog.get_log_level()
    dlog.set_log_level(dlog.LogLevel.INFO)

    # Redirect stdout to both terminal and log file
    old_stdout_fd = None
    log_file_info = None

    if log_file_path and comm.rank == 0:
        old_stdout_fd, log_file_info = redirect_stdout_to_log(log_file_path)

    # Configure linear solver
    ksp = solver.krylov_solver

    if linear_solver == "mumps":
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
    elif linear_solver == "superlu_dist":
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("superlu_dist")
    elif linear_solver == "gmres":
        ksp.setType("gmres")
        ksp.getPC().setType("ilu")
    elif linear_solver == "cg":
        ksp.setType("cg")
        ksp.getPC().setType("ilu")
    else:
        # Custom solver - just set the type
        ksp.setType(linear_solver)

    # Solve
    num_its, converged = solver.solve(u)

    # Restore stdout
    if log_file_info:
        restore_stdout_from_log(old_stdout_fd, log_file_info)

    # Restore original log level
    dlog.set_log_level(original_log_level)

    return num_its, converged


def solve_linear_problem(a, L, u, bcs=None, linear_solver="mumps", comm=None):
    """
    Solve a linear variational problem.

    This is a helper function to simplify the common pattern of setting up
    and solving linear problems in DOLFINx.

    Arguments:
        a: The bilinear form (UFL form)
        L: The linear form (UFL form)
        u: The solution function (fem.Function) - will be updated in place
        bcs: List of Dirichlet boundary conditions (optional)
        linear_solver: Linear solver type ("mumps", "superlu_dist", etc.) (default: "mumps")
        comm: MPI communicator (default: MPI.COMM_WORLD)

    Returns:
        fem.Function: The solution (same as u, which is modified in place)

    Example:
        >>> a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        >>> L = ufl.inner(f, test) * ufl.dx
        >>> u = fem.Function(V)
        >>> solve_linear_problem(a, L, u, bcs=bcs)
    """
    from dolfinx.fem.petsc import LinearProblem

    if comm is None:
        comm = MPI.COMM_WORLD

    if bcs is None:
        bcs = []

    # Set up PETSc options for the linear solver
    petsc_options = {}
    if linear_solver == "mumps":
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    elif linear_solver == "superlu_dist":
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
        }
    elif linear_solver == "gmres":
        petsc_options = {"ksp_type": "gmres", "pc_type": "ilu"}
    elif linear_solver == "cg":
        petsc_options = {"ksp_type": "cg", "pc_type": "ilu"}

    # Create and solve the linear problem
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    u_solution = problem.solve()

    # Copy solution into u
    u.x.array[:] = u_solution.x.array[:]

    return u
