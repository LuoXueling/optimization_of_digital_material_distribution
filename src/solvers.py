"""
Helper functions for solving problems in FEniCSx/DOLFINx
"""

from mpi4py import MPI
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import dolfinx.fem as fem
import dolfinx.fem.petsc
from petsc4py import PETSc
import numpy as np
import ufl
import functools


def solve_nonlinear_problem(
    F,
    u,
    bcs=None,
    J=None,
    rtol=1e-6,
    atol=1e-6,
    max_it=50,
    linear_solver="mumps",
    comm=None,
    logger=print,
    snes_type="newtonls",
    linesearch_type="bt",
    snes_options=None,
):
    """
    Solve nonlinear problem using PETSc SNES (Scalable Nonlinear Equations Solvers).

    This is a backport of DOLFINx 0.10.0's approach to using SNES directly with DOLFINx 0.9.0.

    Arguments:
        F: The residual form (UFL form)
        u: The solution function (fem.Function)
        bcs: List of Dirichlet boundary conditions (optional)
        J: The Jacobian form (optional, will be computed automatically if not provided)
        rtol: Relative tolerance for convergence (default: 1e-6)
        atol: Absolute tolerance for convergence (default: 1e-6)
        max_it: Maximum number of iterations (default: 50)
        linear_solver: Linear solver type ("mumps", "superlu_dist", "gmres", "cg") (default: "mumps")
        comm: MPI communicator (default: MPI.COMM_WORLD)
        logger: A callable used for logging
        snes_type: SNES solver type (default: "newtonls")
            Options: "newtonls", "newtontr" (trust region), "nrichardson", "ngmres",
                    "qn" (quasi-Newton), "ngs", "ncg", "fas", "ms", "nasm", "anderson",
                    "aspin", "composite"
        linesearch_type: Line search algorithm (default: "bt")
            Options: "bt" (backtracking), "l2" (L2-norm), "cp" (critical point),
                    "basic", "nleqerr"
        snes_options: Additional PETSc SNES options as dict (optional)

    Returns:
        tuple: (num_iterations, converged) where num_iterations is the number of iterations
               and converged is a boolean indicating if the solver converged
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if bcs is None:
        bcs = []

    # Compute Jacobian if not provided
    if J is None:
        J = ufl.derivative(F, u)

    # Create forms
    residual_form = fem.form(F)
    jacobian_form = fem.form(J)

    # Create matrix and vectors
    A = dolfinx.fem.petsc.create_matrix(jacobian_form)
    b = dolfinx.fem.petsc.create_vector(residual_form)

    # Callback for residual assembly (compatible with SNES.setFunction)
    def assemble_residual_callback(snes_solver, x, F_vec):
        """Assemble residual vector for SNES"""
        # Update ghost values in input vector
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Copy solution from SNES into function
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()

        # Zero residual vector
        with F_vec.localForm() as F_local:
            F_local.set(0.0)

        # Assemble residual
        dolfinx.fem.petsc.assemble_vector(F_vec, residual_form)
        dolfinx.fem.petsc.apply_lifting(
            F_vec, [jacobian_form], bcs=[bcs], x0=[x], alpha=-1.0
        )
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F_vec, bcs, x, -1.0)

        # Final ghost update to ensure consistency across processes
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Callback for Jacobian assembly (compatible with SNES.setJacobian)
    def assemble_jacobian_callback(snes_solver, x, J_mat, P_mat):
        """Assemble Jacobian matrix for SNES"""
        # Update ghost values in input vector
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Copy solution from SNES into function
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()

        # Zero and assemble Jacobian
        J_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J_mat, jacobian_form, bcs=bcs)
        J_mat.assemble()

        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    # Monitoring callback for logging
    iteration_data = {"count": 0, "residual_norm0": None}

    def monitor_callback(snes_solver, iteration, residual_norm):
        """Monitor callback for logging iteration progress"""
        if iteration == 0:
            iteration_data["residual_norm0"] = (
                residual_norm if residual_norm > 0 else 1.0
            )

        residual_rel = residual_norm / iteration_data["residual_norm0"]
        logger(
            f"  Iteration {iteration}: residual abs norm = {residual_norm:.5e} ({atol}) "
            f"rel norm = {residual_rel:.5e} ({rtol})"
        )
        iteration_data["count"] = iteration

    # Create SNES solver
    snes = PETSc.SNES().create(comm)
    snes.setType(snes_type)

    # Set residual and Jacobian callbacks
    snes.setFunction(assemble_residual_callback, b)
    snes.setJacobian(assemble_jacobian_callback, A, A)

    # Configure line search
    if linesearch_type:
        snes.getLineSearch().setType(linesearch_type)

    # Set tolerances
    snes.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    # Configure linear solver (KSP)
    ksp = snes.getKSP()
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
        ksp.setType(linear_solver)

    # Apply custom SNES options if provided
    if snes_options:
        opts = PETSc.Options()
        for key, value in snes_options.items():
            opts[key] = value
        snes.setFromOptions()

    # Set monitoring callback
    snes.setMonitor(monitor_callback)

    # Log start
    logger(
        f"Starting PETSc SNES solver (type: {snes_type}, line search: {linesearch_type})"
    )

    # Solve
    snes.solve(None, u.x.petsc_vec)
    u.x.scatter_forward()

    # Check convergence
    reason = snes.getConvergedReason()
    num_iterations = snes.getIterationNumber()
    final_residual = snes.getFunctionNorm()

    # Cleanup
    snes.destroy()
    A.destroy()
    b.destroy()

    # Check for NaN
    if np.isnan(final_residual) or np.isinf(final_residual):
        logger(f"  ERROR: Final residual is {final_residual}")
        logger(f"  SNES convergence reason: {reason}")
        raise Exception(f"SNES solver encountered NaN/Inf in final residual")

    # Determine if converged
    converged = reason > 0

    if converged:
        logger(
            f"  SNES solver converged in {num_iterations} iterations (reason: {reason})"
        )
    else:
        logger(f"  SNES solver did not converge (reason: {reason})")
        logger(f"  Final residual norm: {final_residual:.5e}")
        raise Exception(
            f"SNES solver did not converge after {num_iterations} iterations (reason: {reason})"
        )

    return num_iterations, converged


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
