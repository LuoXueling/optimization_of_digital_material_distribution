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


def solve_nonlinear_problem(
    F,
    u,
    bcs=None,
    J=None,
    rtol=1e-9,
    atol=1e-10,
    max_it=50,
    linear_solver="mumps",
    comm=None,
    logger=print,
    use_linesearch=True,
    max_linesearch_it=10,
    alpha_min=1e-4,
    c1=1e-4,
):
    """
    Custom Newton solver for nonlinear problems with backtracking line search.

    This implements Newton's method directly using PETSc matrices and vectors,
    giving more control over the iteration process. Includes optional backtracking
    line search to ensure residual reduction.

    Arguments:
        F: The residual form (UFL form)
        u: The solution function (fem.Function)
        bcs: List of Dirichlet boundary conditions (optional)
        J: The Jacobian form (optional, will be computed automatically if not provided)
        rtol: Relative tolerance for convergence (default: 1e-9)
        atol: Absolute tolerance for convergence (default: 1e-10)
        max_it: Maximum number of Newton iterations (default: 50)
        linear_solver: Linear solver type ("mumps", "superlu_dist", etc.) (default: "mumps")
        comm: MPI communicator (default: MPI.COMM_WORLD)
        logger: A callable used for logging
        use_linesearch: Enable backtracking line search (default: True)
        max_linesearch_it: Maximum line search iterations (default: 10)
        alpha_min: Minimum step size for line search (default: 1e-4)
        c1: Armijo condition parameter (default: 1e-4)

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
    L = dolfinx.fem.petsc.create_vector(residual_form)
    dx = fem.Function(u.function_space)

    # For line search: store trial solution and residual
    if use_linesearch:
        u_trial = fem.Function(u.function_space)
        L_trial = dolfinx.fem.petsc.create_vector(residual_form)

    # Create linear solver
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)

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

    # Newton iterations
    i = 0
    converged = False
    residual_norm0 = None

    logger(f"Starting custom Newton solver")

    while i < max_it:
        # Assemble Jacobian and residual
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()

        dolfinx.fem.petsc.assemble_matrix(A, jacobian_form, bcs=bcs)
        A.assemble()

        dolfinx.fem.petsc.assemble_vector(L, residual_form)

        # Apply lifting for boundary conditions
        dolfinx.fem.petsc.apply_lifting(
            L, [jacobian_form], bcs=[bcs], x0=[u.x.petsc_vec], alpha=-1.0
        )
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set BC values in residual
        fem.petsc.set_bc(L, bcs, u.x.petsc_vec, -1.0)

        # Compute residual norm
        residual_norm = L.norm()

        if i == 0:
            residual_norm0 = residual_norm
            if residual_norm0 == 0:
                residual_norm0 = 1.0  # Avoid division by zero

        # Compute relative residual
        residual_rel = residual_norm / residual_norm0 if residual_norm0 > 0 else 0.0

        # Log iteration info
        logger(
            f"  Iteration {i}: residual abs norm = {residual_norm:.5e} ({atol}) rel norm = {residual_rel:.5e} ({rtol})"
        )

        # Check convergence
        if residual_norm < atol:
            converged = True
            logger(f"  Converged (absolute tolerance)")
            break

        if residual_rel < rtol:
            converged = True
            logger(f"  Converged (relative tolerance)")
            break

        # Solve linear system: A * du = -F, which is A * du = L (since L was set to -F)
        ksp.solve(L, dx.x.petsc_vec)
        dx.x.scatter_forward()

        # Backtracking line search
        if use_linesearch:
            alpha = 1.0  # Start with full Newton step
            ls_iter = 0
            accepted = False

            while ls_iter < max_linesearch_it and not accepted:
                # Compute trial solution: u_trial = u - alpha * dx
                u_trial.x.array[:] = u.x.array[:]
                u_trial.x.petsc_vec.axpy(-alpha, dx.x.petsc_vec)
                u_trial.x.scatter_forward()

                # Temporarily update u to evaluate residual at trial point
                u_backup = u.x.array.copy()
                u.x.array[:] = u_trial.x.array[:]
                u.x.scatter_forward()

                # Assemble residual at trial point
                with L_trial.localForm() as loc_L:
                    loc_L.set(0)
                dolfinx.fem.petsc.assemble_vector(L_trial, residual_form)
                dolfinx.fem.petsc.apply_lifting(
                    L_trial, [jacobian_form], bcs=[bcs], x0=[u.x.petsc_vec], alpha=-1.0
                )
                L_trial.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )
                fem.petsc.set_bc(L_trial, bcs, u.x.petsc_vec, -1.0)

                # Compute trial residual norm
                residual_norm_trial = L_trial.norm()

                # Restore u to original value
                u.x.array[:] = u_backup
                u.x.scatter_forward()

                # Check Armijo condition: ||F(u + alpha*p)|| <= (1 - c1*alpha) * ||F(u)||
                # For Newton, this ensures sufficient decrease in residual
                if (
                    residual_norm_trial <= (1.0 - c1 * alpha) * residual_norm
                    or alpha < alpha_min
                ):
                    accepted = True
                    if ls_iter > 0:
                        logger(
                            f"    Line search: alpha = {alpha:.4e}, residual = {residual_norm_trial:.5e} (after {ls_iter + 1} iterations)"
                        )
                else:
                    # Backtrack: reduce step size
                    alpha *= 0.5
                    ls_iter += 1

            if not accepted:
                logger(
                    f"    Line search failed to find acceptable step size, using alpha = {alpha:.4e}"
                )

            # Update solution with accepted step
            u.x.array[:] = u_trial.x.array[:]
            u.x.scatter_forward()
        else:
            # No line search: use full Newton step
            # Update solution: u = u - dx (Newton correction)
            u.x.petsc_vec.axpy(-1.0, dx.x.petsc_vec)
            u.x.scatter_forward()

        i += 1

    # Cleanup
    ksp.destroy()
    A.destroy()
    L.destroy()
    if use_linesearch:
        L_trial.destroy()

    if not converged:
        logger(f"  Newton solver did not converge after {i} iterations")
        raise Exception(f"Newton solver did not converge after {i} iterations")
    else:
        logger(f"  Newton solver converged in {i} iterations")

    return i, converged


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
