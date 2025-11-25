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
    linear_solver="mumps",
    comm=None,
    logger=print,
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
        linear_solver: Linear solver type (default: "mumps")
            Options: "mumps", "superlu_dist", "gmres", "cg"
        comm: MPI communicator (default: MPI.COMM_WORLD)
        logger: A callable used for logging (default: print)
        snes_options: PETSc SNES options as dict (optional)
            All SNES configuration should be set here. Common options:
            - "snes_type": Solver type (default: "newtonls")
            - "snes_linesearch_type": Line search (default: "basic")
            - "snes_rtol": Relative tolerance (default: 1e-6)
            - "snes_atol": Absolute tolerance (default: 1e-15)
            - "snes_max_it": Maximum iterations (default: 50)
            - "snes_linesearch_max_it": Max line search iterations
            - "snes_linesearch_minlambda": Min step length
            - "snes_monitor": Enable PETSc iteration monitoring
            - "snes_converged_reason": Show convergence reason

            Example:
            {
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-8,
                "snes_atol": 1e-12,
                "snes_max_it": 100
            }

    Returns:
        tuple: (num_iterations, converged) where num_iterations is the number of iterations
               and converged is a boolean indicating if the solver converged

    Notes:
        - Concise line search monitor logs when step size is reduced (alpha < 1.0)
        - Ghost values are properly synchronized for MPI parallel execution
        - Independent solution vector copy ensures line search works correctly
        - For ill-conditioned problems, try: {"snes_type": "newtontr"}
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

    # Extract SNES configuration from snes_options with defaults
    if snes_options is None:
        snes_options = {}

    snes_type = snes_options.get("snes_type", "newtonls")
    linesearch_type = snes_options.get("snes_linesearch_type", "basic")
    rtol = snes_options.get("snes_rtol", 1e-6)
    atol = snes_options.get("snes_atol", 1e-15)
    max_it = snes_options.get("snes_max_it", 50)

    # Monitoring callback for logging
    iteration_data = {"count": 0, "residual_norm0": None, "linesearch_count": 0}

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

        # Check line search status after iteration (except iteration 0)
        if iteration > 0 and linesearch_type and linesearch_type != "basic":
            try:
                ls = snes_solver.getLineSearch()
                alpha = ls.getStepLength()
                # Only log if step was reduced (alpha < 1.0)
                if alpha < 0.999:  # Use 0.999 to account for floating point
                    reason = ls.getReason()
                    logger(f"    Line search: alpha = {alpha:.6e} (reason: {reason})")
                    iteration_data["linesearch_count"] += 1
            except:
                pass  # Silently ignore if line search info unavailable

        iteration_data["count"] = iteration

    # Create SNES solver
    snes = PETSc.SNES().create(comm)

    # Set residual and Jacobian callbacks
    snes.setFunction(assemble_residual_callback, b)
    snes.setJacobian(assemble_jacobian_callback, A, A)

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

    # Apply SNES options to PETSc
    opts = PETSc.Options()
    for key, value in snes_options.items():
        opts[key] = value

    # Set monitoring callback
    snes.setMonitor(monitor_callback)

    # Apply PETSc options (must be called after all configuration)
    # This reads snes_type, snes_linesearch_type, and other options from PETSc.Options()
    snes.setFromOptions()

    # Log start
    logger(
        f"Starting PETSc SNES solver (type: {snes_type}, line search: {linesearch_type})"
    )

    # Create independent solution vector for line search
    # Critical: Line search needs an independent vector to evaluate different step sizes
    # If we pass u.x.petsc_vec directly, it shares memory and line search fails
    x = u.x.petsc_vec.copy()
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Solve
    snes.solve(None, x)

    # Copy solution back to u
    x.copy(u.x.petsc_vec)
    u.x.scatter_forward()

    # Check convergence
    reason = snes.getConvergedReason()
    num_iterations = snes.getIterationNumber()
    final_residual = snes.getFunctionNorm()

    # Cleanup
    snes.destroy()
    A.destroy()
    b.destroy()
    x.destroy()

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


def solve_nonlinear_problem_with_retry(
    F,
    u,
    bcs=None,
    J=None,
    linear_solver="mumps",
    comm=None,
    logger=print,
    snes_options=None,
    num_retries=3,
    reset_to_initial=False,
):
    """
    Wrapper for solve_nonlinear_problem that implements retry logic with state restoration.

    If the solver fails (raises an exception), this function will retry up to num_retries times.
    By default, it continues from the last failed state with perturbation, or optionally resets
    to the initial state with perturbation on each retry.

    Arguments:
        F: The residual form (UFL form)
        u: The solution function (fem.Function)
        bcs: List of Dirichlet boundary conditions (optional)
        J: The Jacobian form (optional, will be computed automatically if not provided)
        linear_solver: Linear solver type (default: "mumps")
        comm: MPI communicator (default: MPI.COMM_WORLD)
        logger: A callable used for logging (default: print)
        snes_options: PETSc SNES options as dict (optional)
        num_retries: Number of retry attempts if solver fails (default: 3)
        reset_to_initial: If True, reset to initial state + perturbation on each retry.
                         If False, continue from last failed state + perturbation (default: False)

    Returns:
        tuple: (num_iterations, converged) from successful solve

    Raises:
        Exception: If all retry attempts fail

    Example:
        >>> # Continue from last failed state (default)
        >>> solve_nonlinear_problem_with_retry(
        ...     F, u, bcs=bcs, J=J,
        ...     snes_options={"snes_rtol": 1e-8},
        ...     num_retries=5
        ... )
        >>>
        >>> # Reset to initial state on each retry
        >>> solve_nonlinear_problem_with_retry(
        ...     F, u, bcs=bcs, J=J,
        ...     snes_options={"snes_rtol": 1e-8},
        ...     num_retries=5,
        ...     reset_to_initial=True
        ... )
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    if num_retries is None or num_retries == 0:
        return solve_nonlinear_problem(
            F=F,
            u=u,
            bcs=bcs,
            J=J,
            linear_solver=linear_solver,
            comm=comm,
            logger=logger,
            snes_options=snes_options,
        )

    # Store the initial state of u
    u_initial = u.x.array.copy()

    for attempt in range(num_retries + 1):
        try:
            if attempt > 0:
                logger(f"Retry attempt {attempt}/{num_retries}...")

                if reset_to_initial:
                    # Strategy 1: Reset to initial state with increasing perturbation
                    logger("  Strategy: Reset to initial state with perturbation")
                    perturbation = 1e-10 * 10**attempt * np.random.randn(len(u_initial))
                    u.x.array[:] = u_initial + perturbation * u_initial
                else:
                    # Strategy 2: Continue from last failed state with small perturbation
                    logger("  Strategy: Continue from last state with perturbation")
                    u_last = u.x.array.copy()
                    perturbation = 1e-15 * 10**attempt * np.random.randn(len(u_last))
                    u.x.array[:] = u_last + perturbation * u_last

                u.x.scatter_forward()

            # Attempt to solve
            result = solve_nonlinear_problem(
                F=F,
                u=u,
                bcs=bcs,
                J=J,
                linear_solver=linear_solver,
                comm=comm,
                logger=logger,
                snes_options=snes_options,
            )

            # If we get here, the solve was successful
            if attempt > 0:
                logger(f"Retry attempt {attempt} succeeded!")
            return result

        except Exception as e:
            if attempt < num_retries:
                logger(f"Solver failed on attempt {attempt + 1}: {str(e)}")
                logger(f"Restoring initial state and retrying...")
            else:
                # Final attempt failed, re-raise the exception
                logger(f"All {num_retries + 1} solver attempts failed. Giving up.")
                raise


def solve_linear_problem(
    a,
    L,
    u,
    bcs=None,
    linear_solver="mumps",
    comm=None,
    snes_options=None,
    logger=print,
    A_cache=None,
    b_cache=None,
):
    """
    Solve a linear variational problem using PETSc SNES (ksponly).

    This mirrors ``solve_nonlinear_problem`` but with a fixed linear operator,
    so it can drop-in replace ``dolfinx.fem.petsc.LinearProblem`` while sharing
    the same assembly logic and boundary-condition handling in parallel.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if bcs is None:
        bcs = []
    if logger is None:
        logger = lambda *args, **kwargs: None

    # Forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Matrix/vector holders (reuse if provided)
    A_external = A_cache is not None
    b_external = b_cache is not None
    A = A_cache if A_cache is not None else fem.petsc.create_matrix(a_form)
    b = b_cache if b_cache is not None else fem.petsc.create_vector(L_form)  # RHS
    r = fem.petsc.create_vector(L_form)  # residual workspace

    # Assemble once (matrix is constant)
    def assemble_matrix(J_mat):
        J_mat.zeroEntries()
        fem.petsc.assemble_matrix(J_mat, a_form, bcs=bcs)
        J_mat.assemble()

    def assemble_rhs(rhs_vec):
        with rhs_vec.localForm() as r_loc:
            r_loc.set(0.0)
        fem.petsc.assemble_vector(rhs_vec, L_form)
        fem.petsc.apply_lifting(rhs_vec, [a_form], bcs=[bcs])
        rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in bcs:
            bc.set(rhs_vec.array_w)

    # Residual: F(x) = A x - b
    def assemble_residual_callback(snes_solver, x, F_vec):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()

        A.mult(x, F_vec)
        F_vec.axpy(-1.0, b)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Jacobian: constant A
    def assemble_jacobian_callback(snes_solver, x, J_mat, P_mat):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()
        # Matrix already assembled once; reuse
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    # SNES setup
    snes = PETSc.SNES().create(comm)
    snes.setFunction(assemble_residual_callback, r)
    snes.setJacobian(assemble_jacobian_callback, A, A)

    # Linear solver options via KSP
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

    # SNES options
    opts = PETSc.Options()
    if snes_options is None:
        snes_options = {}
    snes_type = snes_options.get("snes_type", "ksponly")
    opts["snes_type"] = snes_type
    for key, val in snes_options.items():
        opts[key] = val
    snes.setFromOptions()

    # Assemble A and b once before solve
    assemble_matrix(A)
    assemble_rhs(b)

    logger(f"Starting SNES linear solve (snes_type={snes_type}, ksp={ksp.getType()})")

    x = u.x.petsc_vec.copy()
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    snes.solve(None, x)

    x.copy(u.x.petsc_vec)
    u.x.scatter_forward()

    reason = snes.getConvergedReason()
    its = snes.getIterationNumber()
    if reason <= 0:
        logger(f"SNES linear solve did not converge (reason {reason})")
        raise RuntimeError(
            f"SNES linear solver failed after {its} iterations (reason {reason})"
        )
    logger(f"SNES linear solve converged in {its} iterations (reason {reason})")

    snes.destroy()
    if not A_external:
        A.destroy()
    if not b_external:
        b.destroy()
    x.destroy()

    return u


class BarzilaiBorwein:
    """
    Stateful Barzilai-Borwein step length estimator.
    Call step(x, grad) to get a learning rate; internally stores previous x/grad.
    """

    def __init__(
        self, variant="long", fallback_lr=1e-2, smooth_beta=0.95, comm=MPI.COMM_WORLD
    ):
        self.variant = variant
        self.fallback_lr = fallback_lr
        self.smooth_beta = smooth_beta  # EMA on lr to avoid aggressive jumps
        self.comm = comm
        self.prev_x = None
        self.prev_grad = None
        self.prev_lr = None

    def step(self, x, grad):
        if self.prev_x is None or self.prev_grad is None:
            self.prev_x = x.copy()
            self.prev_grad = grad.copy()
            self.prev_lr = self.fallback_lr
            return self.prev_lr

        s = x - self.prev_x
        y = grad - self.prev_grad
        sty = self.comm.allreduce(np.dot(s, y), op=MPI.SUM)
        if self.variant == "long":
            denom = sty
            numer = self.comm.allreduce(np.dot(s, s), op=MPI.SUM)
        else:  # short
            denom = self.comm.allreduce(np.dot(y, y), op=MPI.SUM)
            numer = sty
        self.prev_x = x.copy()
        self.prev_grad = grad.copy()
        if denom <= 1e-16 or numer <= 0:
            raw_lr = self.fallback_lr
        else:
            raw_lr = numer / denom
        # Exponential smoothing to temper increases
        smoothed_lr = (
            self.smooth_beta * (self.prev_lr if self.prev_lr is not None else raw_lr)
            + (1 - self.smooth_beta) * raw_lr
        )
        self.prev_lr = smoothed_lr
        return smoothed_lr
