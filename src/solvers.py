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
from scipy.optimize import minimize


def solve_nonlinear_problem(
    F,
    u,
    bcs=None,
    J=None,
    linear_solver="mumps",
    comm=None,
    logger=print,
    snes_options=None,
    save_jacobian=False,
    external_jacobian=None,
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
    A = (
        dolfinx.fem.petsc.create_matrix(jacobian_form)
        if external_jacobian is None
        else external_jacobian
    )
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

        if not external_jacobian:
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
        raise Exception("gmres not working at this time")
    elif linear_solver == "cg":
        ksp.setType("cg")
        ksp.getPC().setType("ilu")
        raise Exception("cg not working at this time")
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
    if not save_jacobian:
        A.destroy()
    elif num_iterations == 0 and not external_jacobian:
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, jacobian_form, bcs=bcs)
        A.assemble()
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

    if not save_jacobian:
        return num_iterations, converged
    else:
        return num_iterations, converged, A


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
    if not A_external:
        assemble_matrix(A)
    if not b_external:
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


class MMASolver:
    def __init__(self, local_n_vars, xmin, xmax, move_limit=0.2, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.local_n_vars = local_n_vars
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.move_limit = move_limit

        # History variables for asymptote updates (local to rank)
        self.x_old1 = None
        self.x_old2 = None
        self.L = None
        self.U = None
        self.iter = 0

    def step(
        self, x_local, f0_global, grad_f0_local, f_i_global=None, grad_f_i_local=None
    ):
        """
        Executes one distributed MMA step. Safe for unconstrained problems.
        """
        x_local = np.array(x_local)

        # Safely handle the unconstrained case inputs
        if f_i_global is None or len(f_i_global) == 0:
            n_constraints = 0
        else:
            f_i_global = np.atleast_1d(f_i_global)
            grad_f_i_local = np.atleast_2d(grad_f_i_local)
            n_constraints = len(f_i_global)

        # ---------------------------------------------------------
        # 1. Update Asymptotes (Strictly Local Operations)
        # ---------------------------------------------------------
        if self.iter < 2:
            self.L = x_local - (self.xmax - self.xmin)
            self.U = x_local + (self.xmax - self.xmin)
        else:
            oscillation = (x_local - self.x_old1) * (self.x_old1 - self.x_old2)
            gamma = np.ones(self.local_n_vars)
            gamma[oscillation < 0] = 0.7
            gamma[oscillation > 0] = 1.2
            self.L = x_local - gamma * (self.x_old1 - self.L)
            self.U = x_local + gamma * (self.U - self.x_old1)

        asymp_min_dist = 1e-4 * (self.xmax - self.xmin)
        self.L = np.maximum(self.L, x_local - 100 * (self.xmax - self.xmin))
        self.U = np.minimum(self.U, x_local + 100 * (self.xmax - self.xmin))
        self.L = np.minimum(self.L, x_local - asymp_min_dist)
        self.U = np.maximum(self.U, x_local + asymp_min_dist)

        # Define dynamic bounds (alpha, beta) based on move limits
        alpha = np.maximum(self.xmin, self.L + 0.1 * (x_local - self.L))
        alpha = np.maximum(alpha, x_local - self.move_limit * (self.xmax - self.xmin))
        beta = np.minimum(self.xmax, self.U - 0.1 * (self.U - x_local))
        beta = np.minimum(beta, x_local + self.move_limit * (self.xmax - self.xmin))

        # ---------------------------------------------------------
        # 2. UNCONSTRAINED BYPASS
        # ---------------------------------------------------------
        if n_constraints == 0:
            # In an unconstrained monotonic subproblem, variables are pushed directly
            # to their dynamic bounds based solely on the sign of the local gradient.
            x_new_local = np.where(grad_f0_local > 0, alpha, beta)

            # Update Local History and return immediately (No MPI communication)
            self.x_old2 = self.x_old1 if self.x_old1 is not None else x_local.copy()
            self.x_old1 = x_local.copy()
            self.iter += 1
            return x_new_local

        # ---------------------------------------------------------
        # 3. Compute Approximation Parameters (Requires MPI Reduce)
        # ---------------------------------------------------------
        # print("Approximation", flush=True)
        p0_local = np.maximum(0, grad_f0_local) * (self.U - x_local) ** 2
        q0_local = np.maximum(0, -grad_f0_local) * (x_local - self.L) ** 2

        P_local = np.maximum(0, grad_f_i_local) * (self.U - x_local) ** 2
        Q_local = np.maximum(0, -grad_f_i_local) * (x_local - self.L) ** 2

        obj_sum_local = np.sum(
            p0_local / (self.U - x_local) + q0_local / (x_local - self.L)
        )
        cons_sum_local = np.sum(
            P_local / (self.U - x_local) + Q_local / (x_local - self.L), axis=1
        )

        # Synchronize parameters across the communicator
        obj_sum_global = self.comm.allreduce(obj_sum_local, op=MPI.SUM)
        cons_sum_global = self.comm.allreduce(cons_sum_local, op=MPI.SUM)

        r0_global = f0_global - obj_sum_global
        R_global = f_i_global - cons_sum_global

        # ---------------------------------------------------------
        # 4. The Analytical Primal-Dual Mapping & Dual Solve
        # ---------------------------------------------------------
        def x_from_lambda(lam):
            P_lam = p0_local + lam @ P_local
            Q_lam = q0_local + lam @ Q_local
            sqrt_QP = np.sqrt(Q_lam / np.maximum(P_lam, 1e-15))
            x_unc = (self.L + self.U * sqrt_QP) / (1.0 + sqrt_QP)
            return np.clip(x_unc, alpha, beta)

        def dual_objective(lam):
            x_opt_local = x_from_lambda(lam)

            obj_term_local = np.sum(
                p0_local / (self.U - x_opt_local) + q0_local / (x_opt_local - self.L)
            )
            cons_term_local = np.sum(
                P_local / (self.U - x_opt_local) + Q_local / (x_opt_local - self.L),
                axis=1,
            )

            # Synchronize the evaluation of the dual function
            obj_term_global = self.comm.allreduce(obj_term_local, op=MPI.SUM)
            cons_term_global = self.comm.allreduce(cons_term_local, op=MPI.SUM)

            obj_val_global = r0_global + obj_term_global
            cons_val_global = R_global + cons_term_global

            dual_val = obj_val_global + np.dot(lam, cons_val_global)
            return -dual_val, -cons_val_global

        # print("Start optimization", flush=True)
        # All MPI ranks solve the exact same small dual problem identically
        lam0 = np.zeros(n_constraints)
        res = minimize(
            dual_objective,
            lam0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, None)] * n_constraints,
            options={"ftol": 1e-6, "maxls": 20, "maxiter": 100, "disp": True},
        )
        # print("End optimization", flush=True)
        x_new_local = x_from_lambda(res.x)
        # print("lambda: ", res.x)

        # 5. Update Local History
        self.x_old2 = self.x_old1 if self.x_old1 is not None else x_local.copy()
        self.x_old1 = x_local.copy()
        self.iter += 1

        return x_new_local


class CCSAMMASolver:
    def __init__(
        self,
        local_n_vars,
        xmin,
        xmax,
        move_limit=0.2,
        rel_tolerance=1e-5,
        comm=MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.local_n_vars = local_n_vars
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.move_limit = move_limit
        self.rel_tolerance = rel_tolerance

        # History variables for asymptote updates (local to rank)
        self.x_old1 = None
        self.x_old2 = None

        # NLopt explicitly tracks the asymptote distance (sigma)
        self.sigma = None
        self.L = None
        self.U = None
        self.iter = 0

        # GCMMA Specific: Convexity Penalty Parameters
        self.rho = 1.0
        self.rhoc = None

        self.tol_f0 = None
        self.tol_fc = None

    def step(
        self,
        x_local,
        f0_global,
        grad_f0_local,
        f_i_global=None,
        grad_f_i_local=None,
        eval_func=None,
    ):
        """
        Executes one distributed GCMMA step strictly following NLopt's CCSA implementation.
        """
        x_local = np.array(x_local)

        # Safely handle the unconstrained case inputs
        if f_i_global is None or len(f_i_global) == 0:
            n_constraints = 0
            f_i_global = np.array([])
            grad_f_i_local = np.zeros((0, self.local_n_vars))
        else:
            f_i_global = np.atleast_1d(f_i_global)
            grad_f_i_local = np.atleast_2d(grad_f_i_local)
            n_constraints = len(f_i_global)

        if self.iter == 0:
            # Initialize tolerances based on initial function values
            self.tol_f0 = max(self.rel_tolerance * abs(f0_global), 1e-8)
            self.tol_fc = np.full(
                n_constraints, self.rel_tolerance * np.abs(f_i_global)
            )
            self.tol_fc[self.tol_fc < 1e-8] = 1e-8

        # ---------------------------------------------------------
        # 1. Update Asymptotes (NLopt sigma tracking logic)
        # ---------------------------------------------------------
        if self.iter == 0:
            self.sigma = (self.move_limit / 0.9) * (self.xmax - self.xmin)

            # --- Objective Penalty Initialization ---
            max_g0 = self.comm.allreduce(np.max(np.abs(grad_f0_local)), op=MPI.MAX)
            # Store the minimum floor so we can use it during decay later
            self.rho_min = max(1e-5 * max_g0, 1e-12)
            self.rho = self.rho_min

            # --- Constraint Penalty Initialization (rhoc) ---
            if n_constraints > 0:
                self.rhoc_min = np.zeros(n_constraints)
                self.rhoc = np.zeros(n_constraints)
                for i in range(n_constraints):
                    # Find the max gradient for this specific constraint across all MPI ranks
                    max_gi = self.comm.allreduce(
                        np.max(np.abs(grad_f_i_local[i])), op=MPI.MAX
                    )
                    self.rhoc_min[i] = max(1e-5 * max_gi, 1e-12)
                    self.rhoc[i] = self.rhoc_min[i]
            else:
                self.rhoc_min = np.array([])
                self.rhoc = np.array([])

        else:
            # Decay penalties at the start of a new outer iteration.
            # CRITICAL: Use your calculated `_min` values here, NOT NLopt's hardcoded 1e-5!
            self.rho = max(0.1 * self.rho, self.rho_min)
            if n_constraints > 0:
                self.rhoc = np.maximum(0.1 * self.rhoc, self.rhoc_min)

            # Oscillation tracking for sigma update
            if self.iter > 1:
                dx2 = (x_local - self.x_old1) * (self.x_old1 - self.x_old2)
                gam = np.ones(self.local_n_vars)
                gam[dx2 < 0] = 0.7
                gam[dx2 > 0] = 1.2
                self.sigma = self.sigma * gam

                # NLopt safety bounds for sigma
                self.sigma = np.clip(
                    self.sigma,
                    0.01 * (self.xmax - self.xmin),
                    10.0 * (self.xmax - self.xmin),
                )

        self.L = x_local - self.sigma
        self.U = x_local + self.sigma

        # Dynamic Bounds (NLopt limits the step to 90% of sigma)
        alpha = np.maximum(self.xmin, x_local - 0.9 * self.sigma)
        alpha = np.maximum(alpha, x_local - self.move_limit * (self.xmax - self.xmin))
        beta = np.minimum(self.xmax, x_local + 0.9 * self.sigma)
        beta = np.minimum(beta, x_local + self.move_limit * (self.xmax - self.xmin))

        # ---------------------------------------------------------
        # 2. GCMMA Inner Iteration Loop
        # ---------------------------------------------------------
        inner_iter = 0
        max_inner = 15
        x_new_local = x_local.copy()

        while inner_iter < max_inner:
            # A. Compute CCSA Parameters mapped to SciPy P/Q dual formulation
            # NLopt formulates dual variables u & v, which mathematically equal the following P & Q
            v0 = np.abs(grad_f0_local) * self.sigma + 0.5 * self.rho
            P0 = (
                np.maximum(0, grad_f0_local) * self.sigma**2
                + 0.25 * self.rho * self.sigma
            )
            Q0 = (
                np.maximum(0, -grad_f0_local) * self.sigma**2
                + 0.25 * self.rho * self.sigma
            )

            r0_local = np.sum(v0)
            r0 = f0_global - self.comm.allreduce(r0_local, op=MPI.SUM)

            if n_constraints > 0:
                v_i = (
                    np.abs(grad_f_i_local) * self.sigma + 0.5 * self.rhoc[:, np.newaxis]
                )
                P_i = (
                    np.maximum(0, grad_f_i_local) * self.sigma**2
                    + 0.25 * self.rhoc[:, np.newaxis] * self.sigma
                )
                Q_i = (
                    np.maximum(0, -grad_f_i_local) * self.sigma**2
                    + 0.25 * self.rhoc[:, np.newaxis] * self.sigma
                )

                R_local = np.sum(v_i, axis=1)
                R = f_i_global - self.comm.allreduce(R_local, op=MPI.SUM)
            else:
                P_i = np.zeros((0, self.local_n_vars))
                Q_i = np.zeros((0, self.local_n_vars))
                R = np.array([])

            # B. Solve the Dual Problem
            def x_from_lambda(lam):
                if n_constraints > 0:
                    P_lam = P0 + lam @ P_i
                    Q_lam = Q0 + lam @ Q_i
                else:
                    P_lam = P0
                    Q_lam = Q0

                sqrt_QP = np.sqrt(Q_lam / np.maximum(P_lam, 1e-15))
                x_unc = (self.L + self.U * sqrt_QP) / (1.0 + sqrt_QP)
                return np.clip(x_unc, alpha, beta)

            def dual_objective(lam):
                x_opt_local = x_from_lambda(lam)

                obj_term_local = np.sum(
                    P0 / (self.U - x_opt_local) + Q0 / (x_opt_local - self.L)
                )
                cons_term_local = np.sum(
                    P_i / (self.U - x_opt_local) + Q_i / (x_opt_local - self.L), axis=1
                )

                obj_term_global = self.comm.allreduce(obj_term_local, op=MPI.SUM)
                cons_term_global = self.comm.allreduce(cons_term_local, op=MPI.SUM)

                dual_val = (r0 + obj_term_global) + np.dot(lam, R + cons_term_global)
                return -dual_val, -(R + cons_term_global)

            if n_constraints == 0:
                x_new_local = x_from_lambda(np.array([]))
            else:
                lam0 = np.zeros(n_constraints)
                res = minimize(
                    dual_objective,
                    lam0,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0, None)] * n_constraints,
                    options={"ftol": 1e-6, "maxls": 20, "maxiter": 100, "disp": False},
                )
                x_new_local = x_from_lambda(res.x)

            # C. The Conservativeness Check (Inner Loop Physics Test)
            if eval_func is None:
                break  # Bypass inner loop if no evaluation function is provided

            # 1. Run the forward finite element check
            eval_res = eval_func(x_new_local)

            if isinstance(eval_res, tuple):
                fcur, fcval_cur = eval_res
            else:
                fcur = eval_res
                fcval_cur = np.array([])

            fcval_cur = np.atleast_1d(fcval_cur)

            # 2. Re-evaluate the approximation (gval, gcval) at x_new (NLopt formulation)
            dx = x_new_local - x_local
            dx2 = dx**2
            sigma2 = self.sigma**2
            denominv = 1.0 / (sigma2 - dx2)

            # The linear term multiplier (matches 'c = sigma2 * dx' in mma.c)
            c = sigma2 * dx

            u0 = grad_f0_local
            val0_local = np.sum((u0 * c + v0 * dx2) * denominv)
            gval = f0_global + self.comm.allreduce(val0_local, op=MPI.SUM)

            gcval = np.zeros(n_constraints)
            if n_constraints > 0:
                u_i = grad_f_i_local
                val_i_local = np.sum((u_i * c + v_i * dx2) * denominv, axis=1)
                gcval = f_i_global + self.comm.allreduce(val_i_local, op=MPI.SUM)

            wval_local = np.sum(0.5 * dx2 * denominv)
            wval = self.comm.allreduce(wval_local, op=MPI.SUM)
            wval_safe = max(wval, 1e-15)
            # 3. Check and Penalize if not conservative
            inner_done = (gval + self.tol_f0) >= fcur
            if n_constraints > 0:
                inner_done = inner_done and np.all((gcval + self.tol_fc) >= fcval_cur)

            if inner_done:
                break

            # NLopt: multiply rho by up to 10 if step is rejected
            if fcur > gval:
                self.rho = min(
                    10.0 * self.rho, 1.1 * (self.rho + (fcur - gval) / wval_safe)
                )

            if n_constraints > 0:
                for i in range(n_constraints):
                    if fcval_cur[i] > gcval[i]:
                        self.rhoc[i] = min(
                            10.0 * self.rhoc[i],
                            1.1
                            * (self.rhoc[i] + (fcval_cur[i] - gcval[i]) / wval_safe),
                        )

            inner_iter += 1

        # ---------------------------------------------------------
        # 3. Update History
        # ---------------------------------------------------------
        self.x_old2 = self.x_old1 if self.x_old1 is not None else x_local.copy()
        self.x_old1 = x_local.copy()
        self.iter += 1

        return x_new_local


class BarzilaiBorwein:
    """
    Stateful Barzilai-Borwein step length estimator.
    Call step(x, grad) to get a learning rate; internally stores previous x/grad.
    """

    def __init__(
        self,
        variant="long",
        fallback_lr=1.0,
        smooth_beta=0.95,
        upper_bound_lr=None,
        lower_bound_lr=None,
        comm=MPI.COMM_WORLD,
    ):
        self.variant = variant
        self.fallback_lr = fallback_lr
        self.smooth_beta = smooth_beta  # EMA on lr to avoid aggressive jumps
        self.upper_bound_lr = upper_bound_lr
        self.lower_bound_lr = lower_bound_lr
        if self.fallback_lr > 0.0 and (
            (self.upper_bound_lr is not None and self.fallback_lr > self.upper_bound_lr)
            or (
                self.lower_bound_lr is not None
                and self.fallback_lr < self.lower_bound_lr
            )
        ):
            raise Exception(
                f"lr ({self.fallback_lr}) is not in the range ({self.lower_bound_lr}, {self.upper_bound_lr})"
            )
        self.comm = comm
        self.prev_x = None
        self.prev_grad = None
        self.prev_lr = None

    def step(self, x, grad):
        if self.fallback_lr == 0.0:
            return self.fallback_lr

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
        smoothed_lr = float(
            self.smooth_beta * (self.prev_lr if self.prev_lr is not None else raw_lr)
            + (1 - self.smooth_beta) * raw_lr
        )
        if self.lower_bound_lr is not None:
            smoothed_lr = max(self.lower_bound_lr, smoothed_lr)
        if self.upper_bound_lr is not None:
            smoothed_lr = min(self.upper_bound_lr, smoothed_lr)
        self.prev_lr = smoothed_lr
        return smoothed_lr
