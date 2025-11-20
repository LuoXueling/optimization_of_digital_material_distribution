import dolfinx.fem as fem
import ufl
import basix.ufl
from src.default_config import params
from src.neoHookean import NeoHookean
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.neoHookean_tmc import NeoHookeanTMC
from src.utils import project
import numpy as np
from mpi4py import MPI
from src.solvers import solve_nonlinear_problem_with_retry
from dolfinx.la import InsertMode
import warnings


class NeoHookeanIncompressible(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define function space (mixed element)
        # P2 vector element for displacement
        el_u = basix.ufl.element(
            "Lagrange", self.mesh.basix_cell(), 2, shape=(self.mesh.geometry.dim,)
        )
        # P1 scalar element for pressure (Lagrange multiplier)
        el_p = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
        # Create mixed element
        el_mixed = basix.ufl.mixed_element([el_u, el_p])

        # Create mixed function space
        self.V_w = fem.functionspace(self.mesh, el_mixed)

        # Define test functions
        self.u_test, self.p_test = ufl.TestFunctions(self.V_w)

        self.w = None
        self.u, self.p = None, None  # fields from previous iteration
        self._initialize_fields()

        # Override V_u to point to the displacement subspace for boundary conditions
        self.V_u = self.V_w.sub(0)
        self.V_p = self.V_w.sub(1)

        self.load_boundary_conditions()

    def _initialize_fields(self):
        self.w = fem.Function(self.V_w)
        self.u, self.p = ufl.split(self.w)

    def forward(self):
        psi = self._strain_energy_density()

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Compute first variation of Pi (directional derivative about w in the direction of test functions)
        F = ufl.derivative(Pi, self.w, ufl.TestFunction(self.V_w))

        # Compute Jacobian of F
        J = ufl.derivative(F, self.w, ufl.TrialFunction(self.V_w))

        # Solve variational problem
        solve_nonlinear_problem_with_retry(
            F,
            self.w,
            bcs=self.dirichlet_bcs,
            J=J,
            comm=self.mesh.comm,
            logger=self.log,
            snes_options=self.params.snes_options,
            num_retries=self.params.num_retries,
        )

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        Ic, J, _, _ = self.strain_invariant(F=F, C=C, Eps=Eps)

        # Elasticity parameters
        mu = self.mu  # Lame second coefficient

        # Stored strain energy density
        psi = (mu / 2) * (J ** (-2 / 3) * Ic - 3) + self.p * (J - 1)
        return psi

    def _get_point_data(self, main_field=None):
        # Collapse the mixed function components
        u = self.w.sub(0).collapse()
        p = self.w.sub(1).collapse()
        u.name = "displacement"
        p.name = "pressure"

        # Projection to P1 is handled in write_vtu if needed
        point_data = super()._get_point_data(main_field=u)
        point_data["pressure"] = p
        return point_data


class NeoHookeanIncompressibleTMC(NeoHookeanIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )

    def load_boundary_conditions(self):
        # Call parent to load displacement boundary conditions
        NeoHookeanIncompressible.load_boundary_conditions(self)

        if hasattr(self, "V_p"):
            # Add pressure constraint in third medium region
            pressure_bcs = self._define_domain_dirichlet_constraints(
                function_space=self.V_p,
                domain_marker=self.association_table[self.params.third_medium],
                fixed_value=0.0,
            )
            self.dirichlet_bcs.extend(pressure_bcs)

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        Ic, J, _, _ = self.strain_invariant(F=F, C=C, Eps=Eps)

        # Elasticity parameters
        mu = self.mu  # Lame second coefficient

        kappa = 1 / (2 * (1 + 0.5))
        # Has to add a nearly incompressiblity term? instead of a strict incompressibility
        # Adds convergence with this term.
        psi = (mu / 2) * (J ** (-2 / 3) * Ic - 3) + 0.5 * kappa * (ufl.ln(J)) ** 2
        return psi

    def _potential_energy(self, psi=None):
        return NeoHookeanTMC._potential_energy(self, psi)


class NeoHookeanIncompressibleTopoOpt(NeoHookeanTopoOpt, NeoHookeanIncompressible):
    def __init__(self, *args, **kwargs):
        NeoHookeanIncompressible.__init__(self, *args, **kwargs)
        NeoHookeanTopoOpt._init_topo_opt(self)

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        return (
            self.density**self.penalty
            * NeoHookeanIncompressible._strain_energy_density(self, F=F, C=C, Eps=Eps)
        )

    def _get_point_data(self, main_field=None):
        point_data = NeoHookeanIncompressible._get_point_data(self, main_field)
        point_data["Filtered gradient"] = self.filtered_gradient
        return point_data


class NeoHookeanIncompressibleStaggered(NeoHookeanIncompressible):
    def __init__(self, *args, override_params=None, **kwargs):
        if override_params is None:
            override_params = dict()
        override_params.update(dict(elem_order=2))
        NeoHookean.__init__(self, *args, override_params=override_params, **kwargs)
        self.log("Enforcing element order = 2 for displacement field")

        # define function space
        self.V_p = fem.functionspace(self.mesh, ("Lagrange", 1))

        # Define functions
        self.dproj = ufl.TrialFunction(self.V_p)
        self.p_test = ufl.TestFunction(self.V_p)
        self.p = fem.Function(
            self.V_p, name="incompressibility"
        )  # fields from previous iteration

    def _initialize_fields(self):
        NeoHookean._initialize_fields(self)
        self.p = fem.Function(self.V_p, name="incompressibility")

    def forward(self):
        F = self._deformation_gradient()
        C = self._right_cauchy_green(F=F)
        J = ufl.sqrt(ufl.det(C))
        psi = self._strain_energy_density(F=F, C=C)

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Solve variational problem

        kappa = 10 / (2 * (1 + 0.5))
        max_kappa = 128 * kappa
        last_dp_norm = None
        for i in range(100):
            # Compute first variation of Pi (directional derivative about u in the direction of v)
            F_u = ufl.derivative(
                Pi + kappa / 2 * (J - 1) ** 2 * self.dx, self.u, self.u_test
            )
            # Compute Jacobian of F
            J_u = ufl.derivative(F_u, self.u, self.du)

            # Solve nonlinear problem
            solve_nonlinear_problem_with_retry(
                F_u,
                self.u,
                bcs=self.dirichlet_bcs,
                J=J_u,
                comm=self.mesh.comm,
                logger=self.log,
                snes_options=self.params.snes_options,
                num_retries=self.params.num_retries,
            )

            # Project (J-1) into V_p then we have the increment of p
            dp = project(J - 1, self.V_p, "dp")

            dp_norm = np.sqrt(
                self.mesh.comm.allreduce(
                    fem.assemble_scalar(fem.form(ufl.inner(dp, dp) * self.dx)),
                    op=MPI.SUM,
                )
            )
            self.p.x.array[:] += kappa * dp.x.array[:]

            if dp_norm < 1e-100:
                break

            if last_dp_norm is None:
                last_dp_norm = dp_norm

            self.log(
                f"dp (J-1) l2-norm (abs) = {dp_norm:.3e}, (rel): {dp_norm/last_dp_norm:.3e}"
            )
            if dp_norm < 1e-8:
                break
            if dp_norm / last_dp_norm > 0.8:
                if kappa != max_kappa and i > 0:
                    kappa = min(2 * kappa, max_kappa)
                    self.log(f"Update kappa to {kappa}")
            last_dp_norm = dp_norm
        else:
            raise Exception(f"The staggered scheme does not converge")

    def _strain_energy_density(self, *args, **kwargs):
        return NeoHookeanIncompressible._strain_energy_density(self, *args, **kwargs)

    def _get_point_data(self, main_field=None):
        point_data = NeoHookean._get_point_data(self, main_field)
        point_data["incompressibility"] = self.p
        return point_data
