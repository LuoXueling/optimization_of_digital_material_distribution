import dolfin as fe
from src.default_config import params
from src.neoHookean import NeoHookean
import numpy as np


class NeoHookeanIncompressible(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define function space
        self.V_w = fe.FunctionSpace(
            self.mesh,
            fe.MixedElement(
                [
                    fe.VectorElement(
                        "Lagrange", self.mesh.ufl_cell(), 2
                    ),  # For displacement
                    fe.FiniteElement(
                        "Lagrange", self.mesh.ufl_cell(), 1
                    ),  # For p (lagrangian multiplier of incompressibility)
                ]
            ),
        )
        self.V_u = self.V_w.sub(0)
        self.V_p = self.V_w.sub(1)

        # Define functions
        self.u_test, self.p_test = fe.TestFunctions(self.V_w)  # Test function

        self.w = None
        self.u, self.p = None, None  # fields from previous iteration
        self._initialize_fields()

        self.dirichlet_bcs = self._define_dirichlet_boundaries()

    def _initialize_fields(self):
        self.w = fe.Function(self.V_w)
        self.u, self.p = fe.split(self.w)

    def forward(self):
        psi = self._strain_energy_density()

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = fe.derivative(Pi, self.w, fe.TestFunction(self.V_w))

        # Compute Jacobian of F
        J = fe.derivative(F, self.w, fe.TrialFunction(self.V_w))

        # Solve variational problem
        fe.solve(
            F == 0,
            self.w,
            self.dirichlet_bcs,
            J=J,
            solver_parameters={"newton_solver": {"linear_solver": "mumps"}},
        )

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        to_3d = lambda x: (
            fe.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
            if self.params.dim == 2
            else x
        )

        if Eps is None:
            if F is None:
                F = self._deformation_gradient()
            if C is None:
                C = self._right_cauchy_green(F=F)
            F = to_3d(F)
            C = to_3d(C)
            J = fe.det(F)
        else:
            C = 2.0 * Eps + fe.Identity(len(self.u))
            C = to_3d(C)
            J = fe.sqrt(fe.det(C))
        Ic = fe.tr(C)

        # Elasticity parameters
        mu = self.params.E / (2 * (1 + self.params.nu))  # Lame second coefficient

        # Stored strain energy density (incompressible neo-Hookean model)
        psi = (mu / 2) * (Ic - 3) - mu * fe.ln(J) + self.p * (J - 1)
        return psi

    def save(self, time=0):
        u, p = self.w.split(True)
        u.rename("displacement", "(No description)")
        p.rename("incompressibility", "(No description)")
        self.xdmffile.write(p, time)
        super().save(main_field=u, time=time)


class NeoHookeanIncompressibleStaggered(NeoHookeanIncompressible):
    def __init__(self, *args, override_params=None, **kwargs):
        if override_params is None:
            override_params = dict()
        override_params.update(dict(elem_order=2))
        NeoHookean.__init__(self, *args, override_params=override_params, **kwargs)
        self.log("Enforcing element order = 2 for displacement field")

        # define function space
        self.V_p = fe.FunctionSpace(self.mesh, "Lagrange", 1)

        # Define functions
        self.dproj = fe.TrialFunction(self.V_p)
        self.p_test = fe.TestFunction(self.V_p)
        self.p = fe.Function(
            self.V_p, name="incompressibility"
        )  # fields from previous iteration

    def _initialize_fields(self):
        NeoHookean._initialize_fields(self)
        self.p = fe.Function(self.V_p, name="incompressibility")

    def forward(self):
        F = self._deformation_gradient()
        C = self._right_cauchy_green(F=F)
        J = fe.sqrt(fe.det(C))
        psi = self._strain_energy_density(F=F, C=C)

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Solve variational problem

        kappa = 10 / (2 * (1 + 0.5))
        max_kappa = 128 * kappa
        last_dp_norm = None
        for i in range(100):
            # Compute first variation of Pi (directional derivative about u in the direction of v)
            F_u = fe.derivative(
                Pi + kappa / 2 * (J - 1) ** 2 * self.dx, self.u, self.u_test
            )
            # Compute Jacobian of F
            J_u = fe.derivative(F_u, self.u, self.du)

            fe.solve(
                F_u == 0,
                self.u,
                self.dirichlet_bcs,
                J=J_u,
                solver_parameters={
                    "newton_solver": {"linear_solver": "mumps", "report": True}
                },
            )

            # Project (J-1) into V_p then we have the increment of p
            dp = fe.project(J - 1, self.V_p)

            dp_norm = fe.norm(dp, "L2")
            self.p.vector().axpy(kappa, dp.vector())

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

    def save(self, time=0):
        NeoHookean.save(self, time=time)
        self.xdmffile.write(self.p, time)
