import dolfin as fe
from src.default_config import params
from src.neoHookean import NeoHookean


class NeoHookeanIncompressible(NeoHookean):
    def __init__(self, config_path):
        super().__init__(config_path)

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
