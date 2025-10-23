from src.neoHookean import NeoHookean
import ufl


class NeoHookeanTMC(NeoHookean):
    def __init__(
        self,
        config_path,
        override_params=None,
    ):
        super().__init__(config_path, override_params)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        to_3d = lambda x: (
            ufl.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
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
            J = ufl.det(F)
        else:
            C = 2.0 * Eps + ufl.Identity(len(self.u))
            C = to_3d(C)
            J = ufl.sqrt(ufl.det(C))
        Ic = ufl.tr(C)

        # Elasticity parameters
        mu = self.params.E / (2 * (1 + self.params.nu))  # Lame second coefficient
        kappa = self.params.E / (3 * (1 - 2 * self.params.nu))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (J ** (-2 / 3) * Ic - 3) + (kappa / 2) * (ufl.ln(J)) ** 2
        return psi

    def _potential_energy(self, psi=None):
        to_3d = lambda x: (
            ufl.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
            if self.params.dim == 2
            else x
        )
        if psi is None:
            psi = self._strain_energy_density()

        F = self._deformation_gradient()
        F = to_3d(F)
        J = ufl.det(F)

        kappa = self.params.E / (3 * (1 - 2 * self.params.nu))

        kr = self.params.kr_coef * kappa * ufl.exp(-5 * J)

        hessian = ufl.grad(ufl.grad(self.u))
        psi_reg = 0.5 * ufl.inner(hessian, hessian)

        energy = (
            psi * self.dx
            + (self.params.psi_coef * psi + kr * psi_reg) * self.third_medium_dx
            - ufl.dot(self.params.B, self.u) * self.dx
        )
        # Neumann boundaries
        for i, val in self.neumann_bcs:
            energy -= ufl.dot(val, self.u) * self.boundary_ds(i)
        return energy
