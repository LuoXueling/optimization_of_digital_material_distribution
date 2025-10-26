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
        Ic, J, _, _ = self.strain_invariant(F=F, C=C, Eps=Eps)

        # Elasticity parameters
        mu = self.mu  # Lame second coefficient
        kappa = self.kappa

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (J ** (-2 / 3) * Ic - 3) + (kappa / 2) * (ufl.ln(J)) ** 2
        return psi

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()

        Ic, J, F, C = self.strain_invariant()

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
