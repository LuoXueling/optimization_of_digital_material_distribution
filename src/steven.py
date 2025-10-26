import ufl
from src.neoHookean import NeoHookean
from src.neoHookean_incompressible import (
    NeoHookeanIncompressible,
    NeoHookeanIncompressibleStaggered,
)


class Steven(NeoHookeanIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.x.array[:] = self.params.c

    def _get_c(self):
        return self.params.c

    def _strain_energy_density_wo_constraint(self):
        c = self._get_c()

        a1 = 6.84602
        a2 = 6.86996
        a3 = 6.88268
        a4 = 6.93876
        b1 = 2.37126
        b2 = 2.19610
        b3 = 2.56167
        b4 = 1.98880

        psi_nn = lambda I1, I2: (
            0.083089 * I1
            + 4.01198
            * ufl.ln(
                (
                    ufl.exp(-0.207897 * I1 + 0.135303 * I2)
                    / (
                        (ufl.exp(a1 * c) + 1) ** b1
                        * (ufl.exp(a2 * c) + 1) ** b2
                        * (ufl.exp(a3 * c) + 1) ** b3
                        * (ufl.exp(a4 * c) + 1) ** b4
                        + 1
                    )
                    ** 0.021823
                    + 1
                )
                ** 1.16891
                * (
                    (
                        (ufl.exp(a1 * c) + 1) ** b1
                        * (ufl.exp(a2 * c) + 1) ** b2
                        * (ufl.exp(a3 * c) + 1) ** b3
                        * (ufl.exp(a4 * c) + 1) ** b4
                        + 1
                    )
                    ** 0.049792
                    * ufl.exp(0.228289 * I1 - 0.204157 * I2)
                    + 1
                )
                ** 0.733048
                * ufl.exp(0.033506 * I1)
                + 1
            )
            + 0.809351 * ufl.ln(ufl.exp(0.009634 * I1) + 1)
        )
        return psi_nn

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        Ic, J, F, C = self.strain_invariant(F=F, C=C, Eps=Eps)

        Cbar = J ** (-2 / 3) * C
        I1 = ufl.tr(Cbar)
        I2 = 0.5 * (I1**2 - ufl.tr(Cbar * Cbar))

        psi_nn = self._strain_energy_density_wo_constraint()
        psi = psi_nn(I1, I2) - psi_nn(3, 3) + self.p * (J - 1)
        return psi


class StevenNearlyIncompressible(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.x.array[:] = self.params.c

    def _get_c(self):
        return self.params.c

    def _strain_energy_density_wo_constraint(self):
        return Steven._strain_energy_density_wo_constraint(self)

    @property
    def kappa(self):
        c = self._get_c()
        E = (
            0.30390594 * c**4
            - 0.96196938 * c**3
            + 0.89915784 * c**2
            + 0.39949939 * c
            + 0.69701715
        )
        kappa = E / (3 * (1 - 2 * self.params.nu))
        return kappa

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        Ic, J, F, C = self.strain_invariant(F=F, C=C, Eps=Eps)

        Cbar = J ** (-2 / 3) * C
        I1 = ufl.tr(Cbar)
        I2 = 0.5 * (I1**2 - ufl.tr(Cbar * Cbar))

        psi_nn = self._strain_energy_density_wo_constraint()

        kappa = self.kappa

        psi = psi_nn(I1, I2) - psi_nn(3, 3) + 0.5 * kappa * (ufl.ln(J)) ** 2
        return psi


class StevenStaggered(NeoHookeanIncompressibleStaggered):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.x.array[:] = self.params.c

    def _get_c(self):
        return self.params.c

    def _strain_energy_density(self, *args, **kwargs):
        return Steven._strain_energy_density(self, *args, **kwargs)
