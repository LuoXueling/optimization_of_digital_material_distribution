import dolfin as fe
from src.neoHookean_incompressible import (
    NeoHookeanIncompressible,
    NeoHookeanIncompressibleStaggered,
)


class NeoHookeanSteven(NeoHookeanIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.vector()[:] = self.params.c

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
        Cbar = J ** (-2 / 3) * C
        I1 = fe.tr(Cbar)
        I2 = 0.5 * (I1**2 - fe.tr(Cbar * Cbar))

        c = self.e

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
            * fe.ln(
                (
                    fe.exp(-0.207897 * I1 + 0.135303 * I2)
                    / (
                        (fe.exp(a1 * c) + 1) ** b1
                        * (fe.exp(a2 * c) + 1) ** b2
                        * (fe.exp(a3 * c) + 1) ** b3
                        * (fe.exp(a4 * c) + 1) ** b4
                        + 1
                    )
                    ** 0.021823
                    + 1
                )
                ** 1.16891
                * (
                    (
                        (fe.exp(a1 * c) + 1) ** b1
                        * (fe.exp(a2 * c) + 1) ** b2
                        * (fe.exp(a3 * c) + 1) ** b3
                        * (fe.exp(a4 * c) + 1) ** b4
                        + 1
                    )
                    ** 0.049792
                    * fe.exp(0.228289 * I1 - 0.204157 * I2)
                    + 1
                )
                ** 0.733048
                * fe.exp(0.033506 * I1)
                + 1
            )
            + 0.809351 * fe.ln(fe.exp(0.009634 * I1) + 1)
        )
        psi = psi_nn(I1, I2) - psi_nn(3, 3) + self.p * (J - 1)
        return psi


class NeoHookeanStevenStaggered(NeoHookeanIncompressibleStaggered):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.vector()[:] = self.params.c

    def _strain_energy_density(self, *args, **kwargs):
        return NeoHookeanSteven._strain_energy_density(self, *args, **kwargs)
