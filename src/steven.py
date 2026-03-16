import ufl
from src.neoHookean import NeoHookean
from src.neoHookean_tmc import NeoHookeanTMC
from src.neoHookean_incompressible import (
    NeoHookeanIncompressible,
    NeoHookeanIncompressibleStaggered,
    NeoHookeanIncompressibleTMC,
)
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
import numpy as np
import dolfinx.fem as fem


class Steven(NeoHookeanIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.x.array[:] = self.params.c

    def _get_c(self):
        return self.params.c

    @property
    def E(self):
        c = self._get_c()
        E = (
            0.30390594 * c**4
            - 0.96196938 * c**3
            + 0.89915784 * c**2
            + 0.39949939 * c
            + 0.69701715
        )
        return E

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


class StevenIncompressibleTMC(Steven):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )
        self.area = self._assemble_scalar_parallel(1.0 * self.dx)

    def load_boundary_conditions(self):
        return NeoHookeanIncompressibleTMC.load_boundary_conditions(self)

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

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        return StevenNearlyIncompressible._strain_energy_density(
            self, F=F, C=C, Eps=Eps
        )

    def _potential_energy(self, psi=None):
        return NeoHookeanTMC._potential_energy(self, psi)

    def _get_cell_data(self):
        return NeoHookeanTMC._get_cell_data(self)


class StevenNearlyIncompressibleTMC(StevenNearlyIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )
        self.area = self._assemble_scalar_parallel(1.0 * self.dx)

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        return self._strain_energy_density(F=F, C=C, Eps=Eps)

    def _potential_energy(self, psi=None):
        return NeoHookeanTMC._potential_energy(self, psi)

    def _get_cell_data(self):
        return NeoHookeanTMC._get_cell_data(self)


class StevenNearlyIncompressibleDistOpt(StevenNearlyIncompressible):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_objective_values = {}

    def _get_c(self):
        return self.e

    def _optimize_callback(self, *args) -> bool:
        res = super()._optimize_callback(*args)
        self.record_traction_dirichlet(step=self._optimize_step)
        self.save_records()
        return res

    def _objective_function(self):
        raise NotImplementedError()

    def _initial_values(self):
        return np.array([self.params.c] * len(self.e.x.array))

    def _bounds(self):
        return [(0.0, 1.0)] * len(self.e.x.array)

    def _normalized_scalar_expression(self, expression, name):
        return NeoHookeanTopoOpt._normalized_scalar_expression(
            self, expression=expression, name=name
        )

    def _define_design_dirichlet_boundaries(self):
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self, fixed_value=self.params.get("c", 1.0)
        )


class StevenNearlyIncompressibleTMCDistOpt(StevenNearlyIncompressibleDistOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )
        self.area = self._assemble_scalar_parallel(1.0 * self.dx)

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        return self._strain_energy_density(F=F, C=C, Eps=Eps)

    def _potential_energy(self, psi=None):
        return NeoHookeanTMC._potential_energy(self, psi)

    def _get_cell_data(self):
        return NeoHookeanTMC._get_cell_data(self)


class StevenStaggered(NeoHookeanIncompressibleStaggered):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e.x.array[:] = self.params.c

    def _get_c(self):
        return self.params.c

    def _strain_energy_density(self, *args, **kwargs):
        return Steven._strain_energy_density(self, *args, **kwargs)
