from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanPDEFilteredTopoOpt, NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressible
import numpy as np
import ufl
import dolfinx.fem as fem
from src.utils import project

parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_cantilever_beam.py"
)
args = parser.parse_args()


class StevenTopoOpt(NeoHookeanPDEFilteredTopoOpt):
    def _get_c(self):
        return 1.0

    def _strain_energy_density_wo_constraint(self):
        return StevenNearlyIncompressible._strain_energy_density_wo_constraint(self)

    def _strain_energy_density(self, *args, **kwargs):
        return self._degradation * StevenNearlyIncompressible._strain_energy_density(
            self, *args, **kwargs
        )

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

    @property
    def density(self):
        raw_density = (
            self.e if self.n_component == 1 else self.e[self.density_component_id]
        )
        beta = 3.0
        eta = 0.5
        num = ufl.tanh(beta * eta) + ufl.tanh(beta * (raw_density - eta))
        den = ufl.tanh(beta * eta) + ufl.tanh(beta * (1.0 - eta))
        proj = num / den
        return proj


prob = StevenTopoOpt(config_path=args.config)
prob.optimize()
