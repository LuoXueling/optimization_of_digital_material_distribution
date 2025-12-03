from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanPDEFilteredTopoOpt, NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressible
import numpy as np
import ufl

parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_cantilever_beam.py"
)
args = parser.parse_args()


class StevenPDEFilteredTopoOpt(NeoHookeanPDEFilteredTopoOpt):
    def _get_c(self):
        return 0.5

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


class StevenDistPDEFilteredTopoOpt(StevenPDEFilteredTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.dx
        self.V_c, self._c_map = self.V_e.sub(self.c_component_id).collapse()

    @property
    def n_component(self) -> int:
        return 2

    @property
    def c_component_id(self):
        return 1

    def _get_c(self):
        return self.e[self.c_component_id]

    def _initial_values(self):
        x0 = super()._initial_values()
        x0[self._c_map] = 0.5
        return x0

    def _bounds(self):
        bnds = super()._bounds()
        for idx in self._c_map:
            bnds[idx] = (0.0, 1.0)
        return bnds

    def _gradient_norm_c(self):
        thres = 3000
        smooth = 400
        softplus = lambda x: ufl.conditional(
            ufl.lt(x, thres * 10),
            ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )
        c = self._get_c()
        grad_c = ufl.grad(c)
        grad_norm = (
            softplus(ufl.dot(grad_c, grad_c) - thres) - softplus(0.0 - thres)
        ) * self.dx_grad
        return grad_norm

    def _objective_function_value(self):
        stored_energy = self._stored_energy()
        stored_energy_val = self._assemble_scalar_parallel(stored_energy)

        softplus = lambda x: np.log(1 + np.exp(100 * x)) / 100
        weight = self._weight_diff()
        weight_val = 5 * softplus(self._assemble_scalar_parallel(weight))

        grad_norm = self._gradient_norm_c()
        grad_norm_val = self._assemble_scalar_parallel(grad_norm)

        energy_current = self._assemble_scalar_parallel(
            self._strain_energy_density() * self.dx
        )
        weight_current = self._assemble_scalar_parallel(
            self.density / self.area * self.dx
        )

        self.log(
            f"Stored energy: {stored_energy_val}, weight constraint: {weight_val}, gradient norm: {grad_norm_val}"
        )
        self.log(f"Current energy: {energy_current}")
        self.log(f"Current weight: {weight_current} ({self.params.target_volume})")
        return stored_energy_val + weight_val + grad_norm_val

    def _grad_j_state(self, j, state, state_test):
        stored_energy = self._stored_energy()
        return ufl.derivative(stored_energy, state, state_test)

    def _grad_j_opt(self, j, opt, opt_test):
        stored_energy = self._stored_energy()
        weight_diff_form = self._weight_diff()
        grad_norm = self._gradient_norm_c()
        softplus_diff = (
            5
            * (lambda x: np.exp(100 * x) / (1 + np.exp(100 * x)))(
                self._assemble_scalar_parallel(weight_diff_form)
            )
            * ufl.derivative(self.density * self.dx, opt, opt_test)
        )
        return ufl.derivative(stored_energy + grad_norm, opt, opt_test) + softplus_diff


prob = StevenDistPDEFilteredTopoOpt(config_path=args.config)
prob.optimize()
