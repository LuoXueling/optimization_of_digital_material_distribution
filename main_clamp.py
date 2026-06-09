from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanPDEFilteredTopoOpt, NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressible
from src.neoHookean import NeoHookean
from petsc4py.PETSc import ScalarType
import numpy as np
import ufl
import dolfinx.fem as fem
from src.utils import project
import os
import pandas as pd

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./mesh/config_clamp.py")
parser.add_argument("-m", "--mode", type=str)
parser.add_argument("-s", "--steps", type=int, default=10)
parser.add_argument("-l", "--load", type=str, default=None)
parser.add_argument("--optimizer", type=str, default="CCSAMMA")
parser.add_argument("--ccsamma_rel_tolerance", type=float, default=1e-5)
parser.add_argument("--plane_strain", action="store_true")
parser.add_argument("--disable_rho", action="store_true")
parser.add_argument("--disable_c", action="store_true")
parser.add_argument("--max_penalty", type=float, default=5.0)
parser.add_argument("--target_volume", type=float, default=0.35)
parser.add_argument("--I1_rhs", type=float, default=0.25)
parser.add_argument("--n_filters", type=int, default=0)
args = parser.parse_args()


class StevenTopoOpt(NeoHookeanPDEFilteredTopoOpt):
    def _get_c(self):
        return 1.0

    def _strain_energy_density_wo_constraint(self):
        return StevenNearlyIncompressible._strain_energy_density_wo_constraint(self)

    def _void_indicator(self):
        thres = 0.01
        sharpness = 500
        return 1 - (
            np.tanh(sharpness * thres)
            + ufl.tanh(sharpness * (self._degradation - thres))
        ) / (np.tanh(sharpness * thres) + np.tanh(sharpness * (1 - thres)))
        # self.density**self.params.max_penalty instead of self._degradation:
        # if penalty is not large enough, when using self._degradation,
        # the sharpness of this function creates a noisy void_indicator

    def _objective_function(self):
        # Note the minus sign here, we are maximizing strain energy under disp control
        return -self._normalized_scalar_expression(
            self._degradation
            * StevenNearlyIncompressible._strain_energy_density(self)
            * self.dx,
            "stored_energy",
        )

    def _objective_function_value(self):
        return super()._objective_function_value() + 2

    @property
    def _degradation(self):
        return ufl.conditional(
            self.density**self.penalty < 1e-3, 1e-3, self.density**self.penalty
        )

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        void_indicator_ufl = self._void_indicator()
        void_indicator_space = fem.functionspace(self.mesh, ("CG", 1))
        void_indicator = project(
            void_indicator_ufl, void_indicator_space, "void_indicator"
        )
        self.void_indicator = void_indicator

        grad_u = (1 - void_indicator) * ufl.grad(
            self.u
        )  # Degrade grad(u) instead of u, otherwise it creates a significant term grad(void_indicator)*u
        F = ufl.Identity(len(self.u)) + grad_u
        psi_nonlinear_degraded = StevenNearlyIncompressible._strain_energy_density(
            self, F=F
        )

        Ic, J, _, _ = self.strain_invariant(F=F)
        psi_linear_degraded = (self.mu / 2) * (J ** (-2 / 3) * Ic - 3) + (
            self.kappa / 2
        ) * (J - 1) ** 2

        Ic, J, _, _ = self.strain_invariant()
        psi_linear = (self.mu / 2) * (J ** (-2 / 3) * Ic - 3) + (self.kappa / 2) * (
            J - 1
        ) ** 2

        return self._degradation * (
            psi_nonlinear_degraded - psi_linear_degraded + psi_linear
        )

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

    @property
    def density(self):
        raw_density = (
            self.e if self.n_component == 1 else self.e[self.density_component_id]
        )
        beta = 2**self.penalty
        eta = 0.5
        num = ufl.tanh(beta * eta) + ufl.tanh(beta * (raw_density - eta))
        den = ufl.tanh(beta * eta) + ufl.tanh(beta * (1.0 - eta))
        proj = num / den
        return proj

    def _get_cell_data(self):
        cell_data = NeoHookeanPDEFilteredTopoOpt._get_cell_data(self)
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        cell_data.update(
            {
                "void_indicator": project(
                    self.void_indicator, scalar_space, "void_indicator"
                ),
            }
        )

        return cell_data


class StevenTopoOptI1(StevenTopoOpt):
    def critical_I1(self):
        c = self._get_c()
        I1_crit = 0.22426408 * c**2 - 1.12468456 * c + 4.883886
        I1, _, _, _ = self.strain_invariant()

        # smooth = 0.01
        relu = lambda x: ufl.conditional(
            ufl.lt(x, 0),
            0.0 * x,
            # ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )

        # We only want to compute I1-based failure criterion in regions with high density,
        # so supposedly we should multiply softplus(xxx) by self.density.
        # However, the symbolic differentiation of self.density*softplus(xxx)
        # allows topology optimization to remove material for lower value of this term,
        # causing waves in the density field.
        # Therefore, we remove self.density from the compute graph by a projection operation,
        # which makes it an independent variable in the eyes of symbolic differentiation.
        # scalar_space = fem.functionspace(self.mesh, ("CG", 1))
        # density = project(self.density, scalar_space, "isolated_density")
        safety_factor = 10
        I_1_diff = self.density ** (1 / self.penalty) * (I1 - 3) * safety_factor - (
            I1_crit - 3
        )

        # switch = fem.Constant(self.mesh, ScalarType(1.0))
        # fail_area_percent = (
        #     self._assemble_scalar_parallel(heaviside_ufl(I_1_diff) * self.dx)
        #     / self.area
        # )
        # switch.value = heaviside(fail_area_percent - 0.01)
        # self.log(
        #     f"Switch of I_1 constraint: {switch.value}, failing area percentage: {fail_area_percent}"
        # )

        return self._normalized_scalar_expression(
            relu(I_1_diff) * self.dx, "I1_constraint"
        )

    def _constraints_rhs(self):
        return [0.0, args.I1_rhs]

    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        I_1 = self.critical_I1()
        return [volume, I_1]


class StevenDistTopoOpt(StevenTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.dx
        _, self._c_map = self.V_e.sub(self.c_component_id).collapse()

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
        x0[self._c_map] = 1.0
        return x0

    def _bounds(self):
        bnds = super()._bounds()
        for idx in self._c_map:
            bnds[idx] = (0.0, 1.0)
        return bnds

    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        return [volume]

    def _define_design_dirichlet_boundaries(self, *args, **kwargs):
        kwargs.update(dict(constrained_subspaces=[self.density_component_id]))
        return super()._define_design_dirichlet_boundaries(*args, **kwargs)

    def _set_optimized_variable(self, x: np.ndarray):
        if args.disable_rho:
            return NeoHookean._set_optimized_variable(self, x)
        else:
            return super()._set_optimized_variable(x)


class StevenDistTopoOptI1(StevenDistTopoOpt):
    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        I_1 = StevenTopoOptI1.critical_I1(self)
        return [volume, I_1]

    def _constraints_rhs(self):
        return [0.0, args.I1_rhs]


problem = {
    "topo": StevenTopoOpt,
    "topo-i1": StevenTopoOptI1,
    "dist-topo": StevenDistTopoOpt,
    "dist-topo-i1": StevenDistTopoOptI1,
}[args.mode](config_path=args.config)


if args.disable_rho or args.disable_c:
    problem.params.design_fields_disabled = ([0] if args.disable_rho else []) + (
        [1] if args.disable_c else []
    )

problem.params["max_penalty"] = args.max_penalty
problem.params["optimizer"] = args.optimizer
problem.params["ccsamma_rel_tolerance"] = args.ccsamma_rel_tolerance

if args.load is None:
    total_step = args.steps
    for i in range(total_step):
        problem.log(f"Step: {i+1}/{total_step}")
        if problem.params.dim == 2:
            problem.params.dirichlet_bcs_dict = {
                lambda x: np.isclose(x[1], problem.params.height)
                & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0),
                lambda x: np.isclose(x[0], problem.params.width)
                & (x[1] <= 0.4 * problem.params.height): {
                    0: -0.4 * (i + 1) / total_step
                },
            }
        else:
            if args.plane_strain:
                problem.params.dirichlet_bcs_dict = {
                    lambda x: np.isclose(x[1], problem.params.height)
                    & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                    lambda x: x[0] > -1.0: {2: 0},
                    lambda x: np.isclose(x[0], problem.params.width)
                    & (x[1] <= 0.4 * problem.params.height): {
                        0: -0.1 * (i + 1) / total_step
                    },
                }
            else:
                problem.params.dirichlet_bcs_dict = {
                    lambda x: np.isclose(x[1], problem.params.height)
                    & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                    lambda x: np.isclose(x[2], problem.params.depth): {2: 0},
                    lambda x: np.isclose(x[0], problem.params.width)
                    & (x[1] <= 0.4 * problem.params.height): {
                        0: -0.1 * (i + 1) / total_step
                    },
                }

        problem.load_boundary_conditions()
        problem.forward()
        # problem.record_traction_dirichlet(step=i)
        # problem.save_records()
    problem.save(time=i)
    problem.save_checkpoint()
else:
    if problem.params.dim == 2:
        problem.params.dirichlet_bcs_dict = {
            lambda x: np.isclose(x[1], problem.params.height)
            & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0),
            lambda x: np.isclose(x[0], problem.params.width)
            & (x[1] <= 0.4 * problem.params.height): {0: -0.4},
        }
        problem.load_functions_from_checkpoint(args.load)
        problem.forward()
        problem.save()
    else:
        problem.load_functions_from_checkpoint(args.load)
        problem.u = fem.Function(problem.V_u, name="displacement")

        total_step = args.steps
        for i in range(total_step):
            problem.log(f"Step: {i+1}/{total_step}")
            if args.plane_strain:
                problem.params.dirichlet_bcs_dict = {
                    lambda x: np.isclose(x[1], problem.params.height)
                    & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                    lambda x: x[0] > -1.0: {2: 0},
                    lambda x: np.isclose(x[0], problem.params.width)
                    & (x[1] <= 0.4 * problem.params.height): {
                        0: -0.4 * (i + 1) / total_step
                    },
                }
            else:
                problem.params.dirichlet_bcs_dict = {
                    lambda x: np.isclose(x[1], problem.params.height)
                    & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                    lambda x: np.isclose(x[2], problem.params.depth): {2: 0},
                    lambda x: np.isclose(x[0], problem.params.width)
                    & (x[1] <= 0.4 * problem.params.height): {
                        0: -0.4 * (i + 1) / total_step
                    },
                }
            problem.load_boundary_conditions()
            problem.forward()
            problem.save(time=i)
            problem.record_traction_dirichlet(step=i)
            problem.save_records()

        problem.save_checkpoint()

    if not args.disable_rho and args.n_filters > 0:
        # We apply the filter several times to have a blurred start.
        # Otherwise it's easy for the optimizer to simply remove all the materials and get the lowest loss.
        for k in range(args.n_filters):
            problem.log(
                f"Applying filters to blur the initial guess: {k+1}/{args.n_filters}"
            )
            x = problem.e.x.array[:]
            problem._set_optimized_variable(x)
            problem.forward()
        problem.save()


problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
problem.record_file = os.path.join(problem.path, "records_opt.csv")
if problem.params.mesh.comm.rank == 0:
    os.makedirs(problem.vtu_dir, exist_ok=True)
problem.timesteps = []
problem.records = pd.DataFrame()
problem.records.index.name = "Step"

problem.optimize(warm_start=args.load is not None)
problem.save_checkpoint(filename="state_opt.bp")


# prob = NeoHookeanTopoOpt(config_path=args.config)
# prob.forward()
# prob.save()
