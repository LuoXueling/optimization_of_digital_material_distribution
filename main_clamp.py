from argparse import ArgumentParser
from main_hook import (
    StevenTopoOpt,
    StevenTopoOptI1,
    StevenDistTopoOptI1,
    StevenDistTopoOpt,
)
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
parser.add_argument("--plane_strain", action="store_true")
parser.add_argument("--lr_rho", type=float, default=50.0)
parser.add_argument("--lr_c", type=float, default=0.0)
parser.add_argument("--max_penalty", type=float, default=6.0)
parser.add_argument("--n_filters", type=int, default=0)
args = parser.parse_args()


class StevenTopoOptMaximize(StevenTopoOpt):
    def _stored_energy(self):
        return -StevenTopoOpt._stored_energy(self)

    def _weight_diff(self):
        return self._normalized_scalar_expression(
            (self.density - self.params.target_volume) * self.dx,
            "weight",
            ref_value=(
                (1.0 - self.params.target_volume) * self.area
                if args.load is not None
                else None
            ),
        )


class StevenTopoOptI1Maximize(StevenTopoOptI1):
    def _stored_energy(self):
        return -StevenTopoOptI1._stored_energy(self)

    def _weight_diff(self):
        return self._normalized_scalar_expression(
            (self.density - self.params.target_volume) * self.dx,
            "weight",
            ref_value=(
                (1.0 - self.params.target_volume) * self.area
                if args.load is not None
                else None
            ),
        )


class StevenDistTopoOptMaximize(StevenDistTopoOpt):
    def _stored_energy(self):
        return -StevenTopoOpt._stored_energy(self)

    def _weight_diff(self):
        return self._normalized_scalar_expression(
            (self.density - self.params.target_volume) * self.dx,
            "weight",
            ref_value=(
                (1.0 - self.params.target_volume) * self.area
                if args.load is not None
                else None
            ),
        )


class StevenDistTopoOptI1Maximize(StevenDistTopoOptI1):
    def _stored_energy(self):
        return -StevenTopoOptI1._stored_energy(self)

    def _weight_diff(self):
        return self._normalized_scalar_expression(
            (self.density - self.params.target_volume) * self.dx,
            "weight",
            ref_value=(
                (1.0 - self.params.target_volume) * self.area
                if args.load is not None
                else None
            ),
        )


problem = {
    "topo": StevenTopoOptMaximize,
    "topo-i1": StevenTopoOptI1Maximize,
    "dist-topo": StevenDistTopoOptMaximize,
    "dist-topo-i1": StevenDistTopoOptI1Maximize,
}[args.mode](config_path=args.config)

problem.params["lr_0"] = args.lr_rho
problem.params["lr_1"] = args.lr_c
problem.params["max_penalty"] = args.max_penalty


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
        problem.save(time=i)
        problem.record_traction_dirichlet(step=i)
        problem.save_records()

    problem.save_checkpoint()
else:
    if problem.params.dim == 2:
        problem.params.dirichlet_bcs_dict = {
            lambda x: np.isclose(x[1], problem.params.height)
            & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0),
            lambda x: np.isclose(x[0], problem.params.width)
            & (x[1] <= 0.4 * problem.params.height): {0: -0.4},
        }
    else:
        if args.plane_strain:
            problem.params.dirichlet_bcs_dict = {
                lambda x: np.isclose(x[1], problem.params.height)
                & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                lambda x: x[0] > -1.0: {2: 0},
                lambda x: np.isclose(x[0], problem.params.width)
                & (x[1] <= 0.4 * problem.params.height): {0: -0.1},
            }
        else:
            problem.params.dirichlet_bcs_dict = {
                lambda x: np.isclose(x[1], problem.params.height)
                & (x[0] <= 0.4 * problem.params.width): (0.0, 0.0, 0.0),
                lambda x: np.isclose(x[2], problem.params.depth): {2: 0},
                lambda x: np.isclose(x[0], problem.params.width)
                & (x[1] <= 0.4 * problem.params.height): {0: -0.1},
            }
    problem.load_functions_from_checkpoint(args.load)
    problem.forward()
    problem.save()

    if args.lr_rho > 0 and args.n_filters > 0:
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
