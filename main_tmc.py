from argparse import ArgumentParser
from src.neoHookean import NeoHookean
from src.neoHookean_tmc import NeoHookeanTMC
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.neoHookean_incompressible import (
    NeoHookeanIncompressible,
    NeoHookeanIncompressibleTopoOpt,
    NeoHookeanIncompressibleStaggered,
)
import time
import ufl
import numpy as np
import dolfinx.fem as fem
from petsc4py.PETSc import ScalarType

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./mesh/config_box.py")

args = parser.parse_args()
config_path = args.config

problem = NeoHookeanTMC(config_path=config_path)
total_step = 50
for i in range(total_step):
    print(f"{i+1}/{total_step}")
    # For c-shape
    problem.params.dirichlet_bcs_dict = {
        lambda x: np.isclose(x[0], 0.0): fem.Constant(
            problem.mesh, ScalarType((0.0, 0.0))
        ),  # Modify for 3D
        lambda x: np.logical_and(x[0] >= 10, np.isclose(x[1], problem.params.height)): {
            1: -10 * (i + 1) / total_step
        },
    }

    # For config_box
    # problem.params.neumann_bcs_dict = {
    #     lambda x: np.logical_and(
    #         np.logical_and(x[0] >= 0.8, x[0] <= 1.2),
    #         np.isclose(x[1], problem.params.height),
    #     ): fem.Constant(
    #         problem.mesh, ScalarType((0.0, -1 * (i + 1) / total_step))
    #     ),  # Modify for 3D
    # }

    # For config_ball
    # problem.params.dirichlet_bcs = [
    #     (
    #         problem.association_table["bottom"],
    #         fem.Constant(problem.mesh, ScalarType((0.0, 0.0))),
    #     ),
    #     (
    #         problem.association_table["ball"],
    #         fem.Constant(problem.mesh, ScalarType((0.0, -0.55 * (i + 1) / total_step))),
    #     ),
    # ]
    problem.load_boundary_conditions()
    problem.forward()
    problem.save(time=i)
    problem.record_traction_dirichlet(step=i)
    problem.save_records()
