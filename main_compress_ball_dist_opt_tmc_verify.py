import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean import NeoHookean
from src.neoHookean_topo_opt import (
    NeoHookeanTopoOpt,
    NeoHookeanMappedTopoOpt,
    NeoHookeanPDEFilteredTopoOpt,
)
from src.neoHookean_tmc import NeoHookeanTMC
from src.steven import (
    StevenNearlyIncompressibleTMCDistOpt,
    Steven,
    StevenNearlyIncompressible,
)
from petsc4py.PETSc import ScalarType
import os
import pandas as pd
import numpy as np
from src.utils import project


parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_compress_ball_filled_tmc.py"
)
parser.add_argument(
    "-l",
    "--load",
    type=str,
    default="outputs/compress_ball_filled-2025-11-29-15-55-12/state_opt.bp",
)
parser.add_argument("-e", "--case", type=str, default="topo-pde")
parser.add_argument("--thres", type=int, default=3000)
parser.add_argument("--smooth", type=int, default=400)

args = parser.parse_args()
config_path = args.config
ckpt = args.load
case = args.case
thres = args.thres
smooth = args.smooth


from main_compress_ball_dist_opt_tmc import (
    StevenCompressBallDistOpt,
    StevenCompressBallPDEFilteredTopoOpt,
)

problem = {
    "dist": StevenCompressBallDistOpt,
    "topo-pde": StevenCompressBallPDEFilteredTopoOpt,
    # "dist-topo": StevenCompressBallDistPDEFilteredTopoOpt,
}[case](config_path=config_path)

problem.load_functions_from_checkpoint(ckpt)
problem.u = fem.Function(problem.V_u, name="displacement")

total_step = 50
for i in range(total_step):
    problem.log(f"Step: {i+1}/{total_step}")
    problem.params.dirichlet_bcs = [
        (
            problem.association_table["top"],
            (
                0.0,
                -(0.75 - 0.35 - 0.08) * (i + 1) / total_step,
            ),
        ),
        (problem.association_table["bottom"], {1: 0}),
        (problem.association_table["right"], {0: 0}),
    ]

    problem.load_boundary_conditions()
    problem.forward()
    problem.save(time=i)
    problem.record_traction_dirichlet(step=i)
    problem.save_records()

# problem.save_checkpoint()
