from argparse import ArgumentParser
from src.neoHookean import NeoHookean
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.neoHookean_incompressible import NeoHookeanIncompressible
from src.steven import Steven, StevenStaggered
import numpy as np
import dolfinx.fem as fem
from petsc4py.PETSc import ScalarType

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config.py")

args = parser.parse_args()
config_path = args.config

problem = Steven(config_path=config_path)

for step, stretch in enumerate(np.linspace(0, 2, 20)):
    problem.params.dirichlet_bcs_dict = {
        lambda x: np.isclose(x[1], 0.0): (0.0, 0.0, 0.0),
        lambda x: np.isclose(
            x[1], problem.params.length, atol=problem.params.mesh_size
        ): (0.0, stretch * problem.params.length, 0.0),
    }
    problem.load_boundary_conditions()
    problem.forward()
    problem.save(time=step)
    problem.record(step=step, name="stretch", value=stretch)
    problem.record_traction_dirichlet(step=step)
    problem.save_records()
