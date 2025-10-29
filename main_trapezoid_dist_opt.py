import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressibleTMCDistOpt
from petsc4py.PETSc import ScalarType
import os
import pandas as pd


class StevenTrapezoidDistOpt(StevenNearlyIncompressibleTMCDistOpt):
    def _define_design_dirichlet_boundaries(self):
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self,
            fixed_value=None,
            boundary_markers=[self.association_table["trapezoid_top"]],
        )

    def _objective_function(self):
        pk1_stress = self.params.psi_coef * self._PK1_stress()
        N = ufl.FacetNormal(self.mesh)
        e_x = fem.Constant(self.mesh, ScalarType([1.0, 0.0]))
        e_y = fem.Constant(self.mesh, ScalarType([0.0, 1.0]))
        tx = -ufl.dot(ufl.dot(pk1_stress, N), e_x) * self.boundary_ds(
            self.association_table["top"]
        )
        ty = -ufl.dot(ufl.dot(pk1_stress, N), e_y) * self.boundary_ds(
            self.association_table["top"]
        )
        tx_val = self._assemble_scalar_parallel(tx)
        ty_val = self._assemble_scalar_parallel(ty)

        self.log(
            f"Traction vertical = {ty_val:.10f} Traction horizontal = {tx_val:.10f}"
        )

        return ty - tx


parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_trapezoid_tmc.py"
)

args = parser.parse_args()
config_path = args.config

problem = StevenTrapezoidDistOpt(config_path=config_path)

total_step = 40
for i in range(total_step):
    problem.log(f"Step: {i+1}/{total_step}")
    problem.params.dirichlet_bcs = [
        (
            problem.association_table["bottom"],
            fem.Constant(problem.mesh, ScalarType((0.0, 0.0))),
        ),
        (
            problem.association_table["top"],
            fem.Constant(problem.mesh, ScalarType((0.0, -0.4 * (i + 1) / total_step))),
        ),
        (
            problem.association_table["third_medium_bottom"],
            {1: 0},
        ),
    ]

    problem.load_boundary_conditions()
    problem.forward()
    problem.save(time=i)
    problem.record_traction_dirichlet(step=i)
    problem.save_records()

problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
problem.record_file = os.path.join(problem.path, "records_opt.csv")
if problem.params.mesh.comm.rank == 0:
    os.makedirs(problem.vtu_dir, exist_ok=True)
problem.timesteps = []
problem.records = pd.DataFrame()
problem.records.index.name = "Step"

problem.optimize()
