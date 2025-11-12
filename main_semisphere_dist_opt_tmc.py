import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressibleTMCDistOpt
from petsc4py.PETSc import ScalarType
import os
import pandas as pd
import numpy as np
from src.utils import project


parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_semisphere_tmc.py"
)
parser.add_argument("-l", "--load", type=str, default=None)
parser.add_argument("-e", "--case", type=int, default=1)
parser.add_argument("--thres", type=int, default=3000)
parser.add_argument("--smooth", type=int, default=400)

args = parser.parse_args()
config_path = args.config
load_ckpt = args.load
case = args.case
thres = args.thres
smooth = args.smooth


class StevenSemisphereDistOpt(StevenNearlyIncompressibleTMCDistOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.domain_dx(self.association_table["main"])
        # self.dx_grad = self.domain_dx(self.association_table["grad_integral_domain"])

    def _define_design_dirichlet_boundaries(self):
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self,
            fixed_value=None,
            boundary_markers=[
                self.association_table["bottom"],
                self.association_table["semisphere-top"],
            ],
        )

    def _tractions(self):
        pk1_stress = self.params.psi_coef * self._PK1_stress()
        N = ufl.FacetNormal(self.mesh)
        e_x = fem.Constant(self.mesh, ScalarType([1.0, 0.0]))
        e_y = fem.Constant(self.mesh, ScalarType([0.0, 1.0]))
        tx = ufl.dot(ufl.dot(pk1_stress, N), e_x) * self.boundary_ds(
            self.association_table["bottom"]
        )
        ty = ufl.dot(ufl.dot(pk1_stress, N), e_y) * self.boundary_ds(
            self.association_table["bottom"]
        )
        return tx, ty

    def _objective_function(self):
        tx, ty = self._tractions()
        tx, ty = self._normalized_scalar_expression(
            tx, "tx"
        ), self._normalized_scalar_expression(ty, "ty")
        tx_val = self._assemble_scalar_parallel(tx)
        ty_val = self._assemble_scalar_parallel(ty)

        softplus = lambda x: ufl.conditional(
            ufl.lt(x, thres * 10),
            ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )
        grad_e = ufl.grad(self.e)
        grad_norm = (
            softplus(ufl.dot(grad_e, grad_e) - thres) - softplus(0.0 - thres)
        ) * self.dx_grad
        grad_norm_val = self._assemble_scalar_parallel(grad_norm)

        # self.log(f"Traction x = {tx_val:.10f}, Traction y = {ty_val:.10f}")
        self.log(
            f"Traction x = {tx_val:.10f}, Traction y = {ty_val:.10f}, grad_norm = {grad_norm_val}"
        )
        return ty - tx + grad_norm

    def _get_cell_data(self):
        cell_data = super()._get_cell_data()
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        grad_e = ufl.grad(self.e)
        grad_norm = ufl.dot(grad_e, grad_e)
        grad_norm_proj = project(grad_norm, scalar_space, "Optimized_projected")
        cell_data.update({"e_grad_norm": grad_norm_proj})
        return cell_data


ckpt = (
    {
        1: "outputs/semisphere-2025-11-10-10-47-06-Case1-ckpt/state.bp",
        2: "outputs/semisphere-2025-11-10-10-46-57-Case2-ckpt/state.bp",
        3: "outputs/semisphere-2025-11-10-10-46-48-Case3-ckpt/state.bp",
        4: "outputs/semisphere-2025-11-10-10-41-12-Case4-ckpt/state.bp",
    }.get(case, load_ckpt)
    if load_ckpt is None
    else load_ckpt
)

case_func = {
    1: lambda x: (-0.7 / 1.5 * x[0]),
    2: lambda x: (-0.5 / 1.5 * x[0] - 0.12),
    3: lambda x: (-0.3 / 1.5 * x[0] - 0.24),
    4: lambda x: (-0.1 / 1.5 * x[0] - 0.36),
}[case]

problem = StevenSemisphereDistOpt(config_path=config_path)

if ckpt is None:
    total_step = 40
    for i in range(total_step):
        problem.log(f"Step: {i+1}/{total_step}")
        problem.params.dirichlet_bcs = [
            (problem.association_table["bottom"], (0.0, 0.0)),
            (
                problem.association_table["top"],
                {
                    0: 0.0,
                    1: (lambda x: case_func(x) * (i + 1) / total_step),
                },
            ),
            (problem.association_table["third_medium_bottom"], {1: 0}),
        ]

        problem.load_boundary_conditions()
        problem.forward()
        problem.save(time=i)
        problem.record_traction_dirichlet(step=i)
        problem.save_records()

    problem.save_checkpoint()
else:
    problem.params.dirichlet_bcs = [
        (problem.association_table["bottom"], (0.0, 0.0)),
        (
            problem.association_table["top"],
            {0: 0.0, 1: case_func},
        ),
        (problem.association_table["third_medium_bottom"], {1: 0}),
    ]
    problem.load_boundary_conditions()
    problem.load_functions_from_checkpoint(ckpt)
    problem.forward()

problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
problem.record_file = os.path.join(problem.path, "records_opt.csv")
if problem.params.mesh.comm.rank == 0:
    os.makedirs(problem.vtu_dir, exist_ok=True)
problem.timesteps = []
problem.records = pd.DataFrame()
problem.records.index.name = "Step"

problem.optimize()
problem.save_checkpoint()
