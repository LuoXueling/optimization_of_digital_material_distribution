import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressibleDistOpt
from petsc4py.PETSc import ScalarType
import os
import pandas as pd
import numpy as np
from src.utils import project

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./mesh/config_trapezoid.py")
parser.add_argument("--radius", type=float, default=0.05)
parser.add_argument("-l", "--load", type=str, default=None)

args = parser.parse_args()
config_path = args.config
radius = args.radius


class StevenTrapezoidDistOpt(StevenNearlyIncompressibleDistOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.dx
        # self.dx_grad = self.domain_dx(self.association_table["grad_integral_domain"])

    def _tractions(self):
        pk1_stress = self._PK1_stress()
        N = ufl.FacetNormal(self.mesh)
        e_x = fem.Constant(self.mesh, ScalarType([1.0, 0.0]))
        e_y = fem.Constant(self.mesh, ScalarType([0.0, 1.0]))
        tx = -ufl.dot(ufl.dot(pk1_stress, N), e_x) * self.boundary_ds(
            self.association_table["top"]
        )
        ty = -ufl.dot(ufl.dot(pk1_stress, N), e_y) * self.boundary_ds(
            self.association_table["top"]
        )
        return tx, ty

    def _objective_function(self):
        tx, ty = self._tractions()
        tx, ty = self._normalized_scalar_expression(
            tx, "tx"
        ), self._normalized_scalar_expression(ty, "ty")
        tx_val = self._assemble_scalar_parallel(tx)
        ty_val = self._assemble_scalar_parallel(ty)

        self.log(f"Traction x = {tx_val:.10f}, Traction y = {ty_val:.10f}")
        return ty - tx

    def _objective_function_value(self):
        return super()._objective_function_value() + 10

    def _get_cell_data(self):
        cell_data = super()._get_cell_data()
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        grad_e = ufl.grad(self.e)
        grad_norm = ufl.dot(grad_e, grad_e)
        grad_norm_proj = project(grad_norm, scalar_space, "Optimized_projected")
        cell_data.update({"e_grad_norm": grad_norm_proj})
        return cell_data


problem = StevenTrapezoidDistOpt(
    config_path=config_path, override_params={"filter_radius": radius}
)
if args.load is None:
    problem.optimize()
    problem.save_checkpoint()
else:
    problem.load_functions_from_checkpoint(args.load)
    problem.forward()
    total_step = 30
    for i in range(total_step):
        problem.log(f"Step: {i+1}/{total_step}")
        problem.params.dirichlet_bcs = [
            (problem.association_table["bottom"], (0.0, 0.0)),
            (
                problem.association_table["top"],
                {1: -0.3 * (i + 1) / total_step},
            ),
        ]
        problem.load_boundary_conditions()
        problem.forward()
        problem.save(time=i)
        problem.record_traction_dirichlet(step=i)
        problem.save_records()
