import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
from src.neoHookean_tmc import NeoHookeanTMC
from src.steven import StevenNearlyIncompressibleTMCDistOpt
from petsc4py.PETSc import ScalarType
import os
import pandas as pd
import numpy as np
from src.utils import project


parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="./mesh/config_compress_ball_tmc.py"
)
parser.add_argument("-l", "--load", type=str, default=None)
parser.add_argument("-e", "--case", type=int, default=1)
parser.add_argument("--thres", type=int, default=3000)
parser.add_argument("--smooth", type=int, default=400)

args = parser.parse_args()
config_path = args.config
ckpt = args.load
case = args.case
thres = args.thres
smooth = args.smooth


class StevenCompressBallDistOpt(StevenNearlyIncompressibleTMCDistOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.domain_dx(self.association_table["main"])
        # self.dx_grad = self.domain_dx(self.association_table["grad_integral_domain"])
        self.ds_circle = self._define_inner_surface_measure(
            boundary_names=[
                "outer_circle_top",
                "outer_circle_left",
            ],
            main_obj="main",
        )

    def _define_inner_surface_measure(self, boundary_names, main_obj):
        # The surface of the circle is inside the computational domain, so we need to extract
        # the correct surface (i.e., the facets attached to the circle, not the third medium)
        # To perform correct integration of surface normal
        # Code ref: https://fenicsproject.discourse.group/t/wrong-facetnormal-vector-on-internal-boundaries/12887/2
        subdomain_data = []

        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        facet_map = self.mesh.topology.index_map(fdim)
        f_to_c = self.mesh.topology.connectivity(fdim, tdim)
        c_to_f = self.mesh.topology.connectivity(tdim, fdim)

        for b_name in boundary_names:
            # Give a set of facets marked with a value (in this case 2), get a consistent orientation for an interior integral
            facets_to_integrate = self.boundaries.find(self.association_table[b_name])

            # Compute integration entities for a single facet of a cell.
            # Each facet is represented as a tuple (cell_index, local_facet_index), where cell_index is local to process
            # local_facet_index is the local indexing of a facet for a given cell
            integration_entities = []
            for i, facet in enumerate(facets_to_integrate):
                # Only loop over facets owned by the process to avoid duplicate integration
                if facet >= facet_map.size_local:
                    continue
                # Find cells connected to facet
                cells = f_to_c.links(facet)
                # Get value of cells
                marked_cells = self.domains.values[cells]
                # Get the cell marked with 2
                correct_cell = np.flatnonzero(
                    marked_cells == self.association_table[main_obj]
                )

                assert len(correct_cell) == 1
                # Get local index of facet
                local_facets = c_to_f.links(cells[correct_cell[0]])
                local_index = np.flatnonzero(local_facets == facet)
                assert len(local_index) == 1

                # Append integration entities
                integration_entities.append(cells[correct_cell[0]])
                integration_entities.append(local_index[0])

            subdomain_data.append(
                (
                    self.association_table[b_name],
                    np.array(integration_entities, dtype=np.int32),
                )
            )

        # Create custom integration measure for one-sided integrals
        subdomain_data.sort(key=lambda x: x[0])
        return ufl.ds(
            metadata={"quadrature_degree": self.params.q_deg},
            domain=self.mesh,
            subdomain_data=subdomain_data,
        )

    def _define_design_dirichlet_boundaries(self):
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self,
            fixed_value=None,
            boundary_markers=[self.association_table["outer_circle_top"]],
            fix_boundary_w_bc=False,
        )

    def _objective_function(self):
        F = self._deformation_gradient()
        N = ufl.FacetNormal(self.mesh)
        top = (F[1, 0] * N[1] - F[1, 1] * N[0]) ** 2 * self.ds_circle(
            self.association_table["outer_circle_top"]
        )
        left_right = (F[0, 0] * N[1] - F[0, 1] * N[0]) ** 2 * self.ds_circle(
            self.association_table["outer_circle_left"]
        )

        self.log(
            f"Integration top: {self._assemble_scalar_parallel(top)}, integration left: {self._assemble_scalar_parallel(left_right)}"
        )
        return left_right + top


# class NeoHookeanCompressBallDistOpt(NeoHookeanTMC):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dx_grad = self.domain_dx(self.association_table["main"])
#         # self.dx_grad = self.domain_dx(self.association_table["grad_integral_domain"])
#         self.ds_circle = StevenCompressBallDistOpt._define_inner_surface_measure(
#             self=self,
#             boundary_names=[
#                 "outer_circle_top",
#                 "outer_circle_left",
#             ],
#             main_obj="main",
#         )
#         self.e.x.array[:] = self.params.E

#     @property
#     def mu(self):
#         return self.e / (2 * (1 + self.params.nu))

#     @property
#     def kappa(self):
#         return self.e / (3 * (1 - 2 * self.params.nu))

#     def _optimize_callback(self, *args) -> bool:
#         res = super()._optimize_callback(*args)
#         self.record_traction_dirichlet(step=self._optimize_step)
#         self.save_records()
#         return res

#     def _objective_function(self):
#         return StevenCompressBallDistOpt._objective_function(self)

#     def _initial_values(self):
#         return np.array([self.params.E] * len(self.e.x.array))

#     def _bounds(self):
#         return [(0.7, 1.3)] * len(self.e.x.array)

#     def _define_design_dirichlet_boundaries(self):
#         return StevenCompressBallDistOpt._define_design_dirichlet_boundaries(self)


problem = StevenCompressBallDistOpt(config_path=config_path)

if ckpt is None:
    total_step = 50
    for i in range(total_step):
        problem.log(f"Step: {i+1}/{total_step}")
        problem.params.dirichlet_bcs = [
            (
                problem.association_table["top"],
                (
                    0.0,
                    -(0.75 - 2 * np.pi * 0.35 * np.sqrt(2) / 8 - 0.05)
                    * (i + 1)
                    / total_step,
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

    problem.save_checkpoint()
else:
    problem.params.dirichlet_bcs = [
        (problem.association_table["bottom"], {1: 0}),
        (problem.association_table["right"], {0: 0}),
        (
            problem.association_table["top"],
            (0.0, -(0.75 - 2 * np.pi * 0.35 * np.sqrt(2) / 8 - 0.05)),
        ),
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
