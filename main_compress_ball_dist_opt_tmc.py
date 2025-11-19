import ufl
import dolfinx.fem as fem
from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanTopoOpt
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
    default="outputs/compress_ball_filled-2025-11-19-12-34-13-ckpt/state.bp",
)
parser.add_argument("-e", "--case", type=str, default="dist")
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
        self.e.x.array[:] = self.params.c

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
                try:
                    assert len(correct_cell) == 1
                except Exception as e:
                    self.log(
                        "There is a unknown error related to which cell the facet is connected to. Try a different number of mpi processes"
                    )
                    raise e
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

    def _morphology(self):
        F = self._deformation_gradient()
        N = ufl.FacetNormal(self.mesh)
        top = (F[1, 0] * N[1] - F[1, 1] * N[0]) ** 2 * self.ds_circle(
            self.association_table["outer_circle_top"]
        )
        left = (F[0, 0] * N[1] - F[0, 1] * N[0]) ** 2 * self.ds_circle(
            self.association_table["outer_circle_left"]
        )
        return top, left

    def _objective_function(self):
        top, left = self._morphology()

        self.log(
            f"Integration top: {self._assemble_scalar_parallel(top)}, integration left: {self._assemble_scalar_parallel(left)}"
        )
        return left + top


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


class StevenCompressBallTopoOpt(NeoHookeanTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )
        self.main_area = self._assemble_scalar_parallel(1.0 * self.dx)

        self.ds_circle = self._define_inner_surface_measure(
            boundary_names=[
                "outer_circle_top",
                "outer_circle_left",
            ],
            main_obj="main",
        )

    def _define_inner_surface_measure(self, boundary_names, main_obj):
        return StevenCompressBallDistOpt._define_inner_surface_measure(
            self, boundary_names=boundary_names, main_obj=main_obj
        )

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        return self._strain_energy_density(F=F, C=C, Eps=Eps)

    def _get_c(self):
        return self.params.c

    def _strain_energy_density_wo_constraint(self):
        return StevenNearlyIncompressible._strain_energy_density_wo_constraint(self)

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

    def _void_indicator(self):
        return 1 / (1 + ufl.exp(500 * (self.e - 0.01)))

    def _potential_energy(self, psi=None):
        return (
            NeoHookeanTMC._potential_energy(self, psi)
            + (1 - self.e) * ufl.dot(self.B, self.u) * self.dx
        )

    def _strain_energy_density(self, *args, **kwargs):
        return self.e**self.penalty * StevenNearlyIncompressible._strain_energy_density(
            self, *args, **kwargs
        )

    def _define_design_dirichlet_boundaries(self):
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self,
            fixed_value=None,
            boundary_markers=[
                self.association_table["outer_circle_top"],
                self.association_table["outer_circle_left"],
            ],
            fix_boundary_w_bc=False,
        )

    def _get_cell_data(self):
        cell_data = NeoHookeanTMC._get_cell_data(self)
        topo_cell_data = NeoHookeanTopoOpt._get_cell_data(self)
        cell_data.update(topo_cell_data)
        return cell_data

    def _objective_function_value(self):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = top + left
        morp_val = self._assemble_scalar_parallel(morp)

        softplus = lambda x: np.log(1 + np.exp(100 * x)) / 100
        weight = self._weight_diff()
        weight_val = 50 * softplus(self._assemble_scalar_parallel(weight))

        weight_current = self._assemble_scalar_parallel(
            self.e / self.main_area * self.dx
        )

        self.log(f"Morphology: {morp_val}, weight constraint: {weight_val}")
        self.log(f"Real weight: {weight_current} ({self.params.target_volume})")
        return morp_val + weight_val

    def _grad_j_state(self, j, state, state_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = top + left
        return ufl.derivative(morp, state, state_test)  # Should be nothing left

    def _grad_j_opt(self, j, opt, opt_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = top + left
        weight_diff_form = self._weight_diff()
        softplus_diff = (
            50
            * (lambda x: np.exp(100 * x) / (1 + np.exp(100 * x)))(
                self._assemble_scalar_parallel(weight_diff_form)
            )
            * ufl.derivative(self.e * self.dx, opt, opt_test)
        )
        return ufl.derivative(morp, opt, opt_test) + softplus_diff


problem = {"dist": StevenCompressBallDistOpt, "topo": StevenCompressBallTopoOpt}[case](
    config_path=config_path
)

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
    problem.save(time=0)

# problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
# problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
# problem.record_file = os.path.join(problem.path, "records_opt.csv")
# if problem.params.mesh.comm.rank == 0:
#     os.makedirs(problem.vtu_dir, exist_ok=True)
# problem.timesteps = []
# problem.records = pd.DataFrame()
# problem.records.index.name = "Step"

# problem.optimize()
# problem.save_checkpoint()
