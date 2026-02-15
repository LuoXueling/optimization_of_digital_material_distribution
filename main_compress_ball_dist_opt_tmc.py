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


class StevenCompressBallPDEFilteredTopoOpt(NeoHookeanPDEFilteredTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )
        self.area = self._assemble_scalar_parallel(1.0 * self.dx)

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

    def _strain_energy_density_third_medium(self, *args, **kwargs):
        return StevenNearlyIncompressible._strain_energy_density(self, *args, **kwargs)

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

    @property
    def density(self):
        raw_density = (
            self.e if self.n_component == 1 else self.e[self.density_component_id]
        )
        beta = 6.0
        eta = 0.5
        num = ufl.tanh(beta * eta) + ufl.tanh(beta * (raw_density - eta))
        den = ufl.tanh(beta * eta) + ufl.tanh(beta * (1.0 - eta))
        proj = num / den
        return proj

    @property
    def _degradation(self):
        return ufl.conditional(
            self.density**self.penalty < 5e-2, 5e-2, self.density**self.penalty
        )

    def _void_indicator(self):
        return 1 / (1 + ufl.exp(20 * (self._degradation - 0.2)))

    def _body_force_energy(self):
        return NeoHookeanTopoOpt._body_force_energy(self)

    def _potential_energy(self, psi=None):
        return NeoHookeanTMC._potential_energy(self, psi)

    def _strain_energy_density(self, *args, **kwargs):
        return self._degradation * StevenNearlyIncompressible._strain_energy_density(
            self, *args, **kwargs
        )

    # def _set_optimized_variable(self, x: np.ndarray):
    #     raise NotImplementedError()

    def _define_design_dirichlet_boundaries(self, *args, **kwargs):
        kwargs.update(
            dict(
                fixed_value=1.0,
                boundary_markers=[
                    self.association_table["outer_circle_top"],
                    self.association_table["outer_circle_left"],
                ],
                fix_boundary_w_bc=False,
            )
        )
        return NeoHookeanPDEFilteredTopoOpt._define_design_dirichlet_boundaries(
            self, *args, **kwargs
        )

    def _get_cell_data(self):
        cell_data = NeoHookeanTMC._get_cell_data(self)
        topo_cell_data = NeoHookeanPDEFilteredTopoOpt._get_cell_data(self)
        cell_data.update(topo_cell_data)
        return cell_data

    def _get_point_data(self, main_field=None):
        point_data = super()._get_point_data(main_field)

        # beta = 12.0
        # eta = 0.5
        # num = ufl.tanh(beta * eta) + ufl.tanh(beta * (self.density - eta))
        # den = ufl.tanh(beta * eta) + ufl.tanh(beta * (1.0 - eta))
        # proj = num / den
        # scalar_space = fem.functionspace(self.mesh, ("CG", 1))
        # density_projected = project(proj, scalar_space, "density_projected")
        # point_data["density_projected"] = density_projected
        return point_data

    def _objective_function_value(self):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        morp_val = self._assemble_scalar_parallel(morp)

        softplus = lambda x: np.log(1 + np.exp(100 * x)) / 100
        weight = self._weight_diff()
        weight_val = self.params.get("volume_cons_weight", 10.0) * softplus(
            self._assemble_scalar_parallel(weight)
        )

        weight_current = self._assemble_scalar_parallel(
            self.density / self.area * self.dx
        )

        self.log(f"Morphology: {morp_val}, weight constraint: {weight_val}")
        self.log(f"Real weight: {weight_current} ({self.params.target_volume})")
        return morp_val + weight_val

    def _grad_j_state(self, j, state, state_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        return ufl.derivative(morp, state, state_test)  # Should be nothing left

    def _grad_j_opt(self, j, opt, opt_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        weight_diff_form = self._weight_diff()
        softplus_diff = (
            self.params.get("volume_cons_weight", 10.0)
            * (lambda x: np.exp(100 * x) / (1 + np.exp(100 * x)))(
                self._assemble_scalar_parallel(weight_diff_form)
            )
            * ufl.derivative(weight_diff_form, opt, opt_test)
        )
        return ufl.derivative(morp, opt, opt_test) + softplus_diff


class StevenCompressBallDistPDEFilteredTopoOpt(StevenCompressBallPDEFilteredTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.domain_dx(self.association_table["main"])
        self.V_c, self._c_map = self.V_e.sub(self.c_component_id).collapse()
        x = self._initial_values()
        self._set_optimized_variable(x)

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
        x0[self._c_map] = self.params.c
        return x0

    def _bounds(self):
        bnds = super()._bounds()
        for idx in self._c_map:
            bnds[idx] = (0.0, 1.0)
        return bnds

    def _gradient_norm_c(self):
        softplus = lambda x: ufl.conditional(
            ufl.lt(x, thres * 10),
            ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )
        c = self._get_c()
        grad_c = ufl.grad(c)
        grad_norm = (
            softplus(ufl.dot(grad_c, grad_c) - thres) - softplus(0.0 - thres)
        ) * self.dx_grad
        return grad_norm

    def _objective_function_value(self):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        morp_val = self._assemble_scalar_parallel(morp)

        softplus = lambda x: np.log(1 + np.exp(100 * x)) / 100
        weight = self._weight_diff()
        weight_val = self.params.get("volume_cons_weight", 10.0) * softplus(
            self._assemble_scalar_parallel(weight)
        )

        weight_current = self._assemble_scalar_parallel(
            self.density / self.area * self.dx
        )

        grad_norm = self._gradient_norm_c()
        grad_norm_val = self._assemble_scalar_parallel(grad_norm)

        self.log(
            f"Morphology: {morp_val}, weight constraint: {weight_val}, gradient norm: {grad_norm_val}"
        )
        self.log(f"Real weight: {weight_current} ({self.params.target_volume})")
        return morp_val + weight_val + grad_norm_val

    def _grad_j_state(self, j, state, state_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        return ufl.derivative(morp, state, state_test)  # Should be nothing left

    def _grad_j_opt(self, j, opt, opt_test):
        top, left = StevenCompressBallDistOpt._morphology(self)
        morp = self._normalized_scalar_expression(left, "morp")
        grad_norm = self._gradient_norm_c()
        weight_diff_form = self._weight_diff()
        softplus_diff = (
            self.params.get("volume_cons_weight", 10.0)
            * (lambda x: np.exp(100 * x) / (1 + np.exp(100 * x)))(
                self._assemble_scalar_parallel(weight_diff_form)
            )
            * ufl.derivative(weight_diff_form, opt, opt_test)
        )
        return ufl.derivative(morp + grad_norm, opt, opt_test) + softplus_diff

    def _define_design_dirichlet_boundaries(self, *args, **kwargs):
        kwargs.update(
            dict(
                fixed_value=1.0,
                boundary_markers=[
                    self.association_table["outer_circle_top"],
                    self.association_table["outer_circle_left"],
                ],
                fix_boundary_w_bc=False,
                constrained_subspaces=[self.density_component_id],
            )
        )
        e_bcs, g_bcs = NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self, *args, **kwargs
        )
        e_bcs2, g_bcs2 = NeoHookean._define_design_dirichlet_boundaries(
            self, function_space=self.V_e.sub(self.c_component_id), fixed_value=None
        )
        e_bcs.extend(e_bcs2)
        g_bcs.extend(g_bcs2)
        return e_bcs, g_bcs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./mesh/config_compress_ball_filled_tmc.py"
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default=None,
    )
    parser.add_argument("-e", "--case", type=str, default="topo-dist")
    parser.add_argument("--thres", type=int, default=3000)
    parser.add_argument("--smooth", type=int, default=400)

    args = parser.parse_args()
    config_path = args.config
    ckpt = args.load
    case = args.case
    thres = args.thres
    smooth = args.smooth

    problem = {
        "dist": StevenCompressBallDistOpt,
        "topo-pde": StevenCompressBallPDEFilteredTopoOpt,
        "topo-dist": StevenCompressBallDistPDEFilteredTopoOpt,
        # "dist-topo": StevenCompressBallDistPDEFilteredTopoOpt,
    }[case](config_path=config_path)

    if ckpt is None:
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

        problem.save_checkpoint()
    else:
        problem.params.dirichlet_bcs = [
            (problem.association_table["bottom"], {1: 0}),
            (problem.association_table["right"], {0: 0}),
            (
                problem.association_table["top"],
                (0.0, -(0.75 - 0.35 - 0.08)),
            ),
        ]
        problem.load_boundary_conditions()
        problem.load_functions_from_checkpoint(ckpt)
        problem.forward()
        problem.save(time=0)

    problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
    problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
    problem.record_file = os.path.join(problem.path, "records_opt.csv")
    if problem.params.mesh.comm.rank == 0:
        os.makedirs(problem.vtu_dir, exist_ok=True)
    problem.timesteps = []
    problem.records = pd.DataFrame()
    problem.records.index.name = "Step"

    problem.optimize()
    problem.save_checkpoint(filename="state_opt.bp")
