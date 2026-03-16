import dolfinx.fem as fem
import ufl
from src.neoHookean import NeoHookean
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from src.utils import project
from src.solvers import solve_linear_problem
from scipy.spatial import cKDTree
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType


class NeoHookeanTopoOpt(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_topo_opt()

    def _init_topo_opt(self):
        # Defined optimized variable
        V_scalar = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.filtered_gradient = fem.Function(V_scalar, name="Filtered gradient")
        # Initialize optimization related variables
        self._initial_objective_values = {}
        self.penalty = self.params.init_penalty

    @property
    def density(self):
        return self.e if self.n_component == 1 else self.e[self.density_component_id]

    @property
    def density_component_id(self):
        return 0

    def _normalized_scalar_expression(self, expression, name, ref_value=None):
        if name not in self._initial_objective_values.keys():
            self._initial_objective_values[name] = (
                self._assemble_scalar_parallel(expression)
                if ref_value is None
                else ref_value
            )
        normalized_scalar = 1 / self._initial_objective_values[name] * expression
        return normalized_scalar

    def _objective_function(self):
        return self._normalized_scalar_expression(
            self._strain_energy_density() * self.dx, "stored_energy"
        )

    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        return [volume]

    def _initial_values(self):
        return np.array([1.0] * len(self.e.x.array))

    def _bounds(self):
        return [(self.params.min_density, 1.0)] * len(self.e.x.array)

    def _define_design_dirichlet_boundaries(
        self,
        fixed_value=1.0,
        boundary_markers=None,
        function_space=None,
        fix_boundary_w_bc=True,
        constrained_subspaces=None,
    ):
        """Define Dirichlet boundary conditions for density field on boundaries
        where Neumann or Dirichlet BCs are applied to displacement field.
        Includes DOFs within num_layers element layers from the boundary.
        """
        function_space = self.V_e if function_space is None else function_space

        # If vector-valued, dispatch once per component
        if function_space.dofmap.bs > 1:
            e_bcs_all, g_bcs_all = [], []
            if constrained_subspaces is None:
                constrained_subspaces = range(function_space.dofmap.bs)
            for comp in constrained_subspaces:
                comp_space = function_space.sub(comp)
                e_bcs, g_bcs = NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
                    self,
                    fixed_value=fixed_value,
                    boundary_markers=boundary_markers,
                    function_space=comp_space,
                    fix_boundary_w_bc=fix_boundary_w_bc,
                )
                e_bcs_all.extend(e_bcs)
                g_bcs_all.extend(g_bcs)
            return e_bcs_all, g_bcs_all

        density_bcs, gradient_bcs = NeoHookean._define_design_dirichlet_boundaries(
            self, function_space=function_space, fixed_value=fixed_value
        )

        try:
            V_collapsed, collapse_map = function_space.collapse()
            collapse_map = np.array(collapse_map, dtype=np.int32)
            is_subspace = True
        except RuntimeError:
            is_subspace = False
            V_collapsed = function_space

        if self.params.fix_density_h is None:
            return density_bcs, gradient_bcs

        if not fix_boundary_w_bc and boundary_markers is None:
            self.log(
                f"Warning: No boundary to fix design. Pass either fix_boundary_w_bc=True or boundary_markers to _define_design_dirichlet_boundaries"
            )

        # Collect all boundary marker IDs from Neumann and Dirichlet BCs
        boundary_markers = set() if boundary_markers is None else set(boundary_markers)

        if fix_boundary_w_bc:
            # Add Neumann BC markers
            for marker_id, _ in self.neumann_bcs:
                boundary_markers.add(marker_id)

            # Add Dirichlet BC markers
            if self.params.dirichlet_bcs is not None:
                for marker_id, _ in self.params.dirichlet_bcs:
                    boundary_markers.add(marker_id)

            # Add geometrically-defined Dirichlet BC markers
            if self.params.dirichlet_bcs_dict is not None:
                for i in range(len(self.params.dirichlet_bcs_dict)):
                    marker_key = f"Config Dirichlet BC {i}"
                    if marker_key in self.association_table:
                        boundary_markers.add(self.association_table[marker_key])

        # Get DOF coordinates (includes ghost DOFs for parallel safety)
        dof_coords = V_collapsed.tabulate_dof_coordinates()

        # Get number of locally owned DOFs
        num_owned = V_collapsed.dofmap.index_map.size_local

        # Collect all boundary dofs (union across markers) and DOFs to fix
        all_boundary_dofs = set()
        all_dofs_to_fix = set()

        for marker_id in boundary_markers:
            facets = self.boundaries.find(marker_id)

            # Ensure necessary topology connectivity exists (both directions)
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

            # Call collectively on all ranks (even if local facets array is empty)
            boundary_dofs = fem.locate_dofs_topological(
                V_collapsed, self.mesh.topology.dim - 1, facets
            )

            # Accumulate union of boundary DOFs (may include ghost DOFs)
            all_boundary_dofs.update(boundary_dofs.tolist())

        # Perform a single radius query with the union of boundary DOFs
        boundary_coords = dof_coords[np.fromiter(all_boundary_dofs, dtype=np.int32)]
        search_distance = float(self.params.fix_density_h)

        # Build KD-tree once for efficient radius queries
        kd_tree = cKDTree(dof_coords)

        neighbor_lists = kd_tree.query_ball_point(boundary_coords, r=search_distance)
        for lst in neighbor_lists:
            for idx in lst:
                if idx < num_owned:  # keep only locally owned dofs
                    all_dofs_to_fix.add(idx)

        # Create BCs with all collected DOFs (only locally owned)
        dofs_array = np.array(list(all_dofs_to_fix), dtype=np.int32)
        dofs_array_parent = collapse_map[dofs_array] if is_subspace else dofs_array

        if fixed_value is not None:
            # Create BC for density with fixed value 1.0
            if len(all_dofs_to_fix) > 0:
                bc_density = fem.dirichletbc(
                    ScalarType(fixed_value),
                    dofs_array_parent,
                    function_space if is_subspace else V_collapsed,
                )
                density_bcs.append(bc_density)

        # Create BC for gradient with fixed value 0.0
        if len(all_dofs_to_fix) > 0:
            bc_gradient = fem.dirichletbc(
                ScalarType(0.0),
                dofs_array_parent,
                function_space if is_subspace else V_collapsed,
            )
            gradient_bcs.append(bc_gradient)

        # Parallel-safe: sum across all processes
        total_dofs_fixed = self.mesh.comm.allreduce(len(all_dofs_to_fix), op=MPI.SUM)
        self.log(f"Created density Dirichlet BCs fixing {total_dofs_fixed} DOFs")
        return density_bcs, gradient_bcs

    @property
    def _degradation(self):
        return self.density**self.penalty

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        return self._degradation * super()._strain_energy_density(F=F, C=C, Eps=Eps)

    def _body_force_energy(self):
        if self.params.body_force_applied_on == "deformed":
            _, J, _, _ = self.strain_invariant()
            return self.density * ufl.dot(J * self.B, self.u) * self.dx
        elif self.params.body_force_applied_on == "reference":
            return self.density * ufl.dot(self.B, self.u) * self.dx
        else:
            raise Exception(
                f"Forces are applied on either deformed or reference configuration."
            )

    def grad(self, x=None):
        grad = super().grad(x)
        if self.n_component == 1:
            V_dens = self.V_e
            collapse_map = None
        else:
            V_dens, collapse_map = self.V_e.sub(self.density_component_id).collapse()

        if self.params.gradient_filter:
            # Create temporary function for gradient to access ghost values
            grad_func = fem.Function(V_dens)
            grad_func.x.array[:] = grad if collapse_map is None else grad[collapse_map]
            grad_func.x.scatter_forward()

            dens_func = fem.Function(V_dens)
            dens_func.x.array[:] = (
                self.e.x.array if collapse_map is None else self.e.x.array[collapse_map]
            )
            dens_func.x.scatter_forward()

            # Deal with the gradient of density: filter method (sparse KD-tree approach)
            coord = V_dens.tabulate_dof_coordinates()
            num_owned = V_dens.dofmap.index_map.size_local

            # Build KD-tree for efficient neighbor search
            tree = cKDTree(coord)

            # Query neighbors within r_min for each owned DOF
            owned_coords = coord[:num_owned, :]
            neighbors_list = tree.query_ball_point(
                owned_coords, self.params.gradient_filter
            )

            # Compute filtered gradient using sparse approach
            filtered_grad = grad_func.x.array[:]

            for p in range(num_owned):
                neighbor_indices = neighbors_list[p]
                if len(neighbor_indices) == 0:
                    continue

                # Get neighbor coordinates, distances, and values
                neighbor_coords = coord[neighbor_indices]
                distances = np.linalg.norm(neighbor_coords - owned_coords[p], axis=1)

                # Compute weights
                weights = self.params.gradient_filter - distances

                # Get gradient and density values at neighbors
                grads = grad_func.x.array[neighbor_indices]
                densities = dens_func.x.array[neighbor_indices]

                # Weighted average
                numerator = np.sum(weights * densities * grads)
                denominator = dens_func.x.array[p] * np.sum(weights)

                if denominator > 1e-12:
                    filtered_grad[p] = numerator / denominator
            if collapse_map is None:
                grad[:num_owned] = filtered_grad[:num_owned]
            else:
                grad[collapse_map[:num_owned]] = filtered_grad[:num_owned]
        else:
            filtered_grad = copy.deepcopy(
                grad if collapse_map is None else grad[collapse_map]
            )

        self.filtered_gradient.x.array[:] = filtered_grad
        self.filtered_gradient.x.scatter_forward()
        return grad

    def _optimize_callback(self, *args) -> bool:
        # self._initialize()
        if self._optimize_step % self.params.output_per_step == 0:
            self.save(time=self._optimize_step)
        self._optimize_step += 1
        if len(self._loss_history) > 2:
            self.log(
                f"Convergence criterion: {(1 - self._loss_history[-1] / self._loss_history[-2])
                * np.sign(self._loss_history[-1])} < {self.params.relative_tol}"
            )
            if len(self._loss_history) < 20:
                self.log("Convergence criterion ignored before 20 iterations.")
            elif (1 - self._loss_history[-1] / self._loss_history[-2]) * np.sign(
                self._loss_history[-1]
            ) > (1 - self._loss_history[-2] / self._loss_history[-3]) * np.sign(
                self._loss_history[-2]
            ):
                self.log(
                    "Loss reduction keeps increasing and convergence criterion is ignored"
                )
            elif (
                0
                < (1 - self._loss_history[-1] / self._loss_history[-2])
                * np.sign(self._loss_history[-1])
                < self.params.relative_tol
                and 0
                < (1 - self._loss_history[-2] / self._loss_history[-3])
                * np.sign(self._loss_history[-2])
                < self.params.relative_tol
            ) or (
                len(self._stage_loss_history) > 500
                and self._stage_loss_history[-50] == min(self._stage_loss_history[-50:])
            ):
                if (
                    self.penalty == self.params.init_penalty
                    and self.params.init_penalty != self.params.max_penalty
                ) or (
                    self.params.init_penalty < self.penalty < self.params.max_penalty
                ):
                    gray_level = self._assemble_scalar_parallel(
                        self.density * (1.0 - self.density) * 4 / self.area * self.dx
                    )
                    self.penalty = min(
                        self.penalty * (1 + 0.2 ** (1 + gray_level / 2)),
                        self.params.max_penalty,
                    )
                    self._stage_loss_history = []
                    self.log(f"Updated penalty to {self.penalty}")
                elif self.penalty >= self.params.max_penalty:
                    self.log(f"Reached maximum penalty and converged. ")
                    return True
        return False

    def _get_point_data(self, main_field=None):
        point_data = super()._get_point_data(main_field)
        point_data["Filtered gradient"] = self.filtered_gradient
        return point_data

    def _get_cell_data(self):
        cell_data = super()._get_cell_data()
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))

        grad_projected = project(
            self.filtered_gradient, scalar_space, "Gradient_projected"
        )
        cell_data["Gradient_projected"] = grad_projected
        return cell_data


class NeoHookeanPDEFilteredTopoOpt(NeoHookeanTopoOpt):
    """
    Applies Helmholtz PDE filtering and tanh projection in-place on self.e
    after every design update. The optimized variables correspond to the
    density DOFs; gradients are not filtered.
    """

    @property
    def density(self):
        raw_density = (
            self.e if self.n_component == 1 else self.e[self.density_component_id]
        )
        beta = 3.0
        eta = 0.5
        num = ufl.tanh(beta * eta) + ufl.tanh(beta * (raw_density - eta))
        den = ufl.tanh(beta * eta) + ufl.tanh(beta * (1.0 - eta))
        proj = num / den
        return proj

    def grad(self, x=None):
        return NeoHookean.grad(self, x)

    def _initial_values(self):
        x0 = np.zeros(len(self.e.x.array))
        if (
            self.params.filter_radius is None
            or self._filter_vars[self.density_component_id]["map"] is None
        ):
            x0[:] = 1.0
        else:
            x0[self._filter_vars[self.density_component_id]["map"]] = 1.0
        return x0

    def _bounds(self):
        bnds = [(-np.inf, np.inf)] * len(self.e.x.array)
        if (
            self.params.filter_radius is None
            or self._filter_vars[self.density_component_id]["map"] is None
        ):
            for i in range(len(bnds)):
                bnds[i] = (self.params.min_density, 1.0)
        else:
            for idx in self._filter_vars[self.density_component_id]["map"]:
                bnds[idx] = (self.params.min_density, 1.0)
        return bnds

    @property
    def _degradation(self):
        return self.density**self.penalty

    def _optimize_callback(self, *args) -> bool:
        return NeoHookeanTopoOpt._optimize_callback(self, *args)

    def _get_point_data(self, main_field=None):
        point_data = NeoHookeanTopoOpt._get_point_data(self, main_field)
        scalar_space = fem.functionspace(self.mesh, ("CG", 1))
        density_projected = project(self.density, scalar_space, "density_projected")
        point_data["density_projected"] = density_projected
        return point_data

    def _get_cell_data(self):
        return NeoHookean._get_cell_data(self)

    def _define_design_dirichlet_boundaries(self, *args, **kwargs):
        kwargs.update(dict(fixed_value=1.0))
        return NeoHookeanTopoOpt._define_design_dirichlet_boundaries(
            self, *args, **kwargs
        )
