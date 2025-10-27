import dolfinx.fem as fem
import ufl
from src.neoHookean import NeoHookean
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from src.utils import project
from scipy.spatial import cKDTree
from petsc4py import PETSc


class NeoHookeanTopoOpt(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_topo_opt()

    def _init_topo_opt(self):
        # Defined optimized variable
        self.filtered_gradient = fem.Function(self.V_e, name="Filtered gradient")

        # Initialize optimization related variables
        self._initial_objective_values = {}
        self.penalty = self.params.init_penalty

    def _normalized_scalar_expression(self, expression, name):
        if name not in self._initial_objective_values.keys():
            self._initial_objective_values[name] = self._assemble_scalar_parallel(
                expression
            )
        normalized_scalar = 1 / self._initial_objective_values[name] * expression
        return normalized_scalar

    def _objective_function(self):
        return None

    def _stored_energy(self):
        return self._normalized_scalar_expression(
            self._strain_energy_density() * self.dx, "stored_energy"
        )

    def _weight_diff(self):
        return self._normalized_scalar_expression(
            (self.e - self.params.target_volume) * self.dx, "weight"
        )

    def _objective_function_value(self):
        stored_energy = self._stored_energy()
        stored_energy_val = self._assemble_scalar_parallel(stored_energy)

        softplus = lambda x: np.log(1 + np.exp(100 * x)) / 100
        weight = self._weight_diff()
        weight_val = 50 * softplus(self._assemble_scalar_parallel(weight))

        weight_current = self._assemble_scalar_parallel(self.e / self.area * self.dx)

        self.log(f"Stored energy: {stored_energy_val}, weight constraint: {weight_val}")
        self.log(f"Real weight: {weight_current} ({self.params.target_volume})")
        return stored_energy_val + weight_val

    def _grad_j_state(self, j, state, state_test):
        stored_energy = self._stored_energy()
        return ufl.derivative(stored_energy, state, state_test)

    def _grad_j_opt(self, j, opt, opt_test):
        stored_energy = self._stored_energy()
        weight_diff_form = self._weight_diff()
        softplus_diff = (
            50
            * (lambda x: np.exp(100 * x) / (1 + np.exp(100 * x)))(
                self._assemble_scalar_parallel(weight_diff_form)
            )
            * ufl.derivative(self.e * self.dx, opt, opt_test)
        )
        return ufl.derivative(stored_energy, opt, opt_test) + softplus_diff

    def _initial_values(self):
        return np.array([1.0] * len(self.e.x.array))

    def _bounds(self):
        return [(self.params.min_density, 1.0)] * len(self.e.x.array)

    def _define_design_dirichlet_boundaries(self):
        """Define Dirichlet boundary conditions for density field on boundaries
        where Neumann or Dirichlet BCs are applied to displacement field.
        Includes DOFs within num_layers element layers from the boundary.

        Args:
            num_layers: Number of element layers below the boundary to fix (default=2)
        """
        density_bcs, gradient_bcs = super()._define_design_dirichlet_boundaries()

        if self.params.fix_density_layers is None:
            return density_bcs, gradient_bcs

        # Collect all boundary marker IDs from Neumann and Dirichlet BCs
        boundary_markers = set()

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
        dof_coords = self.V_e.tabulate_dof_coordinates()

        # Get number of locally owned DOFs
        num_owned = self.V_e.dofmap.index_map.size_local

        # Collect all DOFs to fix (only locally owned)
        all_dofs_to_fix = set()

        for marker_id in boundary_markers:
            facets = self.boundaries.find(marker_id)
            if len(facets) > 0:
                # Locate DOFs on boundary facets
                boundary_dofs = fem.locate_dofs_topological(
                    self.V_e, self.mesh.topology.dim - 1, facets
                )

                # Get coordinates of boundary DOFs (may include ghost DOFs)
                boundary_coords = dof_coords[boundary_dofs]

                # Calculate characteristic element size from boundary DOFs
                if len(boundary_coords) > 1:
                    # Estimate element size as minimum distance between boundary DOFs
                    dist_matrix = np.linalg.norm(
                        boundary_coords[:, None, :] - boundary_coords[None, :, :],
                        axis=2,
                    )
                    # Set diagonal to large value to avoid self-distance
                    np.fill_diagonal(dist_matrix, np.inf)
                    min_distances = np.min(dist_matrix, axis=1)
                    h = np.min(min_distances[np.isfinite(min_distances)])
                else:
                    h = 0.0

                # Expand region by num_layers * h
                search_distance = self.params.fix_density_layers * h

                # Find all locally owned DOFs within search_distance from any boundary DOF
                for i in range(num_owned):
                    dof_pos = dof_coords[i]
                    # Calculate minimum distance to any boundary DOF
                    distances = np.linalg.norm(boundary_coords - dof_pos, axis=1)
                    min_dist = np.min(distances)

                    if min_dist <= search_distance:
                        all_dofs_to_fix.add(i)

        # Create BCs with all collected DOFs (only locally owned)
        if len(all_dofs_to_fix) > 0:
            dofs_array = np.array(list(all_dofs_to_fix), dtype=np.int32)

            # Create BC for density with fixed value 1.0
            bc_func_density = fem.Function(self.V_e)
            bc_func_density.x.array[:] = 1.0
            bc_density = fem.dirichletbc(bc_func_density, dofs_array)
            density_bcs.append(bc_density)

            # Create BC for gradient with fixed value 0.0
            bc_func_gradient = fem.Function(self.V_e)
            bc_func_gradient.x.array[:] = 0.0
            bc_gradient = fem.dirichletbc(bc_func_gradient, dofs_array)
            gradient_bcs.append(bc_gradient)

        # Parallel-safe: sum across all processes
        total_dofs_fixed = self.mesh.comm.allreduce(len(all_dofs_to_fix), op=MPI.SUM)
        self.log(
            f"Created density Dirichlet BCs fixing {total_dofs_fixed} DOFs (including {self.params.fix_density_layers} layers from boundaries)"
        )
        return density_bcs, gradient_bcs

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        return self.e**self.penalty * super()._strain_energy_density(F=F, C=C, Eps=Eps)

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()
        energy = psi * self.dx - self.e * ufl.dot(self.params.B, self.u) * self.dx
        # Neumann boundaries
        for i, val in self.neumann_bcs:
            energy -= ufl.dot(val, self.u) * self.boundary_ds(i)
        return energy

    def grad(self, x=None):
        grad = super().grad(x)

        if self.params.gradient_filter:
            # Create temporary function for gradient to access ghost values
            grad_func = fem.Function(self.V_e)
            grad_func.x.array[:] = grad
            grad_func.x.scatter_forward()

            # Deal with the gradient of density: filter method (sparse KD-tree approach)
            coord = self.V_e.tabulate_dof_coordinates()
            num_owned = self.V_e.dofmap.index_map.size_local

            # Build KD-tree for efficient neighbor search
            tree = cKDTree(coord)

            # Query neighbors within r_min for each owned DOF
            owned_coords = coord[:num_owned, :]
            neighbors_list = tree.query_ball_point(
                owned_coords, self.params.gradient_filter
            )

            # Compute filtered gradient using sparse approach
            filtered_grad = copy.deepcopy(grad)

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
                densities = self.e.x.array[neighbor_indices]

                # Weighted average
                numerator = np.sum(weights * densities * grads)
                denominator = self.e.x.array[p] * np.sum(weights)

                if denominator > 1e-12:
                    filtered_grad[p] = numerator / denominator

            # Clip gradient values
            # filtered_grad = np.clip(filtered_grad, a_min=-0.2, a_max=0.2)
            fem.set_bc(filtered_grad, self.gradient_dirichlet_bcs)
            grad = filtered_grad
        else:
            filtered_grad = copy.deepcopy(grad)

        self.filtered_gradient.x.array[:] = filtered_grad
        self.filtered_gradient.x.scatter_forward()
        return grad

    def _optimize_callback(self, *args) -> bool:
        # self._initialize()
        if self._optimize_step % self.params.output_per_step == 0:
            self.save(time=self._optimize_step)
        self._optimize_step += 1

        if len(self._loss_history) > 1 and (
            0
            < (1 - self._loss_history[-1] / self._loss_history[-2])
            * np.sign(self._loss_history[-1])
            < self.params.relative_tol
            or (
                len(self._stage_loss_history) > 10
                and self._stage_loss_history[-10] == min(self._stage_loss_history[-10:])
            )
        ):
            if (self.penalty == self.params.init_penalty) or (
                self.params.init_penalty < self.penalty < self.params.max_penalty
            ):
                gray_level = self._assemble_scalar_parallel(
                    self.e * (1.0 - self.e) * 4 / self.area * self.dx
                )
                self.penalty = min(
                    self.penalty * (1 + 0.3 ** (1 + gray_level / 2)),
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
