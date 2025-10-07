import dolfinx.fem as fem
import ufl
from src.neoHookean import NeoHookean
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


class NeoHookeanTopoOpt(NeoHookean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Defined optimized variable
        self.filtered_gradient = fem.Function(self.V_e, name="Filtered gradient")

        # Initialize optimization related variables
        self._initial_values = {}
        self.penalty = self.params.init_penalty


    def _normalized_scalar_expression(self, expression, name):
        if name not in self._initial_values.keys():
            local_val = fem.assemble_scalar(fem.form(expression))
            self._initial_values[name] = self.mesh.comm.allreduce(local_val, op=MPI.SUM)
        normalized_scalar = 1 / self._initial_values[name] * expression
        return normalized_scalar

    def _objective_function(self):
        stored_energy = self._normalized_scalar_expression(
            self._strain_energy_density() * self.dx
        )
        softplus = lambda x: ufl.ln(1 + ufl.exp(10 * x)) / 10
        weight = 50 * self._normalized_scalar_expression(
            softplus(self.e - self.params.target_volume) * self.dx
        )
        gray_level = 20 * self.e * (1.0 - self.e) * 4 / self.area * self.dx

        stored_energy_val = self._assemble_scalar_parallel(stored_energy)
        weight_val = self._assemble_scalar_parallel(weight)
        gray_level_val = self._assemble_scalar_parallel(gray_level)

        self.log(
            f"Stored energy: {stored_energy_val}, weight constraint: {weight_val}, Gray level: {gray_level_val}"
        )
        return stored_energy + weight + gray_level

    def _initial_values(self):
        return np.array([1.0] * len(self.e.x.array))

    def _bounds(self):
        return [(self.params.min_density, 1.0)] * len(self.e.x.array)

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        to_3d = lambda x: (
            ufl.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
            if self.params.dim == 2
            else x
        )

        if Eps is None:
            if F is None:
                F = self._deformation_gradient()
            if C is None:
                C = self._right_cauchy_green(F=F)
            F = to_3d(F)
            C = to_3d(C)
            J = ufl.det(F)
        else:
            C = 2.0 * Eps + ufl.Identity(len(self.u))
            C = to_3d(C)
            J = ufl.sqrt(ufl.det(C))
        Ic = ufl.tr(C)

        # Elasticity parameters
        degraded_E = self.e**self.penalty * self.params.E
        # degraded_E = E
        mu = degraded_E / (2 * (1 + self.params.nu))  # Lame second coefficient
        lmbda = (
            degraded_E
            * self.params.nu
            / (1 + self.params.nu)
            / (1 - 2 * self.params.nu)
        )  # Lame first coefficient

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
        return psi

    def grad(self, x):
        grad = super().grad(x)

        # Create temporary function for gradient to access ghost values
        grad_func = fem.Function(self.V_e)
        grad_func.x.array[:] = grad
        grad_func.x.scatter_forward()

        # Deal with the gradient of density: filter method
        coord = self.V_e.tabulate_dof_coordinates()

        # Get number of locally owned DOFs
        num_owned = self.V_e.dofmap.index_map.size_local

        filtered_grad = copy.deepcopy(grad)

        # Only compute filtered gradient for locally owned DOFs
        for p in range(num_owned):
            p_pos = coord[p, :]
            pdist2 = np.sqrt(
                np.sum(
                    (coord - np.repeat(p_pos.reshape(1, -1), coord.shape[0], axis=0))
                    ** 2,
                    axis=-1,
                )
            )
            within_filter = np.where(pdist2 <= self.params.r_min)[0]
            distances = pdist2[within_filter]
            grads = grad_func.x.array[within_filter]
            densities = self.e.x.array[within_filter]
            weights = self.params.r_min - distances
            # if max(x[p] - np.min(densities), np.max(densities) - x[p]) > 0.1:
            p_grad = np.sum(weights * densities * grads) / (self.e.x.array[p] * np.sum(weights))
            filtered_grad[p] = p_grad

        filtered_grad = np.clip(filtered_grad, a_min=-0.2, a_max=0.2)
        grad = filtered_grad
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
