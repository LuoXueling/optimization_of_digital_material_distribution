import dolfin as fe
from src.neoHookean import NeoHookean
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


class NeoHookeanTopoOpt(NeoHookean):
    def __init__(self, config_path):
        super().__init__(config_path)

        # Defined optimized variable
        self.filtered_gradient = fe.Function(self.V_e, name="Filtered gradient")

        # Initialize optimization related variables
        self._initial_strain_energy_density = None
        self._initial_weight_constraint = None
        self.penalty = self.params.init_penalty

    def _objective_function(self):
        if self._initial_strain_energy_density is None:
            self._initial_strain_energy_density = fe.assemble(
                self._strain_energy_density() * self.dx
            )
        compliance = (
            self._strain_energy_density()
            / self._initial_strain_energy_density
            * self.dx
        )
        softplus = lambda x: fe.ln(1 + fe.exp(10 * x)) / 10
        if self._initial_weight_constraint is None:
            self._initial_weight_constraint = fe.assemble(
                softplus(self.e - self.params.target_volume) * self.dx
            )
        weight = (
            50
            * softplus(self.e - self.params.target_volume)
            / self._initial_weight_constraint
            * self.dx
        )
        gray_level = 20 * self.e * (1.0 - self.e) * 4 / self.area * self.dx
        self.log(
            f"Compliance: {fe.assemble(compliance)}, Average density squared: {fe.assemble(weight)}, Gray level: {fe.assemble(gray_level)}"
        )
        return compliance + weight + gray_level

    def _initial_values(self):
        return np.array([1.0] * len(self.e.vector()))

    def _bounds(self):
        return [(self.params.min_density, 1.0)] * len(self.e.vector())

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        to_3d = lambda x: (
            fe.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
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
            J = fe.det(F)
        else:
            C = 2.0 * Eps + fe.Identity(len(self.u))
            C = to_3d(C)
            J = fe.sqrt(fe.det(C))
        Ic = fe.tr(C)

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
        psi = (mu / 2) * (Ic - 3) - mu * fe.ln(J) + (lmbda / 2) * (fe.ln(J)) ** 2
        return psi

    def grad(self, x):
        grad = super().grad(x)
        # Deal with the gradient of density: filter method
        coord = self.V_e.tabulate_dof_coordinates()
        filtered_grad = copy.deepcopy(grad)
        for p in range(
            coord.shape[0]
        ):  # Suppose that the first coord.shape[0] elements in grad is for density in topology optimization
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
            grads = grad[within_filter]
            densities = x[within_filter]
            weights = self.params.r_min - distances
            # if max(x[p] - np.min(densities), np.max(densities) - x[p]) > 0.1:
            p_grad = np.sum(weights * densities * grads) / (x[p] * np.sum(weights))
            filtered_grad[p] = p_grad
        filtered_grad = np.clip(filtered_grad, a_min=-0.2, a_max=0.2)
        grad = filtered_grad
        self.filtered_gradient.vector()[:] = filtered_grad
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
                gray_level = (
                    fe.assemble(self.e * (1.0 - self.e) * self.dx) * 4 / self.area
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

    def save(self, time=0):
        super().save(time=time)
        self.xdmffile.write(self.filtered_gradient, time)
