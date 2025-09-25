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
        # https://fenicsproject.discourse.group/t/how-to-modify-the-value-of-a-function-object/4890
        self.e = fe.Function(self.V_e, name="Optimized")
        self.e.vector()[:] = self.params.E
        self.gradient = fe.Function(self.V_e, name="Gradient")
        self.filtered_gradient = fe.Function(self.V_e, name="Filtered gradient")
        # self.e = E
        # Or set values by the following methods
        # e.vector().set_local(my_values)
        # e.interpolate(fe.Expression("(x[0]-width/2)*0.2+10", width=width, degree=1), M)

        # Initialize optimization related variables
        self._optimize_step = 0
        self._initial_strain_energy_density = None
        self._initial_weight_constraint = None
        self._loss_history = []
        self._stage_loss_history = []
        self.penalty = self.params.init_penalty

    def eval(self, x):
        self.log(f"Taking input: min={np.min(x)}, max={np.max(x)}")
        self._set_optimized_variable(x)
        # Solve \partial L/\partial \lambda = 0, i.e. the governing equation it self
        self.log("Solving forward problem")
        try:
            self.forward()
        except:
            self._initialize()
            return np.inf
        loss = fe.assemble(self._objective_function())
        self.log(f"Loss: {loss}")
        return loss

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

    def grad(self, x):
        self.log("Solving gradient")
        self._set_optimized_variable(x)
        # Solve \partial L/\partial u = 0 for \lambda. It's a linear equation.
        j = self._objective_function()
        dj_du = fe.assemble(fe.derivative(j, self.u, self.v))
        rhs = -dj_du
        lagrangian = fe.Function(
            self.V_u
        )  # Lagrangian multipliers share the same space as u
        dl = fe.TrialFunction(self.V_u)
        Pi = self._potential_energy()
        lhs = fe.assemble(
            fe.derivative(
                fe.derivative(
                    fe.derivative(
                        Pi, self.u, lagrangian
                    ),  # Derivative of potential energy w.r.t. u and inner product with lagrangian
                    self.u,
                    self.v,
                ),  # Derivative w.r.t. u
                lagrangian,
                dl,  # Tangent matrix * dl
            )
        )
        bcs = [
            fe.DirichletBC(self.V_u, fe.Constant([0.0] * len(val)), key)
            for key, val in self.params.dirichlet_bcs_dict.items()
        ]  # No values at DirichletBC boundaries
        solver = fe.LUSolver("default")
        for bc in bcs:
            bc.apply(lhs)
            bc.apply(rhs)
        solver.solve(lhs, lagrangian.vector(), rhs)
        # self.log(np.sum(np.abs(np.array(lagrangian.vector()))))

        # Compute \partial L/\partial param
        v_e = fe.TestFunction(self.V_e)
        dj_dp = fe.assemble(fe.derivative(j, self.e, v_e))
        dfl_dp = fe.assemble(
            fe.derivative(fe.derivative(Pi, self.u, lagrangian), self.e, v_e)
        )
        grad = dj_dp + dfl_dp
        self.gradient.vector()[:] = grad
        return grad.get_local()

    def _set_optimized_variable(self, x: np.ndarray):
        self.e.vector()[:] = x

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

    def optimize(self):
        x0 = self._initial_values()
        x = x0
        self.e.vector()[:] = x0

        for i in range(self.params.maxiter):
            print()
            self.log(f"----- Iteration {i} -----")
            res = self.eval(x)
            self._loss_history.append(res)
            self._stage_loss_history.append(res)
            grad = self.grad(x)

            # Deal with the gradient of density: filter method
            coord = self.V_e.tabulate_dof_coordinates()
            filtered_grad = copy.deepcopy(grad)
            for p in range(
                coord.shape[0]
            ):  # Suppose that the first coord.shape[0] elements in grad is for density in topology optimization
                p_pos = coord[p, :]
                pdist2 = np.sqrt(
                    np.sum(
                        (
                            coord
                            - np.repeat(p_pos.reshape(1, -1), coord.shape[0], axis=0)
                        )
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

            x -= self.params.lr * grad
            for i, bound in zip(range(len(x)), self._bounds()):
                if bound[0] is not None:
                    x[i] = max(bound[0], x[i])
                if bound[1] is not None:
                    x[i] = min(bound[1], x[i])
            terminate = self._optimize_callback()
            if terminate:
                break
        plt.figure(figsize=(4, 3), dpi=300)
        ax = plt.subplot(111)
        ax.plot(np.arange(len(self._loss_history)), self._loss_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "loss.pdf"))

        # scipy.optimize.fmin_l_bfgs_b(
        #     func=self.eval,
        #     x0=self._initial_values(),
        #     fprime=self.grad,
        #     bounds=self._bounds(),
        #     maxls=20,
        #     maxiter=50,
        #     factr=1e7,
        #     pgtol=1e-100,
        #     iprint=1,
        #     callback=self._optimize_callback,
        # )

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

        self.xdmffile.write(self.e, time)
        self.xdmffile.write(self.gradient, time)
        self.xdmffile.write(self.filtered_gradient, time)

        scalar_space = fe.FunctionSpace(self.mesh, "DG", 0)

        e_projected = fe.project(self.e, scalar_space)
        e_projected.rename("Optimized (projected)", "(No description)")
        self.xdmffile.write(e_projected, time)

        grad_projected = fe.project(self.filtered_gradient, scalar_space)
        grad_projected.rename("Gradient (projected)", "(No description)")
        self.xdmffile.write(grad_projected, time)
