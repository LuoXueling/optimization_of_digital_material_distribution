from argparse import ArgumentParser
from src.neoHookean_topo_opt import NeoHookeanPDEFilteredTopoOpt, NeoHookeanTopoOpt
from src.steven import StevenNearlyIncompressible
from src.neoHookean import NeoHookean
import numpy as np
import ufl
import dolfinx.fem as fem
from src.utils import project
import os
import pandas as pd


class StevenTopoOpt(NeoHookeanPDEFilteredTopoOpt):
    def _get_c(self):
        return 1.0

    def _strain_energy_density_wo_constraint(self):
        return StevenNearlyIncompressible._strain_energy_density_wo_constraint(self)

    def _void_indicator(self):
        thres = 0.01
        smooth = 500
        return 1 - (
            np.tanh(smooth * thres) + ufl.tanh(smooth * (self._degradation - thres))
        ) / (np.tanh(smooth * thres) + np.tanh(smooth * (1 - thres)))
        # self.density**self.params.max_penalty instead of self._degradation:
        # if penalty is not large enough, when using self._degradation,
        # the sharpness of this function creates a noisy void_indicator

    def _stored_energy(self):
        return self._normalized_scalar_expression(
            self._degradation
            * StevenNearlyIncompressible._strain_energy_density(self)
            * self.dx,
            "stored_energy",
        )

    @property
    def _degradation(self):
        return ufl.conditional(
            self.density**self.penalty < 1e-3, 1e-3, self.density**self.penalty
        )

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        void_indicator = self._void_indicator()
        scalar_space = fem.functionspace(self.mesh, ("CG", 1))
        void_indicator = project(void_indicator, scalar_space, "void_indicator")
        self.void_indicator = void_indicator

        grad_u = (1 - void_indicator) * ufl.grad(
            self.u
        )  # Degrade grad(u) instead of u, otherwise it creates a significant term grad(void_indicator)*u
        F = ufl.Identity(len(self.u)) + grad_u
        psi_nonlinear_degraded = StevenNearlyIncompressible._strain_energy_density(
            self, F=F
        )

        Ic, J, _, _ = self.strain_invariant(F=F)
        psi_linear_degraded = (self.mu / 2) * (J ** (-2 / 3) * Ic - 3) + (
            self.kappa / 2
        ) * (J - 1) ** 2

        Ic, J, _, _ = self.strain_invariant()
        psi_linear = (self.mu / 2) * (J ** (-2 / 3) * Ic - 3) + (self.kappa / 2) * (
            J - 1
        ) ** 2

        return self._degradation * (
            psi_nonlinear_degraded - psi_linear_degraded + psi_linear
        )

    @property
    def E(self):
        c = self._get_c()
        E = (
            0.30390594 * c**4
            - 0.96196938 * c**3
            + 0.89915784 * c**2
            + 0.39949939 * c
            + 0.69701715
        )
        return E

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

    # def _potential_energy(self, psi=None):
    #     if psi is None:
    #         psi = self._strain_energy_density()
    #     energy = psi * self.dx - self._body_force_energy()

    #     energy = self._apply_neumann_bcs(energy)

    #     Ic, J, F, C = self.strain_invariant()

    #     # if self.mesh.topology.dim != 2:
    #     # raise Exception("Rotation angle regularization is only supported in 2D.")
    #     # tan_phi = (F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1])
    #     # grad_tan_phi = ufl.grad(tan_phi)
    #     grad_J = ufl.grad(J)
    #     kr = self.params.kr_coef
    #     # regularization = (
    #     #     kr * 0.5 * (ufl.dot(grad_tan_phi, grad_tan_phi) + ufl.dot(grad_J, grad_J))
    #     # )
    #     regularization = kr * 0.5 * ufl.dot(grad_J, grad_J)

    #     void_indicator = (
    #         self._void_indicator() if hasattr(self, "_void_indicator") else None
    #     )

    #     if void_indicator is not None:
    #         scalar_space = fem.functionspace(self.mesh, ("CG", 1))
    #         void_indicator = project(void_indicator, scalar_space, "void_indicator")
    #         energy = energy + void_indicator * regularization * self.dx
    #         self.void_indicator = void_indicator
    #         self.regularization_residual = fem.Function(
    #             self.V_u, name="regularization_residual"
    #         )
    #         self.regularization_residual.x.array[:] = fem.assemble_vector(
    #             fem.form(
    #                 ufl.derivative(
    #                     void_indicator * regularization * self.dx, self.u, self.u_test
    #                 )
    #             )
    #         ).array

    #     return energy

    def _get_cell_data(self):
        cell_data = NeoHookeanPDEFilteredTopoOpt._get_cell_data(self)
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        cell_data.update(
            {
                "void_indicator": project(
                    self.void_indicator, scalar_space, "void_indicator"
                ),
            }
        )

        #     vector_space = fem.functionspace(
        #         self.mesh, ("DG", 0, (self.mesh.geometry.dim,))
        #     )
        #     cell_data.update(
        #         {
        #             "regularization_residual": project(
        #                 self.regularization_residual,
        #                 vector_space,
        #                 "regularization_residual",
        #             ),
        #         }
        #     )

        return cell_data


class StevenTopoOptI1(StevenTopoOpt):
    def critical_I1(self):
        c = self._get_c()
        I1_crit = 0.22426408 * c**2 - 1.12468456 * c + 4.883886
        I1, _, _, _ = self.strain_invariant()

        smooth = 0.01
        softplus = lambda x: ufl.conditional(
            ufl.lt(x, 1),
            ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )

        # We only want to compute I1-based failure criterion in regions with high density,
        # so supposedly we should multiply softplus(xxx) by self.density.
        # However, the symbolic differentiation of self.density*softplus(xxx)
        # allows topology optimization to remove material for lower value of this term,
        # causing waves in the density field.
        # Therefore, we remove self.density from the compute graph by a projection operation,
        # which makes it an independent variable in the eyes of symbolic differentiation.
        # scalar_space = fem.functionspace(self.mesh, ("CG", 1))
        # density = project(self.density, scalar_space, "isolated_density")

        safety_factor = 15
        return (
            softplus(ufl.sqrt(self.density) * (I1 - 3) * safety_factor - (I1_crit - 3))
        ) * self.dx

    def _objective_function(self):
        return self._normalized_scalar_expression(
            self._strain_energy_density() * self.dx, "stored_energy"
        )

    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        I_1 = self.critical_I1()
        return [volume, I_1]


class StevenDistTopoOpt(StevenTopoOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx_grad = self.dx
        self.V_c, self._c_map = self.V_e.sub(self.c_component_id).collapse()

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
        x0[self._c_map] = 1.0
        return x0

    def _bounds(self):
        bnds = super()._bounds()
        for idx in self._c_map:
            bnds[idx] = (0.0, 1.0)
        return bnds

    def _objective_function(self):
        thres = 150
        smooth = thres / 10
        softplus = lambda x: ufl.conditional(
            ufl.lt(x, thres * 10),
            ufl.ln(1 + ufl.exp(x / smooth)) * smooth,
            x,
        )
        grad_e = ufl.grad(self.e)
        grad_norm = (
            softplus(ufl.dot(grad_e, grad_e) - thres) - softplus(0.0 - thres)
        ) * self.dx_grad

        return (
            self._normalized_scalar_expression(
                self._strain_energy_density() * self.dx, "stored_energy"
            )
            + grad_norm
        )

    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        return [volume]

    def _define_design_dirichlet_boundaries(self, *args, **kwargs):
        kwargs.update(dict(constrained_subspaces=[self.density_component_id]))
        return super()._define_design_dirichlet_boundaries(*args, **kwargs)


class StevenDistTopoOptI1(StevenDistTopoOpt):
    def _constraints(self):
        volume = (self.density - self.params.target_volume) / self.area * self.dx
        I_1 = self.critical_I1()
        return [volume, I_1]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./mesh/config_hook.py")
    args = parser.parse_args()

    problem = StevenTopoOpt(config_path=args.config)

    total_step = 10
    for i in range(total_step):
        problem.log(f"Step: {i+1}/{total_step}")
        problem.params.neumann_bcs_dict = {
            lambda x: np.isclose(x[1], (1 - 0.6) * problem.params.height)
            & (x[0] > problem.params.width - 0.6): (0.0, -0.005 * (i + 1) / total_step)
        }

        problem.load_boundary_conditions()
        problem.forward()
        problem.save(time=i)
        problem.record_traction_dirichlet(step=i)
        problem.save_records()

    problem.save_checkpoint()

    problem.vtu_dir = os.path.join(problem.path, "vtu_opt")
    problem.pvd_file = os.path.join(problem.path, "output_opt.pvd")
    problem.record_file = os.path.join(problem.path, "records_opt.csv")
    if problem.params.mesh.comm.rank == 0:
        os.makedirs(problem.vtu_dir, exist_ok=True)
    problem.timesteps = []
    problem.records = pd.DataFrame()
    problem.records.index.name = "Step"

    problem.optimize()

    # prob = NeoHookeanTopoOpt(config_path=args.config)
    # prob.forward()
    # prob.save()
