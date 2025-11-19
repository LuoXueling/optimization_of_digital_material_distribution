from src.neoHookean import NeoHookean
import ufl
import dolfinx.fem as fem
from src.utils import project


class NeoHookeanTMC(NeoHookean):
    def __init__(
        self,
        config_path,
        override_params=None,
    ):
        super().__init__(config_path, override_params)
        self.dx = self.domain_dx(self.association_table[self.params.main_medium])
        self.third_medium_dx = self.domain_dx(
            self.association_table[self.params.third_medium]
        )

    def _void_indicator(self):
        return None

    def _strain_energy_density_third_medium(self, F=None, C=None, Eps=None):
        return self._strain_energy_density(F=F, C=C, Eps=Eps)

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()
        psi_third_medium = self._strain_energy_density_third_medium()

        Ic, J, F, C = self.strain_invariant()

        if (
            self.params.tmc_regularization is None
            or self.params.tmc_regularization == "HuHu"
        ):
            kr = self.params.kr_coef
            hessian = ufl.grad(ufl.grad(self.u))
            psi_reg = 0.5 * ufl.inner(hessian, hessian)
            regularization = kr * psi_reg
        elif self.params.tmc_regularization == "rotation_angle":
            if self.mesh.topology.dim != 2:
                raise Exception(
                    "Rotation angle regularization is only supported in 2D."
                )
            tan_phi = (F[0, 1] - F[1, 0]) / (F[0, 0] + F[1, 1])
            grad_tan_phi = ufl.grad(tan_phi)
            grad_J = ufl.grad(J)
            kr = self.params.kr_coef
            regularization = (
                kr
                * 0.5
                * (ufl.dot(grad_tan_phi, grad_tan_phi) + ufl.dot(grad_J, grad_J))
            )
        else:
            raise NotImplementedError(
                f"{self.params.tmc_regularization} is not implemented."
            )

        self.psi_third_medium = self.params.psi_coef * psi_third_medium
        self.regularizaiton = regularization

        void_indicator = (
            self._void_indicator() if hasattr(self, "_void_indicator") else None
        )

        energy = (
            psi * self.dx
            + (self.params.psi_coef * psi_third_medium + regularization)
            * self.third_medium_dx
            - ufl.dot(self.B, self.u) * self.dx
        )
        if void_indicator is not None:
            energy = (
                energy
                + void_indicator
                * (self.params.psi_coef * psi_third_medium + regularization)
                * self.dx
            )
        # Neumann boundaries
        for i, val in self.neumann_bcs:
            energy -= ufl.dot(val, self.u) * self.boundary_ds(i)
        return energy

    def _get_cell_data(self):
        cell_data = NeoHookean._get_cell_data(self)
        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        cell_data.update(
            {
                "third_medium_strain_energy_density": project(
                    self.psi_third_medium, scalar_space, "psi_third_medium"
                ),
                "third_medium_regularization": project(
                    self.regularizaiton, scalar_space, "regularization"
                ),
            }
        )
        return cell_data
