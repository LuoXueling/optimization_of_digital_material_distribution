import dolfin as fe
import ufl
from src.default_config import params
import os
import importlib
import shutil
from src.utils import log
from functools import partial
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pickle
import pandas as pd


class NeoHookean:
    def __init__(self, config_path, override_params=None):
        # Load parameters
        params_updated = importlib.import_module(config_path.strip(".py")).params
        self.params = params.copy()
        self.params.update(params_updated)
        if override_params is not None:
            self.params.update(override_params)

        # Initialize for outputs
        self.path = os.path.join(
            self.params.output_root,
            f"{self.params.project_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        )
        os.makedirs(self.path, exist_ok=True)
        self.log = partial(log, output_file=os.path.join(self.path, "log.txt"))
        shutil.copy2(config_path, os.path.join(self.path, "config.py"))
        if override_params is not None:
            self.log(f"Parameters are overriden: {override_params}")
            self.log(f"Saving overriding parameters")
            try:
                with open(os.path.join(self.path, "config_override.pkl"), "wb") as file:
                    pickle.dump(override_params, file)
            except Exception as e:
                self.log(f"Failed to save overriding parameters: {e}")

        self.xdmffile = fe.XDMFFile(MPI.COMM_WORLD, os.path.join(self.path, "out.xdmf"))
        self.xdmffile.parameters["flush_output"] = True
        self.xdmffile.parameters["functions_share_mesh"] = True
        self.xdmffile.parameters["rewrite_function_mesh"] = False

        # Initialize dolfin
        self.dx = fe.dx(metadata={"quadrature_degree": self.params.q_deg})
        self.ds = fe.ds(metadata={"quadrature_degree": self.params.q_deg})
        self.dS = fe.dS(metadata={"quadrature_degree": self.params.q_deg})
        fe.parameters["form_compiler"]["quadrature_degree"] = self.params.q_deg

        # Create mesh
        self.mesh = self.params.mesh
        self.area = fe.assemble(
            1.0
            * self.dx(
                metadata={"quadrature_degree": self.params.q_deg}, domain=self.mesh
            )
        )

        # define function space
        self.V_u = fe.VectorFunctionSpace(
            self.mesh, "Lagrange", self.params.elem_order
        )  # For displacement

        # Define functions
        self.du = fe.TrialFunction(self.V_u)  # Incremental field
        self.u_test = fe.TestFunction(self.V_u)  # Test function
        self.u = fe.Function(
            self.V_u, name="displacement"
        )  # fields from previous iteration
        # self.u.vector()[:] = 10

        # Defined optimized variable
        self.V_e = fe.FunctionSpace(
            self.mesh, "Lagrange", self.params.elem_order
        )  # For the optimized variable
        # https://fenicsproject.discourse.group/t/how-to-modify-the-value-of-a-function-object/4890
        self.e = fe.Function(self.V_e, name="Optimized")
        self.e.vector()[:] = 1
        self.gradient = fe.Function(self.V_e, name="Gradient")
        # self.e = E
        # Or set values by the following methods
        # e.vector().set_local(my_values)
        # e.interpolate(fe.Expression("(x[0]-width/2)*0.2+10", width=width, degree=1), M)

        self.load_boundary_conditions()

        # Initialize optimization related variables
        self._optimize_step = 0
        self._loss_history = []
        self._stage_loss_history = []

        self.optimized = False

        self.records = pd.DataFrame()
        self.records.index.name = "Step"

    def _initialize_fields(self):
        self.u = fe.Function(self.V_u, name="displacement")

    def load_boundary_conditions(self):
        # Mark Neumann boundaries
        if self.params.boundaries is None and (
            self.params.neumann_bcs is not None or self.params.dirichlet_bcs is not None
        ):
            raise Exception(
                "Must provide MeshFunction to params.boundaries if defining boundary conditions using boundary id"
            )
        if (
            (
                self.params.neumann_bcs is not None
                or self.params.dirichlet_bcs is not None
            )
            and (
                self.params.neumann_bcs_dict is not None
                or self.params.dirichlet_bcs_dict is not None
            )
            and self.params.association_table is None
        ):
            raise Exception(
                "Must provide association table if defining boundary conditions using both boundary id and dolfin Domain"
            )

        self.association_table = (
            self.params.association_table
            if self.params.association_table is not None
            else {}
        )
        self.neumann_bcs = (
            self.params.neumann_bcs if self.params.neumann_bcs is not None else []
        )
        self.boundaries = (
            self.params.boundaries
            if self.params.boundaries is not None
            else fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        )

        if self.params.neumann_bcs_dict is not None:
            for i, (key, val) in enumerate(self.params.neumann_bcs_dict.items()):
                # Find a index for this bc
                j = 1
                while j in self.association_table.values():
                    j += 1
                key.mark(self.boundaries, j)
                self.neumann_bcs.append((i + 1, val))
                self.association_table[f"Config Neumann BC {i}"] = j

        # Generate Dirichlet BC
        # Mark boundaries first, then get a list of DirichletBC defined by boundary id
        if self.params.dirichlet_bcs_dict is not None:
            for i, (key, val) in enumerate(self.params.dirichlet_bcs_dict.items()):
                # Find a index for this bc
                j = 1
                while j in self.association_table.values():
                    j += 1
                key.mark(self.boundaries, j)
                self.association_table[f"Config Dirichlet BC {i}"] = j

        self.dirichlet_bcs = self._define_dirichlet_boundaries()

        self.log(f"Association table: {self.association_table}")

        self.boundary_ds = fe.ds(
            metadata={"quadrature_degree": self.params.q_deg},
            domain=self.mesh,
            subdomain_data=self.boundaries,
        )

        for i, val in self.neumann_bcs:
            forces = [fe.assemble(x * self.boundary_ds(i)) for x in val]
            self.log(f"Applying Neumann boundary {i}: {forces}")

    def _define_dirichlet_boundaries(self, fixed_value=None):
        dirichlet_bcs = []
        if self.params.dirichlet_bcs_dict is not None:
            dirichlet_bcs += [
                fe.DirichletBC(
                    self.V_u,
                    val if fixed_value is None else fe.Constant([0.0] * len(val)),
                    self.boundaries,
                    self.association_table[f"Config Dirichlet BC {i}"],
                )
                for i, (_, val) in enumerate(self.params.dirichlet_bcs_dict.items())
            ]

        if self.params.dirichlet_bcs is not None:
            dirichlet_bcs += [
                fe.DirichletBC(
                    self.V_u,
                    val if fixed_value is None else fe.Constant([0.0] * len(val)),
                    self.boundaries,
                    key,
                )
                for key, val in self.params.dirichlet_bcs
            ]

        return dirichlet_bcs

    def forward(self):
        psi = self._strain_energy_density()

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = fe.derivative(Pi, self.u, self.u_test)

        # Compute Jacobian of F
        J = fe.derivative(F, self.u, self.du)

        # Solve variational problem
        fe.solve(
            F == 0,
            self.u,
            self.dirichlet_bcs,
            J=J,
            solver_parameters={"newton_solver": {"linear_solver": "mumps"}},
        )

    def _objective_function(self):
        raise NotImplementedError()

    def _set_optimized_variable(self, x: np.ndarray):
        self.e.vector()[:] = x

    def _bounds(self):
        raise NotImplementedError()

    def _initial_values(self):
        raise NotImplementedError()

    def eval(self, x):
        self.log(
            f"Taking input: min={MPI.COMM_WORLD.allreduce(np.min(x), op=MPI.MIN)}, max={MPI.COMM_WORLD.allreduce(np.max(x), op=MPI.MAX)}"
        )
        self._set_optimized_variable(x)
        # Solve \partial L/\partial \lambda = 0, i.e. the governing equation it self
        self.log("Solving forward problem")
        MPI.COMM_WORLD.Barrier()
        try:
            self.forward()
        except:
            self._initialize_fields()
            return np.inf
        loss = fe.assemble(self._objective_function())
        self.log(f"Loss: {loss}")
        MPI.COMM_WORLD.Barrier()
        return loss

    def grad(self, x):
        self.log("Solving gradient")
        MPI.COMM_WORLD.Barrier()
        self._set_optimized_variable(x)
        # Solve \partial L/\partial u = 0 for \lambda. It's a linear equation.
        j = self._objective_function()
        dj_du = fe.assemble(fe.derivative(j, self.u, self.u_test))
        rhs = -dj_du
        lagrangian = fe.Function(
            self.V_u
        )  # Lagrangian multipliers share the same space as u
        l_test = fe.TrialFunction(self.V_u)
        Pi = self._potential_energy()
        lhs = fe.assemble(
            fe.derivative(
                fe.derivative(
                    fe.derivative(
                        Pi, self.u, lagrangian
                    ),  # Derivative of potential energy w.r.t. u and inner product with lagrangian
                    self.u,
                    self.u_test,
                ),  # Derivative w.r.t. u
                lagrangian,
                l_test,  # Tangent matrix * l_test
            )
        )
        # No values at DirichletBC boundaries
        bcs = self._define_dirichlet_boundaries(fixed_value=0.0)

        solver = fe.LUSolver("mumps")
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
        MPI.COMM_WORLD.Barrier()
        return grad.get_local()

    def optimize(self):
        x0 = self._initial_values()
        x = x0
        self._set_optimized_variable(x0)

        for i in range(self.params.maxiter):
            print()
            MPI.COMM_WORLD.Barrier()
            self.log(f"----- Iteration {i} -----")
            res = self.eval(x)
            self._loss_history.append(res)
            self._stage_loss_history.append(res)
            grad = self.grad(x)
            x -= self.params.lr * grad
            for i, bound in zip(range(len(x)), self._bounds()):
                if bound[0] is not None:
                    x[i] = max(bound[0], x[i])
                if bound[1] is not None:
                    x[i] = min(bound[1], x[i])
            MPI.COMM_WORLD.Barrier()
            self.optimized = True
            terminate = self._optimize_callback()
            if terminate:
                break
        if MPI.COMM_WORLD.Get_rank() == 0:
            plt.figure(figsize=(4, 3), dpi=300)
            ax = plt.subplot(111)
            ax.plot(np.arange(len(self._loss_history)), self._loss_history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, "loss.pdf"))

    def _optimize_callback(self, *args) -> bool:
        # self._initialize_fields()
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
            return True
        return False

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
        mu = self.params.E / (2 * (1 + self.params.nu))  # Lame second coefficient
        lmbda = (
            self.params.E
            * self.params.nu
            / (1 + self.params.nu)
            / (1 - 2 * self.params.nu)
        )  # Lame first coefficient

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (Ic - 3) - mu * fe.ln(J) + (lmbda / 2) * (fe.ln(J)) ** 2
        return psi

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()
        energy = psi * self.dx - fe.dot(self.params.B, self.u) * self.dx
        # Neumann boundaries
        for i, val in self.neumann_bcs:
            energy -= fe.dot(val, self.u) * self.boundary_ds(i)
        return energy

    def _deformation_gradient(self):
        u = self.u
        # Kinematics
        d = len(u)
        I = fe.Identity(d)  # Identity tensor
        F = I + fe.grad(u)  # Deformation gradient
        return F

    def _right_cauchy_green(self, F=None):
        if F is None:
            F = self._deformation_gradient()
        C = F.T * F  # Right Cauchy-Green tensor
        return C

    def _strain(self, F=None, C=None):
        if F is None:
            F = self._deformation_gradient()
        if C is None:
            C = self._right_cauchy_green(F)
        d = len(self.u)
        I = fe.Identity(d)  # Identity tensor
        e = 0.5 * (C - I)
        return e

    def _PK2_stress(self, psi=None, e=None, F=None, C=None):
        if F is None:
            F = self._deformation_gradient()
        # Make them fe.variable to make the strain energy density differentiable w.r.t. F and e
        F = fe.variable(F)
        if e is None:
            e = self._strain(F=F, C=C)
        e = fe.variable(e)
        if psi is None:
            psi = self._strain_energy_density(Eps=e)
        s = fe.diff(psi, e)
        return s

    def _PK1_stress(self, psi=None, F=None):
        if F is None:
            F = self._deformation_gradient()
        F = fe.variable(F)
        if psi is None:
            psi = self._strain_energy_density(F=F)
        p = fe.diff(psi, F)
        return p

    def _Cauchy_stress(self, psi=None, F=None):
        if F is None:
            F = self._deformation_gradient()
        F = fe.variable(F)
        if psi is None:
            psi = self._strain_energy_density(F=F)
        p = fe.diff(psi, F)
        J = fe.det(F)
        sigma = J ** (-1) * p * ufl.transpose(F)
        return sigma

    def save(self, main_field=None, time=0):
        # Save solution in XDMF format
        self.xdmffile.write(self.u if main_field is None else main_field, time)
        if self.optimized:
            self.xdmffile.write(self.e, time)
            self.xdmffile.write(self.gradient, time)

        # For tensors
        tensor_space = fe.TensorFunctionSpace(self.mesh, "DG", 0)
        e = fe.project(self._strain(), tensor_space)
        e.rename("strain", "(No description)")
        sigma = fe.project(self._PK2_stress(), tensor_space)
        sigma.rename("PK2 stress", "(No description)")
        self.xdmffile.write(e, time)
        self.xdmffile.write(sigma, time)

        # For computed scalars
        scalar_space = fe.FunctionSpace(self.mesh, "DG", 0)
        psi = fe.project(self._strain_energy_density(), scalar_space)
        psi.rename("strain_energy_density", "(No description)")
        self.xdmffile.write(psi, time)

        if self.optimized:
            e_projected = fe.project(self.e, scalar_space)
            e_projected.rename("Optimized (projected)", "(No description)")
            self.xdmffile.write(e_projected, time)

            grad_projected = fe.project(self.filtered_gradient, scalar_space)
            grad_projected.rename("Gradient (projected)", "(No description)")
            self.xdmffile.write(grad_projected, time)

    def compute_traction_dirichlet(self, type="PK1"):
        space_for_stress = fe.TensorFunctionSpace(self.mesh, "DG", 0)
        if type == "PK1":
            stress = self._PK1_stress()
        elif type == "Cauchy":
            stress = self._Cauchy_stress()
        else:
            raise Exception(
                f"Stress type {type} not supported for computing tractions."
            )
        pk1_stress = fe.project(stress, space_for_stress)
        N = fe.FacetNormal(self.mesh)
        e_x = fe.Constant([1.0, 0.0] + ([] if self.params.dim == 2 else [0.0]))
        e_y = fe.Constant([0.0, 1.0] + ([] if self.params.dim == 2 else [0.0]))
        e_z = None if self.params.dim == 2 else fe.Constant((0.0, 0.0, 1.0))
        tx = fe.dot(fe.dot(pk1_stress, N), e_x)
        ty = fe.dot(fe.dot(pk1_stress, N), e_y)
        tz = None if self.params.dim == 2 else fe.dot(fe.dot(pk1_stress, N), e_z)
        tractions = {}
        for name, key in self.association_table.items():
            tractions[name] = [
                fe.assemble(tx * self.boundary_ds(key)),
                fe.assemble(ty * self.boundary_ds(key)),
            ] + ([] if tz is None else [fe.assemble(tz * self.boundary_ds(key))])
        return tractions

    def record_traction_dirichlet(self, type="PK1", step=0):
        tractions = self.compute_traction_dirichlet(type="PK1")
        for name, values in tractions.items():
            for val, dir in zip(
                values, ["x", "y"] + (["z"] if len(values) == 3 else [])
            ):
                self.records.loc[step, f"{name}_{dir}"] = val

    def record(self, step, name, value):
        self.records.loc[step, name] = value

    def save_records(self):
        self.records.to_csv(os.path.join(self.path, "records.csv"))
