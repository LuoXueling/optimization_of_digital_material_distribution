from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import dolfinx.io as io
from dolfinx.la import InsertMode
import ufl
from src.default_config import params
from src.solvers import solve_nonlinear_problem
import os
import importlib
import shutil
from src.utils import log, write_vtu, write_pvd, project
from functools import partial
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
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
        self.vtu_dir = os.path.join(self.path, "vtu")
        if self.params.mesh.comm.rank == 0:
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.vtu_dir, exist_ok=True)
            shutil.copy2(config_path, os.path.join(self.path, "config.py"))
        self.pvd_file = os.path.join(self.path, "output.pvd")
        self.timesteps = []
        self.log = partial(
            log,
            output_file=os.path.join(self.path, "log.txt"),
            comm=self.params.mesh.comm,
        )
        if self.params.mesh.comm.rank == 0:
            if override_params is not None:
                self.log(f"Parameters are overriden: {override_params}")
                self.log(f"Saving overriding parameters")
                try:
                    with open(
                        os.path.join(self.path, "config_override.pkl"), "wb"
                    ) as file:
                        pickle.dump(override_params, file)
                except Exception as e:
                    pass

        # Initialize measures
        self.dx = ufl.dx(metadata={"quadrature_degree": self.params.q_deg})
        self.ds = ufl.ds(metadata={"quadrature_degree": self.params.q_deg})
        self.dS = ufl.dS(metadata={"quadrature_degree": self.params.q_deg})

        # Create mesh
        self.mesh = self.params.mesh

        self.area = self._assemble_scalar_parallel(
            1.0
            * self.dx(
                metadata={"quadrature_degree": self.params.q_deg}, domain=self.mesh
            )
        )

        # define function space
        self.V_u = fem.functionspace(
            self.mesh, ("Lagrange", self.params.elem_order, (self.mesh.geometry.dim,))
        )  # For displacement

        # Define functions
        self.du = ufl.TrialFunction(self.V_u)  # Incremental field
        self.u_test = ufl.TestFunction(self.V_u)  # Test function
        self.u = fem.Function(
            self.V_u, name="displacement"
        )  # fields from previous iteration

        # Defined optimized variable
        self.V_e = fem.functionspace(
            self.mesh, ("Lagrange", self.params.elem_order)
        )  # For the optimized variable
        self.e = fem.Function(self.V_e, name="Optimized")
        self.e.x.array[:] = 1.0
        self.gradient = fem.Function(self.V_e, name="Gradient")
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
        self.u = fem.Function(self.V_u, name="displacement")

    def load_boundary_conditions(self):
        # Mark Neumann boundaries
        if self.params.boundaries is None and (
            self.params.neumann_bcs is not None or self.params.dirichlet_bcs is not None
        ):
            raise Exception(
                "Must provide MeshTags to params.boundaries if defining boundary conditions using boundary id"
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
                "Must provide association table if defining boundary conditions using both boundary id and geometrical marker"
            )

        self.association_table = (
            self.params.association_table
            if self.params.association_table is not None
            else {}
        )
        self.neumann_bcs = []
        # Check if V_u is a subspace (from mixed space) by trying to collapse
        try:
            V_u_collapsed, _ = self.V_u.collapse()
        except RuntimeError:
            V_u_collapsed = self.V_u
        for marker_id, val in (
            self.params.neumann_bcs if self.params.neumann_bcs is not None else []
        ):
            if callable(val) and not isinstance(val, (fem.Constant, fem.Expression)):
                # Convert function to fem.Function
                neumann_func = fem.Function(V_u_collapsed)
                neumann_func.interpolate(val)
                self.neumann_bcs.append((marker_id, neumann_func))
            else:
                self.neumann_bcs.append((marker_id, val))

        # For geometrically defined boundaries, create MeshTags
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )

        if self.params.boundaries is not None:
            self.boundaries = self.params.boundaries
        else:
            # Create empty MeshTags
            facet_dim = self.mesh.topology.dim - 1
            num_facets = self.mesh.topology.index_map(facet_dim).size_local
            self.boundaries = dmesh.meshtags(
                self.mesh,
                facet_dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        # Collect all geometrically-defined BCs
        bc_list = []

        # Add Neumann BCs
        if self.params.neumann_bcs_dict is not None:
            for i, (marker_func, val) in enumerate(
                self.params.neumann_bcs_dict.items()
            ):
                bc_list.append((marker_func, val, "Neumann", i))

        # Add Dirichlet BCs
        if self.params.dirichlet_bcs_dict is not None:
            for i, (marker_func, val) in enumerate(
                self.params.dirichlet_bcs_dict.items()
            ):
                bc_list.append((marker_func, val, "Dirichlet", i))

        # Process all geometrically-defined BCs uniformly
        for marker_func, val, bc_type, i in bc_list:
            # Find a unique marker ID
            marker_id = 1
            while marker_id in self.association_table.values():
                marker_id += 1

            # Store in association table
            self.association_table[f"Config {bc_type} BC {i}"] = marker_id

            # Locate boundary facets
            facets = dmesh.locate_entities_boundary(
                self.mesh, self.mesh.topology.dim - 1, marker_func
            )

            # Add to neumann_bcs list directly if Neumann
            if bc_type == "Neumann":
                # Check if val is callable (a function of coordinates)
                if callable(val) and not isinstance(
                    val, (fem.Constant, fem.Expression)
                ):
                    # Convert function to fem.Function
                    neumann_func = fem.Function(self.V_u)
                    neumann_func.interpolate(val)
                    self.neumann_bcs.append((marker_id, neumann_func))
                else:
                    self.neumann_bcs.append((marker_id, val))

            if len(facets) > 0:
                # Update boundaries MeshTags
                values = np.full(len(facets), marker_id, dtype=np.int32)
                if self.boundaries.indices.size == 0:
                    self.boundaries = dmesh.meshtags(
                        self.mesh, self.mesh.topology.dim - 1, facets, values
                    )
                else:
                    all_indices = np.concatenate([self.boundaries.indices, facets])
                    all_values = np.concatenate([self.boundaries.values, values])
                    self.boundaries = dmesh.meshtags(
                        self.mesh, self.mesh.topology.dim - 1, all_indices, all_values
                    )

        self.dirichlet_bcs = self._define_dirichlet_boundaries()

        self.log(f"Association table: {self.association_table}")

        self.boundary_ds = ufl.ds(
            metadata={"quadrature_degree": self.params.q_deg},
            domain=self.mesh,
            subdomain_data=self.boundaries,
        )

        for i, val in self.neumann_bcs:
            forces = []
            # Check if val is a fem.Function (spatially-varying) or has iterable components
            if isinstance(val, fem.Function):
                # For fem.Function, compute integral of each component
                for comp in range(self.mesh.geometry.dim):
                    global_force = self._assemble_scalar_parallel(
                        val[comp] * self.boundary_ds(i)
                    )
                    forces.append(global_force)
            else:
                # For Constant or similar, iterate over components
                for x in val:
                    global_force = self._assemble_scalar_parallel(
                        x * self.boundary_ds(i)
                    )
                    forces.append(global_force)
            self.log(f"Applying Neumann boundary {i}: {forces}")

    def _assemble_scalar_parallel(self, expression):
        local_val = fem.assemble_scalar(fem.form(expression))
        return self.mesh.comm.allreduce(local_val, op=MPI.SUM)

    def _define_dirichlet_boundaries(self, fixed_value=None):
        dirichlet_bcs = []

        # Check if V_u is a subspace (from mixed space) by trying to collapse
        try:
            V_u_collapsed, _ = self.V_u.collapse()
            is_subspace = True
        except RuntimeError:
            is_subspace = False
            V_u_collapsed = None

        # Ensure topology connectivity is created on all ranks
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim - 1, self.mesh.topology.dim
        )

        # Collect all BC marker IDs and values
        bc_list = []

        # Add geometrically-defined BCs (already marked in association_table)
        if self.params.dirichlet_bcs_dict is not None:
            for i, (marker_func, val) in enumerate(
                self.params.dirichlet_bcs_dict.items()
            ):
                marker_id = self.association_table[f"Config Dirichlet BC {i}"]
                bc_list.append((marker_id, val))

        # Add marker ID-defined BCs
        if self.params.dirichlet_bcs is not None:
            bc_list.extend(self.params.dirichlet_bcs)

        # Process all BCs uniformly
        marker_facets_num = {}
        for marker_id, val in bc_list:
            facets = self.boundaries.find(marker_id)
            marker_facets_num[marker_id] = self.mesh.comm.allreduce(
                len(facets), op=MPI.SUM
            )

            # Check if val is a dict (component-wise BC) or Constant (all components)
            if isinstance(val, dict):
                # Component-wise BC: val = {0: value, 1: value, ...}
                for component, component_val in val.items():
                    if is_subspace:
                        V_component, _ = V_u_collapsed.sub(component).collapse()
                    else:
                        V_component, _ = self.V_u.sub(component).collapse()

                    bc_func = fem.Function(V_component)
                    if fixed_value is not None:
                        bc_func.x.array[:] = fixed_value
                    else:
                        # Check if component_val is a Constant or Expression first
                        if hasattr(component_val, "value"):
                            bc_func.x.array[:] = component_val.value
                        elif callable(component_val) and not isinstance(
                            component_val, (fem.Constant, fem.Expression)
                        ):
                            # Function of coordinates
                            bc_func.interpolate(component_val)
                        else:
                            bc_func.x.array[:] = component_val

                    if is_subspace:
                        dofs = fem.locate_dofs_topological(
                            (self.V_u.sub(component), V_component),
                            self.mesh.topology.dim - 1,
                            facets,
                        )
                    else:
                        dofs = fem.locate_dofs_topological(
                            (self.V_u.sub(component), V_component),
                            self.mesh.topology.dim - 1,
                            facets,
                        )

                    if len(facets) > 0:
                        bc = fem.dirichletbc(bc_func, dofs, self.V_u.sub(component))
                        dirichlet_bcs.append(bc)
            else:
                # Full vector BC (existing behavior)
                if is_subspace:
                    # For subspaces, use Function instead of Constant
                    bc_func = fem.Function(V_u_collapsed)
                    if fixed_value is not None:
                        bc_func.x.array[:] = fixed_value
                    else:
                        # Check if val is a Constant or Expression first
                        if hasattr(val, "value"):
                            bc_func.interpolate(
                                lambda x: np.tile(val.value, (x.shape[1], 1)).T
                            )
                        elif callable(val) and not isinstance(
                            val, (fem.Constant, fem.Expression)
                        ):
                            # Function of coordinates
                            bc_func.interpolate(val)
                        else:
                            bc_func.interpolate(
                                lambda x: np.tile(val.value, (x.shape[1], 1)).T
                            )

                    dofs = fem.locate_dofs_topological(
                        (self.V_u, V_u_collapsed), self.mesh.topology.dim - 1, facets
                    )
                    if len(facets) > 0:
                        bc = fem.dirichletbc(bc_func, dofs, self.V_u)
                        dirichlet_bcs.append(bc)
                else:
                    # For regular spaces
                    if fixed_value is not None:
                        bc_val = fem.Constant(
                            self.mesh,
                            ScalarType([fixed_value] * self.mesh.geometry.dim),
                        )
                        dofs = fem.locate_dofs_topological(
                            self.V_u, self.mesh.topology.dim - 1, facets
                        )
                        if len(facets) > 0:
                            bc = fem.dirichletbc(bc_val, dofs, self.V_u)
                            dirichlet_bcs.append(bc)
                    elif callable(val) and not isinstance(
                        val, (fem.Constant, fem.Expression)
                    ):
                        # Function-based BC: create a Function and interpolate
                        bc_func = fem.Function(self.V_u)
                        bc_func.interpolate(val)
                        dofs = fem.locate_dofs_topological(
                            self.V_u, self.mesh.topology.dim - 1, facets
                        )
                        if len(facets) > 0:
                            bc = fem.dirichletbc(bc_func, dofs)
                            dirichlet_bcs.append(bc)
                    else:
                        # Constant value (fem.Constant or similar)
                        bc_val = val
                        dofs = fem.locate_dofs_topological(
                            self.V_u, self.mesh.topology.dim - 1, facets
                        )
                        if len(facets) > 0:
                            bc = fem.dirichletbc(bc_val, dofs, self.V_u)
                            dirichlet_bcs.append(bc)
        self.log(f"Found facets for Dirichlet BCs: {marker_facets_num}")
        return dirichlet_bcs

    def forward(self):
        psi = self._strain_energy_density()

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = ufl.derivative(Pi, self.u, self.u_test)

        # Compute Jacobian of F
        J = ufl.derivative(F, self.u, self.du)

        # Solve nonlinear variational problem
        solve_nonlinear_problem(
            F,
            self.u,
            bcs=self.dirichlet_bcs,
            J=J,
            comm=self.mesh.comm,
            log_file_path=os.path.join(self.path, "log.txt"),
        )

    def _objective_function(self):
        raise NotImplementedError()

    def _set_optimized_variable(self, x: np.ndarray):
        self.e.x.array[:] = x
        self.e.x.scatter_forward()

    def _bounds(self):
        raise NotImplementedError()

    def _initial_values(self):
        raise NotImplementedError()

    def eval(self, x):
        self.log(
            f"Taking input: min={self.mesh.comm.allreduce(np.min(x), op=MPI.MIN)}, max={self.mesh.comm.allreduce(np.max(x), op=MPI.MAX)}"
        )
        self._set_optimized_variable(x)
        # Solve \partial L/\partial \lambda = 0, i.e. the governing equation it self
        self.log("Solving forward problem")
        self.mesh.comm.Barrier()
        try:
            self.forward()
        except:
            self._initialize_fields()
            return np.inf
        loss = self._assemble_scalar_parallel(self._objective_function())
        self.log(f"Loss: {loss}")
        self.mesh.comm.Barrier()
        return loss

    def grad(self, x):
        self.log("Solving gradient")
        self.mesh.comm.Barrier()
        self._set_optimized_variable(x)

        # Check if V_u is a subspace (from mixed space) by trying to collapse
        try:
            _ = self.V_u.collapse()
            state_var = self.w
            state_space = self.V_w
        except (RuntimeError, AttributeError):
            state_var = self.u
            state_space = self.V_u

        # Solve \partial L/\partial u = 0 for \lambda. It's a linear equation.
        j = self._objective_function()
        Pi = self._potential_energy()

        dj_du_ufl = -ufl.derivative(j, state_var, ufl.TestFunction(state_space))

        l_solve = ufl.TrialFunction(state_space)

        # lhs_ufl = ufl.derivative(
        #     ufl.derivative(
        #         ufl.derivative(
        #             Pi, self.u, lagrangian
        #         ),  # Derivative of potential energy w.r.t. u and inner product with lagrangian
        #         self.u,
        #         self.u_test,
        #     ),  # Derivative w.r.t. u
        #     lagrangian,
        #     l_test,  # Tangent matrix * l_test
        # )
        lhs_ufl = ufl.derivative(
            ufl.derivative(Pi, state_var, ufl.TestFunction(state_space)),
            state_var,
            l_solve,
        )
        # No values at DirichletBC boundaries
        bcs = self._define_dirichlet_boundaries(fixed_value=0.0)

        # Solve using LinearProblem
        from dolfinx.fem.petsc import LinearProblem

        problem = LinearProblem(
            lhs_ufl,
            dj_du_ufl,
            bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        lagrangian = problem.solve()

        # Compute \partial L/\partial param
        v_e = ufl.TestFunction(self.V_e)
        dj_dp_form = fem.form(ufl.derivative(j, self.e, v_e))
        dj_dp = fem.assemble_vector(dj_dp_form)
        dj_dp.scatter_reverse(InsertMode.add)

        dfl_dp_form = fem.form(
            ufl.derivative(ufl.derivative(Pi, state_var, lagrangian), self.e, v_e)
        )
        dfl_dp = fem.assemble_vector(dfl_dp_form)
        dfl_dp.scatter_reverse(InsertMode.add)

        grad = dj_dp.array + dfl_dp.array
        self.gradient.x.array[:] = grad
        self.mesh.comm.Barrier()
        return grad.copy()

    def optimize(self):
        x0 = self._initial_values()
        x = x0
        self._set_optimized_variable(x0)

        for i in range(self.params.maxiter):
            print()
            self.mesh.comm.Barrier()
            self.log(f"----- Iteration {i} -----")
            res = self.eval(x)
            self._loss_history.append(res)
            self._stage_loss_history.append(res)
            grad = self.grad(x)
            x -= np.clip(
                self.params.lr * grad,
                a_min=-self.params.clip_gradient,
                a_max=self.params.clip_gradient,
            )
            for j, bound in zip(range(len(x)), self._bounds()):
                if bound[0] is not None:
                    x[j] = max(bound[0], x[j])
                if bound[1] is not None:
                    x[j] = min(bound[1], x[j])
            self.mesh.comm.Barrier()
            self.optimized = True
            terminate = self._optimize_callback()
            if terminate:
                break
        if self.mesh.comm.Get_rank() == 0:
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
        mu = self.params.E / (2 * (1 + self.params.nu))  # Lame second coefficient
        lmbda = (
            self.params.E
            * self.params.nu
            / (1 + self.params.nu)
            / (1 - 2 * self.params.nu)
        )  # Lame first coefficient

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
        return psi

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()
        energy = psi * self.dx - ufl.dot(self.params.B, self.u) * self.dx
        # Neumann boundaries
        for i, val in self.neumann_bcs:
            energy -= ufl.dot(val, self.u) * self.boundary_ds(i)
        return energy

    def _deformation_gradient(self):
        u = self.u
        # Kinematics
        d = len(u)
        I = ufl.Identity(d)  # Identity tensor
        F = I + ufl.grad(u)  # Deformation gradient
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
        I = ufl.Identity(d)  # Identity tensor
        e = 0.5 * (C - I)
        return e

    def _PK2_stress(self, psi=None, e=None, F=None, C=None):
        if F is None:
            F = self._deformation_gradient()
        # Make them ufl.variable to make the strain energy density differentiable w.r.t. F and e
        F = ufl.variable(F)
        if e is None:
            e = self._strain(F=F, C=C)
        e = ufl.variable(e)
        if psi is None:
            psi = self._strain_energy_density(Eps=e)
        s = ufl.diff(psi, e)
        return s

    def _PK1_stress(self, psi=None, F=None):
        if F is None:
            F = self._deformation_gradient()
        F = ufl.variable(F)
        if psi is None:
            psi = self._strain_energy_density(F=F)
        p = ufl.diff(psi, F)
        return p

    def _Cauchy_stress(self, psi=None, F=None):
        if F is None:
            F = self._deformation_gradient()
        F = ufl.variable(F)
        if psi is None:
            psi = self._strain_energy_density(F=F)
        p = ufl.diff(psi, F)
        J = ufl.det(F)
        sigma = J ** (-1) * p * ufl.transpose(F)
        return sigma

    def _get_point_data(self, main_field=None):
        u_field = self.u if main_field is None else main_field
        point_data = {u_field.name: u_field}

        if self.optimized:
            point_data.update({self.e.name: self.e, self.gradient.name: self.gradient})

        return point_data

    def _get_cell_data(self):
        tensor_space = fem.functionspace(
            self.mesh, ("DG", 0, (self.mesh.geometry.dim, self.mesh.geometry.dim))
        )
        e = project(self._strain(), tensor_space, "strain")
        pk2 = project(self._PK2_stress(), tensor_space, "PK2_stress")
        sigma = project(self._Cauchy_stress(), tensor_space, "Cauchy_stress")

        scalar_space = fem.functionspace(self.mesh, ("DG", 0))
        psi = project(
            self._strain_energy_density(), scalar_space, "strain_energy_density"
        )

        # Add MPI rank as subdomain indicator
        rank_field = fem.Function(scalar_space, name="mpi_rank")
        rank_field.x.array[:] = self.mesh.comm.rank

        cell_data = {
            "strain": e,
            "PK2_stress": pk2,
            "Cauchy_stress": sigma,
            "strain_energy_density": psi,
            "mpi_rank": rank_field,
        }

        if self.optimized:
            e_projected = project(self.e, scalar_space, "Optimized_projected")
            grad_projected = project(self.gradient, scalar_space, "Gradient_projected")
            cell_data.update(
                {
                    "Optimized_projected": e_projected,
                    "Gradient_projected": grad_projected,
                }
            )

        return cell_data

    def save(self, main_field=None, time=0):
        point_data = self._get_point_data(main_field)
        cell_data = self._get_cell_data()

        output_name = os.path.join(self.vtu_dir, f"output_{time:06d}")
        vtu_filename = f"{output_name}.vtu"
        write_vtu(self.mesh, vtu_filename, point_data, cell_data, comm=self.mesh.comm)

        if self.mesh.comm.rank == 0:
            if self.mesh.comm.size > 1:
                ref_file = os.path.relpath(f"{output_name}.pvtu", self.path)
            else:
                ref_file = os.path.relpath(vtu_filename, self.path)
            self.timesteps.append((time, ref_file))
        write_pvd(self.pvd_file, self.timesteps, comm=self.mesh.comm)

    def compute_traction_dirichlet(self, type="PK1"):
        space_for_stress = fem.functionspace(
            self.mesh, ("DG", 0, (self.mesh.geometry.dim, self.mesh.geometry.dim))
        )
        if type == "PK1":
            stress = self._PK1_stress()
        elif type == "Cauchy":
            stress = self._Cauchy_stress()
        else:
            raise Exception(
                f"Stress type {type} not supported for computing tractions."
            )
        pk1_stress = project(stress, space_for_stress, "stress")
        N = ufl.FacetNormal(self.mesh)
        e_x = fem.Constant(
            self.mesh, ScalarType([1.0, 0.0] + ([] if self.params.dim == 2 else [0.0]))
        )
        e_y = fem.Constant(
            self.mesh, ScalarType([0.0, 1.0] + ([] if self.params.dim == 2 else [0.0]))
        )
        e_z = (
            None
            if self.params.dim == 2
            else fem.Constant(self.mesh, ScalarType((0.0, 0.0, 1.0)))
        )
        tx = ufl.dot(ufl.dot(pk1_stress, N), e_x)
        ty = ufl.dot(ufl.dot(pk1_stress, N), e_y)
        tz = None if self.params.dim == 2 else ufl.dot(ufl.dot(pk1_stress, N), e_z)
        tractions = {}
        for name, key in self.association_table.items():
            global_tx = self._assemble_scalar_parallel(tx * self.boundary_ds(key))
            global_ty = self._assemble_scalar_parallel(ty * self.boundary_ds(key))

            tractions[name] = [global_tx, global_ty]
            if tz is not None:
                global_tz = self._assemble_scalar_parallel(tz * self.boundary_ds(key))
                tractions[name].append(global_tz)
        return tractions

    def record_traction_dirichlet(self, type="PK1", step=0):
        tractions = self.compute_traction_dirichlet(type=type)
        for name, values in tractions.items():
            for val, dir in zip(
                values, ["x", "y"] + (["z"] if len(values) == 3 else [])
            ):
                self.records.loc[step, f"{name}_{dir}"] = val

    def record(self, step, name, value):
        self.records.loc[step, name] = value

    def save_records(self):
        if self.mesh.comm.rank == 0:
            self.records.to_csv(os.path.join(self.path, "records.csv"))
