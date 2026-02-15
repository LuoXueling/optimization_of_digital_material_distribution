from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import dolfinx.io as io
from dolfinx.la import InsertMode
import ufl
from src.default_config import params
from src.solvers import solve_nonlinear_problem_with_retry
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
    def __init__(self, config_path=None, override_params=None):
        # Load parameters
        params_updated = importlib.import_module(
            config_path.replace("./", "").removesuffix(".py").replace("/", ".")
        ).params
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
        self.record_file = os.path.join(self.path, "records.csv")
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

        self.B = fem.Constant(
            self.mesh,
            ScalarType(
                self.params.B if self.params.B is not None else (0.0,) * self.params.dim
            ),
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
        # self.residual = fem.Function(self.V_u, name="residual")

        # Defined optimized variable (allow multi-component via n_component)
        if self.n_component == 1:
            self.V_e = fem.functionspace(
                self.mesh, ("Lagrange", 1)
            )  # scalar design variable
        else:
            self.V_e = fem.functionspace(
                self.mesh, ("Lagrange", 1, (self.n_component,))
            )  # vector-valued design variable
        self.e = fem.Function(self.V_e, name="Optimized")
        self.e.x.array[:] = 1.0
        self.gradient = fem.Function(self.V_e, name="Gradient")
        # self.e = E
        # Or set values by the following methods
        # e.vector().set_local(my_values)
        # e.interpolate(fe.Expression("(x[0]-width/2)*0.2+10", width=width, degree=1), M)

        self.load_boundary_conditions()

        self.log(f"Association table: {self.association_table}")

        # Initialize optimization related variables
        self._optimize_step = 0
        self._loss_history = []
        self._stage_loss_history = []

        self.optimized = False

        self.records = pd.DataFrame()
        self.records.index.name = "Step"

    def save_checkpoint(self, filename: str | None = None):
        try:
            import adios4dolfinx as a4d
        except Exception as e:
            raise RuntimeError("adios4dolfinx is required to save checkpoint") from e

        filename = os.path.join(self.path, "state.bp" if filename is None else filename)

        for name, val in sorted(self.__dict__.items()):
            if isinstance(val, fem.Function):
                a4d.write_function_on_input_mesh(filename, val, engine="BP4", name=name)

        self.log(f"Checkpoint saved at {filename}")

    def load_functions_from_checkpoint(self, checkpoint_dir):
        import adios4dolfinx as a4d

        for name, val in sorted(self.__dict__.items()):
            if isinstance(val, fem.Function):
                try:
                    a4d.read_function(checkpoint_dir, val, engine="BP4", name=name)
                    self.log(f"Loaded function {name} from checkpoint.")
                except:
                    self.log(f"Failed to load function {name} from checkpoint.")

        self.log(f"Loaded checkpoint from {checkpoint_dir}")

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
            if f"Config {bc_type} BC {i}" in self.association_table.keys():
                marker_id = self.association_table[f"Config {bc_type} BC {i}"]
            else:
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
                    try:
                        neumann_func = fem.Function(self.V_u)
                        neumann_func.interpolate(val)
                    except:
                        # If it doesn't work, it's a scalar and represents pressure
                        neumann_func = fem.Function(
                            fem.functionspace(
                                self.mesh, ("Lagrange", self.params.elem_order)
                            )
                        )
                        neumann_func.interpolate(val)
                    self.neumann_bcs.append((marker_id, neumann_func))
                else:
                    if isinstance(val, (tuple, int, float)):
                        val = fem.Constant(self.mesh, ScalarType(val))
                    self.neumann_bcs.append((marker_id, val))

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
        self.adjoint_bcs = self._define_dirichlet_boundaries(fixed_value=0.0)

        self.boundary_ds = ufl.ds(
            metadata={"quadrature_degree": self.params.q_deg},
            domain=self.mesh,
            subdomain_data=self.boundaries,
        )

        for i, val in self.neumann_bcs:
            forces = []
            N = ufl.FacetNormal(self.mesh)
            # Check if val is a fem.Function (spatially-varying) or has iterable components
            if isinstance(val, fem.Function):
                # For fem.Function, compute integral of each component
                if val.function_space.value_size == 1:
                    # For scalars, it's pressure load, and we are using reference configuration just for this functionality
                    val = val * N
                for comp in range(self.mesh.geometry.dim):
                    global_force = self._assemble_scalar_parallel(
                        val[comp] * self.boundary_ds(i)
                    )
                    forces.append(global_force)
            else:
                if val.value.size == 1:
                    # For scalars, it's pressure load, and we are using reference configuration just for this functionality
                    val = val * N
                # For Constant or similar, iterate over components
                for x in val:
                    global_force = self._assemble_scalar_parallel(
                        x * self.boundary_ds(i)
                    )
                    forces.append(global_force)
            self.log(f"Applying Neumann boundary {i}: {forces}")

        # Create domain meshtags for subdomain integration
        self.domains = self._create_domain_meshtags()
        self.domain_dx = ufl.dx(
            metadata={"quadrature_degree": self.params.q_deg},
            domain=self.mesh,
            subdomain_data=self.domains,
        )

        self.e_dirichlet_bcs, self.gradient_dirichlet_bcs = (
            self._define_design_dirichlet_boundaries()
        )
        self._apply_design_bcs()

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
                            if isinstance(val, tuple):
                                val = fem.Constant(self.mesh, ScalarType(val))
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
                        if isinstance(val, tuple):
                            val = fem.Constant(self.mesh, ScalarType(val))
                        bc_val = val
                        dofs = fem.locate_dofs_topological(
                            self.V_u, self.mesh.topology.dim - 1, facets
                        )
                        if len(facets) > 0:
                            bc = fem.dirichletbc(bc_val, dofs, self.V_u)
                            dirichlet_bcs.append(bc)
        self.log(f"Found facets for Dirichlet BCs: {marker_facets_num}")
        return dirichlet_bcs

    def _define_domain_dirichlet_constraints(
        self, function_space, domain_marker, fixed_value=0.0
    ):
        """
        Define Dirichlet constraints for a field within a specified domain (volumetric region).

        This method is analogous to _define_dirichlet_boundaries() but works on cells (domains)
        instead of facets (boundaries). It can handle both regular function spaces and subspaces
        from mixed formulations.

        Args:
            function_space: The function space to constrain (can be a subspace from mixed element)
            domain_marker: Integer marker ID identifying the domain cells
            fixed_value: The value to constrain the field to (scalar or appropriate for the space)

        Returns:
            List of fem.dirichletbc objects for the specified field
        """
        domain_dirichlet_bcs = []

        # Check if function_space is a subspace (from mixed space) by trying to collapse
        try:
            V_collapsed, _ = function_space.collapse()
            is_subspace = True
        except RuntimeError:
            is_subspace = False
            V_collapsed = function_space

        # Ensure topology connectivity for cells
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim, self.mesh.topology.dim
        )

        # Find cells with the specified domain marker
        cells = self.domains.find(domain_marker)

        num_cells = self.mesh.comm.allreduce(len(cells), op=MPI.SUM)
        self.log(
            f"Constraining field in {num_cells} cells with domain marker {domain_marker}"
        )

        # Create a function with the fixed value (always available)
        bc_func = fem.Function(V_collapsed)
        bc_func.x.array[:] = fixed_value

        # Locate DOFs collectively on all ranks, even if cells is empty locally
        if is_subspace:
            dofs = fem.locate_dofs_topological(
                (function_space, V_collapsed), self.mesh.topology.dim, cells
            )
            if len(cells) > 0:
                try:
                    bc = fem.dirichletbc(bc_func, dofs, function_space)
                except:
                    bc = fem.dirichletbc(bc_func, dofs)
                domain_dirichlet_bcs.append(bc)
        else:
            dofs = fem.locate_dofs_topological(
                function_space, self.mesh.topology.dim, cells
            )
            if len(cells) > 0:
                bc = fem.dirichletbc(bc_func, dofs)
                domain_dirichlet_bcs.append(bc)

        return domain_dirichlet_bcs

    def _define_design_dirichlet_boundaries(
        self, function_space=None, fixed_value=None
    ):
        """
        Define dirichlet BCs for self.e and gradient of self.e
        """
        function_space = self.V_e if function_space is None else function_space

        e_bcs = []
        gradient_bcs = []
        fix_design_domain = self.params.fix_design_domain
        if fix_design_domain is not None:
            # Probably we want the fixed domain unaltered instead of a 1.0
            if fixed_value is not None:
                e_bcs.extend(
                    self._define_domain_dirichlet_constraints(
                        function_space=function_space,
                        domain_marker=self.association_table[fix_design_domain],
                        fixed_value=fixed_value,
                    )
                )
            gradient_bcs.extend(
                self._define_domain_dirichlet_constraints(
                    function_space=function_space,
                    domain_marker=self.association_table[fix_design_domain],
                    fixed_value=0.0,
                )
            )
        return e_bcs, gradient_bcs

    def _apply_design_bcs(self):
        fem.set_bc(self.e.x.array, self.e_dirichlet_bcs)
        self.e.x.scatter_forward()

    def _create_domain_meshtags(self):
        """Create cell-based MeshTags for domain integration.

        Processes domain_dict to mark cells based on lambda functions and merges
        with GMSH-imported cell tags if available.

        Returns:
            dolfinx.mesh.MeshTags: Cell-based meshtags for domains
        """
        # Ensure topology connectivity is created
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim, self.mesh.topology.dim
        )

        # Start with GMSH-imported cell tags if available
        if self.params.cell_tags is not None:
            # Extract existing cell tags from GMSH
            existing_indices = self.params.cell_tags.indices
            existing_values = self.params.cell_tags.values
        else:
            # Create empty arrays
            existing_indices = np.array([], dtype=np.int32)
            existing_values = np.array([], dtype=np.int32)

        # Process domain_dict for geometrically-defined domains
        if self.params.domain_dict is not None:
            for i, (marker_func, domain_name) in enumerate(
                self.params.domain_dict.items()
            ):
                # Determine marker ID
                marker_key = (
                    f"Config Domain {i}" if domain_name is None else domain_name
                )

                if marker_key in self.association_table.keys():
                    marker_id = self.association_table[marker_key]
                else:
                    # Find a unique marker ID
                    marker_id = 1
                    while marker_id in self.association_table.values():
                        marker_id += 1

                    # Store in association table
                    self.association_table[marker_key] = marker_id

                # Locate cells matching the marker function
                cells = dmesh.locate_entities(
                    self.mesh, self.mesh.topology.dim, marker_func
                )

                # Add to existing arrays
                values = np.full(len(cells), marker_id, dtype=np.int32)
                existing_indices = np.concatenate([existing_indices, cells])
                existing_values = np.concatenate([existing_values, values])

                # Log the number of cells marked
                num_cells_global = self.mesh.comm.allreduce(len(cells), op=MPI.SUM)
                if num_cells_global > 0:
                    self.log(
                        f"Marked {num_cells_global} cells for domain '{marker_key}' with ID {marker_id}"
                    )

        # Find all cells that are not marked
        num_cells_local = self.mesh.topology.index_map(
            self.mesh.topology.dim
        ).size_local
        all_cells = np.arange(num_cells_local, dtype=np.int32)

        # Get set of marked cells
        marked_cells = set(existing_indices) if len(existing_indices) > 0 else set()

        # Find unmarked cells
        unmarked_cells = np.array(
            [cell for cell in all_cells if cell not in marked_cells], dtype=np.int32
        )

        # Assign a marker ID for unmarked cells
        unmarked_marker_key = "unmarked_cells"

        if unmarked_marker_key in self.association_table.keys():
            unmarked_marker_id = self.association_table[unmarked_marker_key]
        else:
            # Find a unique marker ID
            unmarked_marker_id = 1  # Default to 0 for unmarked cells
            while unmarked_marker_id in self.association_table.values():
                unmarked_marker_id += 1

            # Store in association table
            self.association_table[unmarked_marker_key] = unmarked_marker_id

        # Add unmarked cells to the arrays
        unmarked_values = np.full(
            len(unmarked_cells), unmarked_marker_id, dtype=np.int32
        )
        existing_indices = np.concatenate([existing_indices, unmarked_cells])
        existing_values = np.concatenate([existing_values, unmarked_values])

        # Log the number of unmarked cells
        num_unmarked_global = self.mesh.comm.allreduce(len(unmarked_cells), op=MPI.SUM)
        if num_unmarked_global > 0:
            self.log(
                f"Marked {num_unmarked_global} unmarked cells with ID {unmarked_marker_id}"
            )

        # Create MeshTags with all collected cells
        if len(existing_indices) > 0:
            domains = dmesh.meshtags(
                self.mesh, self.mesh.topology.dim, existing_indices, existing_values
            )
        else:
            # Create empty MeshTags
            domains = dmesh.meshtags(
                self.mesh,
                self.mesh.topology.dim,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        self.log(f"Created domain MeshTags with {len(existing_indices)} cells")
        return domains

    def forward(self):
        psi = self._strain_energy_density()

        # Total potential energy
        Pi = self._potential_energy(psi=psi)

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = ufl.derivative(Pi, self.u, self.u_test)

        # Compute Jacobian of F
        J = ufl.derivative(F, self.u, self.du)

        # Solve nonlinear variational problem
        try:
            solve_nonlinear_problem_with_retry(
                F,
                self.u,
                bcs=self.dirichlet_bcs,
                J=J,
                comm=self.mesh.comm,
                logger=self.log,
                snes_options=self.params.snes_options,
                num_retries=self.params.num_retries,
            )
            # self.residual.x.array[:] = fem.assemble_vector(fem.form(F)).array
        except Exception as e:
            # self.residual.x.array[:] = fem.assemble_vector(fem.form(F)).array
            self.save(time=int(1e6))
            raise e

    def _objective_function(self):
        raise NotImplementedError()

    def _objective_function_value(self):
        return self._assemble_scalar_parallel(self._objective_function())

    def _set_optimized_variable(self, x: np.ndarray):
        self.e.x.array[:] = x
        self.e.x.scatter_forward()
        self._apply_design_bcs()

    def _bounds(self):
        raise NotImplementedError()

    def _initial_values(self):
        raise NotImplementedError()

    def eval(self, x):
        self._set_optimized_variable(x)
        self.log(
            f"Taking input: min={self.mesh.comm.allreduce(np.min(self.e.x.array[:]), op=MPI.MIN)}, max={self.mesh.comm.allreduce(np.max(self.e.x.array[:]), op=MPI.MAX)}"
        )
        # Solve \partial L/\partial \lambda = 0, i.e. the governing equation it self
        self.log("Solving forward problem")
        self.mesh.comm.Barrier()
        try:
            self.forward()
        except Exception as e:
            self.log(f"Exception occurred: {e}")
            self.log("Return inf as loss value")
            return np.inf
        loss = self._objective_function_value()
        self.log(f"Loss: {loss}")
        self.mesh.comm.Barrier()
        return loss

    def _grad_j_state(self, j, state, state_test):
        return ufl.derivative(j, state, state_test)

    def _grad_j_opt(self, j, opt, opt_test):
        return ufl.derivative(j, opt, opt_test)

    @property
    def gradient_accumulation_steps(self):
        return 1

    def grad(self, x=None):
        self.log("Solving gradient")
        self.mesh.comm.Barrier()
        if x is not None:
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

        dj_du_ufl = -self._grad_j_state(j, state_var, ufl.TestFunction(state_space))

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

        # Solve using SNES-based linear solver (drop-in for LinearProblem)
        from src.solvers import solve_linear_problem

        # solution function for adjoint
        lagrangian = fem.Function(state_space, name="adjoint")

        solve_linear_problem(
            lhs_ufl,
            dj_du_ufl,
            lagrangian,
            bcs=self.adjoint_bcs,
            linear_solver="mumps",
            comm=self.mesh.comm,
            logger=self.log,
        )

        # Compute \partial L/\partial param
        v_e = ufl.TestFunction(self.V_e)
        dj_dp_form = fem.form(self._grad_j_opt(j, self.e, v_e))
        dj_dp = fem.assemble_vector(dj_dp_form)
        dj_dp.scatter_reverse(InsertMode.add)

        dfl_dp_form = fem.form(
            ufl.derivative(ufl.derivative(Pi, state_var, lagrangian), self.e, v_e)
        )
        dfl_dp = fem.assemble_vector(dfl_dp_form)
        dfl_dp.scatter_reverse(InsertMode.add)

        grad = dj_dp.array + dfl_dp.array
        self.mesh.comm.Barrier()

        self.gradient.x.array[:] = grad
        self.gradient.x.scatter_forward()
        return grad.copy()

    def optimize(self):
        from src.solvers import BarzilaiBorwein

        x = self._initial_values()
        self._set_optimized_variable(x)
        if self.n_component == 1:
            optimizer = BarzilaiBorwein(
                variant="short", fallback_lr=self.params.lr, smooth_beta=0.99
            )
            collapse_map = None
        else:
            optimizer = [
                BarzilaiBorwein(
                    variant="short", fallback_lr=self.params.lr, smooth_beta=0.99
                )
                for i in range(self.n_component)
            ]
            collapse_map = []
            for i in range(self.n_component):
                _, m = self.V_e.sub(i).collapse()
                collapse_map.append(m)

        grad = None

        for i in range(self.params.maxiter):
            print()
            self.mesh.comm.Barrier()
            self.log(f"----- Iteration {i} -----")
            res = self.eval(x)
            _grad = self.grad()
            fem.set_bc(_grad, self.gradient_dirichlet_bcs)
            grad = _grad if grad is None else grad + _grad
            if (self._optimize_step + 1) % self.gradient_accumulation_steps == 0:
                self._loss_history.append(res)
                self._stage_loss_history.append(res)
                if self.n_component == 1:
                    lr = optimizer.step(x, grad)
                    grad = lr * grad
                else:
                    lr = []
                    for opt, m in zip(optimizer, collapse_map):
                        l = opt.step(x[m], grad[m])
                        lr.append(l)
                        grad[m] = l * grad[m]
                self.log(f"Learning rate from Barzilai-Borwein: {lr}")
                grad = np.clip(
                    grad,
                    a_min=-self.params.clip_gradient,
                    a_max=self.params.clip_gradient,
                )
                x = x - grad
                grad = None
                for j, bound in zip(range(len(x)), self._bounds()):
                    if bound[0] is not None:
                        x[j] = max(bound[0], x[j])
                    if bound[1] is not None:
                        x[j] = min(bound[1], x[j])
                self.mesh.comm.Barrier()
                self.optimized = True
            else:
                self.log(
                    f"Accumulated gradient {(self._optimize_step + 1) % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}"
                )
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
                len(self._stage_loss_history) > 100
                and self._stage_loss_history[-100]
                == min(self._stage_loss_history[-100:])
            ):
                return True
        return False

    def to_3d(self, x):
        return (
            ufl.as_tensor([[x[0, 0], x[0, 1], 0], [x[1, 0], x[1, 1], 0], [0, 0, 1]])
            if self.params.dim == 2
            else x
        )

    @property
    def n_component(self) -> int:
        return 1

    @property
    def mu(self):
        """
        Lame second constant / shear modulus
        """
        return self.params.E / (2 * (1 + self.params.nu))

    @property
    def lmbda(self):
        """
        Lame first constant
        """
        return (
            self.params.E
            * self.params.nu
            / (1 + self.params.nu)
            / (1 - 2 * self.params.nu)
        )  # Lame first coefficient

    @property
    def kappa(self):
        """
        Bulk modulus
        """
        return self.params.E / (3 * (1 - 2 * self.params.nu))

    def strain_invariant(self, F=None, C=None, Eps=None):
        """
        First invariant, sqrt of third invariant of right Cauchy Green tensor,
        deformation gradient, and right Cauchy Green tensor
        """
        if Eps is None:
            if F is None:
                F = self._deformation_gradient()
            if C is None:
                C = self._right_cauchy_green(F=F)
            F = self.to_3d(F)
            C = self.to_3d(C)
            J = ufl.det(F)
        else:
            C = 2.0 * Eps + ufl.Identity(len(self.u))
            C = self.to_3d(C)
            J = ufl.sqrt(ufl.det(C))
        Ic = ufl.tr(C)
        return Ic, J, F, C

    def _strain_energy_density(self, F=None, C=None, Eps=None):
        Ic, J, _, _ = self.strain_invariant(F=F, C=C, Eps=Eps)

        # Elasticity parameters
        mu = self.mu  # Lame second coefficient
        kappa = self.kappa

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu / 2) * (J ** (-2 / 3) * Ic - 3) + (kappa / 2) * (ufl.ln(J)) ** 2
        return psi

    def _body_force_energy(self):
        if self.params.body_force_applied_on == "deformed":
            scalar_space = fem.functionspace(self.mesh, ("CG", 1))
            _, J, _, _ = self.strain_invariant()
            # J has to be isolated from symbolic differentiation,
            # otherwise the minimization manipulates surface area to minimize external work
            J_projected = project(J, scalar_space, "isolated_J")
            return ufl.dot(self.B, self.u) * J_projected * self.dx
        elif self.params.body_force_applied_on == "reference":
            return ufl.dot(self.B, self.u) * self.dx
        else:
            raise Exception(
                f"Forces are applied on either deformed or reference configuration."
            )

    def _apply_neumann_bcs(self, energy):
        N = ufl.FacetNormal(self.mesh)
        if self.params.tractions_applied_on == "deformed":
            F = self._deformation_gradient()
            _, J, _, _ = self.strain_invariant()
            tensor_space = fem.functionspace(
                self.mesh,
                ("CG", 1, (self.mesh.geometry.dim, self.mesh.geometry.dim)),
            )
            scalar_space = fem.functionspace(self.mesh, ("CG", 1))
            # J and F have to be isolated from symbolic differentiation,
            # otherwise the minimization manipulates surface area to minimize external work
            F = project(F, tensor_space, "isolated_F")
            J = project(J, scalar_space, "isolated_J")
            for i, val in self.neumann_bcs:
                if (
                    isinstance(val, fem.Constant)
                    and val.value.size == 1
                    or isinstance(val, fem.Function)
                    and val.function_space.value_size == 1
                ):
                    energy -= ufl.dot(
                        val * J * ufl.transpose(ufl.inv(F)) * N, self.u
                    ) * self.boundary_ds(i)
                else:
                    norm_cof_F = J * ufl.transpose(ufl.inv(F)) * N
                    norm_cof_F = ufl.sqrt(ufl.dot(norm_cof_F, norm_cof_F))
                    energy -= ufl.dot(val, self.u) * norm_cof_F * self.boundary_ds(i)
            return energy
        elif self.params.tractions_applied_on == "reference":
            for i, val in self.neumann_bcs:
                if (
                    isinstance(val, fem.Constant)
                    and val.value.size == 1
                    or isinstance(val, fem.Function)
                    and val.function_space.value_size == 1
                ):
                    val = val * N
                energy -= ufl.dot(val, self.u) * self.boundary_ds(i)
            return energy
        else:
            raise Exception(
                f"Forces are applied on either deformed or reference configuration."
            )

    def _potential_energy(self, psi=None):
        if psi is None:
            psi = self._strain_energy_density()
        energy = psi * self.dx - self._body_force_energy()
        # Neumann boundaries
        energy = self._apply_neumann_bcs(energy)
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
        point_data = {
            u_field.name: u_field,
            self.e.name: self.e,
            self.gradient.name: self.gradient,
            # self.residual.name: self.residual,
        }

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

        # Add domain markers for visualization
        domain_markers = fem.Function(scalar_space, name="domain_markers")
        num_cells_local = self.mesh.topology.index_map(
            self.mesh.topology.dim
        ).size_local
        domain_markers.x.array[:num_cells_local] = 0  # Initialize to 0

        # Map domain tags to cell values
        if len(self.domains.indices) > 0:
            # Create a mapping from cell index to marker value
            for cell_idx, marker_val in zip(self.domains.indices, self.domains.values):
                if cell_idx < num_cells_local:  # Only set locally owned cells
                    domain_markers.x.array[cell_idx] = marker_val

        cell_data = {
            "strain": e,
            "PK2_stress": pk2,
            "Cauchy_stress": sigma,
            "strain_energy_density": psi,
            "mpi_rank": rank_field,
            "domain_markers": domain_markers,
        }

        if self.optimized:
            bs = self.V_e.dofmap.bs
            space = (
                fem.functionspace(self.mesh, ("DG", 0, (bs,)))
                if bs > 1
                else scalar_space
            )
            e_projected = project(self.e, space, "Optimized_projected")
            grad_projected = project(self.gradient, space, "Gradient_projected")
            cell_data.update(
                {
                    "Optimized_projected": e_projected,
                    "Gradient_projected": grad_projected,
                }
            )

        return cell_data

    def save(self, main_field=None, time=0):
        self.log(f"Start saving time {time}")
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
        self.log(f"Finish saving")

    def compute_traction_dirichlet(self, type="PK1"):
        if type == "PK1":
            stress = self._PK1_stress()
        elif type == "Cauchy":
            stress = self._Cauchy_stress()
        else:
            raise Exception(
                f"Stress type {type} not supported for computing tractions."
            )
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
        tx = ufl.dot(ufl.dot(stress, N), e_x)
        ty = ufl.dot(ufl.dot(stress, N), e_y)
        tz = None if self.params.dim == 2 else ufl.dot(ufl.dot(stress, N), e_z)
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
            self.records.to_csv(self.record_file)
