"""
Mesh import utilities for FEniCSx/DOLFINx
"""

#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI

try:
    from dolfinx.io import gmshio
    import dolfinx.fem as fem
    import ufl
except ImportError:
    print("Could not import dolfinx. Continuing without DOLFINx support.")


def import_mesh(mesh_name, dim=2, directory=".", comm=None):
    """
    Import a GMSH .msh file using dolfinx.io.gmshio.

    This function replaces the legacy msh2xdmf workflow by directly reading
    .msh files with gmshio, which is the recommended approach in FEniCSx.

    Arguments:
        mesh_name (str): Name of the .msh file (e.g., "square.msh" or just "square")
        dim (int): Dimension of the domain (2 or 3). Default: 2
        directory (str): Directory containing the mesh file. Default: "."
        comm: MPI communicator. Default: MPI.COMM_WORLD

    Returns:
        tuple: (mesh, cell_tags, facet_tags, association_table)
            - mesh: dolfinx.mesh.Mesh object
            - cell_tags: dolfinx.mesh.MeshTags for cell/volume regions
            - facet_tags: dolfinx.mesh.MeshTags for boundary facets
            - association_table: dict mapping physical group names to marker values

    Example:
        >>> mesh, cell_tags, facet_tags, assoc_table = import_mesh("square", directory="mesh")
        >>> # Use boundary markers
        >>> upper_id = assoc_table["upper"]
        >>> ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        >>> # Use domain markers
        >>> main_id = assoc_table["main"]
        >>> dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    # Ensure .msh extension
    if not mesh_name.endswith(".msh"):
        mesh_name = mesh_name + ".msh"

    # Full path to mesh file
    mesh_path = os.path.join(directory, mesh_name)

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Read mesh using gmshio
    # Returns: (mesh, cell_tags, facet_tags)
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        mesh_path, comm, rank=0, gdim=dim
    )

    # Extract association table from gmsh model
    # We need to read the msh file again to get physical group names
    association_table = {}

    if comm.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress gmsh output
        gmsh.open(mesh_path)

        # Get physical groups
        physical_groups = gmsh.model.getPhysicalGroups()
        for dim_phys, tag_phys in physical_groups:
            name = gmsh.model.getPhysicalName(dim_phys, tag_phys)
            association_table[name] = tag_phys

        gmsh.finalize()

        # Print association table
        print("\n" + "=" * 50, flush=True)
        print(f"{'Physical Group':<25} {'Marker ID':>20}", flush=True)
        print("=" * 50, flush=True)
        for name, marker_id in association_table.items():
            print(f"{name:<25} {marker_id:>20}", flush=True)
        print("=" * 50 + "\n", flush=True)

    # Broadcast association table to all ranks
    association_table = comm.bcast(association_table, root=0)

    return mesh, cell_tags, facet_tags, association_table
