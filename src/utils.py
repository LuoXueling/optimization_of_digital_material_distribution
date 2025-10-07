from datetime import datetime
from copy import deepcopy
from mpi4py import MPI
import pyvista as pv
from dolfinx.plot import vtk_mesh
import dolfinx.fem as fem
import ufl


def redirect_stdout_to_log(log_file_path):
    """
    Redirect stdout to both terminal and log file, stripping ANSI color codes from log file.

    Returns:
        tuple: (old_stdout_fd, log_file_info) to be passed to restore_stdout_from_log()
    """
    import os
    import threading
    import re

    old_stdout_fd = os.dup(1)  # Save original stdout fd

    def tee_with_ansi_strip(input_fd, output_fd, log_path):
        # Use bytes pattern for bytes data
        ansi_escape = re.compile(rb"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        petsc_krylov_pattern = re.compile(rb".*PETSc Krylov solver.*")
        try:
            with os.fdopen(input_fd, "rb", buffering=0) as inp, os.fdopen(
                output_fd, "wb", buffering=0
            ) as out, open(log_path, "ab", buffering=0) as log:
                while True:
                    data = inp.read(1024)
                    if not data:
                        break
                    # Filter out lines containing "PETSc Krylov solver"
                    lines = data.split(b"\n")
                    filtered_lines = [
                        line for line in lines if not petsc_krylov_pattern.match(line)
                    ]
                    filtered_data = b"\n".join(filtered_lines)
                    out.write(filtered_data)  # Write to terminal with colors
                    out.flush()
                    # Strip ANSI codes for log file
                    cleaned = ansi_escape.sub(b"", filtered_data)
                    log.write(cleaned)
                    log.flush()
        except Exception as e:
            pass  # Silently ignore errors during cleanup

    # Create pipe for capturing
    r, w = os.pipe()

    # Start thread to tee the output
    tee_thread = threading.Thread(
        target=tee_with_ansi_strip, args=(r, old_stdout_fd, log_file_path), daemon=False
    )
    tee_thread.start()

    # Redirect stdout to write end (after thread starts)
    os.dup2(w, 1)

    return old_stdout_fd, (tee_thread, w, r)


def restore_stdout_from_log(old_stdout_fd, log_file_info):
    """
    Restore stdout to original state after redirect_stdout_to_log().

    Args:
        old_stdout_fd: The original stdout file descriptor
        log_file_info: The tuple returned by redirect_stdout_to_log()
    """
    import os
    import sys

    sys.stdout.flush()
    tee_thread, write_fd, read_fd = log_file_info

    # Restore stdout to original FIRST (before closing pipe)
    os.dup2(old_stdout_fd, 1)

    # Close the write end - this signals EOF when we close it
    os.close(write_fd)

    # Wait for thread to finish reading (it's using old_stdout_fd)
    tee_thread.join(timeout=5.0)

    # Now we can safely close the read end and saved fd
    try:
        os.close(read_fd)
    except OSError:
        pass  # Thread may have already closed it

    try:
        os.close(old_stdout_fd)
    except OSError:
        pass  # May already be closed by thread's fdopen context manager


def project(expression, target_space, name="projected"):
    from dolfinx.fem.petsc import LinearProblem

    v = ufl.TestFunction(target_space)
    u_proj = fem.Function(target_space, name=name)

    # Use reduced quadrature degree to avoid excessive memory in JIT compilation
    # For DG0, quadrature degree 2 is sufficient
    dx_proj = ufl.dx(metadata={"quadrature_degree": 2})
    a = ufl.inner(ufl.TrialFunction(target_space), v) * dx_proj
    L = ufl.inner(expression, v) * dx_proj

    # Use efficient solver settings for projection
    problem = LinearProblem(a, L)
    u_proj = problem.solve()
    u_proj.name = name
    return u_proj


def write_vtu(mesh, filename, point_data=None, cell_data=None, comm=None):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    if comm is None:
        comm = mesh.comm

    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    if point_data:
        for name, func in point_data.items():
            # Check if function needs projection to P1 for visualization
            elem_degree = func.function_space.ufl_element().degree
            bs = func.function_space.dofmap.bs
            num_points = mesh.geometry.x.shape[0]

            # If higher order (elem_degree > 1), project to P1 for visualization
            if elem_degree > 1:
                import dolfinx.fem as fem

                if bs > 1:
                    V_viz = fem.functionspace(mesh, ("Lagrange", 1, (bs,)))
                else:
                    V_viz = fem.functionspace(mesh, ("Lagrange", 1))
                func_viz = project(func, V_viz, name)
                func = func_viz

            # Now write the function data
            if len(func.x.array.shape) > 1 or func.function_space.num_sub_spaces > 0:
                # Vector field - always output 3 components for ParaView compatibility
                import numpy as np

                data = func.x.array.reshape(-1, func.function_space.dofmap.bs)
                if func.function_space.dofmap.bs == 2:
                    # Pad 2D vectors with zeros to make them 3D
                    data_3d = np.zeros((data.shape[0], 3))
                    data_3d[:, :2] = data
                    grid.point_data[name] = data_3d
                else:
                    grid.point_data[name] = data
            else:
                grid.point_data[name] = func.x.array

    if cell_data:
        for name, func in cell_data.items():
            # Get number of locally owned cells
            num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
            array = func.x.array

            # Determine data type based on array size and block size
            bs = func.function_space.dofmap.bs
            array_size = len(array)

            # Calculate how many values per cell
            if array_size >= num_cells_local * bs:
                # Has enough data for owned cells
                values_per_cell = bs
            else:
                values_per_cell = (
                    array_size // num_cells_local if num_cells_local > 0 else 1
                )

            if values_per_cell == 1:
                # Scalar cell data
                grid.cell_data[name] = array[:num_cells_local]
            elif values_per_cell == mesh.geometry.dim**2:
                # Tensor cell data (e.g., strain, stress) - reshape to (N, dim, dim)
                array_local = array[: num_cells_local * values_per_cell]
                grid.cell_data[name] = array_local.reshape(
                    num_cells_local, mesh.geometry.dim, mesh.geometry.dim
                )
            else:
                # Vector or other data
                array_local = array[: num_cells_local * values_per_cell]
                grid.cell_data[name] = array_local.reshape(
                    num_cells_local, values_per_cell
                )

    if comm.size > 1:
        rank_filename = filename.replace(".vtu", f"_rank{comm.rank}.vtu")
        grid.save(rank_filename)

        if comm.rank == 0:
            import os

            base_name = os.path.splitext(os.path.basename(filename))[0]
            dir_name = os.path.dirname(filename)

            root = ET.Element("VTKFile", type="PUnstructuredGrid", version="0.1")
            punstructuredgrid = ET.SubElement(root, "PUnstructuredGrid", GhostLevel="0")

            # PPoints element is required (pyvista uses Float64 for points)
            ppoints = ET.SubElement(punstructuredgrid, "PPoints")
            ET.SubElement(
                ppoints,
                "PDataArray",
                type="Float64",
                Name="Points",
                NumberOfComponents="3",
            )

            if point_data:
                ppointdata = ET.SubElement(punstructuredgrid, "PPointData")
                for name, func in point_data.items():
                    if (
                        func.function_space.num_sub_spaces > 0
                        or len(func.x.array.shape) > 1
                    ):
                        # Always use 3 components for vector fields (ParaView compatibility)
                        num_comp = 3
                    else:
                        num_comp = 1
                    ET.SubElement(
                        ppointdata,
                        "PDataArray",
                        type="Float64",
                        Name=name,
                        NumberOfComponents=str(num_comp),
                    )

            if cell_data:
                pcelldata = ET.SubElement(punstructuredgrid, "PCellData")
                for name, func in cell_data.items():
                    # Determine number of components based on block size
                    bs = func.function_space.dofmap.bs

                    if bs == 1:
                        # Scalar - no NumberOfComponents needed
                        ET.SubElement(
                            pcelldata, "PDataArray", type="Float64", Name=name
                        )
                    elif bs == mesh.geometry.dim**2:
                        # Tensor - use 9 components for 3D, 4 for 2D
                        ET.SubElement(
                            pcelldata,
                            "PDataArray",
                            type="Float64",
                            Name=name,
                            NumberOfComponents=str(bs),
                        )
                    else:
                        # Vector or other
                        ET.SubElement(
                            pcelldata,
                            "PDataArray",
                            type="Float64",
                            Name=name,
                            NumberOfComponents=str(bs),
                        )

            for rank in range(comm.size):
                piece_filename = f"{base_name}_rank{rank}.vtu"
                ET.SubElement(punstructuredgrid, "Piece", Source=piece_filename)

            xml_str = ET.tostring(root, encoding="unicode")
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent="  ")

            pvtu_filename = filename.replace(".vtu", ".pvtu")
            with open(pvtu_filename, "w") as f:
                f.write(pretty_xml)
    else:
        grid.save(filename)


def write_pvd(pvd_filename, timesteps, comm=None):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    if comm is None:
        comm = MPI.COMM_WORLD

    if comm.rank == 0:
        root = ET.Element("VTKFile", type="Collection", version="0.1")
        collection = ET.SubElement(root, "Collection")

        for time, vtu_path in timesteps:
            ET.SubElement(collection, "DataSet", timestep=str(time), file=vtu_path)

        xml_str = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        with open(pvd_filename, "w") as f:
            f.write(pretty_xml)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        return dotdict(super().copy())


def log(*args, **kwargs):
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()
    if rank != 0:
        return None
    l = f"[{datetime.now().strftime('%m-%d-%Y,%H:%M:%S')}]"
    output_file = kwargs.get("output_file", None)
    _kwargs = kwargs.copy()
    _kwargs.update(dict(flush=True))
    if "comm" in _kwargs:
        del _kwargs["comm"]
    if output_file is not None:
        del _kwargs["output_file"]
    if len(args) == 0:
        print(*args, **_kwargs)
    else:
        print(l, *args, **_kwargs)
    if output_file is not None:
        with open(output_file, "a") as file:
            _kwargs.update(dict(file=file))
            if len(args) == 0:
                print(*args, **_kwargs)
            else:
                print(l, *args, **_kwargs)
