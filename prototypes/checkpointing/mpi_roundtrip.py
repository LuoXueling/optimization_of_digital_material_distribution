import argparse
import json
import os
import sys
from pathlib import Path

from mpi4py import MPI
from src.neoHookean import NeoHookean
from src.neoHookean_incompressible import NeoHookeanIncompressible


def build_problem(mesh_divisions=(40, 10), output_root=None):
    from petsc4py.PETSc import ScalarType
    import dolfinx.mesh as dmesh
    import dolfinx.fem as fem
    import numpy as np

    # Build a small rectangular mesh for speed
    width, height = 4.0, 1.0
    nx, ny = mesh_divisions
    mesh = dmesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (width, height)),
        n=(nx, ny),
        cell_type=dmesh.CellType.quadrilateral,
    )

    override_params = dict(
        mesh=mesh,
        output_root=output_root or str(Path.cwd() / "outputs_test"),
        project_name="adios_roundtrip",
        snes_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-9,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
        },
        # Keep BCs similar to default_config but adapted to our local mesh
        dirichlet_bcs_dict={
            (lambda x: np.isclose(x[0], 0.0)): (0.0, 0.0),
            (lambda x: np.isclose(x[0], 4.0)): (0.0, -0.001),
        },
        neumann_bcs_dict=None,
    )

    # Use default config for other parameters
    problem = NeoHookeanIncompressible(
        config_path="src/default_config.py",
        override_params=override_params,
    )
    return problem


def save_mode(pointer_path: Path, subfolder: str = "chkpt"):
    problem = build_problem()

    problem.forward()
    tr = problem.compute_traction_dirichlet(type="PK1")
    problem.save()
    # Persist checkpoint and reference data
    problem.save_checkpoint()
    comm = problem.mesh.comm
    if comm.rank == 0:
        with open(os.path.join(problem.path, "tractions.json"), "w") as f:
            json.dump(tr, f)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pointer_path, "w") as fp:
            json.dump(
                {"checkpoint_dir": str(os.path.join(problem.path, "state.bp"))}, fp
            )
    comm.Barrier()


def load_compare_mode(pointer_path: Path):
    # Read pointer on all ranks
    if MPI.COMM_WORLD.rank == 0:
        with open(pointer_path, "r") as fp:
            data = json.load(fp)
        chkpt_dir = data["checkpoint_dir"]
    else:
        chkpt_dir = None
    chkpt_dir = MPI.COMM_WORLD.bcast(chkpt_dir, root=0)

    # Initialize directly from checkpoint
    problem = build_problem()
    problem.load_functions_from_checkpoint(checkpoint_dir=chkpt_dir)

    problem.forward()
    tr2 = problem.compute_traction_dirichlet(type="PK1")
    problem.save()

    # Compare against saved reference
    if problem.mesh.comm.rank == 0:
        with open(Path(chkpt_dir).parent / "tractions.json", "r") as f:
            tr_ref = json.load(f)

        # Compare per boundary (keys)
        mismatches = []
        for key, vals in tr_ref.items():
            v1 = vals
            v2 = tr2.get(key)
            if v2 is None:
                mismatches.append((key, "missing", v1, None))
                continue
            if len(v1) != len(v2):
                mismatches.append((key, "length", v1, v2))
                continue
            for i, (a, b) in enumerate(zip(v1, v2)):
                a = float(a)
                b = float(b)
                diff = abs(a - b)
                tol = 1e-8 + 1e-6 * max(1.0, abs(a))
                if diff > tol:
                    mismatches.append((f"{key}[{i}]", "value", a, b))

        ok = 1 if len(mismatches) == 0 else 0
    else:
        ok = None
        mismatches = None

    ok = MPI.COMM_WORLD.bcast(ok, root=0)
    if ok == 1:
        return
    else:
        if MPI.COMM_WORLD.rank == 0:
            for k, t, a, b in mismatches:
                print(f"Mismatch {k} ({t}): {a} vs {b}")
        # Exit with non-zero code so pytest captures failure
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["save", "load-compare"], type=str, default="save"
    )
    parser.add_argument(
        "--pointer", type=str, required=False, default="tmp/adios_ptr.json"
    )
    args = parser.parse_args()
    pointer_path = Path(args.pointer)

    if args.mode == "save":
        save_mode(pointer_path)
    else:
        load_compare_mode(pointer_path)


if __name__ == "__main__":
    main()
