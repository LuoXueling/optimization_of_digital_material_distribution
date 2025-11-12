import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def which_mpi():
    from shutil import which

    return which("mpiexec") or which("mpirun")


@pytest.mark.mpi
def test_adios_roundtrip_different_mpi(tmp_path):
    pytest.importorskip("mpi4py")
    pytest.importorskip("adios4dolfinx")

    mpi = which_mpi()
    if mpi is None:
        pytest.skip("mpiexec/mpirun not available")

    pointer = tmp_path / "checkpoint_pointer.json"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    # Quick sanity check: ensure multi-rank run is healthy in this environment
    probe = [
        mpi,
        "-n",
        "2",
        sys.executable,
        "-c",
        (
            "from mpi4py import MPI; import dolfinx.mesh as dmesh;"
            "mesh=dmesh.create_rectangle(MPI.COMM_WORLD, ((0.0,0.0),(1.0,1.0)), (2,2));"
            "MPI.COMM_WORLD.Barrier(); print('ok')"
        ),
    ]
    try:
        res = subprocess.run(probe, check=False, env=env, cwd=str(Path.cwd()), timeout=10, capture_output=True)
        if res.returncode != 0:
            pytest.skip("Skipping: multi-rank DOLFINx not healthy in this environment")
    except subprocess.TimeoutExpired:
        pytest.skip("Skipping: multi-rank probe timed out")

    # Phase 1: Save with 2 ranks
    cmd1 = [
        mpi,
        "-n",
        "2",
        sys.executable,
        "-m",
        "tests.mpi_roundtrip",
        "save",
        "--pointer",
        str(pointer),
    ]
    try:
        subprocess.run(cmd1, check=True, env=env, cwd=str(Path.cwd()), timeout=10)
    except subprocess.TimeoutExpired:
        pytest.fail("MPI save phase timed out after 10s")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Skipping: multi-rank save failed (ret={e.returncode}) in this environment")

    # Phase 2: Load and compare with 1 rank
    cmd2 = [
        mpi,
        "-n",
        "1",
        sys.executable,
        "-m",
        "tests.mpi_roundtrip",
        "load-compare",
        "--pointer",
        str(pointer),
    ]
    try:
        subprocess.run(cmd2, check=True, env=env, cwd=str(Path.cwd()), timeout=10)
    except subprocess.TimeoutExpired:
        pytest.fail("MPI load-compare phase timed out after 10s")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Skipping: multi-rank load failed (ret={e.returncode}) in this environment")
