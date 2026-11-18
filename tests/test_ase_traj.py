"""ASE trajectory files (.traj) open in every front end, without ASE installed.

The fixtures were written by ASE itself (scripts/make_ase_traj_fixture.py); the expected
numbers below are what that script printed from ASE, not what this reader returns.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pdb_wizard.ase_traj import check_ase_trajectory, read_ase_trajectory
from pdb_wizard.io import detect_filetype, is_fallback_cell, read_file, read_file_trajectory
from pdb_wizard.trajectory import is_trajectory_file, read_trajectory, trajectory_caveats

ROOT = Path(__file__).resolve().parent.parent
WATER = ROOT / "tests" / "fixtures" / "ase_water_md.traj"
ROTATED = ROOT / "tests" / "fixtures" / "ase_periodic_rotated.traj"

WATER_BASE = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]])
ROTATED_FRAC = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.1, 0.9]])


def coords(mol) -> np.ndarray:
    return np.array([a.x for a in mol.atoms])


# ---- the reader ---------------------------------------------------------------------------


def test_file_type_and_detection():
    assert detect_filetype("md.traj") == "ase_traj"
    assert check_ase_trajectory(str(WATER)) and is_trajectory_file(str(WATER))
    assert not check_ase_trajectory(str(ROOT / "HKUST-1.pdb"))
    assert not check_ase_trajectory(str(ROOT / "no_such_file.traj"))


def test_non_periodic_md_frames_elements_positions_and_charges():
    frames, cells = read_ase_trajectory(str(WATER))
    assert len(frames) == 3 and cells == [None, None, None]
    for k, mol in enumerate(frames):
        assert [a.element.symbol for a in mol.atoms] == ["O", "H", "H"]
        np.testing.assert_allclose(coords(mol), WATER_BASE + 0.1 * k, atol=1e-12)
        expected = np.array([-0.8, 0.4, 0.4]) + 0.01 * k * np.array([-2, 1, 1])
        np.testing.assert_allclose([a.charge for a in mol.atoms], expected, atol=1e-12)
        assert is_fallback_cell(mol.pbc)  # no cell in the file: the stand-in box
        assert [a.id for a in mol.atoms] == [1, 2, 3]


def test_later_frames_reuse_the_first_frames_atom_list():
    """ASE stores atomic numbers once; every frame must still have its elements."""
    frames, _ = read_ase_trajectory(str(WATER))
    assert all(len(m.atoms) == 3 for m in frames)
    assert frames[2].atoms[0].element.symbol == "O"


def test_periodic_cell_in_another_orientation_keeps_the_crystal():
    """The fixture's lattice is (0,a,a), (a,0,a), (a,a,0): not a-along-x. The atoms must
    land at the same fractional coordinates in this program's standard cell."""
    frames, cells = read_ase_trajectory(str(ROTATED))
    assert len(frames) == 2
    for mol, pbc, length in zip(frames, cells, (5.656854, 5.939697)):
        assert mol.pbc is pbc and not is_fallback_cell(pbc)
        assert [a.element.symbol for a in mol.atoms] == ["Na", "Cl", "Cu"]
        assert (pbc.a, pbc.b, pbc.c) == pytest.approx((length,) * 3, abs=1e-5)
        assert (pbc.alpha, pbc.beta, pbc.gamma) == pytest.approx((60.0, 60.0, 60.0), abs=1e-8)
        frac = coords(mol) @ pbc.reciprocal_basis_matrix
        np.testing.assert_allclose(frac, ROTATED_FRAC, atol=1e-9)
    # interatomic distance is orientation-independent: Na-Cl along the body diagonal
    first = frames[0]
    d = np.linalg.norm(coords(first)[1] - coords(first)[0])
    assert d == pytest.approx(np.linalg.norm(0.5 * np.array([8.0, 8.0, 8.0])), abs=1e-9)


def test_progress_is_reported_per_frame():
    seen = []
    read_ase_trajectory(str(WATER), progress_callback=seen.append)
    assert seen == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_shared_trajectory_entry_points_read_it():
    frames = read_trajectory(str(WATER))
    assert len(frames) == 3
    assert "no cell information" in " ".join(trajectory_caveats(frames))
    again = read_file_trajectory(str(WATER))
    assert len(again) == 3 and again[0].bonds  # this entry point also finds the bonds
    with pytest.raises(ValueError, match="trajectories"):
        read_file(str(WATER))


# ---- damaged and unfinished files -----------------------------------------------------------


def test_a_file_that_is_not_ulm_is_refused(tmp_path):
    path = tmp_path / "x.traj"
    path.write_bytes(b"not a trajectory at all, just some bytes" * 4)
    assert not check_ase_trajectory(str(path))
    with pytest.raises(ValueError, match="not an ASE trajectory"):
        read_ase_trajectory(str(path))
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not an ASE trajectory"):
        read_ase_trajectory(str(path))


def test_a_run_still_being_written_gives_its_complete_frames(tmp_path):
    """The last frame's array is cut off (the job was killed or is mid-write)."""
    data = bytearray(WATER.read_bytes())
    _, _, _, n_items, table = struct.unpack("<8s16sqqq", bytes(data[:48]))
    offsets = list(struct.unpack("<%dq" % n_items, bytes(data[table:table + 8 * n_items])))
    # point the last frame's record past the end of the file
    offsets[-1] = len(data) + 1000
    data[table:table + 8 * n_items] = struct.pack("<%dq" % n_items, *offsets)
    path = tmp_path / "running.traj"
    path.write_bytes(bytes(data))
    frames, _ = read_ase_trajectory(str(path))
    assert len(frames) == 2
    np.testing.assert_allclose(coords(frames[1]), WATER_BASE + 0.1, atol=1e-12)


def test_a_file_with_no_readable_frame_is_an_error(tmp_path):
    data = bytearray(WATER.read_bytes())
    _, _, _, n_items, table = struct.unpack("<8s16sqqq", bytes(data[:48]))
    data[table:table + 8 * n_items] = struct.pack("<%dq" % n_items, *([len(data) + 50] * n_items))
    path = tmp_path / "broken.traj"
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="still being written"):
        read_ase_trajectory(str(path))


# ---- the three front ends -------------------------------------------------------------------


def test_classic_menu_opens_it_as_a_trajectory(tmp_path):
    import pdb_wizard.menu as menu
    out = tmp_path / "all.xyz"
    printed: list = []
    answers = iter(["6", str(out), "9"])
    with mock.patch("builtins.input", lambda prompt="": next(answers)), \
            mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        menu.run_classic(str(WATER))
    text = out.read_text().splitlines()
    assert text.count("3") == 3  # three frames of three atoms
    assert "wrote 3 frames" in "\n".join(printed)


def test_batch_msd_reads_it():
    proc = subprocess.run([sys.executable, "-m", "pdb_wizard", str(WATER), "--msd", "-e", "O"],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=120,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr


def test_terminal_viewer_loads_it():
    pytest.importorskip("textual")
    from pdb_wizard.molecule import Molecule
    from pdb_wizard.viewer import PdbWizardApp

    async def go():
        app = PdbWizardApp(molecule=Molecule(atoms=[]), filepath=str(WATER),
                           load_trajectory=True)
        async with app.run_test(size=(160, 50)) as pilot:
            deadline = time.monotonic() + 120
            while not (app._frames and len(app._frames) == 3):
                assert time.monotonic() < deadline, "trajectory did not load"
                await pilot.pause(0.05)
            await pilot.pause()
            return len(app._frames), len(app.molecule.atoms), app.is_running

    assert asyncio.run(go()) == (3, 3, True)


def test_gui_opens_it_with_charges(request):
    pytest.importorskip("PySide6.QtWidgets")
    if importlib_missing("pytestqt.qtbot"):
        pytest.skip("needs pytest-qt")
    qtbot = request.getfixturevalue("qtbot")
    from pdb_wizard import gui

    w = gui.MainWindow()
    w.warnings = []
    w._warn = w.warnings.append
    qtbot.addWidget(w)
    assert w.open_path(str(WATER))
    w.load_task.wait()
    assert len(w.frames) == 3 and len(w.mol.atoms) == 3
    # the same notice any cell-less trajectory gets, and nothing else
    assert len(w.warnings) == 1 and "no cell information" in w.warnings[0]
    assert w.mol.atoms[0].charge == pytest.approx(-0.8)
    w.set_frame(2)
    np.testing.assert_allclose(coords(w.mol), WATER_BASE + 0.2, atol=1e-12)
    assert "*.traj" in gui._OPEN_FILTER
    for d in w._live_windows():
        d.close()


def importlib_missing(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
    except ImportError:
        return True
    return False


def test_single_file_build_reads_it(tmp_path):
    """The one-file version must include the reader (it is a separate module)."""
    dist = ROOT / "dist" / "pdb_wizard.py"  # built by conftest.py
    assert dist.exists()
    copy = tmp_path / "pdb_wizard.py"
    shutil.copy(dist, copy)
    out = tmp_path / "frames.xyz"
    proc = subprocess.run([sys.executable, str(copy), "--classic", str(WATER)],
                          input="6\n%s\n9\n" % out, capture_output=True, text=True,
                          timeout=120, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-500:]
    assert out.read_text().splitlines().count("3") == 3
