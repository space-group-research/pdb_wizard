"""Desktop GUI bugs found in the bug hunt (gui.py, gui_tools.py).

Headless (QT_QPA_PLATFORM=offscreen); skipped without PySide6 and pytest-qt. Nothing here
touches the network: database fetches and searches are replaced by local functions.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from pdb_wizard import gui  # noqa: E402
from pdb_wizard.database import BACKENDS  # noqa: E402
from pdb_wizard.gui_tools import GuiTask  # noqa: E402
from pdb_wizard.io import NO_CELL_FALLBACK_LENGTH  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
WATER = "O {x} 0 {z}\nH {x1} 0 {z}\nH {x2} 0.93 {z}\n"


def water(x=0.0, z=0.0):
    return WATER.format(x=x, x1=x + 0.96, x2=x - 0.24, z=z)


@pytest.fixture
def window(qtbot):
    w = gui.MainWindow()
    w.warnings = []
    w._warn = w.warnings.append
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.frame_bar.stop(notify=False)
    for d in w._live_windows():
        d.close()


def open_traj(w, path):
    assert w.open_path(str(path))
    w.load_task.wait()
    QtWidgets.QApplication.processEvents()
    w.warnings.clear()


@pytest.fixture
def nocell_traj(tmp_path):
    """Three frames of one water, no cell in the file."""
    p = tmp_path / "nocell.xyz"
    p.write_text("".join("3\nframe\n" + water(z=0.1 * k) for k in range(3)))
    return p


@pytest.fixture
def growing_traj(tmp_path):
    """1, 2 and 3 waters: the atom count changes between frames (as in a GCMC run)."""
    p = tmp_path / "growing.xyz"
    frames = []
    for k in range(3):
        frames.append("%d\nframe\n" % (3 * (k + 1)) + "".join(water(3.0 * m) for m in range(k + 1)))
    p.write_text("".join(frames))
    return p


def settle(timeout=10.0):
    end = time.time() + timeout
    while GuiTask._live and time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    QtWidgets.QApplication.processEvents()


# ---- 1: a worker that exits must not take the GUI with it ------------------------------------

def test_worker_system_exit_goes_to_on_error(qtbot):
    got = {}

    def worker():
        raise SystemExit("Error reading XYZ trajectory")

    task = GuiTask(worker, lambda r: got.setdefault("done", r), lambda e: got.setdefault("err", e))
    task.wait()  # at 35d43b8 this re-raised SystemExit on the GUI thread
    assert "done" not in got
    assert isinstance(got["err"], SystemExit)
    assert "Error reading XYZ trajectory" in str(got["err"])
    assert task.finished and task not in GuiTask._live


def test_reader_that_exits_is_reported_and_the_window_survives(window, tmp_path, monkeypatch):
    def reader(path, progress=None):
        raise SystemExit("Error reading XYZ trajectory")

    monkeypatch.setattr(gui, "read_trajectory", reader)
    p = tmp_path / "t.xyz"
    p.write_text(("3\nf\n" + water()) * 3)
    assert window.open_path(str(p))
    window.load_task.wait()
    assert len(window.warnings) == 1 and "Error reading XYZ trajectory" in window.warnings[0]
    assert [d.name for d in window.docs] == ["untitled"]  # no "(loading)" tab left behind


# ---- 2: Update unit cell on a trajectory ---------------------------------------------------

def test_update_cell_applies_to_every_frame_and_undoes_as_one_step(window, nocell_traj):
    w = window
    open_traj(w, nocell_traj)
    assert [f.pbc.a for f in w.frames] == [NO_CELL_FALLBACK_LENGTH] * 3
    w.frame_bar.go(1)
    assert w.update_cell((12.0, 13.0, 14.0, 90.0, 90.0, 90.0))
    assert [(f.pbc.a, f.pbc.b, f.pbc.c) for f in w.frames] == [(12.0, 13.0, 14.0)] * 3
    assert len({id(f.pbc) for f in w.frames}) == 3  # still one cell object per frame
    assert len(w.doc.undo) == 1
    w.frame_bar.go(2)
    w.undo()
    assert [f.pbc.a for f in w.frames] == [NO_CELL_FALLBACK_LENGTH] * 3
    assert w.doc.frame_idx == 1 and not w.doc.undo  # back on the frame it was made on
    w.redo()
    assert [(f.pbc.a, f.pbc.b, f.pbc.c) for f in w.frames] == [(12.0, 13.0, 14.0)] * 3
    assert w.update_cell(None)  # removing the cell is also for the whole trajectory
    assert [f.pbc for f in w.frames] == [None] * 3
    w.undo()
    assert [f.pbc.a for f in w.frames] == [12.0] * 3


def test_trajectory_rdf_after_update_cell_matches_the_ideal_gas(window, tmp_path):
    """Reference: uniform random points in a periodic box have g(r) = 1."""
    rng = np.random.default_rng(0)
    box, n, text = 12.0, 300, ""
    for _ in range(6):
        xyz = rng.uniform(0, box, (n, 3))
        text += "%d\nframe\n" % n + "".join("Ar %.4f %.4f %.4f\n" % tuple(p) for p in xyz)
    p = tmp_path / "gas.xyz"
    p.write_text(text)
    w = window
    open_traj(w, p)
    assert w.update_cell((box, box, box, 90.0, 90.0, 90.0))
    d = w.open_tool("rdf")
    d.inputs["xmax"].setValue(6)
    d.inputs["bins"].setValue(30)
    assert d.traj_box.isChecked()
    d.run().wait()
    r, g = d.result.x, d.result.y
    assert abs(float(np.mean(g[r > 2.0])) - 1.0) < 0.05  # was 3e14


def test_single_frame_edits_still_undo_per_frame(window, nocell_traj):
    w = window
    open_traj(w, nocell_traj)
    w.update_cell((12.0, 12.0, 12.0, 90.0, 90.0, 90.0))
    w.frame_bar.go(2)
    w.edit_coord(0, (5.0, 0.0, 0.0))
    w.frame_bar.go(0)
    w.undo()  # the coordinate edit: frame 3 only, the cell stays
    assert w.doc.frame_idx == 2 and w.frames[2].atoms[0].x[0] == 0.0
    assert [f.pbc.a for f in w.frames] == [12.0] * 3
    w.redo()
    assert w.frames[2].atoms[0].x[0] == 5.0 and [f.atoms[0].x[0] for f in w.frames[:2]] == [0, 0]
    w.undo()
    w.undo()  # now the cell, on every frame; the atoms are untouched
    assert [f.pbc.a for f in w.frames] == [NO_CELL_FALLBACK_LENGTH] * 3
    assert [f.atoms[0].x[2] for f in w.frames] == [0.0, 0.1, 0.2]


# ---- 3: the stand-in cell of a cell-less trajectory is not a cell -------------------------

def test_stand_in_cell_is_refused_like_no_cell(window, nocell_traj, tmp_path):
    w = window
    open_traj(w, nocell_traj)
    before = [tuple(a.x) for a in w.mol.atoms]
    w.extend_spins[0].setValue(2)
    w.build_supercell()
    w.wrap(True)
    w.sort_atoms()
    assert w.pack_sorbates() == 0
    assert w.delete_lone_atoms() == 0
    assert len(w.warnings) == 5 and all("needs a unit cell" in t for t in w.warnings)
    assert [tuple(a.x) for a in w.mol.atoms] == before and not w.doc.undo
    assert w.mol.pbc.a == NO_CELL_FALLBACK_LENGTH
    w.warnings.clear()
    for name in ("x.pqr", "x.cif", "x.vasp", "x.d12"):
        out = tmp_path / name
        assert not w.save_path(str(out)), name
        assert not out.exists(), name
    assert not w.save_crystal(str(tmp_path / "y.d12"))
    assert len(w.warnings) == 5 and all("unit cell" in t for t in w.warnings)
    assert w.save_path(str(tmp_path / "x.xyz"))  # formats that work without a cell still save


def test_input_generator_treats_the_stand_in_cell_as_no_cell(window, nocell_traj, tmp_path):
    w = window
    open_traj(w, nocell_traj)
    ig = w.open_tool("input_generator")
    ig.set_engine("cp2k_cell_opt")
    ig.outdir.setText(str(tmp_path / "cp2k"))
    out = ig.generate()
    assert not out.ok and ig.status.text() == "CP2K requires a periodic box"
    assert not (tmp_path / "cp2k").exists()
    w.update_cell((12.0, 12.0, 12.0, 90.0, 90.0, 90.0))  # a real cell: it works
    assert ig.generate().ok
    text = (tmp_path / "cp2k" / "input.inp").read_text()
    assert "A   12.000000 0.000000 0.000000" in text


# ---- 5, 11, 15: database fetches ------------------------------------------------------------

def fake_fetch(delay=0.0):
    def fetch(rid, cache):
        time.sleep(delay)
        out = os.path.join(cache, rid + ".pdb")
        shutil.copy(HKUST, out)
        return out
    return fetch


def start_fetch(w, rid):
    d = w.open_tool("db_search")
    d.source.setCurrentIndex(d.source.findData("cod"))
    d.query.setText(rid)
    assert d.fetch_by_id() is not None
    return d


def test_fetched_structure_does_not_point_into_the_temp_cache(window, monkeypatch, tmp_path):
    monkeypatch.setitem(BACKENDS["cod"], "fetch", fake_fetch())
    monkeypatch.chdir(tmp_path)
    w = window
    start_fetch(w, "2300380")
    settle()
    assert w.doc.name == "2300380.pdb" and len(w.mol.atoms) == 624
    cache = w._db_cache
    starts = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda parent, caption, start, *a, **k:
                                     (starts.append(start), ("", ""))[1]))
    w.save_dialog()
    ig = w.open_tool("input_generator")
    for where in (starts[0], ig.outdir.text()):
        assert not os.path.abspath(where).startswith(cache), where
    assert ig.generate().ok
    outdir = os.path.abspath(ig.outdir.text())
    w.close()
    assert not os.path.exists(cache) and os.path.isdir(outdir)  # the inputs survive closing


def test_escape_cancels_a_running_fetch(window, monkeypatch, qtbot):
    monkeypatch.setitem(BACKENDS["cod"], "fetch", fake_fetch(0.3))
    d = start_fetch(window, "222")
    qtbot.keyClick(d, QtCore.Qt.Key_Escape)
    settle()
    assert [x.name for x in window.docs] == ["untitled"]  # as when closed with the X button


def test_database_dialog_stays_open_when_the_download_cannot_be_read(window, monkeypatch):
    monkeypatch.setitem(BACKENDS["cod"], "fetch",
                        lambda rid, cache: os.path.join(cache, "missing.cif"))
    d = start_fetch(window, "333")
    settle()
    assert len(window.warnings) == 1 and "Could not read COD 333" in window.warnings[0]
    assert gui._qt_alive(d) and d.isVisible()
    assert "333" in d.status.text() and "Fetching" not in d.status.text()

    def empty(rid, cache):
        out = os.path.join(cache, "empty.cif")
        Path(out).write_text("data_x\n")
        return out

    monkeypatch.setitem(BACKENDS["cod"], "fetch", empty)
    d.fetch_by_id()
    settle()
    assert "No atoms" in window.warnings[-1]
    assert [x.name for x in window.docs] == ["untitled"] and d.isVisible()


@pytest.mark.parametrize("name,text", [("empty.pdb", ""), ("notes.pqr", "not a structure\n"),
                                       ("cut.cif", "data_x\n_cell_length_a 10\n"),
                                       ("empty.xyz", "")])
def test_file_without_atoms_is_not_opened(window, tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    assert window.open_path(str(p)) is False
    assert len(window.warnings) == 1 and "No atoms" in window.warnings[0] and name in \
        window.warnings[0]
    assert [d.name for d in window.docs] == ["untitled"]
    assert window.windowTitle() == "PDB Wizard — no structure"


# ---- 7: energy plot -------------------------------------------------------------------------

def test_energy_plot_reports_unreadable_files(window, tmp_path, monkeypatch):
    (tmp_path / "latin1.dat").write_bytes(b"#step #energy\n1 -5.0\n# 1 \xc5 step\n")
    (tmp_path / "adir.dat").mkdir()
    for name in ("latin1.dat", "adir.dat"):
        window.warnings.clear()
        assert window.energy_plot(str(tmp_path / name)) is None  # raised at 35d43b8
        assert len(window.warnings) == 1 and name in window.warnings[0]

    def broken(path):
        raise ValueError("inhomogeneous shape")

    monkeypatch.setattr(gui, "read_energy_dat", broken)
    window.warnings.clear()
    assert window.energy_plot(str(tmp_path / "x.dat")) is None
    assert "inhomogeneous shape" in window.warnings[0]


# ---- 8: picking between a structure change and the repaint ----------------------------------

def test_no_stale_pick_after_the_atom_count_drops(window, growing_traj):
    w = window
    open_traj(w, growing_traj)
    w.frame_bar.go(2)
    w.view.grab()
    x, y = w.view._screen[8]
    assert w.view.atom_at(x, y) == 8
    w.frame_bar.go(0)  # 3 atoms now; the repaint has not happened yet
    assert w.view.atom_at(x, y) is None
    pos = QtCore.QPointF(x, y)
    glob = QtCore.QPointF(w.view.mapToGlobal(pos.toPoint()))
    left, none = QtCore.Qt.LeftButton, QtCore.Qt.NoButton
    for kind, button in ((QtCore.QEvent.MouseMove, none), (QtCore.QEvent.MouseButtonPress, left),
                         (QtCore.QEvent.MouseButtonRelease, left)):
        event = QtGui.QMouseEvent(kind, pos, glob, button, button, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(w.view, event)  # IndexError at 35d43b8
    assert w.view.selected is None
    w.view.grab()
    x, y = w.view._screen[2]
    assert w.view.atom_at(x, y) == 2  # picking works again once painted


# ---- 9: panels after playback is stopped by closing another tab -----------------------------

def test_closing_another_tab_during_playback_refreshes_the_panels(window, growing_traj, tmp_path):
    w = window
    open_traj(w, growing_traj)
    other = tmp_path / "ar.xyz"
    other.write_text("1\n\nAr 0 0 0\n")
    assert w.open_path(str(other))
    w.tab_bar.setCurrentIndex(0)
    w.frame_bar.toggle_play()
    w.frame_bar.step()
    w.frame_bar.step()
    assert w.doc.frame_idx == 2 and len(w.mol.atoms) == 9
    w.close_tab(1)
    assert not w.frame_bar.playing
    assert w.table_model.rowCount() == 9 and w.table_model.mol is w.mol
    assert w.info_labels["Atoms"].text() == "9" and w.info_labels["Frame"].text() == "3 / 3"


# ---- 10: an edit that does nothing keeps the whole undo history -----------------------------

def test_noop_edit_keeps_the_oldest_undo_step(window, tmp_path):
    p = tmp_path / "w.xyz"
    p.write_text("3\n\n" + water())
    w = window
    assert w.open_path(str(p))
    for k in range(1, 21):
        w.edit_coord(0, (float(k), 0.0, 0.0))
    assert len(w.doc.undo) == 20
    assert w.substitute_element("Xe", "Ar") == 0  # nothing to substitute
    assert len(w.doc.undo) == 20
    while w.doc.undo:
        w.undo()
    assert w.mol.atoms[0].x[0] == 0.0  # the file's value; was 1.0
    w.redo()
    w.edit_coord(0, (50.0, 0.0, 0.0))
    for k in range(25):  # a real edit still caps the history at 20
        w.edit_coord(0, (float(k), 0.0, 0.0))
    assert len(w.doc.undo) == 20


# ---- 12, 13: background tasks ----------------------------------------------------------------

def test_slow_tasks_do_not_block_the_next_one_or_interpreter_exit(qtbot):
    release = threading.Event()
    names = []

    def slow():
        names.append(threading.current_thread())
        release.wait(20)
        return "slow"

    got = []
    try:
        slow_tasks = [GuiTask(slow, got.append, got.append) for _ in range(3)]
        t0 = time.time()
        quick = GuiTask(lambda: "quick", got.append, got.append)
        quick.wait(5)
        assert got == ["quick"] and time.time() - t0 < 2.0  # queued behind them at 35d43b8
        end = time.time() + 5
        while len(names) < 3 and time.time() < end:
            time.sleep(0.01)
        assert len(names) == 3 and all(t.daemon for t in names)  # never joined at exit
    finally:
        release.set()
    for t in slow_tasks:
        t.wait(5)
    assert got == ["quick", "slow", "slow", "slow"]


def test_no_report_window_after_the_main_window_closed(window, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(gui, "void_volume_text", lambda mol: (release.wait(10), "text")[1])
    w = window
    assert w.open_path(str(HKUST))
    task = w.open_tool("void_vol")
    w.close()
    release.set()
    task.wait(5)
    QtWidgets.QApplication.processEvents()
    assert w._live_windows() == []
    assert QtWidgets.QApplication.overrideCursor() is None


# ---- 14: the trajectory keys belong to the 3D view and the frame bar ------------------------

def test_space_home_end_reach_the_focused_widget_outside_the_view(window, nocell_traj, qtbot):
    w = window
    w.activateWindow()
    w.tools.setCurrentIndex(2)
    w.sorbate_fill.setFocus()
    qtbot.keyClick(w.sorbate_fill, QtCore.Qt.Key_Space)
    assert w.sorbate_fill.isChecked()  # swallowed by the window shortcut at 35d43b8
    open_traj(w, nocell_traj)
    w.table_dock.raise_()
    w.table.setFocus()
    w.table.setCurrentIndex(w.table_model.index(0, 0))
    qtbot.keyClick(w.table, QtCore.Qt.Key_End)
    assert w.doc.frame_idx == 0 and not w.frame_bar.playing
    assert w.table.currentIndex().row() == 0  # End: last column of the row, same frame
    assert w.table.currentIndex().column() == w.table_model.columnCount() - 1


def test_trajectory_keys_work_in_the_view_and_the_frame_bar(window, nocell_traj, qtbot):
    w = window
    w.activateWindow()
    open_traj(w, nocell_traj)
    for target in (w.view, w.frame_bar.slider, w.frame_bar.buttons["next"]):
        w.frame_bar.go(0)
        target.setFocus()
        QtWidgets.QApplication.processEvents()
        qtbot.keyClick(target, QtCore.Qt.Key_BracketRight)
        assert w.doc.frame_idx == 1, target
        qtbot.keyClick(target, QtCore.Qt.Key_End)
        assert w.doc.frame_idx == 2, target
        qtbot.keyClick(target, QtCore.Qt.Key_BracketLeft)
        assert w.doc.frame_idx == 1, target
        qtbot.keyClick(target, QtCore.Qt.Key_Home)
        assert w.doc.frame_idx == 0, target
        qtbot.keyClick(target, QtCore.Qt.Key_Space)
        assert w.frame_bar.playing, target
        qtbot.keyClick(target, QtCore.Qt.Key_Space)
        assert not w.frame_bar.playing, target
