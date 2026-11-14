"""Regression tests for desktop-GUI bugs found in the 2026-11-10 debugging round.

Headless; skipped without PySide6 and pytest-qt.
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")
sys.path.insert(0, str(Path(__file__).parent))  # for the helpers in test_gui_features

from PySide6 import QtCore, QtWidgets  # noqa: E402
from test_gui_features import (  # noqa: E402
    HKUST,
    WOFOUR,
    write_pdb_trajectory,
    write_xyz_trajectory,
)

from pdb_wizard import database, gui, gui_tools  # noqa: E402
from pdb_wizard.atom import Atom  # noqa: E402
from pdb_wizard.gui_tools import (  # noqa: E402
    AnalysisDialog,
    CellDialog,
    DatabaseDialog,
    EnergyDialog,
    GuiFrameBar,
    GuiTask,
    InputGeneratorDialog,
    QtPlot,
    ReduceCellDialog,
)
from pdb_wizard.io import read_file  # noqa: E402
from pdb_wizard.pbc import PBC  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = gui.MainWindow()
    w.warnings = []
    w._warn = w.warnings.append
    qtbot.addWidget(w)
    w.resize(1000, 700)
    w.show()
    qtbot.waitExposed(w)
    yield w
    for d in w._live_windows():
        d.close()


@pytest.fixture
def traj(window, tmp_path):
    path = tmp_path / "water.pdb"
    write_pdb_trajectory(path, 6)
    window.open_path(str(path))
    window.load_task.wait()
    return window


def flush_deletes():
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    gc.collect()


# ---- main window ------------------------------------------------------------------------


def test_undo_restores_the_frame_that_was_edited(traj):
    """Undo after switching frames wrote frame 0's copy over the frame being shown."""
    w = traj
    x3 = w.frames[3].atoms[0].x.copy()
    w.substitute_element("O", "N")
    w.set_frame(3)
    w.undo()
    assert w.doc.frame_idx == 0 and w.frame_bar.index == 0
    assert {a.element.symbol for a in w.frames[0].atoms} == {"O", "H"}
    np.testing.assert_array_equal(w.frames[3].atoms[0].x, x3)


def test_closing_a_loading_tab_and_failed_loads(window, tmp_path, monkeypatch):
    path = tmp_path / "t.pdb"
    write_pdb_trajectory(path, 4)
    window.open_path(str(HKUST))
    window.open_path(str(path))
    task = window.load_task
    window.close_tab(1)
    task.wait()  # used to raise ValueError: document not in list
    assert window.tab_bar.count() == 1 and window.doc.name == "HKUST-1.pdb"

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gui, "read_trajectory", boom)
    window.open_path(str(path))
    window.load_task.wait()
    assert window.tab_bar.count() == 1  # no dead "(loading)" tab left
    assert "(loading)" not in window.windowTitle() and "boom" in window.warnings[-1]


def test_two_trajectories_loading_at_once_both_finish(window, tmp_path):
    """The second load dropped the only reference to the first task."""
    for k in range(2):
        path = tmp_path / f"t{k}.pdb"
        write_pdb_trajectory(path, 3 + k)
        window.open_path(str(path))
        gc.collect()
    for task in list(GuiTask._live):
        task.wait()
    assert [len(d.frames or []) for d in window.docs] == [3, 4]
    assert all("(loading)" not in d.name for d in window.docs)


def test_trajectory_loaded_in_a_background_tab_is_fitted(window, tmp_path):
    path = tmp_path / "t.xyz"  # no cell: the reader's stand-in box must not be drawn
    write_xyz_trajectory(path, 4)
    window.open_path(str(HKUST))
    window.open_path(str(path))
    window.tab_bar.setCurrentIndex(0)
    window.load_task.wait()
    window.tab_bar.setCurrentIndex(1)
    v = window.view
    v.grab()
    pts = np.array([a.x for a in window.mol.atoms])
    assert np.allclose(v._center, pts.mean(axis=0))
    assert not v.has_cell()
    xs = v._screen[:, 0]
    assert xs.max() - xs.min() > 0.4 * v.width()


def test_edits_clear_stale_marks(window, qtbot):
    window.open_path(str(HKUST))
    v = window.view
    v.grab()
    n = len(window.mol.atoms)
    v.pick(n - 1, shift=True)
    window._selected(n - 1)
    v.highlighted = {5, n - 1}
    v.measurements = [((n - 1, n - 2), "x")]
    window._checkpoint()
    del window.mol.atoms[-50:]
    window._changed("deleted")
    assert (v.selected, v.picked, v.highlighted, v.measurements) == (None, [], set(), [])
    assert "Click an atom" in window.selection_label.text()
    assert v.pick(n - 1, shift=True) == ""  # out of range: ignored, no IndexError
    v.grab()


def test_pinned_distances_follow_the_frames(traj):
    v = traj.view
    v.grab()
    v.set_measure_mode(True)
    v.pick(0)
    v.pick(3)
    traj.set_frame(5)
    d5 = v.distance(0, 3)
    labels = []
    original = v._paint_measurements

    def spy(painter, mol, screen):
        labels.extend(f"{v.distance(*idx):.3f} Å" for idx, _ in v.measurements)
        original(painter, mol, screen)

    v._paint_measurements = spy
    v.grab()
    assert labels == [f"{d5:.3f} Å"]


def test_density_goes_to_the_tab_it_was_computed_for(traj):
    task = traj.density_3d("O")
    source = traj.doc
    traj.open_path(str(HKUST))
    task.wait()
    assert source.density is not None and traj.doc.density is None
    assert traj.view.density is None
    traj.tab_bar.setCurrentIndex(0)
    assert traj.view.density is not None


def test_export_errors_are_reported(window, tmp_path):
    window.open_path(str(HKUST))
    assert window.export_image("/nonexistent_dir/x.png") is None
    assert "Could not write" in window.warnings[-1]
    pytest.importorskip("PIL")
    assert window.export_gif("/nonexistent_dir/x.gif", 2) is None
    assert "Could not write" in window.warnings[-1]
    from PIL import Image
    out = window.export_gif(str(tmp_path / "one.gif"), n_frames=0)
    with Image.open(out) as img:
        assert img.n_frames == 1
    assert QtWidgets.QApplication.overrideCursor() is None


def test_closing_the_window_stops_playback_and_closes_tools(traj):
    traj.frame_bar.toggle_play()
    report = traj.open_tool("info")
    ref = weakref.ref(report)
    traj.close()
    flush_deletes()
    assert not traj.frame_bar.playing
    assert ref() is None or not gui._qt_alive(ref())


@pytest.mark.parametrize("cell", [(0, 0, 0, 90, 90, 90), (10, 10, 10, 0, 90, 90),
                                  (10, 10, 10, 90, 90, 180), (10, 10, 10, 10, 10, 170),
                                  (np.nan, 10, 10, 90, 90, 90)])
def test_impossible_cells_are_rejected(window, cell):
    window.open_path(str(HKUST))
    assert not window.update_cell(cell)
    assert window.mol.pbc.a == pytest.approx(26.343) and not window.undo_action.isEnabled()


def test_non_finite_coordinates_are_rejected(window):
    window.open_path(str(HKUST))
    window.edit_coord(5, (np.nan, 1.0, 2.0))
    assert "finite" in window.warnings[-1] and np.all(np.isfinite(window.mol.atoms[5].x))
    window.geometry_dock.show()
    panel = window.geometry
    panel.tabs.setCurrentIndex(panel.tabs.indexOf(panel.tables["Coords"]))
    before = window.mol.atoms[5].x.copy()
    for text in ("nan", "inf", "1e400"):
        panel.tables["Coords"].item(5, 2).setText(text)
        np.testing.assert_array_equal(window.mol.atoms[5].x, before)
    window.view.grab()  # used to raise in paintEvent and then crash Qt


def test_view_survives_nan_coordinates_from_a_file(window):
    window.open_path(str(HKUST))
    window.mol.atoms[3].x = np.array([np.nan, 0.0, 0.0])
    window.view.refresh_bonds()
    window.view.grab()
    window.view.grab()  # a painter left active by an exception crashed the second paint


def test_overlapping_background_reports_all_arrive(window):
    """The second task dropped the first: one report missing and the wait cursor stuck."""
    window.open_path(str(HKUST))
    t1 = window.open_tool("void_vol")
    t2 = window.open_tool("surface_area")
    t1.wait()
    t2.wait()
    titles = sorted(d.windowTitle() for d in window._live_windows())
    assert titles == ["Surface Area", "Void Volume"]
    assert QtWidgets.QApplication.overrideCursor() is None


# ---- tool windows -----------------------------------------------------------------------


def test_analysis_runs_on_a_snapshot(window, monkeypatch):
    """Edits on the GUI thread raced with analyses reading the same Molecule."""
    window.open_path(str(HKUST))
    dlg = window.open_tool("hbonds")
    seen = []
    spec = gui_tools.ANALYSES["hbonds"]
    real = spec.run
    def spy(mol, fr, p, prog):
        seen.append(mol)
        return real(mol, fr, p, prog)

    monkeypatch.setattr(spec, "run", spy)
    dlg.run().wait()
    assert seen and seen[0] is not window.mol and len(seen[0].atoms) == len(window.mol.atoms)


def test_trajectory_snapshot_copies_only_the_shown_frame(traj):
    dlg = traj.open_tool("rmsd")
    mol, frames = dlg._snapshot()
    shown = traj.doc.frame_idx
    assert frames[shown] is mol and mol is not traj.mol
    assert all(f is g for k, (f, g) in enumerate(zip(frames, traj.frames)) if k != shown)


def test_tool_windows_follow_their_tab(window):
    window.open_path(str(HKUST))
    dlg = window.open_tool("coordination")
    window.substitute_element("Cu", "Zn")
    items = [dlg.inputs["center"].itemText(i) for i in range(dlg.inputs["center"].count())]
    assert "Zn" in items and "Cu" not in items
    window.undo()
    assert dlg.mol is window.mol
    dlg.inputs["center"].setCurrentText("Cu")
    dlg.inputs["neighbor"].setCurrentText("O")
    dlg.run().wait()
    assert dlg.result.summary.startswith("Avg coordination at 5.0 A:")
    ref = weakref.ref(dlg)
    window.close_tab(0)  # the tab's windows close with it
    flush_deletes()
    assert ref() is None or not gui._qt_alive(ref())


def test_closed_tool_windows_are_freed(window):
    window.open_path(str(HKUST))
    refs = []
    for _ in range(4):
        d = window.open_tool("rdf")
        refs.append(weakref.ref(d))
        d.close()
    flush_deletes()
    assert not window.findChildren(AnalysisDialog)
    assert all(r() is None or not gui._qt_alive(r()) for r in refs)


def test_closing_a_window_while_it_computes(window):
    window.open_path(str(HKUST))
    dlg = window.open_tool("pxrd")
    task = dlg.run()
    dlg.close()
    flush_deletes()
    task.wait()  # its callbacks are dropped: no "C++ object already deleted"
    assert task.finished


def test_cell_dialog_keeps_values_it_cannot_show(qtbot):
    pbc = PBC(26.343567, 26.343567, 12000.0, 89.99996, 90.0, 120.000004)
    dlg = CellDialog(pbc)
    qtbot.addWidget(dlg)
    assert dlg.values() == (26.343567, 26.343567, 12000.0, 89.99996, 90.0, 120.000004)
    dlg.spins[0].setValue(30.0)
    assert dlg.values()[0] == 30.0 and dlg.values()[1] == 26.343567


def test_input_generator_keeps_pressures_exactly(qtbot, tmp_path):
    """Pressures were rounded to 6 digits when switching engines (1234567 -> 1234570)."""
    mol = read_file(str(HKUST))
    dlg = InputGeneratorDialog(mol, str(HKUST))
    qtbot.addWidget(dlg)
    dlg.widgets["temperature"].setText("-5")
    assert dlg.values()["temperature"] == -5.0  # accepted as before (flagged, not changed)
    dlg.widgets["pressures"].setText("0.00123456789, 1234567, 0.1")
    dlg.outdir.setText(str(tmp_path / "mine"))
    dlg.set_engine("cp2k_cell_opt")
    assert dlg.outdir.text().endswith("cp2k_cell_opt")  # engine default, as in the TUI
    dlg.set_engine("mpmc_isotherm")
    assert dlg.values()["pressures"] == [0.00123456789, 1234567.0, 0.1]


def test_database_odd_results(qtbot, monkeypatch, tmp_path):
    monkeypatch.setitem(database.BACKENDS["cod"], "search", lambda q, mode="text": [
        {"source": "cod", "id": "1", "formula": None, "title": None},
        {"source": "cod", "formula": "Cu"}])

    def bad_open(path, label, rid):
        raise ValueError("not a structure")

    monkeypatch.setitem(database.BACKENDS["cod"], "fetch", lambda rid, d=".": str(tmp_path / "x"))
    dlg = DatabaseDialog(str(tmp_path), bad_open)
    qtbot.addWidget(dlg)
    dlg.source.setCurrentIndex(dlg.source.findData("cod"))
    dlg.query.setText("q")
    dlg.search().wait()
    rows = {dlg.table.item(r, 1).text(): r for r in range(dlg.table.rowCount())}
    assert set(rows) == {"", "1"}  # results are sorted by id, so the id-less one is first
    assert dlg.table.item(rows["1"], 2).text() == ""  # formula None shows as blank
    dlg.table.setCurrentCell(rows[""], 0)
    assert dlg.fetch_selected() is None and "no ID" in dlg.status.text()
    dlg.table.setCurrentCell(rows["1"], 0)
    dlg.fetch_selected().wait()
    assert "Error opening 1: not a structure" in dlg.status.text()


def test_reduce_dialog_guards(window, qtbot):
    window.open_path(str(HKUST))
    # a real 2 x 1 x 1 supercell: one HKUST-1 cell is not a supercell of anything, and
    # analysing it as one is now refused (ValueError) instead of returning garbage
    from pdb_wizard.operations import extend_axis
    extend_axis(window.mol, 0, 1)
    dlg = ReduceCellDialog(window.mol, window)
    qtbot.addWidget(dlg)
    for spin, value in zip(dlg.spins, (2, 1, 1)):
        spin.setValue(value)
    dlg.analyze()
    assert dlg.reduce_button.isEnabled()
    dlg.spins[0].setValue(1)
    assert not dlg.reduce_button.isEnabled() and "Analyze again" in dlg.status.text()
    assert dlg.reduce() is None
    window.mol.pbc = None
    dlg.analyze()
    assert "periodic cell" in dlg.status.text() and not dlg.reduce_button.isEnabled()


def test_frame_ranges_cannot_be_reversed(traj, qtbot):
    dlg = traj.open_tool("rdf")
    dlg.start_spin.setValue(5)
    dlg.end_spin.setValue(2)
    assert dlg.end_spin.value() == 5
    data = {"step": np.arange(1.0, 11.0), "N": np.arange(10.0)}
    edlg = EnergyDialog(data, "e.dat")
    qtbot.addWidget(edlg)
    edlg.start.setValue(8)
    edlg.end.setValue(3)
    assert edlg.end.value() == 8


def test_pxrd_overlay_survives_recompute_and_zooms(window, tmp_path):
    window.open_path(str(HKUST))
    dlg = window.open_tool("pxrd")
    dlg.run().wait()
    exp = tmp_path / "exp.xy"
    exp.write_text("5 10\n60 100\n90 20\n")  # wider than the simulated 0-50
    dlg.load_experimental(str(exp))
    dlg.inputs["width"].setValue(20)
    dlg.run().wait()
    assert dlg.plot.overlay is not None
    dlg.plot.zoom_x(1.25, 25)
    assert dlg.plot.x_range is not None  # used to stay None when the overlay was wider
    assert dlg.export_png("/nonexistent_dir/x.png") is None
    assert "Could not write" in dlg.status.text()


def test_plot_hover_ignores_nan(qtbot):
    plot = QtPlot()
    qtbot.addWidget(plot)
    plot.resize(500, 300)
    plot.show()
    plot.set_data([np.nan, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
    plot.grab()
    qtbot.mouseMove(plot, QtCore.QPoint(250, 150))
    assert plot._hover in (1, 2, 3)


def test_frame_bar_speed_label(qtbot):
    bar = GuiFrameBar(lambda i: None)
    qtbot.addWidget(bar)
    for speed, text in ((0.125, "0.125x"), (0.25, "0.25x"), (1, "1x"), (16, "16x")):
        bar.set_speed(speed)
        assert bar.speed_label.text() == text


def test_gui_pack_after_a_supercell_uses_fresh_ids(window):
    window.open_path(str(Path(HKUST).parent / "HKUST-1_out_mpmc.pdb"))
    for spin, n in zip(window.extend_spins, (2, 1, 1)):
        spin.setValue(n)
    window.build_supercell()
    window.sorbate_combo.setCurrentIndex(window.sorbate_combo.findData("h2o_tip3p"))
    window.sorbate_count.setValue(2)
    window.pack_sorbates()
    ids = sorted({a.mpmc_mol for a in window.mol.atoms if a.mpmc_flag == "M"})
    assert ids == [2, 4, 5, 6]


def test_wofour_still_opens(window):
    assert window.open_path(str(WOFOUR)) and len(window.mol.atoms) > 0
    assert Atom is not None
