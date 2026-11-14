"""Desktop GUI features added to match the terminal viewer: tabs, trajectories, every edit,
the geometry tables, view styles and options, measuring, 3D density, analysis windows, the
input generator, database search and GIF export.

Headless (QT_QPA_PLATFORM=offscreen); skipped without PySide6 and pytest-qt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from pdb_wizard import database, gui  # noqa: E402
from pdb_wizard.analyses import charge_report  # noqa: E402
from pdb_wizard.atom import Atom  # noqa: E402
from pdb_wizard.gui_tools import (  # noqa: E402
    AnalysisDialog,
    DatabaseDialog,
    GuiTask,
    InputGeneratorDialog,
    QtPlot,
    ReduceCellDialog,
    geometry_rows,
)
from pdb_wizard.inputgen import generate_engine_inputs  # noqa: E402
from pdb_wizard.io import read_file  # noqa: E402
from pdb_wizard.menu_tools import cm_signature  # noqa: E402
from pdb_wizard.molecule import Molecule  # noqa: E402
from pdb_wizard.pbc import PBC  # noqa: E402
from pdb_wizard.reduce_cell import reduce_to_primitive  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
WOFOUR = ROOT / "WOFOUR-1-Ni.pqr"


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
    for d in w._live_windows():
        d.close()


@pytest.fixture
def hkust(window):
    assert window.open_path(str(HKUST))
    return window


def pixels(widget):
    from PySide6 import QtGui
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
    return np.frombuffer(bytes(img.constBits()), np.uint8).reshape(img.height(), img.width(), 4)


def differs(a, b, frac=0.0005):
    return a.shape != b.shape or (a != b).mean() > frac


def water_frames(n_frames=12):
    rng = np.random.default_rng(5)
    centres = rng.uniform(2, 13, (8, 3))
    for k in range(n_frames):
        rows = []
        for c in centres:
            o = c + [0.2 * k, 0, 0]
            for el, d in (("O", (0, 0, 0)), ("H", (0.96, 0, 0)), ("H", (-0.24, 0.93, 0))):
                rows.append((el, o + d))
        yield rows


def write_pdb_trajectory(path, n_frames=12):
    """Multi-model PDB with a 15 Å cubic cell."""
    with open(path, "w") as f:
        for k, rows in enumerate(water_frames(n_frames)):
            f.write("CRYST1   15.000   15.000   15.000  90.00  90.00  90.00 P 1           1\n")
            f.write(f"MODEL     {k + 1:4d}\n")
            for i, (el, x) in enumerate(rows):
                f.write(f"ATOM  {i + 1:5d} {el:<4s} HOH A{i // 3 + 1:4d}    "
                        f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {el:>2s}\n")
            f.write("ENDMDL\n")


def write_xyz_trajectory(path, n_frames=12):
    """Multi-frame XYZ: no cell."""
    with open(path, "w") as f:
        for rows in water_frames(n_frames):
            f.write(f"{len(rows)}\nframe\n")
            for el, x in rows:
                f.write(f"{el} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f}\n")


@pytest.fixture
def traj(window, tmp_path):
    path = tmp_path / "water.pdb"
    write_pdb_trajectory(path)
    assert window.open_path(str(path))
    window.load_task.wait()
    return window


# ---- tabs and trajectories --------------------------------------------------------------


def test_tabs_open_switch_and_close(hkust):
    w = hkust
    assert w.tab_bar.count() == 1 and w.tab_bar.tabText(0) == "HKUST-1.pdb"
    w.view.rotate(0.7, 0.2)
    rot = w.view._rot.copy()
    assert w.open_path(str(WOFOUR))
    assert w.tab_bar.count() == 2 and w.doc_index == 1
    n_wofour = len(w.mol.atoms)
    w.tab_bar.setCurrentIndex(0)
    assert len(w.mol.atoms) == 624 and np.allclose(w.view._rot, rot)  # camera kept per tab
    w.tab_bar.setCurrentIndex(1)
    w.close_tab(0)
    assert w.tab_bar.count() == 1 and len(w.mol.atoms) == n_wofour
    w.close_tab(0)
    assert w.mol is None and w.windowTitle() == "PDB Wizard — no structure"
    assert w.open_path(str(HKUST)) and w.tab_bar.count() == 1  # reuses the empty tab


def test_close_a_background_tab_keeps_the_current_one(hkust):
    w = hkust
    w.open_path(str(WOFOUR))
    w.open_path(str(HKUST))
    assert w.doc_index == 2
    w.close_tab(1)
    assert w.doc_index == 1 and w.tab_bar.count() == 2 and len(w.mol.atoms) == 624


def test_trajectory_loads_with_frame_bar(traj):
    w = traj
    assert len(w.frames) == 12 and w.frame_bar.isVisible()
    assert "(frame 1/12)" in w.windowTitle()
    w.frame_bar.go(4)
    assert w.mol is w.frames[4] and w.doc.frame_idx == 4
    assert w.mol.bonds is w.frames[0].bonds  # same topology: bonds reused
    assert w.info_labels["Frame"].text() == "5 / 12"
    w.frame_bar.buttons["next"].click()
    w.frame_bar.buttons["last"].click()
    assert w.doc.frame_idx == 11
    w.frame_bar.buttons["first"].click()
    assert w.doc.frame_idx == 0


def test_play_and_speed(traj, qtbot):
    bar = traj.frame_bar
    bar.set_speed(4)
    assert bar.speed_label.text() == "4x" and bar.timer.interval() == 25
    bar.set_speed(100)
    assert bar.speed == 16.0
    bar.set_speed(1)
    bar.toggle_play()
    assert bar.playing
    qtbot.waitUntil(lambda: traj.doc.frame_idx >= 2, timeout=3000)
    bar.toggle_play()
    assert not bar.playing
    assert traj.table_model.mol is traj.mol  # tables caught up when playback stopped


def test_frame_to_tab_and_trajectory_analyses(traj):
    w = traj
    w.frame_bar.go(3)
    d = w.frame_to_tab()
    assert d.name == "water_frame4" and w.doc_index == 1 and not w.frames
    assert w.mol is not d.frames and len(w.mol.atoms) == 24
    assert w.frame_to_tab() is None  # not a trajectory any more
    w.tab_bar.setCurrentIndex(0)
    dlg = w.open_tool("msd")
    assert dlg is not None and dlg.start_spin.maximum() == 12
    dlg.run().wait()
    assert dlg.result is not None and not dlg.result.empty


def test_trajectory_without_cell_warns(window, tmp_path):
    path = tmp_path / "nocell.xyz"
    write_xyz_trajectory(path, 3)
    window.open_path(str(path))
    window.load_task.wait()
    assert any("no cell information" in m for m in window.warnings)


def test_open_missing_file_and_drop_several(window, qtbot, tmp_path):
    assert not window.open_path(str(tmp_path / "nope.pdb"))
    assert "No such file" in window.warnings[-1]
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(HKUST)), QtCore.QUrl.fromLocalFile(str(WOFOUR))])
    from PySide6 import QtGui
    ev = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.CopyAction, mime,
                          QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    window.dropEvent(ev)
    assert window.tab_bar.count() == 2


# ---- edits ------------------------------------------------------------------------------


def test_sort_and_undo(hkust):
    w = hkust
    before = [a.element.symbol for a in w.mol.atoms]
    w.sort_atoms()
    assert len(w.mol.atoms) == 624 and "Sorted atoms" in w.statusBar().currentMessage()
    w.undo()
    assert [a.element.symbol for a in w.mol.atoms] == before


def water_box(window):
    rng = np.random.default_rng(1)
    atoms = []
    for c in rng.uniform(3, 12, (4, 3)):
        atoms += [Atom(*c, "O"), Atom(*(c + [0.96, 0, 0]), "H"), Atom(*(c + [0, 0.96, 0]), "H")]
    window.new_tab(Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90)), name="w")
    return window.mol


def test_delete_lone_atoms(window):
    mol = water_box(window)
    assert window.delete_lone_atoms() == 0 and not window.undo_action.isEnabled()
    assert "No lone atoms found" in window.statusBar().currentMessage()
    far = mol.atoms[0].x + [7.5, 7.5, 7.5]
    mol.atoms.append(Atom(*far, "Ar"))
    assert window.delete_lone_atoms() == 1 and len(window.mol.atoms) == 12
    window.undo()
    assert len(window.mol.atoms) == 13


def test_edit_h_distances(window):
    water_box(window)
    assert window.edit_h_distances("o", 1.05) == 8
    mol = window.mol
    assert np.linalg.norm(mol.atoms[1].x - mol.atoms[0].x) == pytest.approx(1.05)
    assert window.edit_h_distances("N", 1.0) == 0
    assert "No bonded N-H pairs" in window.statusBar().currentMessage()


def test_update_and_remove_cell(hkust):
    w = hkust
    assert w.update_cell((27.0, 27.0, 27.0, 90.0, 90.0, 90.0))
    assert w.mol.pbc.a == 27.0 and w.info_labels["a, b, c (Å)"].text().startswith("27.000")
    assert w.update_cell(None) and w.mol.pbc is None
    assert not w.update_cell(None)
    assert w.update_cell((20, 20, 20, 90, 90, 90)) and w.mol.pbc.c == 20
    w.undo()
    w.undo()
    assert w.mol.pbc.a == 27.0


def test_cell_dialog_values(qtbot):
    from pdb_wizard.gui_tools import CellDialog
    dlg = CellDialog(None)
    qtbot.addWidget(dlg)
    assert dlg.values() == (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    assert not dlg.remove.isEnabled()
    dlg = CellDialog(PBC(5, 6, 7, 80, 90, 100))
    qtbot.addWidget(dlg)
    dlg.remove.setChecked(True)
    assert dlg.values() is None and not dlg.spins[0].isEnabled()


def test_substitute_element(hkust):
    w = hkust
    assert w.substitute_element("Cu", "zn") == 48
    assert sum(a.element.symbol == "Zn" for a in w.mol.atoms) == 48
    assert w.substitute_element("Cu", "Zn") == 0
    assert w.substitute_element("O", "Qq") == 0 and "Unknown element" in w.warnings[-1]


def test_reduce_supercell(hkust, qtbot):
    w = hkust
    for spin in w.extend_spins:
        spin.setValue(2)
    w.build_supercell()
    w.sorbate_combo.setCurrentIndex(w.sorbate_combo.findData("h2o_dc97"))  # 4 sites
    w.sorbate_count.setValue(3)
    assert w.pack_sorbates() == 3
    dlg = ReduceCellDialog(w.mol, w)
    qtbot.addWidget(dlg)
    assert dlg.reduce() is None and "Analyze first" in dlg.status.text()
    dlg.analyze()
    assert dlg.table.rowCount() == 3 and "Sorbate molecules: 3" in dlg.status.text()
    dlg.table.item(1, 0).setCheckState(QtCore.Qt.Unchecked)
    assert dlg.keep() == [0, 2]
    expected = len(reduce_to_primitive(w.mol, 2, 2, 2, [0, 2]).atoms)
    dlg.reduce()
    assert w.reduce_supercell(dlg)
    assert len(w.mol.atoms) == expected == 624 + 2 * 4
    assert "kept 2/3 sorbate molecules" in w.statusBar().currentMessage()


def test_qeq_shows_the_report(window):
    window.open_path(str(WOFOUR))
    window.qeq_charges()
    report = window.dialogs[-1]
    assert report.windowTitle() == "QEq Charges"
    assert report.text.toPlainText() == charge_report(window.mol)


def test_edits_need_a_cell(window):
    window.new_tab(Molecule(atoms=[Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C")]), name="m")
    for fn in (window.sort_atoms, window.delete_lone_atoms, window.reduce_supercell,
               lambda: window.edit_h_distances("C", 1.0)):
        fn()
        assert "needs a unit cell" in window.warnings[-1]


# ---- redo -------------------------------------------------------------------------------


def history(w):
    """(Undo enabled, Redo enabled)."""
    return w.undo_action.isEnabled(), w.redo_action.isEnabled()


def zinc(w):
    return sum(a.element.symbol == "Zn" for a in w.mol.atoms)


def test_redo_sits_under_undo_with_its_shortcuts(window):
    w = window
    top = w.menuBar().actions()  # held: older PySide6 drops a menu whose action wrapper dies
    edit = next(a for a in top if a.text() == "&Edit").menu()
    first, second = edit.actions()[:2]
    assert first is w.undo_action and second is w.redo_action
    assert w.actions_by_name["Redo"] is w.redo_action and w.redo_action.text() == "Re&do"
    assert [s.toString() for s in w.redo_action.shortcuts()] == ["Ctrl+Shift+Z", "Ctrl+Y"]
    bar = w.findChild(QtWidgets.QToolBar, "main")
    tools = bar.actions()
    assert tools.index(w.redo_action) == tools.index(w.undo_action) + 1
    assert history(w) == (False, False)


def test_undo_then_redo_restores_the_edits_exactly(hkust):
    w = hkust
    states = [cm_signature(w.mol)]
    w.edit_coord(0, (1.25, 2.5, 3.75))
    states.append(cm_signature(w.mol))
    w.substitute_element("Cu", "Zn")
    states.append(cm_signature(w.mol))
    w.update_cell((27.0, 27.0, 27.0, 90.0, 90.0, 90.0))
    states.append(cm_signature(w.mol))
    assert history(w) == (True, False)
    for k in (2, 1, 0):
        w.undo()
        assert cm_signature(w.mol) == states[k]
    assert history(w) == (False, True)
    for k in (1, 2, 3):
        w.redo()
        assert cm_signature(w.mol) == states[k]
        assert w.statusBar().currentMessage() == "Redone"
    assert history(w) == (True, False)
    w.redo()  # nothing left to redo
    assert cm_signature(w.mol) == states[3]
    assert w.table_model.mol is w.mol and w.geometry.mol is w.mol
    assert w.info_labels["a, b, c (Å)"].text().startswith("27.000")
    w.undo()
    w.undo()
    w.redo()
    assert cm_signature(w.mol) == states[2] and history(w) == (True, True)


def test_redo_keys(hkust, qtbot):
    w = hkust
    w.substitute_element("Cu", "Zn")
    w.view.setFocus()
    ctrl = QtCore.Qt.ControlModifier
    for key, mods in ((QtCore.Qt.Key_Z, ctrl | QtCore.Qt.ShiftModifier), (QtCore.Qt.Key_Y, ctrl)):
        qtbot.keyClick(w.view, QtCore.Qt.Key_Z, ctrl)
        assert zinc(w) == 0
        qtbot.keyClick(w.view, key, mods)
        assert zinc(w) == 48, key


def test_tool_windows_follow_redo(hkust):
    w = hkust
    dlg = w.open_tool("coordination")
    w.substitute_element("Cu", "Zn")
    w.undo()
    assert dlg.mol is w.mol
    w.redo()
    items = [dlg.inputs["center"].itemText(i) for i in range(dlg.inputs["center"].count())]
    assert dlg.mol is w.mol and "Zn" in items and "Cu" not in items


def test_a_new_edit_clears_redo_and_a_no_op_keeps_it(hkust, tmp_path):
    w = hkust
    w.edit_coord(0, (1.0, 2.0, 3.0))
    moved = cm_signature(w.mol)
    w.substitute_element("Cu", "Zn")
    substituted = cm_signature(w.mol)
    w.undo()
    assert history(w) == (True, True) and len(w.doc.undo) == len(w.doc.redo) == 1
    # edits that find nothing to do drop their checkpoint: both histories stay as they were
    assert w.substitute_element("Xe", "Kr") == 0
    charges = tmp_path / "five.txt"
    charges.write_text("0.1\n0.2\n0.3\n0.4\n0.5\n")  # 624 atoms: does not divide
    assert not w.load_charges(str(charges))
    assert history(w) == (True, True) and len(w.doc.undo) == len(w.doc.redo) == 1
    w.redo()
    assert cm_signature(w.mol) == substituted
    w.undo()
    w.update_cell((27.0, 27.0, 27.0, 90.0, 90.0, 90.0))  # a real edit
    assert history(w) == (True, False) and w.doc.redo == []
    w.redo()
    assert w.mol.pbc.a == 27.0 and zinc(w) == 0
    w.undo()
    assert cm_signature(w.mol) == moved


def test_redo_is_per_tab(hkust):
    w = hkust
    w.substitute_element("Cu", "Zn")
    w.undo()
    assert history(w) == (False, True)
    assert w.open_path(str(WOFOUR)) and w.doc_index == 1  # a new tab: its own history
    assert history(w) == (False, False)
    wofour = cm_signature(w.mol)
    w.redo()  # the first tab's redo does not apply here
    assert cm_signature(w.mol) == wofour
    w.update_cell((30.0, 30.0, 30.0, 90.0, 90.0, 90.0))
    assert history(w) == (True, False)
    w.tab_bar.setCurrentIndex(0)
    assert history(w) == (False, True)
    w.redo()
    assert zinc(w) == 48 and history(w) == (True, False)
    w.undo()
    w.tab_bar.setCurrentIndex(1)
    assert history(w) == (True, False) and w.mol.pbc.a == 30.0
    assert w.open_path(str(HKUST)) and w.doc_index == 2
    w.substitute_element("Cu", "Zn")
    w.undo()
    w.tab_bar.setCurrentIndex(1)
    w.close_tab(0)  # a background tab: the shown one keeps its history
    assert w.doc.name == "WOFOUR-1-Ni.pqr" and history(w) == (True, False)
    w.close_tab(0)  # the shown tab: the next one's history takes over
    assert w.doc.name == "HKUST-1.pdb" and history(w) == (False, True)
    w.redo()
    assert zinc(w) == 48
    w.undo()
    w.close_tab(0)  # the last tab, with a redo pending: an empty tab with no history
    assert w.mol is None and history(w) == (False, False)
    assert w.open_path(str(HKUST)) and history(w) == (False, False)


def test_redo_on_a_trajectory_frame(traj):
    """Like undo, redo goes back to the frame that was edited."""
    w = traj
    w.set_frame(2)
    others = {k: cm_signature(w.frames[k]) for k in (0, 5, 7)}
    w.substitute_element("O", "N")
    edited = cm_signature(w.frames[2])
    w.set_frame(5)
    w.undo()
    assert w.doc.frame_idx == 2 and w.frame_bar.index == 2
    assert {a.element.symbol for a in w.frames[2].atoms} == {"O", "H"}
    w.set_frame(7)  # changing frames keeps the tab's history
    assert history(w) == (False, True)
    w.redo()
    assert w.doc.frame_idx == 2 and w.frame_bar.index == 2 and w.mol is w.frames[2]
    assert cm_signature(w.frames[2]) == edited and "(frame 3/12)" in w.windowTitle()
    assert {k: cm_signature(w.frames[k]) for k in others} == others
    assert history(w) == (True, False)
    w.set_frame(0)
    assert history(w) == (True, False)
    w.undo()
    assert w.doc.frame_idx == 2 and {a.element.symbol for a in w.mol.atoms} == {"O", "H"}
    assert cm_signature(w.frames[0]) == others[0]


# ---- geometry tables --------------------------------------------------------------------


def test_geometry_rows_for_every_tab():
    mol = read_file(str(HKUST))
    for tab in ("Bonds", "Angles", "Dihedrals", "Contacts", "Mols", "Lone", "Coords"):
        headers, rows = geometry_rows(mol, tab)
        assert headers and all(len(c) == len(headers) for c, _ in rows), tab
    _, bonds = geometry_rows(mol, "Bonds", ascending=True)
    lengths = [float(c[2]) for c, _ in bonds]
    assert lengths == sorted(lengths)
    assert geometry_rows(None, "Bonds") == ([], [])
    with pytest.raises(ValueError):
        geometry_rows(mol, "Nope")


def test_geometry_panel_highlights_sorts_and_edits(hkust):
    w = hkust
    w.geometry_dock.show()
    w.geometry_dock.raise_()
    panel = w.geometry
    assert panel.tables["Bonds"].rowCount() > 0
    panel.tables["Bonds"].setCurrentCell(3, 0)
    assert len(w.view.highlighted) == 2
    panel.sort_button.setChecked(True)
    col = [float(panel.tables["Bonds"].item(r, 2).text())
           for r in range(panel.tables["Bonds"].rowCount())]
    assert col == sorted(col)
    panel.tabs.setCurrentIndex(panel.tabs.indexOf(panel.tables["Coords"]))
    t = panel.tables["Coords"]
    assert t.rowCount() == 624 and not panel.sort_button.isEnabled()
    t.item(0, 2).setText("1.5")
    assert w.mol.atoms[0].x[0] == pytest.approx(1.5)
    t.item(0, 3).setText("oops")  # not a number: nothing moves, the old value comes back
    assert w.mol.atoms[0].x[1] != "oops" and t.item(0, 3).text() != "oops"
    w.undo()
    assert w.mol.atoms[0].x[0] != pytest.approx(1.5)


# ---- view -------------------------------------------------------------------------------


def test_styles_and_colour_modes_change_the_picture(hkust):
    v = hkust.view
    base = pixels(v)
    for key in ("licorice", "vdw"):
        hkust.style_combo.setCurrentIndex(hkust.style_combo.findData(key))
        assert v.style_name == key
        assert differs(base, pixels(v)), key
    hkust.style_combo.setCurrentIndex(hkust.style_combo.findData("ribbon"))
    hkust.set_style("cpk")
    base = pixels(v)
    hkust.set_style("ribbon")  # a MOF has no backbone: ball and stick, and a note why
    assert not differs(base, pixels(v))
    assert "No protein backbone" in hkust.statusBar().currentMessage()
    hkust.style_combo.setCurrentIndex(0)
    assert v.bond_radius == pytest.approx(0.11)
    hkust.set_style("licorice")
    assert v.bond_radius == pytest.approx(0.15) and hkust.sliders["bond_radius"].value() == 15
    hkust.set_style("cpk")
    base = pixels(v)
    for key, _ in gui.COLOR_MODES[1:]:
        hkust.color_combo.setCurrentIndex(hkust.color_combo.findData(key))
        if key != "chain":  # one connected framework: a single chain colour
            assert differs(base, pixels(v)), key


def test_ribbon_traces_a_protein_backbone(window, qtbot):
    atoms = []
    for k in range(12):  # a helix-like CA trace
        a = Atom(2.3 * np.cos(k), 2.3 * np.sin(k), 1.5 * k, "C")  # CA-CA ~2.7 Å
        a.name = "CA"
        atoms.append(a)
    window.new_tab(Molecule(atoms=atoms), name="protein")
    window.set_style("cpk")
    base = pixels(window.view)
    window.set_style("ribbon")
    assert gui._ca_chains(window.mol) == [list(range(12))]
    assert differs(base, pixels(window.view))


def test_view_options_change_the_picture(hkust):
    v = hkust.view
    for attr in ("show_atom_numbers", "show_polyhedra", "orthographic", "depth_cue",
                 "show_bonds", "show_cell"):
        base = pixels(v)
        hkust.view_checks[attr].toggle()
        assert differs(base, pixels(v)), attr
    for key in ("dark", "light"):
        hkust.bg_combo.setCurrentIndex(hkust.bg_combo.findData(key))
        corner = pixels(v)[2, 2, :3].mean()
        assert (corner > 200) if key == "light" else (corner < 60)
    for attr in ("ambient", "specular", "shininess", "atom_scale"):
        base = pixels(v)
        s = hkust.sliders[attr]
        s.setValue(s.minimum() if s.value() > s.minimum() + 5 else s.maximum())
        assert differs(base, pixels(v)), attr


def test_hide_water(hkust):
    w = hkust
    w.sorbate_combo.setCurrentIndex(w.sorbate_combo.findData("h2o_tip3p"))
    w.sorbate_count.setValue(4)
    w.pack_sorbates()
    w.view_checks["hide_water"].setChecked(True)
    pixels(w.view)
    assert (~w.view._visible).sum() == 12 and w.view._visible[:624].all()
    front = int(np.argmax(np.where(w.view._visible, -1e9, w.view._depth)))
    x, y = w.view._screen[front]
    hit = w.view.atom_at(x, y)
    assert hit is None or hit < 624  # hidden atoms cannot be picked


def test_measure_and_shift_click(hkust, qtbot):
    w, v = hkust, hkust.view
    pixels(v)
    w.view_checks["measure_mode"].setChecked(True)
    i, j = 0, 1
    assert "click another" in v.pick(i)
    msg = v.pick(j)
    d = w.mol.pbc.min_image(w.mol.atoms[i].x - w.mol.atoms[j].x)
    assert msg == f"Distance 1-2 = {d:.4f} Å (pinned)"
    assert v.measurements == [((0, 1), f"{d:.3f} Å")]
    base = pixels(v)
    v.clear_measurements()
    assert differs(base, pixels(v))
    w.view_checks["measure_mode"].setChecked(False)
    v.pick(0, shift=True)
    v.pick(1, shift=True)
    msg = v.pick(2, shift=True)
    assert msg.startswith("Angle 1-2-3 = ") and msg.endswith("°")
    assert v.angle(0, 1, 2) == pytest.approx(float(msg.split("= ")[1][:-1]), abs=0.01)
    front = int(np.argmax(v._depth))
    x, y = (int(c) for c in v._screen[front])
    qtbot.mouseClick(v, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier, QtCore.QPoint(x, y))
    assert w.statusBar().currentMessage().startswith(("Distance", "Angle", "Dihedral", "Atom"))


def same_view(a, b) -> bool:
    (rot_a, zoom_a, pan_a, centre_a), (rot_b, zoom_b, pan_b, centre_b) = a, b
    return (np.array_equal(rot_a, rot_b) and zoom_a == zoom_b and pan_a == pan_b
            and np.array_equal(centre_a, centre_b))


def double_click(qtbot, widget, pos, modifier=QtCore.Qt.NoModifier):
    """What Qt delivers for a double click: press, release, double click, release
    (qtbot.mouseDClick alone sends only the double-click event)."""
    left = QtCore.Qt.LeftButton
    qtbot.mousePress(widget, left, modifier, pos)
    qtbot.mouseRelease(widget, left, modifier, pos)
    qtbot.mouseDClick(widget, left, modifier, pos)
    qtbot.mouseRelease(widget, left, modifier, pos)


def test_double_click_is_a_click_and_keeps_the_view(hkust, qtbot):
    w, v = hkust, hkust.view
    v.rotate(0.6, -0.3)
    v.zoom(0.6)
    v._pan = QtCore.QPointF(25, -15)
    pixels(v)  # project the atoms where they are now drawn
    state = v.view_state()
    front = int(np.argmax(np.where(v._visible, v._depth, -np.inf)))
    on_atom = QtCore.QPoint(*(int(round(c)) for c in v._screen[front]))
    assert v.atom_at(on_atom.x(), on_atom.y()) == front and v.atom_at(3, 3) is None

    double_click(qtbot, v, on_atom)
    assert same_view(state, v.view_state())
    assert v.selected == front and v._drag_pos is None
    assert w.selection_label.text().startswith(f"#{front + 1} ")
    double_click(qtbot, v, QtCore.QPoint(3, 3))  # on the background: deselects
    assert same_view(state, v.view_state()) and v.selected is None and v._drag_pos is None

    double_click(qtbot, v, on_atom, QtCore.Qt.ShiftModifier)  # picks the atom once
    assert v.picked == [front] and same_view(state, v.view_state())
    assert w.statusBar().currentMessage().startswith(f"Atom {front + 1}:")
    w.view_checks["measure_mode"].setChecked(True)
    double_click(qtbot, v, on_atom)  # no zero-length distance pinned
    assert v.picked == [front] and v.measurements == []
    w.view_checks["measure_mode"].setChecked(False)

    # holding the second press and dragging rotates, as after any press
    left = QtCore.Qt.LeftButton
    qtbot.mousePress(v, left, pos=on_atom)
    qtbot.mouseRelease(v, left, pos=on_atom)
    qtbot.mouseDClick(v, left, pos=on_atom)
    assert v._drag_pos is not None
    qtbot.mouseMove(v, on_atom + QtCore.QPoint(40, 25))
    qtbot.mouseRelease(v, left, pos=on_atom + QtCore.QPoint(40, 25))
    assert not np.allclose(v._rot, state[0]) and v._drag_pos is None
    w.actions_by_name["Reset view"].trigger()  # resetting is still one key or click away
    assert not same_view(state, v.view_state())


def dihedral_chain(phi, start=(0.0, 0.0, 0.0)):
    """Atoms i-j-k-m with 1.5 Å bonds and 110° angles; signed dihedral ``phi`` degrees."""
    s, c = 1.5 * np.sin(np.radians(70)), 1.5 * np.cos(np.radians(70))
    p = np.radians(phi)
    pts = np.array([[-c, s, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                    [1.5 + c, s * np.cos(p), s * np.sin(p)]])
    return pts + np.asarray(start)


def chain_tab(window, pts, pbc=None):
    window.new_tab(Molecule(atoms=[Atom(*p, "C") for p in pts], pbc=pbc), name="chain")
    return window.view


@pytest.mark.parametrize("phi", [0, 60, 90, 135, 180, -60, -135])
def test_shift_click_four_atoms_gives_an_unsigned_dihedral(window, phi):
    v = chain_tab(window, dihedral_chain(phi))
    # (get_dihedrals' 1e-10 guard against zero normals moves 0 and 180 by ~1e-3 degrees)
    assert v.dihedral(0, 1, 2, 3) == pytest.approx(abs(phi), abs=0.005)
    assert v.dihedral(3, 2, 1, 0) == pytest.approx(abs(phi), abs=0.005)
    for i in range(3):
        v.pick(i, shift=True)
    assert v.pick(3, shift=True) == f"Dihedral 1-2-3-4 = {abs(phi):.2f}°"
    assert v.picked == [0, 1, 2, 3]


def test_fifth_shift_click_slides_and_all_four_are_marked(window):
    pts = dihedral_chain(60)
    v = chain_tab(window, np.vstack([pts, pts[3] + [1.5, 0.0, 0.0]]))
    for i in range(3):
        v.pick(i, shift=True)
    three = pixels(v)
    assert v.pick(3, shift=True).startswith("Dihedral 1-2-3-4 = 60.00°")
    four = pixels(v)
    assert differs(three, four)  # the fourth atom gets its ring too
    msg = v.pick(4, shift=True)
    assert v.picked == [1, 2, 3, 4]
    assert msg == f"Dihedral 2-3-4-5 = {v.dihedral(1, 2, 3, 4):.2f}°"
    assert differs(four, pixels(v))  # the first atom's ring goes, the fifth's comes
    v.pick(2, shift=True)  # an atom already picked is not added twice
    assert v.picked == [1, 2, 3, 4]
    window.set_molecule(window.mol)  # a new structure clears the picks
    assert v.picked == []


@pytest.mark.parametrize("cell", [(10, 10, 10, 90, 90, 90), (10, 11, 12, 80, 95, 110)])
def test_dihedral_uses_the_minimum_image(window, cell):
    pbc = PBC(*cell)
    pts = dihedral_chain(50, start=(-0.3, -0.2, 0.1))  # three of the atoms outside the cell
    frac = pts @ pbc.reciprocal_basis_matrix
    pts = (frac - np.floor(frac)) @ pbc.basis_matrix  # wrapped in: bonds cross the faces
    v = chain_tab(window, pts, pbc)
    assert v.dihedral(0, 1, 2, 3) == pytest.approx(50.0, abs=1e-6)
    assert v.pick(0, shift=True) and v.pick(1, shift=True) and v.pick(2, shift=True)
    assert v.pick(3, shift=True) == "Dihedral 1-2-3-4 = 50.00°"
    window.mol.bonds = [(0, 1), (1, 2), (2, 3)]
    assert window.mol.get_dihedrals() == [(0, 1, 2, 3, v.dihedral(0, 1, 2, 3))]
    window.mol.pbc = None
    assert abs(v.dihedral(0, 1, 2, 3) - 50.0) > 5  # without the cell it would be wrong


def test_dihedral_matches_the_geometry_panel(hkust):
    """The same number as Molecule.get_dihedrals and the Geometry panel's Dihedrals tab."""
    w, v = hkust, hkust.view
    mol = w.mol
    cell = mol.pbc
    for a in mol.atoms:  # shift and wrap each atom on its own: linkers straddle the faces
        frac = a.x @ cell.reciprocal_basis_matrix + 0.1
        a.x = (frac - np.floor(frac)) @ cell.basis_matrix
    mol.detect_bonds()
    _, rows = geometry_rows(mol, "Dihedrals")
    shown = {idx: cells[4] for cells, idx in rows}
    pos = np.array([a.x for a in mol.atoms])
    half = min(mol.pbc.a, mol.pbc.b, mol.pbc.c) / 2
    data = mol.get_dihedrals()
    across = 0
    for i, j, k, m, angle in data:
        assert v.dihedral(i, j, k, m) == angle
        assert v.dihedral(m, k, j, i) == pytest.approx(angle, abs=1e-9)
        assert f"{v.dihedral(i, j, k, m):.3f}" == shown[(i, j, k, m)]
        across += any(np.abs(pos[b] - pos[a]).max() > half for a, b in ((i, j), (j, k), (k, m)))
    assert len(data) > 100 and across > 0  # some of them span the periodic boundary
    i, j, k, m, angle = data[0]
    for idx in (i, j, k):
        v.pick(idx, shift=True)
    assert v.pick(m, shift=True) == f"Dihedral {i + 1}-{j + 1}-{k + 1}-{m + 1} = {angle:.2f}°"


def test_view_keys_match_the_terminal_viewer(hkust, qtbot):
    v = hkust.view
    v.setFocus()
    steps = [("v", "style_name", "licorice"), ("C", "color_mode", "charge"),
             ("#", "show_atom_numbers", True), ("w", "hide_water", True),
             ("P", "show_polyhedra", True), ("O", "orthographic", False),
             ("m", "measure_mode", True), ("t", "pan_mode", True)]
    for key, attr, value in steps:
        qtbot.keyClicks(v, key)
        assert getattr(v, attr) == value, key
    assert hkust.style_combo.currentData() == "licorice"  # the View tab follows the keys
    assert hkust.view_checks["measure_mode"].isChecked()
    pan = QtCore.QPointF(v._pan)
    qtbot.keyClick(v, QtCore.Qt.Key_Left)  # pan mode: arrows pan
    assert v._pan != pan
    qtbot.keyClicks(v, "c")
    assert v._pan == QtCore.QPointF(0, 0)
    rot = v._rot.copy()
    qtbot.keyClicks(v, ",")
    assert not np.allclose(rot, v._rot)
    qtbot.keyClicks(v, "i")
    assert v.background in ("light", "dark")
    zoom = v._zoom
    qtbot.keyClicks(v, "K")
    assert v._zoom > zoom
    qtbot.keyClicks(v, "J")
    assert v._zoom == pytest.approx(zoom)


def test_density_overlay(traj):
    w = traj
    w.density_3d("O").wait()
    assert w.view.density is not None and w.density_action.isEnabled()
    base = pixels(w.view)
    w.density_action.setChecked(False)
    assert differs(base, pixels(w.view))
    w.density_action.setChecked(True)
    w.set_density(None, None)
    assert w.view.density is None and not w.density_action.isEnabled()
    w.density_3d("Xe").wait()
    assert "No Xe atoms" in w.statusBar().currentMessage()


# ---- analysis windows -------------------------------------------------------------------


def test_analysis_menu_entries_open(hkust):
    for key in ("rdf", "coordination", "hbonds", "density", "pxrd", "isotherm_plot"):
        dlg = hkust.open_tool(key)
        assert isinstance(dlg, AnalysisDialog), key
    dlg = hkust.open_tool("pore_size")  # nothing to set: computes straight away
    dlg.task.wait()
    assert dlg.result.summary.startswith("Dominant pore diameter")
    assert hkust.open_tool("msd") is None
    assert hkust.warnings[-1] == "Need a trajectory with 3+ frames for MSD"
    assert isinstance(hkust.open_tool("input_generator"), InputGeneratorDialog)
    assert isinstance(hkust.open_tool("db_search"), DatabaseDialog)


def test_analysis_dialog_exports_and_options(hkust, tmp_path):
    dlg = hkust.open_tool("rdf")
    assert dlg.export_csv(str(tmp_path / "none.csv")) is None  # nothing computed yet
    dlg.inputs["el1"].setCurrentText("Cu")
    assert dlg.inputs["name1"].count() >= 2  # names follow the element
    dlg.inputs["el2"].setCurrentText("O")
    dlg.run().wait()
    res = dlg.result
    assert res.title == "g(r) Cu-O" and dlg.csv_button.isEnabled()
    out = dlg.export_csv(str(tmp_path / "rdf.csv"))
    lines = Path(out).read_text().splitlines()
    assert lines[0] == "r(A),g(r)" and len(lines) == len(res.x) + 1
    assert dlg.export_png(str(tmp_path / "rdf.png")) and (tmp_path / "rdf.png").stat().st_size
    dlg.inputs["ymax"].setValue(40)
    assert dlg.plot.y_max == 40
    dlg.inputs["el1"].setCurrentText("H")
    dlg.inputs["el2"].setCurrentText("H")
    dlg.inputs["xmax"].setValue(2)
    dlg.run().wait()
    assert dlg.result.summary.startswith("max g(r)") or "pairs found" in dlg.result.summary


def test_pxrd_experimental_overlay(hkust, tmp_path):
    dlg = hkust.open_tool("pxrd")
    dlg.run().wait()
    exp = tmp_path / "exp.csv"
    exp.write_text("two_theta,intensity\n6,50\n9,100\n")
    assert dlg.load_experimental(str(exp))
    assert dlg.plot.overlay is not None and "2 points" in dlg.status.text()
    bad = tmp_path / "bad.xy"
    bad.write_text("x\n")
    assert not dlg.load_experimental(str(bad)) and "Error loading" in dlg.status.text()


def test_analysis_failure_is_reported(qtbot, monkeypatch):
    from pdb_wizard import gui_tools
    spec = gui_tools.ANALYSES["pore_size"]

    def boom(*a):
        raise RuntimeError("kaput")

    monkeypatch.setattr(spec, "run", boom)
    dlg = AnalysisDialog("pore_size", read_file(str(HKUST)))
    qtbot.addWidget(dlg)
    dlg.run().wait()
    assert dlg.status.text() == "Error: kaput" and dlg.compute_button.isEnabled()


def test_plot_zoom_hover_and_overlay(qtbot):
    plot = QtPlot()
    qtbot.addWidget(plot)
    plot.resize(600, 400)
    plot.show()
    x = np.linspace(0, 10, 101)
    plot.set_data(x, np.sin(x), "t", "x")
    plot.grab()
    plot.zoom_x(2.0)
    assert plot.limits()[1] - plot.limits()[0] == pytest.approx(5.0)
    qtbot.keyClick(plot, QtCore.Qt.Key_R)
    assert plot.x_range is None
    qtbot.keyClick(plot, QtCore.Qt.Key_Plus)
    qtbot.keyClick(plot, QtCore.Qt.Key_Minus)
    qtbot.keyClick(plot, QtCore.Qt.Key_Minus)
    assert plot.x_range is None  # zoomed back out past everything
    qtbot.mouseMove(plot, QtCore.QPoint(300, 200))
    assert plot._hover is not None
    plot.set_overlay(x, np.cos(x), "exp")
    plot.grab()
    plot.y_min = None
    assert plot.limits()[2] == pytest.approx(-1.0, abs=0.01)
    plot.clear()
    assert plot.limits() == (0.0, 1.0, 0.0, 1.0)
    plot.grab()


def test_energy_plot_from_the_window(hkust, tmp_path):
    shutil.copy(HKUST, tmp_path / "s.pdb")
    (tmp_path / "energy.dat").write_text(
        "\n".join(f"{s} {-5 - s:.2f} 0 0 0 0 0 0 {s % 4} 0 0 0" for s in range(1, 21)) + "\n")
    hkust.open_path(str(tmp_path / "s.pdb"))
    dlg = hkust.open_tool("energy_plot")
    assert dlg.column.currentText() == "N"
    dlg.column.setCurrentText("energy")
    dlg.start.setValue(5)
    assert dlg.plot.title == "energy (steps 5-20)"
    out = dlg.export_csv(str(tmp_path / "e.csv"))
    assert len(Path(out).read_text().splitlines()) == 17
    assert "Exported steps 5-20" in dlg.status.text()
    empty = tmp_path / "empty.dat"
    empty.write_text("# nothing\n")
    assert hkust.energy_plot(str(empty)) is None and "Could not read" in hkust.warnings[-1]


def test_void_and_surface_reports(hkust):
    hkust.open_tool("void_vol").wait()
    assert hkust.dialogs[-1].windowTitle() == "Void Volume"
    hkust.open_tool("surface_area").wait()
    assert "Surface area:" in hkust.dialogs[-1].text.toPlainText()
    hkust.open_tool("info")
    assert hkust.dialogs[-1].text.toPlainText().startswith("Filename:")


def test_background_task_failure_warns(hkust):
    hkust._busy("Doing it...", lambda: 1 / 0, lambda r: None).wait()
    assert hkust.warnings[-1].startswith("Doing it failed:")


# ---- input generator --------------------------------------------------------------------


def test_input_generator_writes_the_same_files_as_the_shared_generator(hkust, tmp_path):
    dlg = hkust.open_tool("input_generator")
    assert dlg.engine == "mpmc_isotherm" and dlg.status.text()
    for engine in ("cp2k_cell_opt", "cp2k_dft_md", "openmm_npt", "openmm_nvt"):
        dlg.engine_box.setCurrentIndex(dlg.engine_box.findData(engine))
        assert dlg.engine == engine and dlg.status.text() == "Edit values, then Generate."
        dlg.outdir.setText(str(tmp_path / "gui" / engine))
        out = dlg.generate()
        assert out.ok, out.status
        ref = generate_engine_inputs(hkust.mol, engine, dlg.values(),
                                     str(tmp_path / "ref" / engine))
        for f in (tmp_path / "ref" / engine).iterdir():
            assert (tmp_path / "gui" / engine / f.name).read_text() == f.read_text()
        assert ref.ok
    dlg.show_template()
    assert dlg.preview.toPlainText().startswith("-- openmm_nvt.py --")
    dlg.validate()  # OpenMM's stock force field has no templates for a MOF: a caution
    assert dlg.status.text() == "1 warning — review then Generate"
    assert "No template found for residue" in dlg.preview.toPlainText()


def test_input_generator_keeps_values_per_engine_and_rejects_bad_input(hkust):
    dlg = InputGeneratorDialog(hkust.mol, str(HKUST), hkust)
    dlg.widgets["temperature"].setText("298")
    dlg.set_engine("cp2k_cell_opt")
    dlg.widgets["cutoff_ry"].setText("")
    assert dlg.generate() is None and dlg.status.text().startswith("Invalid input:")
    dlg.set_engine("mpmc_isotherm")
    assert float(dlg.widgets["temperature"].text()) == 298.0
    assert dlg.outdir.text().endswith("isotherm")
    dlg.widgets["pressures"].setText("0.1, 1")
    assert dlg.values()["pressures"] == [0.1, 1.0]
    dlg.close()


# ---- database ---------------------------------------------------------------------------


@pytest.fixture
def fake_backends(monkeypatch, tmp_path):
    def search(code):
        return lambda q, mode="text": [{"source": code, "id": f"{code}1", "formula": "Cu O",
                                        "title": f"{q} ({mode})"}]

    def fetch(rid, out_dir="."):
        path = Path(out_dir) / f"{rid}.pdb"
        shutil.copy(HKUST, path)
        return str(path)

    for code in database.BACKENDS:
        monkeypatch.setitem(database.BACKENDS[code], "search", search(code))
        monkeypatch.setitem(database.BACKENDS[code], "fetch", fetch)


def test_database_search_and_fetch(hkust, fake_backends):
    dlg = hkust.open_tool("db_search")
    assert dlg.mode.count() == 2  # all databases: text and formula
    assert dlg.search() is None  # empty query
    dlg.query.setText("cuprite")
    dlg.search().wait()
    # text search goes to the three databases that have it (not Materials Project)
    assert dlg.table.rowCount() == 3 and "3 results" in dlg.status.text()
    dlg.table.setCurrentCell(0, 0)
    assert "<b>COD</b> cod1" in dlg.preview.text()
    assert dlg.fetch_by_id() is None and "Pick a specific database" in dlg.status.text()
    dlg.fetch_selected().wait()
    assert hkust.tab_bar.count() == 2 and hkust.doc.name == "cod1.pdb"
    assert "Opened COD cod1 (624 atoms)" in hkust.statusBar().currentMessage()
    cache = hkust._db_cache
    hkust.close()
    assert not Path(cache).exists()  # downloads are cleaned up with the window


def test_database_modes_and_ids(window, fake_backends):
    dlg = window.open_tool("db_search")
    dlg.source.setCurrentIndex(dlg.source.findData("mp"))
    assert [dlg.mode.itemData(i) for i in range(dlg.mode.count())] == ["formula", "id"]
    assert dlg.query.placeholderText() == "Si O2"
    dlg.mode.setCurrentIndex(dlg.mode.findData("id"))
    assert dlg.search() is None and "Enter an ID" in dlg.status.text()
    dlg.query.setText("mp-149")
    dlg.search().wait()  # ID mode fetches directly
    assert window.doc.name == "mp-149.pdb"
    dlg = window.open_tool("db_search")
    assert dlg.fetch_selected() is None and "Select a result first" in dlg.status.text()
    dlg.source.setCurrentIndex(dlg.source.findData("rcsb"))
    assert dlg.fetch_by_id() is None and "Enter an ID" in dlg.status.text()
    assert dlg._fetch("nope", "x") is None and "Unknown source" in dlg.status.text()


# ---- exports ----------------------------------------------------------------------------


def test_export_gif(hkust, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    out = hkust.export_gif(str(tmp_path / "spin.gif"), n_frames=6)
    with Image.open(out) as img:
        assert img.n_frames == 6 and img.info["duration"] == 80
    rot = hkust.view._rot.copy()
    hkust.export_gif(str(tmp_path / "again.gif"), n_frames=2)
    assert np.allclose(rot, hkust.view._rot)  # the view is left as it was


def test_export_gif_without_pillow(hkust, tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pil(name, *a, **k):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    assert hkust.export_gif(str(tmp_path / "x.gif")) is None
    assert hkust.warnings[-1] == "Pillow required: pip install Pillow"


def test_gui_task_reports_errors(qtbot):
    seen = []
    GuiTask(lambda: 1 / 0, seen.append, seen.append).wait()
    assert isinstance(seen[0], ZeroDivisionError)


def test_menus_list_every_tool(window):
    names = set(window.actions_by_name)
    for text in ("Radial distribution", "Powder XRD", "3D density map", "Energy plot (energy.dat)",
                 "Input generator", "Fetch from database", "Substitute element",
                 "Keyboard and mouse", "Bonds", "Measure mode",
                 "Update unit cell", "Reduce supercell", "Edit H distances", "Delete lone atoms",
                 "Sort atoms", "Open frame in new tab", "Export rotation GIF", "Cycle style",
                 "Cycle colour mode", "Clear measurements", "Show 3D density"):
        assert text in names, text
    window.actions_by_name["Cycle style"].trigger()
    assert window.view.style_name == "licorice" and window.style_combo.currentData() == "licorice"
    window.actions_by_name["Build supercell"].trigger()
    assert window.tools.currentIndex() == 1
    window.actions_by_name["Pack sorbates"].trigger()
    assert window.tools.currentIndex() == 2
    window.actions_by_name["Force field and charges"].trigger()
    assert window.tools.currentIndex() == 3


def test_no_menu_entry_is_listed_twice(window):
    """Each command lives in one menu, and no menu reuses an Alt-key letter."""
    seen = {}
    for top in window.menuBar().actions():
        letters = {}
        for act in top.menu().actions():
            if act.isSeparator():
                continue
            text = act.text().split("\t")[0]
            plain = text.replace("&", "").rstrip("…").lower()
            assert plain not in seen, f"{plain!r} is in both {seen.get(plain)} and {top.text()}"
            seen[plain] = top.text()
            if "&" in text:
                letter = text[text.index("&") + 1].lower()
                assert letter not in letters, (top.text(), text, letters[letter])
                letters[letter] = text
    assert "generate qeq charges" in seen and "qeq charges" not in seen


def test_window_opens_on_general_defaults(window):
    """The first thing shown is the View tab, and the sorbate default is the group's
    standard H2 model, not one user's water model."""
    assert window.tools.currentIndex() == 0
    assert window.sorbate_combo.currentData() == "h2_bss"
    assert [s.value() for s in window.extend_spins] == [1, 1, 1]


def test_supercell_counts_are_whole_cells(hkust):
    w = hkust
    assert "1 × 1 × 1 cells, 624 atoms (no change)" in w.supercell_preview.text()
    w.build_supercell()
    assert len(w.mol.atoms) == 624  # 1×1×1 does nothing
    w.extend_spins[0].setValue(2)
    w.extend_spins[2].setValue(3)
    assert "2 × 1 × 3 cells, 3,744 atoms" in w.supercell_preview.text()
    w.build_supercell()
    assert len(w.mol.atoms) == 3744
    assert w.mol.pbc.a == pytest.approx(2 * 26.343, abs=1e-3)
    assert w.mol.pbc.c == pytest.approx(3 * 26.343, abs=1e-3)
    assert [s.value() for s in w.extend_spins] == [1, 1, 1]  # ready for the next build
    assert "3,744 atoms (no change)" in w.supercell_preview.text()
    w.undo()
    assert "624 atoms" in w.supercell_preview.text()


def test_supercell_preview_without_a_cell(window):
    assert "unit cell" in window.supercell_preview.text()


def test_view_menu_toggles_stay_in_step_with_the_view_tab(hkust, qtbot):
    w = hkust
    for attr, act in w.view_actions.items():
        assert act.isCheckable() and act.isChecked() == bool(getattr(w.view, attr)), attr
    w.view_actions["show_bonds"].trigger()
    assert not w.view.show_bonds and not w.view_checks["show_bonds"].isChecked()
    w.view_checks["show_bonds"].setChecked(True)
    w._sync_view_controls()
    assert w.view.show_bonds and w.view_actions["show_bonds"].isChecked()
    w.view.setFocus()
    qtbot.keyClick(w.view, "w")  # the view's own key
    assert w.view.hide_water and w.view_actions["hide_water"].isChecked()
    assert w.view_checks["hide_water"].isChecked()
    w.view_actions["measure_mode"].trigger()
    assert w.view.measure_mode and w.view_checks["measure_mode"].isChecked()


def test_controls_help_matches_the_view(hkust, qtbot):
    """Every key the help lists does something, and the help is reachable from the menu."""
    from pdb_wizard.gui import CONTROLS_HELP
    dlg = hkust.actions_by_name["Keyboard and mouse"]
    assert dlg.shortcut().toString() == "F1"
    report = hkust.show_controls()
    assert report.text.toPlainText().strip() == CONTROLS_HELP.strip()
    assert "Double click" not in CONTROLS_HELP  # it is an ordinary click now
    assert "3 an angle, 4 a dihedral" in CONTROLS_HELP
    assert f"{hkust.undo_action.shortcut().toString()} undo" in CONTROLS_HELP
    redo_keys = " or ".join(s.toString() for s in hkust.redo_action.shortcuts())
    assert f"\n  Ctrl+Z undo    {redo_keys} redo\n" in CONTROLS_HELP
    v = hkust.view
    for key, attr in (("b", "show_bonds"), ("#", "show_atom_numbers"), ("w", "hide_water"),
                      ("P", "show_polyhedra"), ("O", "orthographic"), ("t", "pan_mode"),
                      ("m", "measure_mode")):
        assert f"\n  {key}  " in CONTROLS_HELP, key
        before = getattr(v, attr)
        qtbot.keyClicks(v, key)
        assert getattr(v, attr) != before, key
    style = v.style_name
    qtbot.keyClicks(v, "v")
    assert v.style_name != style


def test_analysis_form_keeps_pairs_together(qtbot):
    """Element and name of one atom type share a row; the frame range has its own."""
    from pdb_wizard.atom import Atom
    from pdb_wizard.molecule import Molecule
    from pdb_wizard.pbc import PBC
    frames = [Molecule(atoms=[Atom(0, 0, f * 0.1, "O"), Atom(1, 0, 0, "H")],
                       pbc=PBC(10, 10, 10, 90, 90, 90)) for f in range(4)]
    dlg = AnalysisDialog("rdf", frames[0], frames)
    qtbot.addWidget(dlg)
    grid = dlg.layout().itemAt(0).layout()

    def row_of(w):
        return grid.getItemPosition(grid.indexOf(w))[0]

    assert row_of(dlg.inputs["el1"]) == row_of(dlg.inputs["name1"])
    assert row_of(dlg.inputs["el2"]) == row_of(dlg.inputs["name2"])
    assert row_of(dlg.inputs["el1"]) != row_of(dlg.inputs["el2"])
    assert row_of(dlg.start_spin) == row_of(dlg.end_spin)
    assert row_of(dlg.start_spin) > max(row_of(w) for w in dlg.inputs.values())


def test_input_generator_names_the_sorbates(hkust, qtbot):
    from pdb_wizard.gui_tools import InputGeneratorDialog
    from pdb_wizard.sorbates import SORBATE_MODELS
    dlg = InputGeneratorDialog(hkust.mol, "h.pdb", hkust)
    qtbot.addWidget(dlg)
    box = dlg.widgets["sorbate"]
    assert box.count() == len(SORBATE_MODELS) and box.currentData() == "h2_bss"
    assert SORBATE_MODELS["h2_bss"].description in box.currentText()
    box.setCurrentIndex(box.findData("co2_trappe"))
    assert dlg.values()["sorbate"] == "co2_trappe"
    assert dlg.values()["ensemble"] == "uvt"
    assert QtWidgets.QApplication.instance() is not None
