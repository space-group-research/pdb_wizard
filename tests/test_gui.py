"""Tests for the desktop GUI (pdb_wizard.gui), run headless with Qt's offscreen platform.

Needs PySide6 and pytest-qt; skipped otherwise. The tests drive the real window: they
open files, press the tool buttons' slots, drag and click in the 3D view, and check both
the window's state and the files it writes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtGui  # noqa: E402

from pdb_wizard import gui  # noqa: E402
from pdb_wizard.io import read_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
WOFOUR = ROOT / "WOFOUR-1-Ni.pqr"


@pytest.fixture
def window(qtbot):
    w = gui.MainWindow()
    w.warnings = []
    w._warn = w.warnings.append  # no modal dialogs in tests
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)
    yield w
    for d in w._live_windows():  # tool and report windows the test opened
        d.close()


@pytest.fixture
def hkust(window):
    assert window.open_path(str(HKUST))
    return window


def movable_ids(path):
    return [
        ln.split()[5]
        for ln in Path(path).read_text().splitlines()
        if ln.startswith("ATOM") and ln.split()[4] == "M"
    ]


def image(widget):
    return widget.grab().toImage()


def image_array(widget):
    img = image(widget).convertToFormat(QtGui.QImage.Format_RGB32)
    ptr = img.constBits()
    return np.frombuffer(ptr, np.uint8, img.sizeInBytes()).reshape(img.height(), -1).copy()


# ---- opening and showing ----------------------------------------------------------------


def test_empty_window_shows_a_hint(window):
    assert window.mol is None
    assert "no structure" in window.windowTitle()
    arr = image_array(window.view)
    assert arr.std() > 0  # background gradient + hint text drawn


def test_open_fills_info_table_and_title(hkust):
    labels = hkust.info_labels
    assert labels["Atoms"].text() == str(len(hkust.mol.atoms)) == "624"
    assert "Cu<sub>48</sub>" in labels["Formula"].text()
    assert labels["a, b, c (Å)"].text().startswith("26.343")
    assert hkust.table_model.rowCount() == 624
    assert hkust.legend.count() == 4  # C, O, H, Cu
    assert "HKUST-1.pdb" in hkust.windowTitle()


def test_open_bad_file_warns(window, tmp_path):
    bad = tmp_path / "bad.cif"
    bad.write_text("not a cif at all\n")
    assert not window.open_path(str(bad)) or window.mol is not None
    missing = tmp_path / "missing.pdb"
    assert not window.open_path(str(missing))
    assert window.warnings and "missing.pdb" in window.warnings[-1]


def test_view_draws_the_structure_and_rotation_changes_it(hkust):
    before = image_array(hkust.view)
    assert before.std() > 5
    hkust.view.rotate(0.6, 0.2)
    after = image_array(hkust.view)
    assert (before != after).mean() > 0.01


def test_reset_view_centres_the_cell(hkust):
    v = hkust.view
    v.rotate(1.0, 0.5)
    v.reset_view()
    corners = v._project(v.cell_corners())[0]
    centre = corners.mean(axis=0)
    assert abs(centre[0] - v.width() / 2) < 2 and abs(centre[1] - v.height() / 2) < 2
    assert corners[:, 0].min() >= 0 and corners[:, 0].max() <= v.width()


def test_cell_box_sits_on_a_centred_structure(hkust):
    """HKUST-1.pdb is stored around the origin; the box must be drawn there too."""
    frac = np.array([a.x for a in hkust.mol.atoms]) @ hkust.mol.pbc.reciprocal_basis_matrix
    corners = hkust.view.cell_corners() @ hkust.mol.pbc.reciprocal_basis_matrix
    assert np.allclose(corners.min(axis=0), np.floor(frac.min(axis=0) * 2) / 2, atol=0.51)
    assert corners.min() <= frac.min() and corners.max() >= frac.max()


def test_picking_hits_the_front_atom(hkust, qtbot):
    v = hkust.view
    image(v)  # paint once so the projection is cached
    front = int(np.argmax(v._depth))
    x, y = v._screen[front]
    assert v.atom_at(x, y) == front
    qtbot.mouseClick(v, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(x), int(y)))
    assert v.selected == front
    assert f"#{front + 1}" in hkust.selection_label.text()
    assert v.atom_at(-50, -50) is None


def test_mouse_drag_rotates_and_wheel_zooms(hkust, qtbot):
    v = hkust.view
    rot = v._rot.copy()
    c = v.rect().center()
    qtbot.mousePress(v, QtCore.Qt.LeftButton, pos=c)
    qtbot.mouseMove(v, c + QtCore.QPoint(40, 10))
    qtbot.mouseRelease(v, QtCore.Qt.LeftButton, pos=c + QtCore.QPoint(40, 10))
    assert not np.allclose(rot, v._rot)
    zoom = v._zoom
    v.zoom(1.5)
    assert v._zoom == pytest.approx(zoom * 1.5)
    qtbot.keyClick(v, QtCore.Qt.Key_R)
    assert v._zoom == pytest.approx(zoom, rel=1e-6)


def test_view_toggles(hkust):
    v = hkust.view
    base = image_array(v)
    hkust.chk_bonds.setChecked(False)
    no_bonds = image_array(v)
    assert (base != no_bonds).mean() > 0.001
    hkust.color_combo.setCurrentIndex(hkust.color_combo.findData("charge"))
    assert v.color_mode == "charge"
    hkust.size_slider.setValue(150)
    assert v.atom_scale == pytest.approx(1.5)
    hkust.chk_cell.setChecked(False)
    assert not v.show_cell


# ---- tools ------------------------------------------------------------------------------


def test_pack_sorbates_adds_movable_waters_and_undo(hkust):
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("h2o_dc97"))
    hkust.sorbate_count.setValue(5)
    assert hkust.pack_sorbates() == 5
    moved = [a for a in hkust.mol.atoms if a.mpmc_flag == "M"]
    assert len(moved) == 20 and sorted({a.mpmc_mol for a in moved}) == [2, 3, 4, 5, 6]
    assert hkust.info_labels["Sorbates"].text() == "5"
    assert "Cu<sub>48</sub>" in hkust.info_labels["Formula"].text()
    assert "X" not in hkust.info_labels["Formula"].text()  # model sites are not "X atoms"
    hkust.undo()
    assert len(hkust.mol.atoms) == 624 and not hkust.undo_action.isEnabled()


def test_pack_then_save_mpmc_keeps_them_movable(hkust, tmp_path):
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("h2o_dc97"))
    hkust.sorbate_count.setValue(4)
    hkust.pack_sorbates()
    out = tmp_path / "packed.pqr"
    assert hkust.save_path(str(out))
    ids = movable_ids(out)
    assert sorted(set(ids)) == ["2", "3", "4", "5"] and len(ids) == 16
    text = out.read_text()
    assert all(ln.split()[3] == "=" for ln in text.splitlines() if ln.startswith("REMARK BOX"))
    again = read_file(str(out))
    assert sum(a.mpmc_flag == "M" for a in again.atoms) == 16


def test_pack_more_continues_molecule_ids(hkust):
    hkust.sorbate_count.setValue(2)
    hkust.pack_sorbates()
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("co2_trappe"))
    hkust.pack_sorbates()
    ids = sorted({a.mpmc_mol for a in hkust.mol.atoms if a.mpmc_flag == "M"})
    assert ids == [2, 3, 4, 5]


def test_pack_fill_and_template_file(hkust, tmp_path):
    tpl = tmp_path / "co2.pqr"
    tpl.write_text(
        "ATOM 1 C CO2 M 2 0.0 0.0 0.0 12.011 0.70 0.0 27.0 2.80\n"
        "ATOM 2 O CO2 M 2 1.16 0.0 0.0 15.999 -0.35 0.0 79.0 3.05\n"
        "ATOM 3 O CO2 M 2 -1.16 0.0 0.0 15.999 -0.35 0.0 79.0 3.05\n"
    )
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("__file__"))
    hkust.sorbate_count.setValue(3)
    assert hkust.pack_sorbates(model_path=str(tpl)) == 3
    assert {a.mpmc_res for a in hkust.mol.atoms if a.mpmc_flag == "M"} == {"CO2"}
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("he"))
    hkust.sorbate_fill.setChecked(True)
    assert not hkust.sorbate_count.isEnabled()
    assert hkust.pack_sorbates() > 20


def test_pack_without_room_warns(window):
    from pdb_wizard.atom import Atom
    from pdb_wizard.molecule import Molecule
    from pdb_wizard.pbc import PBC

    pts = np.array(np.meshgrid(*[np.arange(0, 6, 1.5)] * 3)).reshape(3, -1).T
    atoms = []
    for x, y, z in pts:
        a = Atom(x, y, z, "C")
        a.sigma = 3.4
        atoms.append(a)
    window.set_molecule(Molecule(atoms=atoms, pbc=PBC(6, 6, 6, 90, 90, 90)))
    assert window.pack_sorbates() == 0
    assert "No room" in window.warnings[-1]


def test_force_field_leaves_sorbates_alone(hkust):
    hkust.sorbate_count.setValue(3)
    hkust.pack_sorbates()
    before = [
        (a.charge, a.alpha, a.epsilon, a.sigma) for a in hkust.mol.atoms if a.mpmc_flag == "M"
    ]
    hkust.ff_combo.setCurrentIndex(hkust.ff_combo.findData(0))
    hkust.apply_force_field()
    after = [(a.charge, a.alpha, a.epsilon, a.sigma) for a in hkust.mol.atoms if a.mpmc_flag == "M"]
    assert after == before
    cu = next(a for a in hkust.mol.atoms if a.element.symbol == "Cu")
    assert cu.epsilon > 0
    assert not hkust.warnings  # every framework element is in OPLS-AA/UFF


def test_charges_file_framework_only(hkust, tmp_path):
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("h2o_dc97"))
    hkust.sorbate_count.setValue(2)
    hkust.pack_sorbates()
    q = tmp_path / "q.txt"
    q.write_text("".join("0.0100\n" for _ in range(624)))
    assert hkust.load_charges(str(q))
    assert {a.charge for a in hkust.mol.atoms if a.mpmc_flag != "M"} == {0.01}
    assert -1.038 in {a.charge for a in hkust.mol.atoms if a.mpmc_flag == "M"}
    bad = tmp_path / "bad.txt"
    bad.write_text("0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n0.7\n")
    assert not hkust.load_charges(str(bad))
    assert "framework atom count" in hkust.warnings[-1]


def test_supercell_and_wrap_with_undo(hkust):
    hkust.extend_spins[0].setValue(2)
    hkust.build_supercell()
    assert len(hkust.mol.atoms) == 1248
    assert hkust.mol.pbc.a == pytest.approx(2 * 26.343, abs=1e-3)
    hkust.wrap(forward=True)
    frac = np.array([a.x for a in hkust.mol.atoms]) @ hkust.mol.pbc.reciprocal_basis_matrix
    assert frac.min() >= -1e-9 and frac.max() < 1 + 1e-9
    hkust.undo()
    hkust.undo()
    assert len(hkust.mol.atoms) == 624


def test_supercell_needs_a_count(hkust):
    hkust.build_supercell()
    assert len(hkust.mol.atoms) == 624


def test_tools_without_a_structure_warn(window):
    window.build_supercell()
    window.pack_sorbates()
    window.apply_force_field()
    window.qeq_charges()
    assert len(window.warnings) == 4


@pytest.mark.parametrize("suffix", [".pqr", ".pdb", ".xyz", ".cif", ".vasp", ".lmp"])
def test_save_every_format_and_read_back(hkust, tmp_path, suffix):
    out = tmp_path / f"out{suffix}"
    assert hkust.save_path(str(out)) and out.stat().st_size > 1000
    if suffix != ".lmp":
        assert len(read_file(str(out)).atoms) == 624


def test_save_unknown_suffix_warns(hkust, tmp_path):
    assert not hkust.save_path(str(tmp_path / "x.unknown"))
    assert "unknown output format" in hkust.warnings[-1]


def test_export_image(hkust, tmp_path):
    out = hkust.export_image(str(tmp_path / "view.png"))
    img = QtGui.QImage(out)
    assert not img.isNull() and img.width() == hkust.view.width()


def test_qeq_on_a_small_framework(window, tmp_path):
    assert window.open_path(str(WOFOUR))
    window.qeq_charges()
    assert abs(sum(a.charge for a in window.mol.atoms)) < 1e-6


def test_table_shows_sorbate_rows(hkust):
    hkust.sorbate_count.setValue(1)
    hkust.pack_sorbates()
    m = hkust.table_model
    last = m.rowCount() - 1
    assert m.data(m.index(last, 3), QtCore.Qt.DisplayRole).startswith("M 2")
    assert m.data(m.index(0, 3), QtCore.Qt.DisplayRole) == "framework"
    assert m.headerData(7, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "q (e)"


# ---- without Qt -------------------------------------------------------------------------


def test_module_and_cli_work_without_pyside6(tmp_path):
    """Block PySide6 and check the module imports, run_gui explains, and the single-file
    build still loads (its --gui prints the install hint)."""
    code = (
        "import sys; sys.modules['PySide6'] = None; "
        "from pdb_wizard import gui; print(gui._HAS_QT); print(gui.run_gui())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["False", gui.GUI_INSTALL_HINT, "1"]
    dist = ROOT / "dist" / "pdb_wizard.py"
    runner = tmp_path / "run_dist.py"
    runner.write_text(
        "import runpy, sys\nsys.modules['PySide6'] = None\n"
        f"sys.argv = ['pdb_wizard', '--gui']\nrunpy.run_path({str(dist)!r}, "
        "run_name='__main__')\n"
    )
    out = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True)
    assert out.returncode == 1 and gui.GUI_INSTALL_HINT in out.stdout


# ---- dialogs, keyboard, drag and drop, theme --------------------------------------------


def test_file_dialogs(window, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(HKUST), ""))
    )
    window.open_dialog()
    assert len(window.mol.atoms) == 624
    out = tmp_path / "saved.xyz"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    window.save_dialog()
    assert out.exists()
    png = tmp_path / "shot.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(png), ""))
    )
    assert window.export_image() == str(png) and png.exists()
    # cancelled dialogs change nothing
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", ""))
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )
    window.open_dialog()
    window.save_dialog()
    assert window.export_image() is None and not window.load_charges()
    window.sorbate_combo.setCurrentIndex(window.sorbate_combo.findData("__file__"))
    assert window.pack_sorbates() == 0


def test_bad_template_and_charges_files_warn(hkust, tmp_path):
    empty = tmp_path / "empty.pqr"
    empty.write_text("ATOM 1 C MOF F 1 0 0 0 12 0 0 0 0\n")
    hkust.sorbate_combo.setCurrentIndex(hkust.sorbate_combo.findData("__file__"))
    assert hkust.pack_sorbates(model_path=str(empty)) == 0
    assert "no movable" in hkust.warnings[-1]
    assert not hkust.load_charges(str(tmp_path / "nope.txt"))
    assert "Could not read charges" in hkust.warnings[-1]


def test_save_without_structure_and_ff_none(window, hkust):
    blank = gui.MainWindow()
    blank.warnings = []
    blank._warn = blank.warnings.append
    blank.save_dialog()
    assert blank.warnings
    hkust.ff_combo.setCurrentIndex(0)  # "None"
    hkust.apply_force_field()
    assert not hkust.undo_action.isEnabled()  # nothing changed, nothing to undo


def test_keyboard(hkust, qtbot):
    v = hkust.view
    rot, zoom = v._rot.copy(), v._zoom
    qtbot.keyClick(v, QtCore.Qt.Key_Left)
    assert not np.allclose(v._rot, rot)
    for key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
        qtbot.keyClick(v, key)
    assert np.allclose(v._rot @ v._rot.T, np.eye(3))
    qtbot.keyClick(v, QtCore.Qt.Key_Plus)
    assert v._zoom > zoom
    qtbot.keyClick(v, QtCore.Qt.Key_Minus)
    qtbot.keyClick(v, QtCore.Qt.Key_Minus)
    assert v._zoom < zoom
    qtbot.keyClick(v, QtCore.Qt.Key_Q)  # unhandled keys pass through


def test_hover_updates_status_and_pan(hkust, qtbot):
    v = hkust.view
    image(v)
    front = int(np.argmax(v._depth))
    x, y = (int(c) for c in v._screen[front])
    qtbot.mouseMove(v, QtCore.QPoint(x, y))
    assert f"#{front + 1}" in hkust.statusBar().currentMessage()
    qtbot.mouseMove(v, QtCore.QPoint(2, 2))
    pan = QtCore.QPointF(v._pan)
    qtbot.mousePress(v, QtCore.Qt.RightButton, pos=QtCore.QPoint(100, 100))
    qtbot.mouseMove(v, QtCore.QPoint(130, 120))
    qtbot.mouseRelease(v, QtCore.Qt.RightButton, pos=QtCore.QPoint(130, 120))
    assert v._pan != pan
    qtbot.mouseDClick(v, QtCore.Qt.LeftButton, pos=QtCore.QPoint(5, 5))
    assert v._pan == QtCore.QPointF(0, 0)


def test_table_click_selects_in_view(hkust):
    hkust._selected(10, from_table=True)
    assert hkust.view.selected == 10
    hkust._selected(None)
    assert "Click an atom" in hkust.selection_label.text()


def test_drop_a_file(window):
    from PySide6 import QtGui as G

    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(HKUST))])
    pos = QtCore.QPointF(50, 50)
    enter = G.QDragEnterEvent(
        pos.toPoint(), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    drop = G.QDropEvent(pos, QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    window.dropEvent(drop)
    assert window.mol is not None and len(window.mol.atoms) == 624


def test_about_and_theme(window, monkeypatch):
    from PySide6 import QtWidgets

    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "about", staticmethod(lambda *a: shown.append(a)))
    window.about()
    assert shown and "PDB Wizard" in shown[0][2]
    app = QtWidgets.QApplication.instance()
    style, palette = app.style().name(), app.palette()
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    gui._apply_theme(app)
    assert app.style().name() == style  # KDE keeps its own style
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.delenv("KDE_FULL_SESSION", raising=False)
    gui._apply_theme(app)
    assert app.style().name().lower() == "fusion"
    assert app.palette().color(QtGui.QPalette.Window).lightness() < 80
    app.setPalette(palette)


def test_run_gui_starts_and_quits(monkeypatch):
    from PySide6 import QtWidgets

    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)
    windows = []
    real_show = gui.MainWindow.show
    monkeypatch.setattr(
        gui.MainWindow, "show", lambda self: windows.append(self) or real_show(self)
    )
    assert gui.run_gui(str(HKUST)) == 0
    assert windows and len(windows[0].mol.atoms) == 624
    windows[0].close()


def test_resizing_keeps_the_fit(hkust, qtbot):
    v = hkust.view
    zoom = v._zoom
    hkust.resize(1600, 1000)
    qtbot.wait(50)
    assert v._zoom > zoom
