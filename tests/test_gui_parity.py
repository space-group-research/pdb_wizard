"""The desktop GUI and the terminal viewer must offer the same features and give the same
numbers.

Every analysis is run in both front ends on the same structure or trajectory, with each
window's default options, and the plotted curve, title and status line must match exactly.
A feature map also checks that every terminal Operations-panel and View-menu entry has a
GUI counterpart, so a feature added to one front end alone fails here.

Needs PySide6, pytest-qt and textual; skipped otherwise. Runs headless.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")
pytest.importorskip("textual")

from textual.widgets import Label  # noqa: E402

from pdb_wizard import gui  # noqa: E402
from pdb_wizard.analyses import (  # noqa: E402
    run_isotherm,
    surface_area_text,
    system_info_text,
    void_volume_report,
)
from pdb_wizard.atom import Atom  # noqa: E402
from pdb_wizard.gui_tools import ANALYSES, AnalysisDialog, EnergyDialog  # noqa: E402
from pdb_wizard.io import read_file  # noqa: E402
from pdb_wizard.molecule import Molecule  # noqa: E402
from pdb_wizard.operations import pore_size_distribution, void_volume  # noqa: E402
from pdb_wizard.panels import OPERATIONS, OperationsPanel  # noqa: E402
from pdb_wizard.pbc import PBC  # noqa: E402
from pdb_wizard.viewer import (  # noqa: E402
    MENU_VIEW,
    CoordinationScreen,
    DensityScreen,
    EnergyPlotScreen,
    GyrationScreen,
    HbondScreen,
    InfoModal,
    MsdScreen,
    PdbWizardApp,
    PlotWidget,
    PxrdScreen,
    RdfScreen,
    RmsdScreen,
)

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"


# ---- structures -------------------------------------------------------------------------


def mixed_mol() -> Molecule:
    rng = np.random.default_rng(42)
    atoms = [Atom(*rng.uniform(0, 14, 3), el) for el in ("C", "O", "N", "H") for _ in range(8)]
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()
    return mol


def water_trajectory(n_frames: int = 12) -> list:
    """Waters (O named OW) drifting along x, so MSD, RMSD and H-bonds all have signal."""
    rng = np.random.default_rng(7)
    centres = rng.uniform(2, 13, (10, 3))
    pbc = PBC(15, 15, 15, 90, 90, 90)
    frames = []
    for f in range(n_frames):
        atoms = []
        for c in centres:
            o = c + np.array([0.25 * f, 0, 0]) + rng.normal(0, 0.03, 3)
            ow = Atom(*o, "O")
            ow.name = "OW"
            atoms.append(ow)
            for d in ((0.96, 0, 0), (-0.24, 0.93, 0)):
                atoms.append(Atom(*(o + np.array(d)), "H"))
        m = Molecule(atoms=atoms, pbc=pbc)
        m.detect_bonds()
        frames.append(m)
    return frames


# ---- drivers ----------------------------------------------------------------------------


async def _tui(mol, screen, prefix: str, compute: bool = True):
    mol = mol if mol is not None else Molecule(atoms=[])
    app = PdbWizardApp(molecule=mol, filepath="t.pdb")
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()
        if compute:
            await pilot.click(f"#{prefix}-compute")
            for _ in range(1200):
                await asyncio.sleep(0.05)
                task = screen._compute_task
                if task is not None and task.done():
                    break
            await pilot.pause()
            task.result()
        plot = screen.query_one(f"#{prefix}-plot", PlotWidget)
        status = str(screen.query_one(f"#{prefix}-status", Label).render())
        return plot._x.copy(), plot._y.copy(), plot._title, status


def tui_run(mol, screen, prefix: str, compute: bool = True):
    return asyncio.run(_tui(mol, screen, prefix, compute))


def gui_run(qtbot, key, mol, frames=None):
    dlg = AnalysisDialog(key, mol, frames)
    qtbot.addWidget(dlg)
    dlg.run().wait()
    return dlg.plot.x.copy(), dlg.plot.y.copy(), dlg.plot.title, dlg.status.text()


def assert_same(tui, gui_out):
    tx, ty, ttitle, tstatus = tui
    gx, gy, gtitle, gstatus = gui_out
    assert len(tx) > 0, tstatus
    np.testing.assert_array_equal(tx, gx)
    np.testing.assert_array_equal(ty, gy)
    assert ttitle == gtitle
    assert tstatus == gstatus


# ---- analyses: identical results --------------------------------------------------------


def test_rdf_single_structure(qtbot):
    mol = mixed_mol()
    assert_same(tui_run(mol, RdfScreen(mol), "rdf"), gui_run(qtbot, "rdf", mol))


def test_rdf_trajectory(qtbot):
    frames = water_trajectory()
    t = tui_run(frames[0], RdfScreen(frames[0], frames), "rdf")
    assert "frames 1-12" in t[2]
    assert_same(t, gui_run(qtbot, "rdf", frames[0], frames))


def test_coordination(qtbot):
    mol = mixed_mol()
    assert_same(tui_run(mol, CoordinationScreen(mol), "coord"),
                gui_run(qtbot, "coordination", mol))


@pytest.mark.parametrize("traj", [False, True])
def test_hbonds(qtbot, traj):
    frames = water_trajectory()
    fr = frames if traj else None
    assert_same(tui_run(frames[0], HbondScreen(frames[0], fr), "hb"),
                gui_run(qtbot, "hbonds", frames[0], fr))


@pytest.mark.parametrize("key, screen_cls, prefix", [
    ("msd", MsdScreen, "msd"), ("rmsd", RmsdScreen, "rmsd"), ("gyration", GyrationScreen, "rg"),
])
def test_trajectory_analyses(qtbot, key, screen_cls, prefix):
    frames = water_trajectory()
    assert_same(tui_run(frames[0], screen_cls(frames[0], frames), prefix),
                gui_run(qtbot, key, frames[0], frames))


@pytest.mark.parametrize("traj", [False, True])
def test_density_profile(qtbot, traj):
    frames = water_trajectory()
    fr = frames if traj else None
    assert_same(tui_run(frames[0], DensityScreen(frames[0], fr), "dens"),
                gui_run(qtbot, "density", frames[0], fr))


def test_pxrd(qtbot):
    mol = read_file(str(HKUST))
    assert_same(tui_run(mol, PxrdScreen(mol), "pxrd"), gui_run(qtbot, "pxrd", mol))


def test_energy_plot(qtbot, tmp_path):
    path = tmp_path / "energy.dat"
    rows = [f"{s} {-100 - s * 0.5:.3f} {0.1 * s:.3f} 1.0 2.0 3.0 0 0 {s // 3} 0 0 0"
            for s in range(1, 41)]
    path.write_text("\n".join(rows) + "\n")
    from pdb_wizard.energy import read_energy_dat
    data = read_energy_dat(str(path))
    t = tui_run(None, EnergyPlotScreen(data, str(path)), "energy", compute=False)
    dlg = EnergyDialog(data, str(path))
    qtbot.addWidget(dlg)
    assert dlg.column.currentText() == "N"  # same default column
    assert_same(t, (dlg.plot.x, dlg.plot.y, dlg.plot.title, dlg.status.text()))


def test_pore_size_and_isotherm_use_the_same_numbers(qtbot, tmp_path):
    mol = read_file(str(HKUST))
    bins, hist = pore_size_distribution(mol, 50000, 50, 20.0)
    x, y, _, status = gui_run(qtbot, "pore_size", mol)
    np.testing.assert_array_equal(x, bins * 2)  # the terminal plot uses diameters too
    np.testing.assert_array_equal(y, hist)
    assert status.startswith("Dominant pore diameter:")
    for p, n in ((1.0, 3), (5.0, 7)):
        d = tmp_path / f"P_{p:g}_atm".replace(".", "p")
        d.mkdir()
        (d / "histogram.dat").write_text(f"{n} 1.0\n")
    dlg = AnalysisDialog("isotherm_plot", None)
    qtbot.addWidget(dlg)
    dlg.inputs["directory"].setText(str(tmp_path))
    dlg.run().wait()
    ref = run_isotherm(str(tmp_path))
    np.testing.assert_array_equal(dlg.plot.x, ref.x)
    np.testing.assert_array_equal(dlg.plot.y, [3.0, 7.0])


# ---- reports: identical text ------------------------------------------------------------


async def _tui_report(mol, command: str, path: str) -> str:
    app = PdbWizardApp(molecule=mol, filepath=path)
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        app.on_operations_panel_run_command(OperationsPanel.RunCommand(command))
        for _ in range(1200):
            await asyncio.sleep(0.05)
            if isinstance(app.screen, InfoModal):
                return app.screen._content
    raise AssertionError(f"{command}: no report appeared")


@pytest.mark.parametrize("command", ["info", "void_vol", "surface_area"])
def test_reports_match(qtbot, command):
    mol = read_file(str(HKUST))
    tui_text = asyncio.run(_tui_report(read_file(str(HKUST)), command, str(HKUST)))
    w = gui.MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(HKUST))
    out = w.open_tool(command)
    if hasattr(out, "wait"):
        out.wait()
    gui_text = w.dialogs[-1].text.toPlainText()
    assert gui_text == tui_text
    reference = {
        "info": system_info_text(mol, str(HKUST)),
        "void_vol": void_volume_report(void_volume(mol, 100000, 0.0),
                                       void_volume(mol, 100000, 1.14)),
        "surface_area": surface_area_text(mol),
    }[command]
    assert gui_text == reference
    for d in w.dialogs:
        d.close()


# ---- feature map ------------------------------------------------------------------------

# terminal Operations-panel command -> what the GUI calls for it
GUI_FOR_TUI_COMMAND = {
    "info": "open_tool", "void_vol": "open_tool", "surface_area": "open_tool",
    "pore_size": "open_tool", "rdf": "open_tool", "coordination": "open_tool",
    "hbonds": "open_tool", "msd": "open_tool", "rmsd": "open_tool", "gyration": "open_tool",
    "density": "open_tool", "density3d": "density_3d", "pxrd": "open_tool",
    "energy_plot": "energy_plot", "input_generator": "open_tool",
    "isotherm_plot": "open_tool", "db_search": "open_tool",
    "undo": "undo", "wrap_center": "wrap", "wrap_forward": "wrap", "sort": "sort_atoms",
    "extend": "build_supercell", "del_lone": "delete_lone_atoms",
    "edit_h": "edit_h_distances", "update_cell": "update_cell",
    "substitute": "substitute_element", "reduce_cell": "reduce_supercell",
    "pack_sorbates": "pack_sorbates",
    "qeq_charges": "qeq_charges", "open_file": "open_dialog", "write_xyz": "save_path",
    "write_pdb": "save_path", "write_cif": "save_path", "write_poscar": "save_path",
    "write_lammps": "save_path", "write_mpmc": "save_path", "frame_to_tab": "frame_to_tab",
    "export_png": "export_image", "export_gif": "export_gif", "quit": "close",
}

# terminal View-menu entry -> the StructureView option or window attribute behind it
GUI_FOR_TUI_VIEW = {
    "toggle_bonds": "view.show_bonds", "toggle_style": "view.cycle_style",
    "toggle_bg": "view.background", "toggle_atom_numbers": "view.show_atom_numbers",
    "cycle_color_mode": "view.cycle_color_mode", "toggle_hide_water": "view.hide_water",
    "toggle_measure": "view.measure_mode", "clear_measurements": "view.clear_measurements",
    "toggle_polyhedra": "view.show_polyhedra", "toggle_orthographic": "view.orthographic",
    "toggle_operations": "tools_dock", "toggle_geometry": "geometry_dock",
    "toggle_visual": "tools", "reset_view": "view.reset_view",
}


def test_every_terminal_operation_has_a_gui_counterpart(qtbot):
    w = gui.MainWindow()
    qtbot.addWidget(w)
    missing = [cmd for cmd, _, _ in OPERATIONS if cmd not in GUI_FOR_TUI_COMMAND]
    assert not missing, f"terminal operations without a GUI counterpart: {missing}"
    for cmd, attr in GUI_FOR_TUI_COMMAND.items():
        assert callable(getattr(w, attr)), (cmd, attr)
    gui_keys = set(ANALYSES) | {"info", "void_vol", "surface_area", "density3d",
                                              "energy_plot", "input_generator", "db_search"}
    analysis_cmds = {c for c, _, group in OPERATIONS if group == "analysis"}
    assert analysis_cmds <= gui_keys


def test_every_terminal_view_option_has_a_gui_counterpart(qtbot):
    w = gui.MainWindow()
    qtbot.addWidget(w)
    entries = [key for key, _ in MENU_VIEW if key != "---"]
    missing = [key for key in entries if key not in GUI_FOR_TUI_VIEW]
    assert not missing, f"terminal view options without a GUI counterpart: {missing}"
    for key, path in GUI_FOR_TUI_VIEW.items():
        obj = w
        for part in path.split("."):
            obj = getattr(obj, part)
        assert obj is not None, (key, path)
    # styles and colour modes cycle through the same lists as the terminal viewer
    assert [k for k, _ in gui.STYLES] == ["cpk", "licorice", "vdw", "ribbon"]
    assert [k for k, _ in gui.COLOR_MODES] == ["element", "charge", "residue", "chain", "index"]
