"""Terminal viewer bugs found in the bug hunt: crashes on Escape, on imperfect files and on
stale atom indices; background loads and dialogs landing on the wrong tab; saves going to a
temporary folder; undo and cell updates on trajectories; guards for structures without a cell.

Every test drives ``PdbWizardApp`` with textual's pilot. Waits use a wall-clock deadline.
Slow background work is held with a ``threading.Event`` gate, never a sleep, so the order of
events is the same on every machine.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.widgets import Button, DirectoryTree, Input, Label, Select  # noqa: E402

import pdb_wizard.analyses as analyses  # noqa: E402
import pdb_wizard.charges as charges  # noqa: E402
import pdb_wizard.database as database  # noqa: E402
import pdb_wizard.trajectory as trajectory  # noqa: E402
from pdb_wizard import viewer as V  # noqa: E402
from pdb_wizard.io import is_fallback_cell, read_file  # noqa: E402
from pdb_wizard.molecule import Molecule  # noqa: E402
from pdb_wizard.panels import GeometryPanel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WAIT_SECONDS = 60


# ---- inputs -----------------------------------------------------------------------------

def _water(i, ox, oy, oz, serial):
    return [
        f"HETATM{serial:5d}  O1  HOH A{i:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}"
        "  1.00  0.00           O  ",
        f"HETATM{serial + 1:5d}  H1  HOH A{i:4d}    {ox + 0.957:8.3f}{oy:8.3f}{oz:8.3f}"
        "  1.00  0.00           H  ",
        f"HETATM{serial + 2:5d}  H2  HOH A{i:4d}    {ox - 0.240:8.3f}{oy + 0.927:8.3f}{oz:8.3f}"
        "  1.00  0.00           H  ",
    ]


CRYST = "CRYST1   15.000   15.000   15.000  90.00  90.00  90.00 P 1           1"


def pdb_traj(path, counts) -> str:
    """A PDB trajectory of water molecules in a 15 A box; ``counts`` waters per frame."""
    lines = []
    for fi, n in enumerate(counts):
        lines += [f"MODEL     {fi + 1:4d}", CRYST]
        for i in range(n):
            lines += _water(i + 1, 3.0 + 3.0 * (i % 4) + 0.2 * fi, 5.0 + 3.0 * (i // 4),
                            5.0 + 0.1 * fi, 3 * i + 1)
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def water_box(path) -> str:
    """One water in a 15 A box (a structure with a cell)."""
    Path(path).write_text("\n".join([CRYST] + _water(1, 3.0, 5.0, 5.0, 1) + ["END"]) + "\n")
    return str(path)


def water_xyz(path) -> str:
    Path(path).write_text("3\nwater\nO 0.000 0.000 0.000\nH 0.957 0.000 0.000\n"
                          "H -0.240 0.927 0.000\n")
    return str(path)


def argon_xyz(path) -> str:
    Path(path).write_text("1\nargon\nAr 0.0 0.0 0.0\n")
    return str(path)


def xyz_traj_nocell(path) -> str:
    out = []
    for fi in range(3):
        out += ["6", f"frame {fi}"]
        for i in range(2):
            ox = 3.0 * i + 0.2 * fi
            out += [f"O {ox:.3f} 0.000 0.000", f"H {ox + 0.957:.3f} 0.000 0.000",
                    f"H {ox - 0.240:.3f} 0.927 0.000"]
    Path(path).write_text("\n".join(out) + "\n")
    return str(path)


# ---- driving the app --------------------------------------------------------------------

def app_for(path=None, trajectory_file=False) -> V.PdbWizardApp:
    """The app as ``pdb_wizard [file]`` builds it."""
    if path is None:
        return V.PdbWizardApp(molecule=Molecule(atoms=[]), filepath="")
    if trajectory_file:
        return V.PdbWizardApp(molecule=Molecule(atoms=[]), filepath=str(path),
                              load_trajectory=True)
    return V.PdbWizardApp(molecule=read_file(str(path)), filepath=str(path))


def drive(app, fn):
    async def go():
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            return await fn(pilot, app)
    return asyncio.run(go())


async def wait_for(pilot, cond, what, seconds=WAIT_SECONDS):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError(f"timed out after {seconds} s waiting for {what}")


async def idle(pilot, app):
    """Every background task of the app and of its open screens has finished."""
    def done():
        if any(not t.done() for t in list(app._bg_tasks)):
            return False
        for scr in app.screen_stack:
            task = getattr(scr, "_compute_task", None)
            if task is not None and not task.done():
                return False
        return True
    await wait_for(pilot, done, "background tasks")
    await pilot.pause()


async def on_screen(pilot, app, cls):
    await wait_for(pilot, lambda: isinstance(app.screen, cls) and app.screen.is_mounted,
                   cls.__name__)
    await pilot.pause()
    await pilot.pause()
    return app.screen


async def main_screen(pilot, app):
    await wait_for(pilot, lambda: len(app.screen_stack) == 1, "the dialogs to close")
    await pilot.pause()


def command(app, name: str) -> None:
    app.on_operations_panel_run_command(SimpleNamespace(command=name))


def press(screen, button_id: str) -> None:
    screen.query_one("#" + button_id, Button).press()


def toasts(app, severity=None) -> list:
    return [n.message for n in app._notifications if severity is None or n.severity == severity]


async def answer_file(pilot, app, path) -> None:
    """Type ``path`` into the file dialog on top and accept it."""
    modal = await on_screen(pilot, app, V.FileSaveModal)
    modal.query_one("#file-input", Input).value = str(path)
    press(modal, "file-save")
    await pilot.pause()


async def answer_text(pilot, app, text) -> None:
    modal = await on_screen(pilot, app, V.InputModal)
    modal.query_one("#modal-input", Input).value = text
    press(modal, "input-ok")
    await pilot.pause()


class Gate:
    """Wrap a function so its call blocks (in its worker thread) until ``release()``."""

    def __init__(self, fn):
        self.fn = fn
        self.event = threading.Event()
        self.entered = threading.Event()

    def __call__(self, *args, **kwargs):
        self.entered.set()
        assert self.event.wait(WAIT_SECONDS), "the test never released the gate"
        return self.fn(*args, **kwargs)

    def release(self):
        self.event.set()


def describe_tabs(app) -> list:
    return [(t.name, len(t.molecule.atoms), len(t.frames) if t.frames else 0)
            for t in app._tabs]


async def close_active_tab(pilot, app) -> None:
    """Click the active tab label, then 'Close <name>'."""
    app.on_menu_bar_tab_close(V.MenuBar.TabClose(app._active_tab, screen_x=30))
    menu = await on_screen(pilot, app, V.MenuDropdown)
    menu._already_dismissed = True
    menu.dismiss("close_tab")
    await main_screen(pilot, app)


ENERGY = "#step energy N\n" + "".join(f"{i} {-100.0 - i} {i % 7}\n" for i in range(1, 30))


# ---- 1. Energy Plot closes on Escape and q ----------------------------------------------

@pytest.mark.parametrize("key", ["escape", "q"])
def test_energy_plot_closes_on_escape_and_q(tmp_path, key):
    path = water_xyz(tmp_path / "water.xyz")
    (tmp_path / "energy.dat").write_text(ENERGY)

    async def run(pilot, app):
        command(app, "energy_plot")
        await on_screen(pilot, app, V.EnergyPlotScreen)
        await pilot.press(key)
        await main_screen(pilot, app)
        return len(app.screen_stack)

    assert drive(app_for(path), run) == 1


# ---- 2. A background trajectory load goes to its own tab --------------------------------

def test_trajectory_survives_closing_the_only_other_tab(tmp_path, monkeypatch):
    water = water_xyz(tmp_path / "water.xyz")
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])
    gate = Gate(trajectory.read_trajectory)
    monkeypatch.setattr(trajectory, "read_trajectory", gate)

    async def run(pilot, app):
        app._open_new_tab(traj)
        await pilot.pause()
        await close_active_tab(pilot, app)            # the water tab
        view = app.query_one(V.MoleculeView)
        drawn_while_loading = len(view.molecule.atoms) if view.molecule else 0
        gate.release()
        await idle(pilot, app)
        return drawn_while_loading, describe_tabs(app), len(view.molecule.atoms)

    drawn_while_loading, tabs, drawn = drive(app_for(water), run)
    assert drawn_while_loading == 0       # not the closed water molecule
    assert tabs == [("traj.pdb", 9, 3)]
    assert drawn == 9


def test_trajectory_does_not_overwrite_another_tab(tmp_path, monkeypatch):
    water = water_xyz(tmp_path / "water.xyz")
    argon = argon_xyz(tmp_path / "argon.xyz")
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])
    gate = Gate(trajectory.read_trajectory)
    monkeypatch.setattr(trajectory, "read_trajectory", gate)

    async def run(pilot, app):
        app._open_new_tab(traj)                       # tab 1, loading
        await pilot.pause()
        app._open_new_tab(argon)                      # tab 2
        await pilot.pause()
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(0))
        await pilot.pause()
        await close_active_tab(pilot, app)            # close water: indices shift down
        gate.release()
        await idle(pilot, app)
        app._save_current_tab_state()
        return describe_tabs(app)

    assert drive(app_for(water), run) == [("traj.pdb", 9, 3), ("argon.xyz", 1, 0)]


def test_startup_trajectory_does_not_overwrite_a_file_opened_meanwhile(tmp_path, monkeypatch):
    water = water_xyz(tmp_path / "water.xyz")
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])
    gate = Gate(trajectory.read_trajectory)
    monkeypatch.setattr(trajectory, "read_trajectory", gate)

    async def run(pilot, app):
        app._open_new_tab(water)
        await pilot.pause()
        gate.release()
        await idle(pilot, app)
        app._save_current_tab_state()
        tabs = describe_tabs(app)
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(0))
        await pilot.pause()
        return tabs, len(app.molecule.atoms), len(app._frames or [])

    tabs, atoms, frames = drive(app_for(traj, trajectory_file=True), run)
    assert tabs == [("traj.pdb", 9, 3), ("water.xyz", 3, 0)]
    assert (atoms, frames) == (9, 3)


# ---- 3. A dialog's answer belongs to the structure it was opened for --------------------

def atoms_in_xyz(path) -> int:
    return int(Path(path).read_text().splitlines()[0])


def test_trajectory_finishing_under_a_dialog_does_not_switch_tabs(tmp_path, monkeypatch):
    water = water_xyz(tmp_path / "water.xyz")
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])
    out = tmp_path / "saved.xyz"
    gate = Gate(trajectory.read_trajectory)
    monkeypatch.setattr(trajectory, "read_trajectory", gate)

    async def run(pilot, app):
        app._open_new_tab(traj)
        await pilot.pause()
        command(app, "write_xyz")
        modal = await on_screen(pilot, app, V.FileSaveModal)
        gate.release()
        await idle(pilot, app)
        active_under_dialog = app._tabs[app._active_tab].name
        modal.query_one("#file-input", Input).value = str(out)
        press(modal, "file-save")
        await main_screen(pilot, app)
        return active_under_dialog, describe_tabs(app)

    active, tabs = drive(app_for(water), run)
    assert active == "water.xyz"
    assert tabs == [("water.xyz", 3, 0), ("traj.pdb", 9, 3)]    # loaded, just not shown
    assert atoms_in_xyz(out) == 3


@pytest.mark.parametrize("cmd, suffix", [("write_xyz", ".xyz"), ("write_pdb", ".pdb"),
                                         ("write_lammps", ".data")])
def test_save_writes_the_structure_the_dialog_was_opened_for(tmp_path, cmd, suffix):
    water = water_xyz(tmp_path / "water.xyz")
    argon = argon_xyz(tmp_path / "argon.xyz")
    out = tmp_path / ("saved" + suffix)

    async def run(pilot, app):
        app._open_new_tab(argon)
        await pilot.pause()
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(0))     # back to water
        await pilot.pause()
        command(app, cmd)
        modal = await on_screen(pilot, app, V.FileSaveModal)
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(1))     # the tab changes underneath
        await pilot.pause()
        modal.query_one("#file-input", Input).value = str(out)
        press(modal, "file-save")
        await main_screen(pilot, app)

    drive(app_for(water), run)
    written = read_file(str(out))
    assert sorted(a.element.symbol for a in written.atoms) == ["H", "H", "O"]


def test_mpmc_wizard_stays_with_its_structure_when_the_tab_changes(tmp_path, monkeypatch):
    box = water_box(tmp_path / "wbox.pdb")
    argon = argon_xyz(tmp_path / "argon.xyz")
    out = tmp_path / "export.pqr"
    gate = Gate(charges.apply_qeq_charges)
    monkeypatch.setattr(charges, "apply_qeq_charges", gate)

    async def run(pilot, app):
        app._open_new_tab(argon)
        await pilot.pause()
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(0))     # back to the water box
        await pilot.pause()
        command(app, "write_mpmc")
        modal = await on_screen(pilot, app, V.ConfirmModal)        # Charge Source
        press(modal, "confirm-cancel")                             # No: generate QEq
        await wait_for(pilot, gate.entered.is_set, "QEq to start")
        await main_screen(pilot, app)
        app.on_menu_bar_tab_selected(V.MenuBar.TabSelected(1))     # user clicks the argon tab
        await pilot.pause()
        gate.release()
        ff = await on_screen(pilot, app, V.ForceFieldModal)
        ff_atoms = len(ff._molecule.atoms)
        press(ff, "ff-apply")
        sorb = await on_screen(pilot, app, V.SorbateModal)
        press(sorb, "sorb-skip")
        save = await on_screen(pilot, app, V.FileSaveModal)
        offered = Path(save.query_one("#file-input", Input).value).name
        save.query_one("#file-input", Input).value = str(out)
        press(save, "file-save")
        await main_screen(pilot, app)
        return ff_atoms, offered

    ff_atoms, offered = drive(app_for(box), run)
    assert ff_atoms == 3
    assert offered == "wbox_out_mpmc.pqr"
    atoms = [ln for ln in out.read_text().splitlines()
             if ln.startswith("ATOM") and " BOX " not in ln]
    assert len(atoms) == 3


# ---- 4. Fetched structures save into the working directory ------------------------------

def _fake_backend(monkeypatch, tmp_path, gate=None):
    src = tmp_path / "source_water.xyz"
    water_xyz(src)

    def fetch(rid, output_dir="."):
        out = os.path.join(output_dir, f"COD_{rid}.xyz")
        shutil.copy(src, out)
        return out

    spec = dict(database.BACKENDS["cod"])
    spec["fetch"] = fetch if gate is None else Gate(fetch)
    monkeypatch.setitem(database.BACKENDS, "cod", spec)
    return spec["fetch"]


async def _start_fetch(pilot, app):
    command(app, "db_search")
    scr = await on_screen(pilot, app, V.DatabaseSearchScreen)
    scr.query_one("#db-source", Select).value = "cod"
    await pilot.pause()
    scr.query_one("#db-query", Input).value = "123"
    press(scr, "db-fetch-id")
    await pilot.pause()
    return scr


def test_fetched_structure_saves_default_to_the_working_directory(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _fake_backend(monkeypatch, tmp_path)

    async def run(pilot, app):
        await _start_fetch(pilot, app)
        await idle(pilot, app)
        await main_screen(pilot, app)
        name = app._tabs[app._active_tab].name
        command(app, "write_pdb")
        modal = await on_screen(pilot, app, V.FileSaveModal)
        offered = modal.query_one("#file-input", Input).value
        press(modal, "file-save")
        await main_screen(pilot, app)
        app.action_export_png()
        await idle(pilot, app)
        return name, offered

    name, offered = drive(app_for(), run)
    assert name == "COD_123.xyz"
    assert Path(offered).resolve() == (work / "COD_123_out.pdb").resolve()
    # still there after the app (and its download cache) is gone
    assert sorted(p.name for p in work.iterdir()) == ["COD_123.png", "COD_123_out.pdb"]


# ---- 5. Atom picks do not outlive the atoms ---------------------------------------------

def _pick(app, idx, shift=False):
    app.query_one(V.MoleculeView).post_message(V.MoleculeView.AtomPicked(idx, shift))


@pytest.mark.parametrize("shift", [False, True])
def test_pick_after_stepping_to_a_smaller_frame(tmp_path, shift):
    traj = pdb_traj(tmp_path / "var.pdb", [4, 2, 5])         # 12, 6, 15 atoms

    async def run(pilot, app):
        await idle(pilot, app)
        if not shift:
            await pilot.press("m")
        _pick(app, 11, shift)                                # last atom of frame 1
        await pilot.pause()
        await pilot.press("right_square_bracket")            # 6 atoms now
        await pilot.pause()
        _pick(app, 2, shift)
        await pilot.pause()
        await pilot.pause()
        view = app.query_one(V.MoleculeView)
        return len(app.molecule.atoms), list(view._picked_atoms), set(view.highlighted_atoms)

    atoms, picked, highlighted = drive(app_for(traj, trajectory_file=True), run)
    assert atoms == 6
    assert picked == [2] and highlighted == {2}              # a fresh first pick


def test_pick_after_undo_shrinks_the_structure(tmp_path):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        app._do_extend((1, 0, 0))                            # 6 atoms
        await idle(pilot, app)
        await pilot.press("m")
        _pick(app, 5)
        await pilot.pause()
        app._do_undo()                                       # 3 atoms
        await pilot.pause()
        _pick(app, 1)
        await pilot.pause()
        await pilot.pause()
        return len(app.molecule.atoms), list(app.query_one(V.MoleculeView)._picked_atoms)

    assert drive(app_for(box), run) == (3, [1])


# ---- 7. MPMC export stops when QEq fails ------------------------------------------------

def test_mpmc_wizard_stops_when_qeq_fails(tmp_path, monkeypatch):
    box = water_box(tmp_path / "wbox.pdb")

    def failing(mol, progress_callback=None):
        raise ValueError("System too large for QEq")

    monkeypatch.setattr(charges, "apply_qeq_charges", failing)

    async def run(pilot, app):
        command(app, "write_mpmc")
        modal = await on_screen(pilot, app, V.ConfirmModal)
        press(modal, "confirm-cancel")                       # No: generate QEq
        await idle(pilot, app)
        await pilot.pause(0.2)
        return [type(s).__name__ for s in app.screen_stack[1:]], toasts(app, "error")

    screens, errors = drive(app_for(box), run)
    assert screens == []                                     # no force-field step
    assert any("QEq failed" in m and "too large" in m for m in errors)
    assert list(tmp_path.glob("*.pqr")) == []


# ---- 8. Energy Plot with imperfect files ------------------------------------------------

def _open_energy(tmp_path, content):
    water = water_xyz(tmp_path / "water.xyz")
    target = tmp_path / "run" / "my_energy.dat"
    target.parent.mkdir()
    if content is None:
        target.mkdir()
    elif isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content)

    async def run(pilot, app):
        command(app, "energy_plot")
        await answer_file(pilot, app, target)
        await pilot.pause()
        await pilot.pause()
        top = type(app.screen).__name__
        drawn = None
        if isinstance(app.screen, V.EnergyPlotScreen):
            sel = app.screen.query_one("#energy-col", Select)
            for col in ("energy", "N"):
                sel.value = col
                await pilot.pause()
                await pilot.pause()
            plot = app.screen.query_one("#energy-plot", V.PlotWidget)
            drawn = [plot.render_line(y) for y in range(plot.size.height)]
            await pilot.press("escape")
            await main_screen(pilot, app)
        return top, drawn is not None, toasts(app)

    return drive(app_for(water), run)


@pytest.mark.parametrize("content", [
    "#step\n1\n2\n3\n",                                      # nothing to plot against step
    bytes(range(256)) * 4,                                   # not a text file
    None,                                                    # a directory
], ids=["only-step-column", "binary", "directory"])
def test_energy_plot_refuses_unusable_files_with_a_message(tmp_path, content):
    top, _, messages = _open_energy(tmp_path, content)
    assert top == "Screen"                                   # no plot screen, app alive
    assert any("my_energy.dat" in m for m in messages)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_energy_plot_draws_files_with_non_finite_values(tmp_path, bad):
    top, drawn, _ = _open_energy(tmp_path, ENERGY.replace("-105.0", bad))
    assert top == "EnergyPlotScreen" and drawn


def test_energy_plot_survives_a_line_cut_short(tmp_path):
    """MPMC still writing: the last row is incomplete. Either the complete rows are plotted
    or the file is refused with a message; the app must not die."""
    top, drawn, messages = _open_energy(tmp_path, ENERGY + "30 -130.0\n")
    assert (top == "EnergyPlotScreen" and drawn) or any("my_energy.dat" in m for m in messages)


class _PlotApp(App):
    def compose(self):
        yield V.PlotWidget(id="p")


def _overlay_picture(x, y, overlay=None):
    """The braille rows of a PlotWidget showing ``x, y`` (and an overlay), colours dropped."""
    async def go():
        app = _PlotApp()
        async with app.run_test(size=(60, 14)) as pilot:
            plot = app.query_one(V.PlotWidget)
            plot.set_data(np.asarray(x, float), np.asarray(y, float), y_min=0.0, y_max=100.0)
            if overlay is not None:
                plot.set_overlay(np.asarray(overlay[0], float), np.asarray(overlay[1], float))
            await pilot.pause()
            return ["".join(seg.text for seg in plot.render_line(row))
                    for row in range(plot.size.height)]
    return asyncio.run(go())


def test_plot_skips_non_finite_samples():
    x = np.linspace(0.0, 30.0, 31)
    clean = np.full(31, 50.0)
    holed = clean.copy()
    holed[10] = np.nan
    holed[20] = np.inf
    ref = _overlay_picture(x, clean)
    got = _overlay_picture(x, holed)
    # same flat line, on the same text row; only cells next to the holes may be empty
    ref_rows = [i for i, r in enumerate(ref[1:-3], 1) if any(0x2800 < ord(c) <= 0x28FF for c in r)]
    got_rows = [i for i, r in enumerate(got[1:-3], 1) if any(0x2800 < ord(c) <= 0x28FF for c in r)]
    assert got_rows == ref_rows and len(ref_rows) == 1
    row = ref_rows[0]
    same = sum(1 for a, b in zip(ref[row], got[row]) if a == b)
    assert same >= len(ref[row]) - 12


# ---- 17. Overlay stored from high to low 2-theta ----------------------------------------

def test_overlay_is_drawn_the_same_in_either_file_order():
    x = np.linspace(0.0, 30.0, 200)
    base = np.zeros_like(x)
    ascending = _overlay_picture(x, base, ([5, 10, 20], [20, 100, 10]))
    descending = _overlay_picture(x, base, ([20, 10, 5], [10, 100, 20]))
    assert descending == ascending


# ---- 9. Charges file browser: a file that is not text -----------------------------------

def test_charges_browser_survives_a_binary_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    box = water_box(tmp_path / "wbox.pdb")
    binary = tmp_path / "traj.dcd"
    binary.write_bytes(bytes(range(256)) * 4)

    async def run(pilot, app):
        command(app, "write_mpmc")
        modal = await on_screen(pilot, app, V.ConfirmModal)
        press(modal, "confirm-ok")                           # Yes: load from file
        browser = await on_screen(pilot, app, V.ChargesFileModal)
        tree = browser.query_one("#charges-tree", DirectoryTree)
        tree.post_message(DirectoryTree.FileSelected(tree.root, binary))
        await pilot.pause()
        await pilot.pause()
        still_open = app.screen is browser
        lines = list(browser._raw_lines)
        await pilot.press("escape")
        await main_screen(pilot, app)
        return still_open, lines

    assert drive(app_for(box), run) == (True, [])


# ---- 10. No drop-down offers an empty choice --------------------------------------------

@pytest.mark.parametrize("cmd, screen_cls, trajectory_input", [
    ("input_generator", "InputGeneratorScreen", False),
    ("db_search", "DatabaseSearchScreen", False),
    ("rdf", "RdfScreen", False),
    ("coordination", "CoordinationScreen", False),
    ("density", "DensityScreen", False),
    ("pxrd", "PxrdScreen", False),
    ("energy_plot", "EnergyPlotScreen", False),
    ("msd", "MsdScreen", True),
    ("rmsd", "RmsdScreen", True),
    ("gyration", "GyrationScreen", True),
])
def test_dropdowns_have_no_blank_entry(tmp_path, cmd, screen_cls, trajectory_input):
    if trajectory_input:
        app = app_for(pdb_traj(tmp_path / "traj.pdb", [3, 3, 3]), trajectory_file=True)
    else:
        app = app_for(water_box(tmp_path / "wbox.pdb"))
    (tmp_path / "energy.dat").write_text(ENERGY)

    async def run(pilot, app):
        await idle(pilot, app)
        command(app, cmd)
        scr = await on_screen(pilot, app, getattr(V, screen_cls))
        selects = list(scr.query(Select))
        blank = []
        for sel in selects:
            try:
                sel.clear()                # only possible when a blank choice exists
            except Exception:
                continue
            blank.append(sel.id)
        # and by keyboard on the first one: open, go to the top row, choose it
        first = selects[0]
        before = first.value
        if first.id in blank:
            return len(selects), blank, True, True
        first.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("home")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        return len(selects), blank, not isinstance(before, str), not isinstance(first.value, str)

    count, blank, blank_before, blank_after = drive(app, run)
    assert count >= 1
    assert blank == []
    assert not blank_before and not blank_after


def test_color_mode_dropdown_has_no_blank_entry(tmp_path):
    async def run(pilot, app):
        await pilot.press("V")
        await pilot.pause()
        sel = app.query_one("#sel-colormode", Select)
        try:
            sel.clear()                    # only possible when a blank choice exists
        except Exception:
            return False
        return True

    assert drive(app_for(water_xyz(tmp_path / "water.xyz")), run) is False


# ---- 11. Update Unit Cell ---------------------------------------------------------------

def _metric_determinant(alpha, beta, gamma) -> float:
    """1 - cos^2(a) - cos^2(b) - cos^2(g) + 2 cos(a) cos(b) cos(g): (V / abc)^2 of a cell."""
    ca, cb, cg = (np.cos(np.radians(v)) for v in (alpha, beta, gamma))
    return float(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)


@pytest.mark.parametrize("cell, ok", [
    ((10, 10, 10, 90, 90, 90), True),
    ((26.343, 26.343, 26.343, 90, 90, 90), True),
    ((25.8, 25.8, 6.8, 90, 90, 120), True),
    ((5, 6, 7, 80, 95, 110), True),
    ((0, 10, 10, 90, 90, 90), False),
    ((-5, 10, 10, 90, 90, 90), False),
    ((10, 10, float("nan"), 90, 90, 90), False),
    ((float("inf"), 10, 10, 90, 90, 90), False),
    ((10, 10, 10, 0, 90, 90), False),
    ((10, 10, 10, 180, 90, 90), False),
    ((10, 10, 10, 90, 90, 200), False),
    ((10, 10, 10, 90, float("nan"), 90), False),
    ((10, 10, 10, 60, 60, 150), False),                      # angles that close no cell
])
def test_cell_problem(cell, ok):
    from pdb_wizard.pbc import cell_problem
    problem = cell_problem(*cell)
    if ok:
        assert _metric_determinant(*cell[3:]) > 0            # the reference agrees it is a cell
        assert problem is None
    else:
        assert isinstance(problem, str) and problem


@pytest.mark.parametrize("text", [
    "0 0 0 90 90 90", "10 10 10 0 0 0", "10 10 10 180 90 90", "10 10 10 90 90 200",
    "nan nan nan 90 90 90", "inf 10 10 90 90 90", "-5 10 10 90 90 90", "10 10 10 60 60 150",
])
def test_update_cell_refuses_cells_that_are_not_cells(tmp_path, text):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        command(app, "update_cell")
        await answer_text(pilot, app, text)
        await main_screen(pilot, app)
        await idle(pilot, app)
        pbc = app.molecule.pbc
        return ((pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma), len(app._undo_stack),
                toasts(app))

    cell, undo_depth, messages = drive(app_for(box), run)
    assert cell == (15.0, 15.0, 15.0, 90.0, 90.0, 90.0)
    assert undo_depth == 0
    assert messages and not any(m.startswith("Cell:") for m in messages)


def test_update_cell_still_accepts_a_real_cell(tmp_path):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        command(app, "update_cell")
        await answer_text(pilot, app, "12 13 14 80 95 110")
        await main_screen(pilot, app)
        await idle(pilot, app)
        pbc = app.molecule.pbc
        return (pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma), len(app._undo_stack)

    assert drive(app_for(box), run) == ((12.0, 13.0, 14.0, 80.0, 95.0, 110.0), 1)


def _cells(frames) -> list:
    return [None if f.pbc is None else (f.pbc.a, f.pbc.b, f.pbc.c) for f in frames]


def test_update_cell_applies_to_every_frame_and_undo_restores_every_frame(tmp_path):
    traj = xyz_traj_nocell(tmp_path / "nocell.xyz")

    async def run(pilot, app):
        await idle(pilot, app)
        before = [is_fallback_cell(f.pbc) for f in app._frames]
        app._do_update_cell("20 21 22 90 90 90")
        await idle(pilot, app)
        after = _cells(app._frames)
        await pilot.press("right_square_bracket")
        await pilot.pause()
        shown = (app.molecule.pbc.a, app.molecule.pbc.b, app.molecule.pbc.c)
        await pilot.press("left_square_bracket")
        await pilot.pause()
        app._do_undo()
        await pilot.pause()
        undone = [is_fallback_cell(f.pbc) for f in app._frames]
        app._do_redo()
        await pilot.pause()
        return before, after, shown, undone, _cells(app._frames)

    before, after, shown, undone, redone = drive(app_for(traj, trajectory_file=True), run)
    assert before == [True, True, True]
    assert after == [(20.0, 21.0, 22.0)] * 3
    assert shown == (20.0, 21.0, 22.0)
    assert undone == [True, True, True]
    assert redone == [(20.0, 21.0, 22.0)] * 3


def test_removing_the_cell_applies_to_every_frame(tmp_path):
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])

    async def run(pilot, app):
        await idle(pilot, app)
        app._do_update_cell("none")
        await idle(pilot, app)
        removed = _cells(app._frames)
        app._do_undo()
        await pilot.pause()
        return removed, _cells(app._frames)

    removed, restored = drive(app_for(traj, trajectory_file=True), run)
    assert removed == [None, None, None]
    assert restored == [(15.0, 15.0, 15.0)] * 3


# ---- the 1e6 A stand-in cell is "no cell" -----------------------------------------------

@pytest.mark.parametrize("cmd", [
    "wrap_center", "wrap_forward", "sort", "extend", "del_lone", "edit_h", "pack_sorbates",
    "write_cif", "write_poscar", "write_crystal", "write_mpmc", "reduce_cell", "pxrd",
    "density3d", "void_vol", "surface_area", "pore_size",
])
def test_stand_in_cell_counts_as_no_cell(tmp_path, cmd):
    traj = xyz_traj_nocell(tmp_path / "nocell.xyz")

    async def run(pilot, app):
        await idle(pilot, app)
        assert is_fallback_cell(app.molecule.pbc)
        before = len(app._notifications)
        command(app, cmd)
        await pilot.pause()
        await idle(pilot, app)
        await pilot.pause()
        return ([type(s).__name__ for s in app.screen_stack[1:]],
                [n.message for n in app._notifications][before:], len(app._undo_stack))

    screens, messages, undo_depth = drive(app_for(traj, trajectory_file=True), run)
    assert screens == []
    assert len(messages) == 1 and ("unit cell" in messages[0] or "PBC" in messages[0])
    assert undo_depth == 0


# ---- 12. Undo on a trajectory frame -----------------------------------------------------

def test_undo_on_a_trajectory_frame_undoes_the_frame(tmp_path):
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])

    async def run(pilot, app):
        await idle(pilot, app)
        original = app.molecule.atoms[0].x.copy()
        app._do_edit_coord(0, "1.0 2.0 3.0")
        await idle(pilot, app)
        edited = app._frames[0].atoms[0].x.copy()
        app._do_undo()
        await pilot.pause()
        same_object = app.molecule is app._frames[0]
        await pilot.press("right_square_bracket")
        await pilot.pause()
        await pilot.press("left_square_bracket")
        await pilot.pause()
        return original, edited, same_object, app.molecule.atoms[0].x.copy()

    original, edited, same_object, back = drive(app_for(traj, trajectory_file=True), run)
    np.testing.assert_allclose(edited, [1.0, 2.0, 3.0])
    assert same_object
    np.testing.assert_allclose(back, original)


def test_undo_after_stepping_away_goes_back_to_the_edited_frame(tmp_path):
    traj = pdb_traj(tmp_path / "traj.pdb", [3, 3, 3])

    async def run(pilot, app):
        await idle(pilot, app)
        first = np.array([a.x for a in app._frames[0].atoms])
        await pilot.press("right_square_bracket")            # frame 2
        await pilot.pause()
        original = app.molecule.atoms[0].x.copy()
        app._do_edit_coord(0, "1.0 2.0 3.0")
        await idle(pilot, app)
        await pilot.press("left_square_bracket")             # back to frame 1, then Undo
        await pilot.pause()
        app._do_undo()
        await pilot.pause()
        return (first, np.array([a.x for a in app._frames[0].atoms]), original,
                app._frames[1].atoms[0].x.copy(), app._frame_idx, app.molecule is app._frames[1])

    first, first_after, original, restored, shown, same = drive(
        app_for(traj, trajectory_file=True), run)
    np.testing.assert_array_equal(first_after, first)        # frame 1 was never edited
    np.testing.assert_allclose(restored, original)           # frame 2 got its atom back
    assert shown == 1 and same


# ---- 13. No file loaded: Save as PDB / LAMMPS -------------------------------------------

@pytest.mark.parametrize("cmd", ["write_pdb", "write_lammps"])
def test_saving_an_empty_buffer_is_refused(tmp_path, monkeypatch, cmd):
    monkeypatch.chdir(tmp_path)

    async def run(pilot, app):
        command(app, cmd)
        await pilot.pause()
        await pilot.pause()
        return [type(s).__name__ for s in app.screen_stack[1:]], toasts(app)

    screens, messages = drive(app_for(), run)
    assert screens == []
    assert len(messages) == 1 and "no atoms" in messages[0].lower()
    assert list(tmp_path.iterdir()) == []


# ---- 14. The Close button cancels a running computation ---------------------------------

@pytest.mark.parametrize("cmd, prefix, fn_name", [
    ("rdf", "rdf", "run_rdf"),
    ("coordination", "coord", "run_coordination"),
    ("hbonds", "hb", "run_hbonds"),
    ("density", "dens", "run_density_profile"),
    ("pxrd", "pxrd", "run_pxrd"),
])
def test_close_button_cancels_the_computation(tmp_path, monkeypatch, cmd, prefix, fn_name):
    box = water_box(tmp_path / "wbox.pdb")
    gate = Gate(getattr(analyses, fn_name))
    monkeypatch.setattr(analyses, fn_name, gate)

    async def run(pilot, app):
        command(app, cmd)
        scr = await wait_for_screen_with(pilot, app, prefix)
        press(scr, prefix + "-compute")
        await wait_for(pilot, gate.entered.is_set, "the computation to start")
        task = scr._compute_task
        press(scr, prefix + "-close")
        await main_screen(pilot, app)
        gate.release()
        await wait_for(pilot, task.done, "the task to end")
        await pilot.pause(0.2)
        return task.cancelled(), toasts(app, "error")

    assert drive(app_for(box), run) == (True, [])


async def wait_for_screen_with(pilot, app, prefix):
    await wait_for(pilot, lambda: len(app.screen_stack) == 2 and app.screen.is_mounted
                   and bool(app.screen.query("#" + prefix + "-compute")), prefix + " screen")
    await pilot.pause()
    return app.screen


def test_database_close_button_cancels_the_fetch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gate = _fake_backend(monkeypatch, tmp_path, gate=True)

    async def run(pilot, app):
        scr = await _start_fetch(pilot, app)
        await wait_for(pilot, gate.entered.is_set, "the download to start")
        press(scr, "db-close")
        await main_screen(pilot, app)
        gate.release()
        await pilot.pause(0.5)
        return len(app._tabs), toasts(app, "error")

    assert drive(app_for(), run) == (1, [])


# ---- 15. Void Volume without a cell -----------------------------------------------------

@pytest.mark.parametrize("with_file", [True, False])
def test_void_volume_needs_a_cell(tmp_path, with_file):
    app = app_for(water_xyz(tmp_path / "water.xyz")) if with_file else app_for()

    async def run(pilot, app):
        command(app, "void_vol")
        await idle(pilot, app)
        await pilot.pause(0.2)
        return [type(s).__name__ for s in app.screen_stack[1:]], toasts(app)

    screens, messages = drive(app, run)
    assert screens == []                                     # no "0.00 A^3, 0.0%" report
    assert len(messages) == 1 and "PBC" in messages[0]


# ---- 16. Edit H Distances ---------------------------------------------------------------

@pytest.mark.parametrize("text", ["O nan", "O inf", "O -1", "O 0", "O -inf"])
def test_edit_h_refuses_distances_that_are_not_lengths(tmp_path, text):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        before = np.array([a.x for a in app.molecule.atoms])
        command(app, "edit_h")
        await answer_text(pilot, app, text)
        await main_screen(pilot, app)
        await idle(pilot, app)
        return before, np.array([a.x for a in app.molecule.atoms]), len(app._undo_stack), \
            toasts(app)

    before, after, undo_depth, messages = drive(app_for(box), run)
    np.testing.assert_array_equal(after, before)
    assert undo_depth == 0
    assert messages and not any(m.startswith("Set ") for m in messages)


def test_edit_h_still_sets_a_real_distance(tmp_path):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        command(app, "edit_h")
        await answer_text(pilot, app, "O 1.1")
        await main_screen(pilot, app)
        await idle(pilot, app)
        xyz = np.array([a.x for a in app.molecule.atoms])
        return [float(np.linalg.norm(xyz[i] - xyz[0])) for i in (1, 2)]

    np.testing.assert_allclose(drive(app_for(box), run), [1.1, 1.1], atol=1e-6)


# ---- 18. Atom-number labels follow their atoms across frames ----------------------------

def _expected_label_cell(view, atom, centroid):
    w, h = view.size.width, view.size.height
    px_w, px_h = w * 2, h * 4
    scale = min(px_w, px_h) / 2
    pos = view.rot_matrix @ (atom.position - centroid)
    pos[0] += view.pan_x
    pos[1] += view.pan_y
    pos[2] += view.camera_distance
    sx = px_w / 2 + pos[0] * 1.5 / pos[2] * scale
    sy = px_h / 2 - pos[1] * 1.5 / pos[2] * scale
    return int(sy / 4) - 1, int(sx / 2)


def _drawn_label_cell(view, text):
    for row, strip in enumerate(view._cached_strips):
        col = 0
        for seg in strip:
            style = seg.style
            if style and style.bold and style.color and style.color.triplet == (255, 255, 0):
                i = seg.text.find(text)
                if i >= 0:
                    return row, col + i
            col += len(seg.text)
    return None


def test_atom_number_labels_use_the_locked_centroid(tmp_path):
    traj = pdb_traj(tmp_path / "var.pdb", [4, 2, 5])

    async def run(pilot, app):
        await idle(pilot, app)
        view = app.query_one(V.MoleculeView)
        await pilot.press("o")
        await pilot.press("number_sign")
        first_centroid = app._frames[0].center()
        out = []
        for _ in range(3):
            await pilot.pause()
            await pilot.pause()
            view._rebuild(view.size.width, view.size.height)
            last = len(app.molecule.atoms)
            # atoms are drawn about frame 1's centroid once a frame step has locked it
            centroid = first_centroid if app._frame_idx else app.molecule.center()
            out.append((_expected_label_cell(view, app.molecule.atoms[last - 1], centroid),
                        _drawn_label_cell(view, str(last))))
            await pilot.press("right_square_bracket")
        return out

    for expected, drawn in drive(app_for(traj, trajectory_file=True), run):
        assert drawn == expected


# ---- 19. Square brackets in a file name -------------------------------------------------

def test_tab_label_shows_brackets_in_a_file_name(tmp_path):
    path = water_xyz(tmp_path / "MOF[Cu].xyz")

    async def run(pilot, app):
        await wait_for(pilot, lambda: bool(app.query("#tab-0")), "the tab label")
        await pilot.pause()
        return str(app.query_one("#tab-0", Label).render())

    assert drive(app_for(path), run).strip() == "MOF[Cu].xyz"


# ---- 20 / 21. One edit at a time --------------------------------------------------------

def _symbols(mol) -> str:
    return "".join(sorted(a.element.symbol for a in mol.atoms))


def test_edits_and_undo_wait_for_the_running_edit(tmp_path, monkeypatch):
    box = water_box(tmp_path / "wbox.pdb")
    gate = Gate(V.PdbWizardApp._precompute_geometry)
    monkeypatch.setattr(V.PdbWizardApp, "_precompute_geometry", staticmethod(gate))

    async def run(pilot, app):
        app._do_substitute("O S")                            # refresh starts and is held
        await wait_for(pilot, gate.entered.is_set, "the refresh to start")
        app._do_undo()                                       # pressed while it is running
        app._do_substitute("H F")
        app._do_extend((1, 0, 0))
        await pilot.pause()
        during = _symbols(app.molecule), len(app.molecule.atoms)
        refused = toasts(app, "warning")
        gate.release()
        await idle(pilot, app)
        panel = app.query_one(GeometryPanel)
        after = _symbols(app.molecule), panel._molecule is app.molecule
        app._do_undo()                                       # now it works
        await pilot.pause()
        return during, refused, after, _symbols(app.molecule)

    during, refused, after, undone = drive(app_for(box), run)
    assert during == ("HHS", 3)                              # nothing else touched it
    assert len(refused) == 3
    assert after == ("HHS", True)                            # panel shows what Save writes
    assert undone == "HHO"


# ---- 22. Export PNG without a file name -------------------------------------------------

def test_export_png_without_a_file_name(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.chdir(tmp_path)

    async def run(pilot, app):
        app._do_update_cell("20 20 20 90 90 90")             # an empty box: no file path
        await idle(pilot, app)
        app.action_export_png()
        await idle(pilot, app)
        return toasts(app, "error")

    assert drive(app_for(), run) == []
    assert [p.name for p in tmp_path.iterdir()] == ["untitled.png"]


# ---- Delete Lone Atoms dialog text ------------------------------------------------------

def test_delete_lone_atoms_dialog_does_not_claim_it_cannot_be_undone(tmp_path):
    box = water_box(tmp_path / "wbox.pdb")

    async def run(pilot, app):
        command(app, "del_lone")
        modal = await on_screen(pilot, app, V.ConfirmModal)
        text = modal._description
        await pilot.press("escape")
        await main_screen(pilot, app)
        return text

    assert "cannot be undone" not in drive(app_for(box), run)
