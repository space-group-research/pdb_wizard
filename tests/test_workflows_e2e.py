"""The workflows a student runs, end to end, in all three front ends.

Each test does one real job (CIF to an MPMC-ready framework; an isotherm set up, run in
MPMC and plotted; CP2K and OpenMM inputs; analysing a Monte Carlo trajectory; pictures)
three times: through the classic text menu (scripted input), the terminal viewer (textual's
pilot, answering every dialog) and the desktop GUI (offscreen Qt). The three must write the
same files and show the same numbers and the same warnings.

Needs PySide6, pytest-qt and textual. The parts that run MPMC itself need ``--mpmc PATH``
(or ``mpmc`` on PATH) and are skipped otherwise.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pytestqt")
pytest.importorskip("textual")

from textual.widgets import Input, Label, RadioButton, TextArea  # noqa: E402

import pdb_wizard.menu as menu  # noqa: E402
from pdb_wizard import gui  # noqa: E402
from pdb_wizard.analyses import check_text, run_isotherm, run_msd, run_rdf  # noqa: E402
from pdb_wizard.charges import apply_qeq_charges  # noqa: E402
from pdb_wizard.forcefields import apply_ff_to_system_report, get_forcefield  # noqa: E402
from pdb_wizard.formats import write_cif  # noqa: E402
from pdb_wizard.gui_tools import AnalysisDialog  # noqa: E402
from pdb_wizard.io import read_file, write_mpmc_pdb  # noqa: E402
from pdb_wizard.templates import ENGINES  # noqa: E402
from pdb_wizard.trajectory import read_trajectory  # noqa: E402
from pdb_wizard.viewer import (  # noqa: E402
    ConfirmModal,
    FileSaveModal,
    ForceFieldModal,
    InfoModal,
    InputGeneratorScreen,
    MsdScreen,
    PdbWizardApp,
    SorbateModal,
    SpinBox,
)

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
PRESSURES = "0.1, 1.0"
STEPS, CORR = 1000, 100  # the smallest step count the forms allow


# ---- inputs -----------------------------------------------------------------------------


def bare_hkust_cif(folder: Path) -> str:
    """HKUST-1 as a student gets it: a P1 CIF, no charges, no force field."""
    mol = read_file(str(HKUST))
    for a in mol.atoms:
        a.charge = a.epsilon = a.sigma = a.alpha = 0.0
    path = folder / "hkust.cif"
    with open(path, "w") as f:
        write_cif(mol, f)
    return str(path)


def prepared_pqr(folder: Path) -> str:
    """HKUST-1 with QEq charges and OPLS-AA/UFF parameters, as an MPMC file."""
    mol = read_file(str(HKUST))
    apply_qeq_charges(mol)
    apply_ff_to_system_report(mol.atoms, get_forcefield(0))
    path = folder / "framework.pqr"
    write_mpmc_pdb(mol, str(path), write_charges=True, write_params=True)
    return str(path)


def atom_records(path) -> list:
    return [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.startswith("ATOM") and " BOX " not in ln]


def tree(folder) -> dict:
    """Every file under ``folder`` by relative path, for comparing whole output folders."""
    folder = Path(folder)
    return {str(p.relative_to(folder)): p.read_text() for p in sorted(folder.rglob("*"))
            if p.is_file()}


@pytest.fixture
def mpmc_bin(request):
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("needs an mpmc binary (--mpmc PATH)")
    return path


# ---- classic menu driver --------------------------------------------------------------------


def classic(path: str, answers: list) -> str:
    """Run the classic menu on ``path`` with scripted answers; returns what it printed."""
    printed: list = []
    it = iter(answers)

    def fake_input(prompt: str = "") -> str:
        printed.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"classic menu asked for more input: {prompt!r}")

    with mock.patch("builtins.input", fake_input), \
            mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        menu.run_classic(path)
    assert next(it, None) is None, "scripted answers left over"
    return "\n".join(printed)


def engine_number(engine: str) -> str:
    return str(list(ENGINES).index(engine) + 1)


def classic_form(engine: str, outdir, **changed) -> list:
    """Answers for main-menu 14: engine, one answer per form field, the output folder."""
    answers = [engine_number(engine)]
    answers += [str(changed.get(v.name, "")) for v in ENGINES[engine].user_vars]
    return answers + [str(outdir)]


# ---- terminal viewer driver ---------------------------------------------------------------


WAIT_SECONDS = 180  # generous: shared CI runners are several times slower than a desktop


async def wait_for(pilot, cond, what, seconds=WAIT_SECONDS):
    """Poll ``cond`` until it holds. The limit is wall-clock time, not a number of polls,
    so a slow machine gets the same chance as a fast one."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError(f"timed out after {seconds} s waiting for {what}")


async def on_screen(pilot, app, cls):
    """The screen of type ``cls`` once it is on top and has finished mounting, so that a
    click sent next lands on a laid-out widget."""
    await wait_for(pilot, lambda: isinstance(app.screen, cls) and app.screen.is_mounted,
                   cls.__name__)
    await pilot.pause()
    await pilot.pause()
    return app.screen


def command(app, name: str) -> None:
    app.on_operations_panel_run_command(SimpleNamespace(command=name))


async def read_info(pilot, app) -> str:
    """The text of the report on screen, then OK."""
    modal = await on_screen(pilot, app, InfoModal)
    text = modal.query_one("#info-text", TextArea).text
    await close_screen(pilot, app, modal, "#info-ok")
    return text


async def close_screen(pilot, app, screen, button: str) -> None:
    """Click ``button`` until ``screen`` is gone. A click sent while the screen is still
    being laid out can be lost on a loaded machine, so it is repeated."""
    deadline = time.monotonic() + WAIT_SECONDS
    while app.screen is screen and time.monotonic() < deadline:
        await pilot.click(button)
        for _ in range(20):
            await pilot.pause(0.05)
            if app.screen is not screen:
                return
    assert app.screen is not screen, f"{type(screen).__name__} did not close"


async def save_as(pilot, app, path) -> None:
    modal = await on_screen(pilot, app, FileSaveModal)
    modal.query_one("#file-input", Input).value = str(path)
    await pilot.pause()
    await pilot.click("#file-save")


def notices(app) -> str:
    return "\n".join(str(n.message) for n in app._notifications)


def tui(coro_fn, mol, path, frames=None):
    """Run ``coro_fn(pilot, app)`` inside a live terminal viewer on ``mol``."""
    async def go():
        app = PdbWizardApp(molecule=mol, filepath=str(path))
        if frames is not None:
            app._frames = frames
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            return await coro_fn(pilot, app)
    return asyncio.run(go())


async def tui_generate(pilot, app, engine: str, outdir, **changed) -> tuple:
    """Input Generator: pick the engine, fill the form, Generate. Returns (status, preview)."""
    command(app, "input_generator")
    screen = await on_screen(pilot, app, InputGeneratorScreen)
    if screen._engine_key != engine:
        screen.query_one("#gen-engine").value = engine
        await wait_for(pilot, lambda: screen._engine_key == engine, f"engine {engine}")
        await pilot.pause()
    for name, value in changed.items():
        widget = screen._form_widgets[name]
        if isinstance(widget, SpinBox):
            widget._update(int(value))
        else:
            widget.value = str(value)
    screen.query_one("#gen-outdir", Input).value = str(outdir)
    await pilot.pause()
    await pilot.click("#gen-validate")
    await pilot.pause()
    checked = screen.query_one("#gen-preview", TextArea).text
    await pilot.click("#gen-generate")
    await pilot.pause()
    await pilot.pause()
    status = str(screen.query_one("#gen-status", Label).render())
    preview = screen.query_one("#gen-preview", TextArea).text
    await pilot.click("#gen-close")
    await pilot.pause()
    return status, checked + "\n" + preview


# ---- desktop GUI driver -----------------------------------------------------------------


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


def gui_generate(w, engine: str, outdir, **changed) -> tuple:
    dlg = w.open_tool("input_generator")
    dlg.engine_box.setCurrentIndex(dlg.engine_box.findData(engine))
    for name, value in changed.items():
        widget = dlg.widgets[name]
        if hasattr(widget, "setValue"):
            widget.setValue(int(value))
        else:
            widget.setText(str(value))
    dlg.outdir.setText(str(outdir))
    dlg.validate()
    checked = dlg.preview.toPlainText()
    out = dlg.generate()
    preview = dlg.preview.toPlainText()
    dlg.close()
    return out.status, checked + "\n" + preview


# ======================================================================
# 1. CIF -> checked, charged, parameterised MPMC framework
# ======================================================================


def test_cif_to_mpmc_framework_in_all_three_front_ends(window, tmp_path):
    cif = bare_hkust_cif(tmp_path)
    expected_check = check_text(read_file(cif))
    assert "No charges" in expected_check and "No force-field parameters" in expected_check
    out = {name: tmp_path / f"{name}.pqr" for name in ("classic", "tui", "gui")}

    # classic: check (11 > 15), QEq (12 > 5), write MPMC with OPLS-AA/UFF (7)
    text = classic(cif, ["11", "15", "16", "12", "5", "6", "11", "15", "16",
                         "7", "n", "y", "0", str(out["classic"]), "9"])
    assert expected_check in text
    assert "QEq charges computed for 624 atoms" in text
    classic_after = text.split("QEq charges computed")[1]

    # terminal viewer
    async def run_tui(pilot, app):
        command(app, "check")
        before = await read_info(pilot, app)
        command(app, "qeq_charges")
        await on_screen(pilot, app, ConfirmModal)
        await pilot.click("#confirm-ok")
        report = await read_info(pilot, app)
        command(app, "check")
        after = await read_info(pilot, app)
        command(app, "write_mpmc")
        await on_screen(pilot, app, ConfirmModal)  # keep the charges just made?
        await pilot.click("#confirm-ok")
        modal = await on_screen(pilot, app, ForceFieldModal)
        modal.query_one("#ff-oplsaa", RadioButton).value = True
        await pilot.pause()
        await pilot.click("#ff-apply")
        await on_screen(pilot, app, SorbateModal)
        await pilot.click("#sorb-skip")
        await save_as(pilot, app, out["tui"])
        await wait_for(pilot, lambda: out["tui"].exists(), "the MPMC file")
        return before, report, after

    tui_before, tui_report, tui_after = tui(run_tui, read_file(cif), cif)
    assert tui_before == expected_check
    assert "QEq charges computed for 624 atoms" in tui_report

    # desktop GUI
    assert window.open_path(cif) and not window.warnings
    assert window.open_tool("check").text.toPlainText() == expected_check
    window.qeq_charges()
    gui_after = window.open_tool("check").text.toPlainText()
    window.ff_combo.setCurrentIndex(window.ff_combo.findData(0))
    window.apply_force_field()
    assert window.save_path(str(out["gui"]))

    # the same check after charging, in all three
    assert "No charges" not in tui_after and "Charges present" in tui_after
    assert tui_after == gui_after
    assert gui_after in classic_after

    # the same framework file from all three
    records = {name: atom_records(path) for name, path in out.items()}
    assert len(records["classic"]) == 624
    assert records["classic"] == records["tui"] == records["gui"]
    back = read_file(str(out["gui"]))
    charges = np.array([a.charge for a in back.atoms])
    assert abs(charges.sum()) < 1e-3 and np.all(np.abs(charges) > 0)
    assert all(a.epsilon > 0 and a.sigma > 0 for a in back.atoms)
    final = check_text(back)
    assert final.startswith("Structure check: nothing suspicious found"), final


# ======================================================================
# 2. Isotherm: set up, run MPMC, plot
# ======================================================================


def generate_isotherms(window, tmp_path) -> dict:
    """The same isotherm set up from each front end. Returns {name: folder}."""
    pqr = prepared_pqr(tmp_path)
    changed = dict(pressures=PRESSURES, numsteps=STEPS, corrtime=CORR)
    dirs = {name: tmp_path / name / "isotherm" for name in ("classic", "tui", "gui")}

    text = classic(pqr, ["14", *classic_form("mpmc_isotherm", dirs["classic"], **changed), "9"])
    assert "Generated 2 directories" in text and "All checks passed" in text

    status, preview = tui(lambda pilot, app: tui_generate(pilot, app, "mpmc_isotherm",
                                                          dirs["tui"], **changed),
                          read_file(pqr), pqr)
    assert "Generated 2 directories" in status, status

    assert window.open_path(pqr)
    status, preview = gui_generate(window, "mpmc_isotherm", dirs["gui"], **changed)
    assert "Generated 2 directories" in status, status
    return dirs


def test_isotherm_setup_is_identical_in_all_three_front_ends(window, tmp_path):
    dirs = generate_isotherms(window, tmp_path)
    trees = {name: tree(d) for name, d in dirs.items()}
    assert sorted(trees["classic"]) == sorted(trees["tui"]) == sorted(trees["gui"])
    assert {"P_0p1_atm/mpmc.inp", "P_1_atm/input.pqr", "run_all.sh"} <= set(trees["classic"])
    for name in ("tui", "gui"):
        for rel, body in trees["classic"].items():
            assert trees[name][rel] == body, f"{name}: {rel} differs from the classic menu's"
    inp = trees["gui"]["P_1_atm/mpmc.inp"]
    assert "pbc_cutoff" in inp and "polarization          on" in inp
    assert f"numsteps              {STEPS}" in inp


def test_isotherm_runs_in_mpmc_and_plots_the_same_everywhere(window, tmp_path, mpmc_bin):
    dirs = generate_isotherms(window, tmp_path)
    base = dirs["gui"]  # the three folders are identical (previous test)
    for p_dir in sorted(base.glob("P_*")):
        with open(p_dir / "mpmc.log", "w") as log:
            run = subprocess.run([mpmc_bin, "mpmc.inp"], cwd=p_dir, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=900)
        assert run.returncode == 0, (p_dir / "mpmc.log").read_text()[-2000:]
        assert "simulation exiting successfully" in (p_dir / "mpmc.log").read_text()

    ref = run_isotherm(str(base))
    assert list(ref.x) == [0.1, 1.0] and np.all(ref.y > 0) and ref.y[1] > ref.y[0]
    assert "MPMC's own averages" in ref.text and "has not finished" not in ref.text
    assert ref.text.count("±") >= 4  # N and wt% with error bars at both pressures

    # classic (15)
    text = classic(str(HKUST), ["15", str(base), "", "9"])
    assert ref.summary in text and ref.text in text

    # terminal viewer
    async def run_tui(pilot, app):
        command(app, "isotherm_plot")
        await save_as(pilot, app, base)
        await wait_for(pilot, lambda: type(app.screen).__name__ == "IsothermPlotScreen", "plot")
        screen = app.screen
        status = str(screen.query_one("#iso-status", Label).render())
        points = (screen._p.copy(), screen._u.copy())
        await pilot.click("#iso-log")
        return status, points, await read_info(pilot, app)

    status, (p, u), log_text = tui(run_tui, read_file(str(HKUST)), HKUST)
    assert status == ref.summary and log_text == ref.text
    np.testing.assert_array_equal(p, ref.x)
    np.testing.assert_array_equal(u, ref.y)

    # desktop GUI
    dlg = window.open_tool("isotherm_plot")
    dlg.inputs["directory"].setText(str(base))
    dlg.run().wait()
    np.testing.assert_array_equal(dlg.plot.x, ref.x)
    np.testing.assert_array_equal(dlg.plot.y, ref.y)
    assert dlg.status.text() == ref.summary and dlg.text.toPlainText() == ref.text

    # the trajectory MPMC wrote is a Monte Carlo one: every front end must say so
    traj = base / "P_1_atm" / "trajectory.pqr"
    frames = read_trajectory(str(traj))
    assert len(frames) >= 3
    warn = run_msd(frames, "Cu").text
    assert "Monte Carlo trajectory" in warn

    text = classic(str(traj), ["11", "8", "Cu", "", "", "", "", "", "15", "16", "9"])
    assert warn in text and "Monte Carlo trajectory" in text.split("Structure check")[1]

    async def msd_tui(pilot, app):
        screen = MsdScreen(frames[0], frames)
        app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#msd-el").value = "Cu"
        await pilot.pause()
        await pilot.click("#msd-compute")
        await wait_for(pilot, lambda: screen._compute_task is not None
                       and screen._compute_task.done(), "MSD")
        await pilot.pause()
        return notices(app)

    assert warn in tui(msd_tui, frames[0], traj, frames)

    assert window.open_path(str(traj))
    window.load_task.wait()
    dlg = window.open_tool("msd")
    dlg.inputs["element"].setCurrentText("Cu")
    dlg.run().wait()
    assert dlg.text.toPlainText() == warn and dlg.text.isVisible()
    assert "Monte Carlo trajectory" in window.open_tool("check").text.toPlainText()

    # an analysis that is meaningful on Monte Carlo frames gives the same curve in the GUI
    # as the shared function, with no warning
    hydrogen = sorted({a.element.symbol for a in frames[-1].atoms} - {"Cu", "C", "O"})[0]
    rdf = run_rdf(frames[0], "Cu", hydrogen, frames=frames)
    dlg = AnalysisDialog("rdf", frames[0], frames, window)
    dlg.inputs["el1"].setCurrentText("Cu")
    dlg.inputs["el2"].setCurrentText(hydrogen)
    dlg.run().wait()
    np.testing.assert_array_equal(dlg.plot.y, rdf.y)
    assert rdf.text == "" and not dlg.text.isVisible()
    dlg.close()


# ======================================================================
# 3. CP2K and OpenMM inputs, with their cautions
# ======================================================================


@pytest.mark.parametrize("engine,caution", [
    ("cp2k_cell_opt", "Cu usually has unpaired electrons"),
    ("cp2k_dft_md", "Cu usually has unpaired electrons"),
    ("openmm_npt", "No template found for residue"),
])
def test_engine_inputs_and_cautions_match_in_all_three_front_ends(window, tmp_path, engine,
                                                                  caution):
    dirs = {name: tmp_path / name for name in ("classic", "tui", "gui")}

    text = classic(str(HKUST), ["14", *classic_form(engine, dirs["classic"]), "9"])
    assert caution in text and "Wrote" in text

    status, preview = tui(lambda pilot, app: tui_generate(pilot, app, engine, dirs["tui"]),
                          read_file(str(HKUST)), HKUST)
    assert status.startswith("Wrote"), status
    assert caution in preview

    assert window.open_path(str(HKUST))
    status, preview = gui_generate(window, engine, dirs["gui"])
    assert status.startswith("Wrote"), status
    assert preview.count(caution) == 2  # at Validate and again after Generate

    trees = {name: tree(d) for name, d in dirs.items()}
    assert trees["classic"] and trees["classic"] == trees["tui"] == trees["gui"]
    if engine.startswith("cp2k"):
        inp = next(body for rel, body in trees["gui"].items() if rel.endswith(".inp"))
        assert "&CELL" in inp and "ALPHA_BETA_GAMMA" not in inp
        assert "UKS" not in inp  # the template is still closed-shell: hence the caution


# ======================================================================
# 4. A CIF that is only the asymmetric unit
# ======================================================================

ASYMMETRIC_CIF = """data_asym
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 26.343
_cell_length_b 26.343
_cell_length_c 26.343
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cu1 Cu 0.2850 0.2850 0.0000 1.0
O1 O 0.3160 0.2430 0.0520 1.0
C1 C 0.2970 0.2030 0.0700 1.0
C2 C 0.3220 0.1780 0.1130 1.0
C3 C 0.3650 0.1990 0.1350 0.5
"""


def test_asymmetric_unit_cif_warns_in_all_three_front_ends(window, tmp_path):
    path = tmp_path / "asym.cif"
    path.write_text(ASYMMETRIC_CIF)
    sg = "names space group F m -3 m but lists no symmetry operators"
    disorder = "1 site with occupancy below 1"

    text = classic(str(path), ["11", "15", "16", "9"])
    assert text.count(sg) == 2 and disorder in text  # on loading and in the check

    async def run_tui(pilot, app):
        loaded = notices(app)
        command(app, "check")
        return loaded, await read_info(pilot, app)

    loaded, report = tui(run_tui, read_file(str(path)), path)
    assert sg in loaded and disorder in loaded
    assert sg in report and disorder in report

    assert window.open_path(str(path))
    assert len(window.warnings) == 1 and sg in window.warnings[0] and disorder in window.warnings[0]
    report_gui = window.open_tool("check").text.toPlainText()
    assert report_gui == report
    assert report.startswith("Structure check: ") and "warning" in report.splitlines()[0]


# ======================================================================
# 5. Deleting lone atoms from a MOF
# ======================================================================


def test_lone_atom_deletion_is_guarded_in_all_three_front_ends(window, tmp_path, monkeypatch):
    out = tmp_path / "after.xyz"
    # classic keeps its original behaviour (scripts rely on it) but says what it did, and
    # the new undo brings the copper back
    text = classic(str(HKUST), ["1", "5", "9", "17", "6", str(out), "9"])
    assert "Warning: This includes metal atoms (Cu)" in text and "Undid the last change" in text
    assert out.read_text().splitlines()[0].strip() == "624"

    async def run_tui(pilot, app):
        command(app, "del_lone")
        modal = await on_screen(pilot, app, ConfirmModal)
        shown = "\n".join(str(w.render()) for w in modal.query(Label))
        await pilot.click("#confirm-cancel")
        await pilot.pause()
        return shown, len(app.molecule.atoms)

    shown, n_atoms = tui(run_tui, read_file(str(HKUST)), HKUST)
    assert "Would delete: 48 Cu" in shown and "metal atoms (Cu)" in shown and n_atoms == 624

    assert window.open_path(str(HKUST))
    asked = []
    monkeypatch.setattr(window, "_confirm", lambda title, body: asked.append(body) or False)
    assert window.delete_lone_atoms(ask=True) == 0 and len(window.mol.atoms) == 624
    assert "48 Cu" in asked[0] and "metal atoms (Cu)" in asked[0]
    monkeypatch.setattr(window, "_confirm", lambda title, body: True)
    assert window.delete_lone_atoms(ask=True) == 48
    window.undo()
    assert len(window.mol.atoms) == 624


# ======================================================================
# 6. Packing sorbates and making a picture
# ======================================================================


def test_packing_and_pictures_in_all_three_front_ends(window, tmp_path):
    image = pytest.importorskip("PIL.Image")
    pqr = prepared_pqr(tmp_path)
    n_sites = 5  # h2_bss

    png = tmp_path / "classic.png"
    packed = tmp_path / "classic_packed.pqr"
    text = classic(pqr, ["12", "4", "h2_bss", "6", "6", "11", "15", "16", "13", "4", str(png),
                         "6", "7", "n", "n", str(packed), "9"])
    assert "Placed 6" in text and "nothing suspicious found" in text
    assert len(atom_records(packed)) == 624 + 6 * n_sites

    async def run_tui(pilot, app):
        command(app, "pack_sorbates")
        modal = await on_screen(pilot, app, SorbateModal)
        table = modal.query_one("#sorbate-table")
        table.move_cursor(row=[k.value for k in table.rows].index("h2_bss"))
        await pilot.pause()
        await pilot.click("#sorb-ok")
        await on_screen(pilot, app, type(app.screen))
        await wait_for(pilot, lambda: app.screen.query("#modal-input"), "count prompt")
        app.screen.query_one("#modal-input", Input).value = "6"
        await pilot.pause()
        await pilot.click("#input-ok")
        await wait_for(pilot, lambda: len(app.molecule.atoms) == 624 + 6 * n_sites, "packing")
        command(app, "check")
        report = await read_info(pilot, app)
        command(app, "export_png")
        target = Path(app.filepath).with_suffix(".png")
        await wait_for(pilot, target.exists, "the PNG")
        return report, target

    work = tmp_path / "tui"
    work.mkdir()
    copy = work / "framework.pqr"
    copy.write_text(Path(pqr).read_text())
    report, tui_png = tui(run_tui, read_file(str(copy)), copy)
    assert "nothing suspicious found" in report

    assert window.open_path(pqr)
    window.sorbate_count.setValue(6)
    assert window.sorbate_combo.currentData() == "h2_bss"
    assert window.pack_sorbates() == 6
    assert "nothing suspicious found" in window.open_tool("check").text.toPlainText()
    gui_png = tmp_path / "gui.png"
    assert window.export_image(str(gui_png))
    gui_packed = tmp_path / "gui_packed.pqr"
    assert window.save_path(str(gui_packed))
    assert len(atom_records(gui_packed)) == 624 + 6 * n_sites

    for path in (png, tui_png, gui_png):
        with image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        assert arr.shape[0] > 300 and arr.std() > 5, f"{path} looks empty"
