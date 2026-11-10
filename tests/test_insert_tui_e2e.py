"""End-to-end tests of the TUI's Save as MPMC PDB flow, driven with textual's pilot.

Each test starts the real app, runs the export command and answers every dialog
the way a user would (buttons, radio buttons, table rows, text inputs), then
checks the file that lands on disk. Needs textual and pytest-asyncio.

Run from the repo root:  python -m pytest tests/
"""
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

SRC = Path(__file__).resolve().parent.parent / "dist" / "pdb_wizard.py"  # built by conftest.py
spec = importlib.util.spec_from_file_location("pdb_wizard_dist", SRC)
pw = importlib.util.module_from_spec(spec)
sys.modules["pdb_wizard_dist"] = pw  # dataclasses look the module up while it loads; not "pdb_wizard", which is the package
spec.loader.exec_module(pw)


def framework(charged=True):
    pbc = pw.PBC(20.0, 15.0, 18.0, 90, 99.0, 90)
    atoms = []
    for k, f in enumerate(np.array(np.meshgrid(*[np.arange(0, 1, 0.25)] * 3)).reshape(3, -1).T):
        atom = pw.Atom(*(f @ pbc.basis_matrix), "C")
        atom.charge = (0.1 if k % 2 else -0.1) if charged else 0.0
        atom.sigma, atom.epsilon, atom.alpha = 3.431, 52.84, 1.288
        atoms.append(atom)
    return pw.Molecule(atoms=atoms, pbc=pbc)


def records(path):
    return [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.startswith("ATOM") and " BOX " not in ln]


async def wait_for(pilot, cond, what, tries=100):
    for _ in range(tries):
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError(f"timed out waiting for {what}")


async def on_screen(pilot, app, cls):
    await wait_for(pilot, lambda: isinstance(app.screen, cls), cls.__name__)
    return app.screen


async def choose_sorbate(pilot, app, name):
    modal = await on_screen(pilot, app, pw.SorbateModal)
    table = modal.query_one("#sorbate-table", pw.DataTable)
    keys = [k.value for k in table.rows]
    table.move_cursor(row=keys.index(name))
    await pilot.pause()
    await pilot.click("#sorb-ok")


async def type_into(pilot, app, cls, selector, text, button):
    modal = await on_screen(pilot, app, cls)
    modal.query_one(selector, pw.Input).value = text
    await pilot.pause()
    await pilot.click(button)


async def export(pilot, app, *, keep_charges=True, ff="#ff-none", sorbate=None, count=None, path):
    """Answer the whole Save as MPMC PDB wizard."""
    app.on_operations_panel_run_command(SimpleNamespace(command="write_mpmc"))
    if keep_charges:
        await on_screen(pilot, app, pw.ConfirmModal)
        await pilot.click("#confirm-ok")
    modal = await on_screen(pilot, app, pw.ForceFieldModal)
    if ff != "#ff-none":
        modal.query_one(ff, pw.RadioButton).value = True
        await pilot.pause()
    await pilot.click("#ff-apply")
    if sorbate is None:
        await on_screen(pilot, app, pw.SorbateModal)
        await pilot.click("#sorb-skip")
    else:
        await choose_sorbate(pilot, app, sorbate)
        for answer in ([count] if isinstance(count, str) else count):
            await type_into(pilot, app, pw.InputModal, "#modal-input", answer, "#input-ok")
    await type_into(pilot, app, pw.FileSaveModal, "#file-input", str(path), "#file-save")
    await wait_for(pilot, lambda: Path(path).exists(), f"{path} to be written")


@pytest.mark.asyncio
async def test_tui_inserts_n_waters(tmp_path):
    out = tmp_path / "packed.pqr"
    app = pw.PdbWizardApp(framework(), filepath=str(tmp_path / "mof.pqr"))
    async with app.run_test(size=(160, 50)) as pilot:
        await export(pilot, app, sorbate="h2o_dc97", count="5", path=out)
    recs = records(out)
    assert len({r[5] for r in recs if r[4] == "M"}) == 5
    assert sum(r[4] == "F" for r in recs) == 64
    assert all(ln.split()[3] == "=" for ln in out.read_text().splitlines() if ln.startswith("REMARK BOX"))
    assert [float(r[10]) for r in recs if r[4] == "F"][:2] in ([-0.1, 0.1], [0.1, -0.1])  # charges kept


@pytest.mark.asyncio
async def test_tui_fill_and_a_bad_count_first(tmp_path):
    out = tmp_path / "full.pqr"
    app = pw.PdbWizardApp(framework(), filepath=str(tmp_path / "mof.pqr"))
    async with app.run_test(size=(160, 50)) as pilot:
        await export(pilot, app, sorbate="h2o_spce", count=["lots", "fill"], path=out)
    assert len({r[5] for r in records(out) if r[4] == "M"}) > 5


@pytest.mark.asyncio
async def test_tui_default_file_name_is_pqr(tmp_path):
    app = pw.PdbWizardApp(framework(), filepath=str(tmp_path / "mof.cif"))
    async with app.run_test(size=(160, 50)) as pilot:
        app.on_operations_panel_run_command(SimpleNamespace(command="write_mpmc"))
        await on_screen(pilot, app, pw.ConfirmModal)
        await pilot.click("#confirm-ok")
        await on_screen(pilot, app, pw.ForceFieldModal)
        await pilot.click("#ff-apply")
        await on_screen(pilot, app, pw.SorbateModal)
        await pilot.click("#sorb-skip")
        modal = await on_screen(pilot, app, pw.FileSaveModal)
        assert modal.query_one("#file-input", pw.Input).value == str(tmp_path / "mof_out_mpmc.pqr")


@pytest.mark.asyncio
async def test_tui_resave_with_force_field_keeps_the_water_model(tmp_path):
    packed = tmp_path / "packed.pqr"
    mof = framework()
    placed = pw.insert_sorbates(mof, "h2o_dc97", 4, seed=3)
    lines = pw.format_inserted_sorbates("h2o_dc97", placed, start_atom_id=len(mof.atoms) + 1)
    pw.write_mpmc_pdb(mof, str(packed), sorbate_lines=lines)
    mol = pw.read_file(str(packed))
    mol = mol[0] if isinstance(mol, (list, tuple)) else mol
    out = tmp_path / "resaved.pqr"
    app = pw.PdbWizardApp(mol, filepath=str(packed))
    async with app.run_test(size=(160, 50)) as pilot:
        await export(pilot, app, ff="#ff-oplsaa", path=out)
    before = [r[2:3] + r[9:14] for r in records(packed) if r[4] == "M"]
    after = [r[2:3] + r[9:14] for r in records(out) if r[4] == "M"]
    assert after == before and len(after) == 16  # movable, same model parameters
    c = next(r for r in records(out) if r[4] == "F")
    opls_c = pw.get_forcefield(0)["C"]
    assert float(c[12]) == pytest.approx(opls_c["epsilon"], abs=1e-4)  # framework did get the FF


@pytest.mark.asyncio
async def test_tui_cancel_at_the_count_writes_nothing(tmp_path):
    app = pw.PdbWizardApp(framework(), filepath=str(tmp_path / "mof.pqr"))
    async with app.run_test(size=(160, 50)) as pilot:
        app.on_operations_panel_run_command(SimpleNamespace(command="write_mpmc"))
        await on_screen(pilot, app, pw.ConfirmModal)
        await pilot.click("#confirm-ok")
        await on_screen(pilot, app, pw.ForceFieldModal)
        await pilot.click("#ff-apply")
        await choose_sorbate(pilot, app, "h2o_tip3p")
        await on_screen(pilot, app, pw.InputModal)
        await pilot.click("#input-cancel")
        await wait_for(pilot, lambda: not isinstance(app.screen, pw.InputModal), "the dialog to close")
        await pilot.pause(0.2)
        assert not isinstance(app.screen, pw.FileSaveModal)
    assert not list(tmp_path.glob("*.pqr"))


@pytest.mark.asyncio
async def test_tui_charges_from_file_cover_framework_only(tmp_path):
    packed = tmp_path / "packed.pqr"
    mof = framework(charged=False)
    placed = pw.insert_sorbates(mof, "h2o_dc97", 2, seed=4)
    lines = pw.format_inserted_sorbates("h2o_dc97", placed, start_atom_id=len(mof.atoms) + 1)
    buf = io.StringIO()
    pw.write_mpmc_pdb(mof, buf, sorbate_lines=lines)
    mol = pw.read_pdb(io.StringIO(buf.getvalue()), "packed.pqr")
    mol = pw.Molecule(atoms=mol[0], pbc=mol[1])
    # waters carry charges, so the wizard asks whether to keep them; say no, load a file
    app = pw.PdbWizardApp(mol, filepath=str(packed))
    q = tmp_path / "q.txt"
    q.write_text("\n".join(["0.05", "-0.05"] * 32) + "\n")  # 64 framework atoms
    out = tmp_path / "charged.pqr"
    async with app.run_test(size=(160, 50)) as pilot:
        app.on_operations_panel_run_command(SimpleNamespace(command="write_mpmc"))
        await on_screen(pilot, app, pw.ConfirmModal)
        await pilot.click("#confirm-cancel")  # don't keep: choose new charges
        await on_screen(pilot, app, pw.ConfirmModal)  # file or QEq?
        await pilot.click("#confirm-ok")  # from a file
        modal = await on_screen(pilot, app, pw.ChargesFileModal)
        # what picking the file in the dialog's directory tree does
        modal.on_directory_tree_file_selected(SimpleNamespace(path=q))
        await pilot.pause()
        await pilot.click("#charges-ok")
        await on_screen(pilot, app, pw.ForceFieldModal)
        await pilot.click("#ff-apply")
        await on_screen(pilot, app, pw.SorbateModal)
        await pilot.click("#sorb-skip")
        await type_into(pilot, app, pw.FileSaveModal, "#file-input", str(out), "#file-save")
        await wait_for(pilot, lambda: out.exists(), "the file")
    recs = records(out)
    assert sorted({float(r[10]) for r in recs if r[4] == "F"}) == [-0.05, 0.05]
    assert [float(r[10]) for r in recs if r[4] == "M" and r[2] == "M"] == [-1.038, -1.038]
