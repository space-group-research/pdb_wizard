"""Edit > Pack Sorbates in the terminal viewer: the same packing as the desktop GUI's
Sorbates tab, from a built-in model or an MPMC .pqr template, with undo."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from pdb_wizard.io import read_file, write_mpmc_pdb  # noqa: E402
from pdb_wizard.panels import OPERATIONS, OperationsPanel  # noqa: E402
from pdb_wizard.viewer import (  # noqa: E402
    MENU_EDIT,
    FileSaveModal,
    InputModal,
    PdbWizardApp,
    SorbateModal,
)

ROOT = Path(__file__).resolve().parent.parent


def test_pack_sorbates_is_in_the_menus():
    assert ("pack_sorbates", "Pack Sorbates", "modify") in OPERATIONS
    assert ("pack_sorbates", "Pack Sorbates") in MENU_EDIT


async def _run(steps, mol=None, after=None):
    """Start Pack Sorbates, answer each dialog in turn, wait for the packing task.

    Returns (atom count, movable atoms, name of the screen left showing, after(app))."""
    mol = mol if mol is not None else read_file(str(ROOT / "HKUST-1.pdb"))
    app = PdbWizardApp(molecule=mol, filepath=str(ROOT / "HKUST-1.pdb"))
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        app.on_operations_panel_run_command(OperationsPanel.RunCommand("pack_sorbates"))
        await pilot.pause()
        for expect, answer in steps:
            screen = app.screen
            assert isinstance(screen, expect), type(screen).__name__
            if isinstance(screen, SorbateModal):
                assert screen.query("#sorb-file") and not screen.query("#sorb-skip")
            screen.dismiss(answer)
            await pilot.pause()
        for _ in range(400):
            await pilot.pause()
            if not app._bg_tasks:
                break
            await asyncio.sleep(0.05)
        n_atoms = len(app.molecule.atoms)
        movable = [a for a in app.molecule.atoms if a.mpmc_flag == "M"]
        extra = after(app) if after else None
        return n_atoms, movable, type(app.screen).__name__, extra


def test_pack_builtin_model_with_undo():
    def undo(app):
        app._do_undo()
        return len(app.molecule.atoms)

    n, movable, _, after_undo = asyncio.run(
        _run([(SorbateModal, "h2o_tip3p"), (InputModal, "4")], after=undo))
    assert n == 624 + 12 and len(movable) == 12
    assert sorted({a.mpmc_mol for a in movable}) == [2, 3, 4, 5]
    assert after_undo == 624


def test_pack_from_template_and_fill(tmp_path):
    ar = tmp_path / "ar.pqr"
    ar.write_text("ATOM      1 Ar   ARG M   1       0.000   0.000   0.000 39.948 0.0 1.6418 "
                  "119.8 3.4 0.0 0.0\nEND\n")

    def save(app):
        out = tmp_path / "out.pqr"
        with open(out, "w") as f:
            write_mpmc_pdb(app.molecule, f)
        return sum(ln.split()[4] == "M" for ln in out.read_text().splitlines()
                   if ln.startswith("ATOM"))

    n, movable, _, written = asyncio.run(_run(
        [(SorbateModal, SorbateModal.FROM_FILE), (FileSaveModal, str(ar)), (InputModal, "fill")],
        after=save))
    assert len(movable) > 50 and all(a.element.symbol == "Ar" for a in movable)
    assert written == len(movable)  # still movable in the MPMC file


def test_pack_cancel_bad_count_and_bad_template(tmp_path):
    bad = tmp_path / "bad.pqr"
    bad.write_text("nothing here\n")
    runs = [
        [(SorbateModal, None)],
        [(SorbateModal, "h2o_tip3p"), (InputModal, "")],
        [(SorbateModal, "h2o_tip3p"), (InputModal, "two")],
        [(SorbateModal, SorbateModal.FROM_FILE), (FileSaveModal, str(bad))],
    ]
    results = [asyncio.run(_run(steps)) for steps in runs]
    assert [r[0] for r in results] == [624] * 4
    assert results[2][2] == "InputModal"  # an invalid count asks again
