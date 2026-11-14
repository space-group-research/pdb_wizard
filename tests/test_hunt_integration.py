"""Bug hunt, integration: fixes that span the shared code and all three front ends.

1. Reducing a structure that is not the requested supercell used to return garbage with a
   success message (one HKUST-1 cell "reduced" as 2x2x2 gave 456 atoms at 5.08 g/cm3).
   The shared code now refuses, and each front end must show that refusal, not crash.
2. A trajectory without cell data carries a huge stand-in box. The shared availability
   check must treat it as "no cell" for the analyses that need a real one.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest import mock

import pytest

from pdb_wizard.analyses import unavailable
from pdb_wizard.atom import Atom
from pdb_wizard.io import fallback_cell, read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis
from pdb_wizard.reduce_cell import identify_framework_and_sorbates, reduce_to_primitive

HKUST = Path(__file__).resolve().parent.parent / "HKUST-1.pdb"
NOT_A_SUPERCELL = "this is not a 2 x 2 x 2 supercell"


def hkust() -> Molecule:
    return read_file(str(HKUST))


# ---- the shared rule ----------------------------------------------------------------------


def test_one_cell_is_not_a_supercell_but_an_extended_one_is():
    with pytest.raises(ValueError, match=NOT_A_SUPERCELL):
        identify_framework_and_sorbates(hkust(), 2, 2, 2)
    with pytest.raises(ValueError, match="not a 3 x 1 x 1 supercell"):
        reduce_to_primitive(hkust(), 3, 1, 1)
    doubled = extend_axis(hkust(), 0, 1)
    back = reduce_to_primitive(doubled, 2, 1, 1)
    assert len(back.atoms) == 624 and back.pbc.a == pytest.approx(26.343, abs=1e-3)


# ---- classic menu -------------------------------------------------------------------------


def test_classic_menu_prints_the_refusal_and_keeps_the_structure():
    import pdb_wizard.menu_tools as mt
    printed: list = []
    answers = iter(["3", "2", "2", "2", "6"])
    mol = hkust()
    with mock.patch("builtins.input", lambda prompt="": next(answers)), \
            mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        result = mt.cm_menu_edit(mol)
    assert result is mol and len(result.atoms) == 624
    assert NOT_A_SUPERCELL in "\n".join(printed)
    assert next(answers, None) is None


# ---- terminal viewer ----------------------------------------------------------------------


def test_terminal_viewer_shows_the_refusal_and_stays_up():
    pytest.importorskip("textual")
    from textual.widgets import Label

    from pdb_wizard.viewer import PdbWizardApp, ReduceCellScreen

    async def go():
        mol = hkust()
        app = PdbWizardApp(molecule=mol, filepath=str(HKUST))
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            screen = ReduceCellScreen(mol)
            app.push_screen(screen)
            deadline = time.monotonic() + 120
            while not (app.screen is screen and screen.is_mounted):
                assert time.monotonic() < deadline
                await pilot.pause(0.05)
            await pilot.pause()
            screen._do_analyze()  # the dialog opens at 2 x 2 x 2
            await pilot.pause()
            status = str(screen.query_one("#rc-status", Label).render())
            screen._do_reduce()
            await pilot.pause()
            return status, app.is_running, len(app.molecule.atoms), app.screen is screen

    status, running, n_atoms, still_open = asyncio.run(go())
    assert NOT_A_SUPERCELL in status
    assert running and n_atoms == 624 and still_open


# ---- desktop GUI --------------------------------------------------------------------------


def test_gui_dialog_shows_the_refusal_and_disables_reduce(request):
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pytestqt")
    qtbot = request.getfixturevalue("qtbot")
    from pdb_wizard.gui_tools import ReduceCellDialog

    mol = hkust()
    dlg = ReduceCellDialog(mol)
    qtbot.addWidget(dlg)
    dlg.analyze()  # opens at 2 x 2 x 2
    assert NOT_A_SUPERCELL in dlg.status.text()
    assert not dlg.reduce_button.isEnabled() and dlg.table.rowCount() == 0
    assert dlg.reduce() is None and len(mol.atoms) == 624

    doubled = extend_axis(hkust(), 0, 1)
    dlg = ReduceCellDialog(doubled)
    qtbot.addWidget(dlg)
    for spin, value in zip(dlg.spins, (2, 1, 1)):
        spin.setValue(value)
    dlg.analyze()
    assert dlg.reduce_button.isEnabled()
    assert len(dlg.reduce().atoms) == 624


# ---- the stand-in cell --------------------------------------------------------------------


def cell_less_frames() -> list:
    return [Molecule(atoms=[Atom(0, 0, 0.1 * f, "C"), Atom(1.5, 0, 0, "O")], pbc=fallback_cell())
            for f in range(4)]


@pytest.mark.parametrize("key,message", [
    ("pxrd", "Need a crystal structure with PBC for PXRD"),
    ("void_vol", "Need PBC for void volume"),
    ("surface_area", "Need PBC for surface area"),
    ("pore_size", "Need PBC for pore size"),
    ("density3d", "Need PBC for 3D density map"),
])
def test_stand_in_cell_is_no_cell_for_cell_analyses(key, message):
    frames = cell_less_frames()
    assert unavailable(key, frames[0], frames) == message


@pytest.mark.parametrize("key", ["pxrd", "void_vol", "surface_area", "pore_size", "rdf",
                                 "coordination", "info", "check"])
def test_a_real_cell_is_still_accepted(key):
    assert unavailable(key, hkust(), None) is None


@pytest.mark.parametrize("key", ["rdf", "coordination", "hbonds", "density", "msd", "rmsd"])
def test_analyses_that_never_needed_a_cell_still_run_without_one(key):
    """The owner's ruling: these keep running on cell-less trajectories."""
    frames = cell_less_frames()
    assert unavailable(key, frames[0], frames) is None
