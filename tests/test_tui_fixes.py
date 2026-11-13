"""Regression tests for terminal-viewer bugs found while matching the desktop GUI to it.

1. RDF: same-type g(r) was half the correct value (pairs counted once, normalised as if
   counted twice).
2. Reduce Supercell: the keep/discard toggle never fired (cell cursor) and would have
   crashed if it had (a row's values were passed as its key).
3. PXRD "Load Exp.": no CSV file could be read, with or without a header.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, Label  # noqa: E402

from pdb_wizard.atom import Atom  # noqa: E402
from pdb_wizard.io import read_file  # noqa: E402
from pdb_wizard.molecule import Molecule  # noqa: E402
from pdb_wizard.operations import extend_axis  # noqa: E402
from pdb_wizard.pbc import PBC  # noqa: E402
from pdb_wizard.reduce_cell import reduce_to_primitive  # noqa: E402
from pdb_wizard.sorbates import SORBATE_MODELS, insert_sorbates, sorbate_atoms  # noqa: E402
from pdb_wizard.viewer import (  # noqa: E402
    PdbWizardApp,
    PlotWidget,
    PxrdScreen,
    RdfScreen,
    ReduceCellScreen,
)

ROOT = Path(__file__).resolve().parent.parent


async def _wait(screen):
    for _ in range(1200):
        await asyncio.sleep(0.05)
        if screen._compute_task is not None and screen._compute_task.done():
            return screen._compute_task.result()
    raise AssertionError("compute did not finish")


def test_rdf_screen_same_type_tends_to_one():
    rng = np.random.default_rng(0)
    L = 25.0
    mol = Molecule(atoms=[Atom(*x, "Ar") for x in rng.random((1200, 3)) * L],
                   pbc=PBC(L, L, L, 90, 90, 90))

    async def run():
        app = PdbWizardApp(molecule=mol, filepath="gas.pdb")
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            screen = RdfScreen(mol)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#rdf-compute")
            await _wait(screen)
            plot = screen.query_one("#rdf-plot", PlotWidget)
            return plot._x.copy(), plot._y.copy()

    r, g = asyncio.run(run())
    assert g[r > 6].mean() == pytest.approx(1.0, abs=0.03)


def test_reduce_supercell_keep_toggle_and_reduce():
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    for axis in range(3):
        extend_axis(mol, axis, 1)
    model = SORBATE_MODELS["h2o_dc97"]
    mol.atoms.extend(sorbate_atoms(model, insert_sorbates(mol, model, 3), 2))
    mol.detect_bonds()
    expected = len(reduce_to_primitive(mol, 2, 2, 2, [1, 2]).atoms)

    async def run():
        app = PdbWizardApp(molecule=mol, filepath="x.pqr")
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            results = []
            screen = ReduceCellScreen(mol)
            app.push_screen(screen, callback=results.append)
            await pilot.pause()
            await pilot.click("#rc-analyze")
            await pilot.pause()
            table = screen.query_one("#rc-table", DataTable)
            assert table.row_count == 3
            table.focus()
            table.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
            keep, cell = list(screen._keep), table.get_cell_at((0, 0))
            await pilot.click("#rc-reduce")
            await pilot.pause()
            return keep, cell, results

    keep, cell, results = asyncio.run(run())
    assert keep == [False, True, True] and str(cell) == "[ ]"
    assert len(results[0].atoms) == expected


@pytest.mark.parametrize("text", ["two_theta,intensity\n5,10\n6,40\n7,20\n",
                                  "5,10\n6,40\n7,20\n", "5 10\n6 40\n7 20\n"])
def test_pxrd_loads_experimental_csv_and_xy(tmp_path, text):
    path = tmp_path / "exp.csv"
    path.write_text(text)
    mol = read_file(str(ROOT / "HKUST-1.pdb"))

    async def run():
        app = PdbWizardApp(molecule=mol, filepath="HKUST-1.pdb")
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            screen = PxrdScreen(mol)
            app.push_screen(screen)
            await pilot.pause()
            screen._load_experimental(str(path))
            await pilot.pause()
            plot = screen.query_one("#pxrd-plot", PlotWidget)
            return str(screen.query_one("#pxrd-status", Label).render()), plot._overlay_y

    status, overlay = asyncio.run(run())
    assert status.startswith("Experimental data loaded (3 points)")
    assert overlay is not None and overlay.max() == pytest.approx(100.0)
