"""Cancel buttons must actually cancel the operation.

Regression: in the MPMC export flow the SorbateModal's Cancel button was wired
identically to its Skip button (both dismissed ""), and ForceFieldModal's Cancel
matched "None selected" — so pressing Cancel continued the flow and still wrote
the file. Cancel now returns None (abort) while Skip / "no FF" return "" (proceed
without that option), and the flow callbacks abort on None.
"""

from __future__ import annotations

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import ForceFieldModal, PdbWizardApp, SorbateModal


def _mol(n=3):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _app():
    return PdbWizardApp(molecule=_mol(), filepath="t.pdb")


# ---------------------------------------------------------------------------
# Modal dismiss values: Cancel -> None, Skip / no-selection -> ""
# ---------------------------------------------------------------------------

class TestSorbateModalCancel:
    @pytest.mark.asyncio
    async def test_cancel_button_returns_none(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            results = []
            app.push_screen(SorbateModal(), callback=results.append)
            await pilot.pause()
            await pilot.click("#sorb-cancel")
            await pilot.pause()
            assert results == [None], "Cancel must abort (dismiss None)"

    @pytest.mark.asyncio
    async def test_skip_button_returns_empty_string(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            results = []
            app.push_screen(SorbateModal(), callback=results.append)
            await pilot.pause()
            await pilot.click("#sorb-skip")
            await pilot.pause()
            assert results == [""], "Skip must proceed without a sorbate (dismiss '')"

    @pytest.mark.asyncio
    async def test_escape_returns_none(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            results = []
            app.push_screen(SorbateModal(), callback=results.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert results == [None]


class TestForceFieldModalCancel:
    @pytest.mark.asyncio
    async def test_cancel_button_returns_none(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            results = []
            app.push_screen(ForceFieldModal(_mol()), callback=results.append)
            await pilot.pause()
            await pilot.click("#ff-cancel")
            await pilot.pause()
            assert results == [None], "Cancel must abort (dismiss None)"

    @pytest.mark.asyncio
    async def test_ok_with_none_selected_returns_empty_string(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            results = []
            app.push_screen(ForceFieldModal(_mol()), callback=results.append)
            await pilot.pause()
            # Default radio is "None (skip)"; OK should proceed without a FF.
            await pilot.click("#ff-apply")
            await pilot.pause()
            assert results == [""], "OK + None selected must proceed (dismiss '')"


# ---------------------------------------------------------------------------
# Flow level: Cancel aborts the whole MPMC export, Skip continues
# ---------------------------------------------------------------------------

class TestMpmcFlowAbortsOnCancel:
    @pytest.mark.asyncio
    async def test_sorbate_cancel_does_not_proceed_to_filename(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            reached = []
            app._mpmc_step_filename = lambda: reached.append(True)
            app._mpmc_state = {}
            app._mpmc_step_sorbate_result(None)  # Cancel
            assert not reached, "Cancel must not write the MPMC file"

    @pytest.mark.asyncio
    async def test_sorbate_skip_proceeds_to_filename(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            reached = []
            app._mpmc_step_filename = lambda: reached.append(True)
            app._mpmc_state = {}
            app._mpmc_step_sorbate_result("")  # Skip
            assert reached, "Skip must continue to the filename step"
            assert app._mpmc_state["sorbate"] is None

    @pytest.mark.asyncio
    async def test_ff_cancel_does_not_proceed_to_sorbate(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            reached = []
            app._mpmc_step_sorbate = lambda: reached.append(True)
            app._mpmc_state = {}
            app._mpmc_step_ff_apply(None)  # Cancel
            assert not reached, "Cancel must abort before the sorbate step"

    @pytest.mark.asyncio
    async def test_ff_none_proceeds_to_sorbate(self):
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            reached = []
            app._mpmc_step_sorbate = lambda: reached.append(True)
            app._mpmc_state = {}
            app._mpmc_step_ff_apply("")  # skip FF
            assert reached, "Skipping the FF must still continue the flow"
            assert app._mpmc_state["write_ff"] is False
