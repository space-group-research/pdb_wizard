"""Tests for EnergyPlotScreen — including the Start/End step range selector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Label, Select

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import EnergyPlotScreen, PdbWizardApp, SpinBox


def _mol():
    atoms = [Atom(0.0, 0.0, 0.0, "C")]
    return Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))


def _energy_data(n: int = 50) -> dict:
    """Synthetic MPMC-like energy.dat content with an upward trend."""
    return {
        "step":   np.arange(n, dtype=float),
        "energy": np.linspace(-100.0, -50.0, n),
        "N":      np.linspace(0, 5.0, n),
    }


# ---------------------------------------------------------------------------
# Step range UI
# ---------------------------------------------------------------------------

class TestStepRangeWidgets:
    @pytest.mark.asyncio
    async def test_start_end_spinboxes_appear_when_data_has_multiple_steps(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(_energy_data(50), "energy.dat"))
            await pilot.pause()
            screen = app.screen
            # Both SpinBoxes must exist with default values 1 and 50
            start = screen.query_one("#energy-start", SpinBox)
            end = screen.query_one("#energy-end", SpinBox)
            assert start.value == 1
            assert end.value == 50

    @pytest.mark.asyncio
    async def test_step_spinboxes_omitted_for_single_step_data(self):
        """Don't clutter the bar with Start/End SpinBoxes when there's
        nothing to slice."""
        from textual.css.query import NoMatches
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            single = {"step": np.array([0.0]), "energy": np.array([-1.0])}
            app.push_screen(EnergyPlotScreen(single, "energy.dat"))
            await pilot.pause()
            screen = app.screen
            with pytest.raises(NoMatches):
                screen.query_one("#energy-start", SpinBox)


# ---------------------------------------------------------------------------
# Step range affects the rendered plot + statistics
# ---------------------------------------------------------------------------

class TestStepRangeBehavior:
    @pytest.mark.asyncio
    async def test_changing_end_reduces_plot_data(self):
        """Setting End from 50 → 25 must re-plot using only the first 25 rows.
        We verify by inspecting the status string's reported range."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(_energy_data(50), "energy.dat"))
            await pilot.pause()
            screen = app.screen

            initial = str(screen.query_one("#energy-status", Label).render())
            assert "[1-50]" in initial, f"initial range not 1-50; got {initial!r}"

            # Halve the End spinbox
            screen.query_one("#energy-end", SpinBox)._update(25)
            await pilot.pause()
            updated = str(screen.query_one("#energy-status", Label).render())
            assert "[1-25]" in updated, (
                f"after End=25 the status should report [1-25]; got {updated!r}"
            )

    @pytest.mark.asyncio
    async def test_changing_start_skips_initial_steps(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(_energy_data(50), "energy.dat"))
            await pilot.pause()
            screen = app.screen
            screen.query_one("#energy-start", SpinBox)._update(20)
            await pilot.pause()
            status = str(screen.query_one("#energy-status", Label).render())
            assert "[20-50]" in status, (
                f"after Start=20 the status should report [20-50]; got {status!r}"
            )

    @pytest.mark.asyncio
    async def test_last_quarter_avg_uses_only_selected_range(self):
        """The 'last-quarter avg' must be computed from the SELECTED slice,
        not the entire dataset. Otherwise the user can't isolate the
        equilibrated portion of an MD run."""
        # Build data where steps 0-49 hover around -100 and steps 50-99 around 0.
        n = 100
        data = {
            "step": np.arange(n, dtype=float),
            "energy": np.concatenate([
                np.full(50, -100.0),
                np.full(50, 0.0),
            ]),
        }
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(data, "energy.dat"))
            await pilot.pause()
            screen = app.screen
            # Restrict to steps 51..100 — last-quarter avg should be 0.0
            screen.query_one("#energy-start", SpinBox)._update(51)
            await pilot.pause()
            status = str(screen.query_one("#energy-status", Label).render())
            # Must mention "0.0" for last-quarter avg of the slice
            assert "= 0.0000" in status, (
                f"last-quarter avg should be 0.0 for steps 51-100; got {status!r}"
            )

    @pytest.mark.asyncio
    async def test_step_slice_helper_clamps_invalid_input(self):
        """If End < Start, the helper must produce a non-empty slice (at
        least one step) instead of an empty plot."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(_energy_data(50), "energy.dat"))
            await pilot.pause()
            screen = app.screen
            # Force an inverted range
            screen.query_one("#energy-start", SpinBox)._update(40)
            screen.query_one("#energy-end", SpinBox)._update(10)
            await pilot.pause()
            sl = screen._step_slice()
            assert sl.start < sl.stop, (
                f"slice must be non-empty even with inverted bounds; got {sl}"
            )


# ---------------------------------------------------------------------------
# CSV export honors the step range
# ---------------------------------------------------------------------------

class TestCsvExportRespectsRange:
    @pytest.mark.asyncio
    async def test_export_writes_only_selected_rows(self):
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            app.push_screen(EnergyPlotScreen(_energy_data(50), "energy.dat"))
            await pilot.pause()
            screen = app.screen
            # Restrict to steps 11-30 — that's 20 rows of CSV
            screen.query_one("#energy-start", SpinBox)._update(11)
            screen.query_one("#energy-end", SpinBox)._update(30)
            await pilot.pause()

            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
                csv_path = f.name
            screen._export_csv(csv_path)
            await pilot.pause()

            lines = Path(csv_path).read_text().splitlines()
            # First line is the header; remaining lines are data rows
            assert len(lines) - 1 == 20, (
                f"expected 20 data rows for steps 11-30, got {len(lines) - 1}"
            )
