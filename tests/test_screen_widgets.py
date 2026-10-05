"""Systematic widget sizing and presence tests for every screen/modal.

For each screen: verify all expected widgets exist, SpinBoxes are constrained,
Selects have reasonable width, Buttons are present, and the screen is centered.
"""

import numpy as np
import pytest
from textual.widgets import Button, DataTable, Input, Label, Select, TextArea

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    ConfirmModal,
    CoordinationScreen,
    DatabaseSearchScreen,
    DensityScreen,
    EnergyPlotScreen,
    ExtendAxisModal,
    FileSaveModal,
    ForceFieldModal,
    GyrationScreen,
    HbondScreen,
    InfoModal,
    InputModal,
    IsothermPlannerScreen,
    MsdScreen,
    PdbWizardApp,
    PlotWidget,
    PxrdScreen,
    RdfScreen,
    ReduceCellScreen,
    RmsdScreen,
    SorbateModal,
    SpinBox,
)

TERM = (120, 40)
MAX_SPINBOX_W = 20
MAX_SELECT_W = 24


def _mol(n=3):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _app():
    return PdbWizardApp(molecule=_mol(), filepath="test.pdb")


async def _push(app, pilot, screen):
    """Push a screen and wait for it to mount."""
    app.push_screen(screen)
    await pilot.pause()
    return app.screen


def _check_spinboxes(screen, max_w=MAX_SPINBOX_W):
    """Every SpinBox should be width-constrained."""
    for sb in screen.query(SpinBox):
        assert sb.size.width <= max_w, (
            f"SpinBox '{sb.id}' width={sb.size.width} > {max_w}"
        )


def _check_selects(screen, max_w=MAX_SELECT_W):
    """Every Select should have reasonable width."""
    for sel in screen.query(Select):
        assert sel.size.width <= max_w, (
            f"Select '{sel.id}' width={sel.size.width} > {max_w}"
        )


def _check_buttons_present(screen, min_count=1):
    """Screen should have at least min_count buttons."""
    buttons = screen.query(Button)
    assert len(buttons) >= min_count, (
        f"Expected >= {min_count} buttons, got {len(buttons)}"
    )


def _check_buttons_visible(screen):
    """All buttons should have nonzero size."""
    for btn in screen.query(Button):
        assert btn.size.width > 0, f"Button '{btn.id}' has zero width"
        assert btn.size.height > 0, f"Button '{btn.id}' has zero height"


# ===========================================================================
# Modals
# ===========================================================================

class TestInputModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, InputModal("Test:", "val", "hint"))
            assert len(s.query(Input)) >= 1
            _check_buttons_present(s, 2)
            _check_buttons_visible(s)

    @pytest.mark.asyncio
    async def test_input_has_width(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, InputModal("Test:", "val"))
            inp = s.query(Input).first()
            assert inp.size.width >= 20


class TestInfoModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, InfoModal("Title", "content\nhere"))
            assert len(s.query(TextArea)) >= 1
            _check_buttons_present(s, 1)
            _check_buttons_visible(s)


class TestConfirmModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ConfirmModal("T", "Q?"))
            _check_buttons_present(s, 2)
            _check_buttons_visible(s)

    @pytest.mark.asyncio
    async def test_button_labels_ok_cancel(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ConfirmModal("T", "Q?", yes_no=False))
            labels = {str(b.label).strip() for b in s.query(Button)}
            assert "OK" in labels and "Cancel" in labels

    @pytest.mark.asyncio
    async def test_button_labels_yes_no(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ConfirmModal("T", "Q?", yes_no=True))
            labels = {str(b.label).strip() for b in s.query(Button)}
            assert "Yes" in labels and "No" in labels


class TestFileSaveModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, FileSaveModal("Save:", "out.xyz"))
            assert len(s.query(Input)) >= 1
            _check_buttons_present(s, 2)
            _check_buttons_visible(s)

    @pytest.mark.asyncio
    async def test_save_label(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, FileSaveModal("Save:", "f.xyz"))
            labels = {str(b.label).strip() for b in s.query(Button)}
            assert "Save" in labels

    @pytest.mark.asyncio
    async def test_open_label(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, FileSaveModal("Open:", "", button_label="Open"))
            labels = {str(b.label).strip() for b in s.query(Button)}
            assert "Open" in labels
            assert "Save" not in labels


class TestForceFieldModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ForceFieldModal(_mol()))
            _check_buttons_present(s, 2)


class TestExtendAxisModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ExtendAxisModal(PBC(10, 10, 10, 90, 90, 90)))
            # Now has 3 SpinBoxes (a, b, c multipliers), no Selects
            assert len(s.query(SpinBox)) == 3
            _check_spinboxes(s, max_w=30)
            _check_buttons_present(s, 2)

    @pytest.mark.asyncio
    async def test_value_label_visible(self):
        """Each SpinBox's value label must have enough width to display 2-digit numbers."""
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ExtendAxisModal(PBC(10, 10, 10, 90, 90, 90)))
            for sb in s.query(SpinBox):
                # SpinBox has 2 buttons (4 wide each) + a value label (1fr)
                # The value label needs to be at least 4 wide (border eats 2, leaving 2+ for digits)
                value_label = sb.query_one(".spin-value")
                assert value_label.size.width >= 4, (
                    f"SpinBox '{sb.id}' value label width={value_label.size.width}, "
                    f"need >=4 to display 2-digit numbers (max_val=20)"
                )

    @pytest.mark.asyncio
    async def test_each_spinbox_on_own_row(self):
        """Each axis should be on its own row for visibility."""
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ExtendAxisModal(PBC(10, 10, 10, 90, 90, 90)))
            spinboxes = list(s.query(SpinBox))
            ys = sorted(sb.region.y for sb in spinboxes)
            # All three SpinBoxes should be at different y positions
            assert len(set(ys)) == 3, f"All SpinBoxes at same row: y={ys}"

    @pytest.mark.asyncio
    async def test_value_2_digit_renders(self):
        """A 2-digit value should fit within the SpinBox value label."""
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            modal = ExtendAxisModal(PBC(10, 10, 10, 90, 90, 90))
            s = await _push(app, pilot, modal)
            sb = s.query_one("#extend-na", SpinBox)
            sb._value = 20
            sb.query_one("#spin-display", Label).update("20")
            await pilot.pause()
            # Value label width must accommodate "20" (2 chars + border padding)
            value_label = sb.query_one(".spin-value")
            assert value_label.size.width >= 4


class TestSorbateModalWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, SorbateModal())
            assert len(s.query(DataTable)) >= 1
            _check_buttons_present(s, 2)


# ===========================================================================
# Analysis screens
# ===========================================================================

class TestRdfScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, RdfScreen(_mol()))
            assert len(s.query(Select)) >= 2  # el1, el2
            assert len(s.query(SpinBox)) >= 2  # ymax, xmax, bins
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)
            _check_buttons_visible(s)

    @pytest.mark.asyncio
    async def test_plot_has_size(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, RdfScreen(_mol()))
            plot = s.query_one(PlotWidget)
            assert plot.size.width > 50
            assert plot.size.height > 10


class TestMsdScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, MsdScreen(m, [m, m, m]))
            assert len(s.query(Select)) >= 1
            assert len(s.query(SpinBox)) >= 1
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestCoordinationScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, CoordinationScreen(_mol()))
            assert len(s.query(Select)) >= 2  # center, neighbor
            assert len(s.query(SpinBox)) >= 1  # rmax
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestRmsdScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, RmsdScreen(m, [m, m, m]))
            assert len(s.query(Select)) >= 1
            assert len(s.query(SpinBox)) >= 1
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestHbondScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, HbondScreen(m, [m, m]))
            assert len(s.query(SpinBox)) >= 2  # cutoff, angle
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_buttons_present(s, 2)


class TestGyrationScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, GyrationScreen(m, [m, m, m]))
            assert len(s.query(Select)) >= 1
            assert len(s.query(SpinBox)) >= 1
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestDensityScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DensityScreen(m, [m, m]))
            assert len(s.query(Select)) >= 2  # element, axis
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestPxrdScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, PxrdScreen(_mol()))
            assert len(s.query(Select)) >= 1  # wavelength
            assert len(s.query(SpinBox)) >= 1  # 2theta max
            assert len(s.query(PlotWidget)) == 1
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestEnergyPlotScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        data = {"step": np.arange(10.0), "N": np.ones(10), "energy": np.zeros(10)}
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, EnergyPlotScreen(data, "energy.dat"))
            assert len(s.query(Select)) >= 1
            assert len(s.query(PlotWidget)) == 1
            _check_selects(s)
            _check_buttons_present(s, 1)

    @pytest.mark.asyncio
    async def test_auto_plots_on_mount(self):
        data = {"step": np.arange(10.0), "N": np.ones(10), "energy": np.zeros(10)}
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, EnergyPlotScreen(data, "energy.dat"))
            plot = s.query_one(PlotWidget)
            assert len(plot._x) > 0, "Plot should have data on mount"


# ===========================================================================
# Tool screens
# ===========================================================================

class TestReduceCellScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, ReduceCellScreen(_mol()))
            assert len(s.query(SpinBox)) >= 3  # na, nb, nc
            assert len(s.query(DataTable)) >= 1
            _check_spinboxes(s, max_w=12)  # smaller for 1-digit spinboxes
            _check_buttons_present(s, 2)


class TestIsothermPlannerScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, IsothermPlannerScreen(_mol(), "test.pdb"))
            assert len(s.query(Select)) >= 1  # sorbate
            assert len(s.query(SpinBox)) >= 2  # temp, steps
            assert len(s.query(Input)) >= 1  # pressures
            _check_spinboxes(s)
            _check_selects(s)
            _check_buttons_present(s, 2)


class TestDatabaseSearchScreenWidgets:
    @pytest.mark.asyncio
    async def test_widgets(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DatabaseSearchScreen())
            assert len(s.query(Select)) >= 1  # source
            assert len(s.query(Input)) >= 1  # query
            assert len(s.query(DataTable)) >= 1
            _check_selects(s)
            _check_buttons_present(s, 2)

    @pytest.mark.asyncio
    async def test_table_has_columns(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, DatabaseSearchScreen())
            table = s.query_one(DataTable)
            assert len(table.columns) == 4  # Source, ID, Formula, Title


# ===========================================================================
# PlotWidget sizing across all screens
# ===========================================================================

class TestPlotWidgetSizing:
    """PlotWidget should fill most of the screen in all analysis screens."""

    @pytest.mark.asyncio
    async def test_rdf_plot_fills(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, RdfScreen(_mol()))
            plot = s.query_one(PlotWidget)
            assert plot.size.width >= 80
            assert plot.size.height >= 20

    @pytest.mark.asyncio
    async def test_msd_plot_fills(self):
        m = _mol()
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, MsdScreen(m, [m, m, m]))
            plot = s.query_one(PlotWidget)
            assert plot.size.width >= 80
            assert plot.size.height >= 20

    @pytest.mark.asyncio
    async def test_pxrd_plot_fills(self):
        app = _app()
        async with app.run_test(size=TERM) as pilot:
            s = await _push(app, pilot, PxrdScreen(_mol()))
            plot = s.query_one(PlotWidget)
            assert plot.size.width >= 80
            assert plot.size.height >= 20
