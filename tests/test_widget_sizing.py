"""Tests for widget sizing, layout constraints, and modal dimensions.

Verifies that all UI components render at their specified CSS sizes and
that modals/panels/controls meet their min/max constraints.
"""

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.panels import GeometryPanel, OperationsPanel, VisualPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    ConfirmModal,
    FileSaveModal,
    InfoModal,
    InputModal,
    MenuBar,
    MoleculeView,
    PdbWizardApp,
    SpinBox,
)


def _make_mol(n=3):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _make_app(**kw):
    return PdbWizardApp(molecule=_make_mol(), filepath="test.pdb", **kw)


# ===========================================================================
# MenuBar
# ===========================================================================

class TestMenuBarSizing:
    @pytest.mark.asyncio
    async def test_height_1(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.size.height == 1

    @pytest.mark.asyncio
    async def test_full_width(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.size.width == 120

    @pytest.mark.asyncio
    async def test_at_top(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.region.y == 0

    @pytest.mark.asyncio
    async def test_labels_have_width(self):
        """Each menu label should have nonzero width."""
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#app-menu", MenuBar)
            for child in bar.children:
                if child.id and child.id.startswith("menu-"):
                    assert child.size.width > 0, f"{child.id} has zero width"
                    assert child.size.height == 1, f"{child.id} height != 1"

    @pytest.mark.asyncio
    async def test_narrow_terminal(self):
        """Menu bar should still be width==terminal in narrow terminals."""
        app = _make_app()
        async with app.run_test(size=(60, 20)):
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.size.width == 60
            assert bar.size.height == 1


# ===========================================================================
# Panels
# ===========================================================================

class TestPanelSizing:
    @pytest.mark.asyncio
    async def test_operations_panel_width(self):
        """Operations panel CSS specifies width: 30 (minus border = 29)."""
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            ops = app.query_one(OperationsPanel)
            assert ops.has_class("visible")
            assert 28 <= ops.size.width <= 30

    @pytest.mark.asyncio
    async def test_operations_panel_docked_left(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            ops = app.query_one(OperationsPanel)
            assert ops.region.x == 0

    @pytest.mark.asyncio
    async def test_geometry_panel_width(self):
        """Geometry panel CSS specifies width: 50 (minus border = 49)."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("g")
            geom = app.query_one(GeometryPanel)
            assert geom.has_class("visible")
            assert 48 <= geom.size.width <= 50

    @pytest.mark.asyncio
    async def test_geometry_panel_docked_right(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("g")
            geom = app.query_one(GeometryPanel)
            assert geom.region.x + geom.region.width == 120

    @pytest.mark.asyncio
    async def test_visual_panel_width(self):
        """Visual panel CSS specifies width: 30 (minus border/padding)."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("V")
            vis = app.query_one(VisualPanel)
            assert vis.has_class("visible")
            assert 25 <= vis.size.width <= 30

    @pytest.mark.asyncio
    async def test_visual_panel_docked_right(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("V")
            vis = app.query_one(VisualPanel)
            assert vis.region.x + vis.region.width == 120

    @pytest.mark.asyncio
    async def test_hidden_panels_zero_display(self):
        """Hidden panels should not take up screen space."""
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            geom = app.query_one(GeometryPanel)
            vis = app.query_one(VisualPanel)
            assert not geom.has_class("visible")
            assert not vis.has_class("visible")


# ===========================================================================
# MoleculeView
# ===========================================================================

class TestMoleculeViewSizing:
    @pytest.mark.asyncio
    async def test_fills_remaining_space(self):
        """MoleculeView should fill space between panels."""
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            view = app.query_one(MoleculeView)
            ops = app.query_one(OperationsPanel)
            # View should start after ops panel
            assert view.region.x >= ops.size.width
            # View + ops should span the width (minus borders)
            total = ops.size.width + view.size.width
            assert total >= 100  # most of 120

    @pytest.mark.asyncio
    async def test_grows_when_panels_close(self):
        """Closing panels should grow MoleculeView."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            view = app.query_one(MoleculeView)
            w_with_ops = view.size.width
            await pilot.press("o")  # close ops panel
            await pilot.pause()
            w_without_ops = view.size.width
            assert w_without_ops > w_with_ops

    @pytest.mark.asyncio
    async def test_height_fills_between_menu_and_footer(self):
        """View height should be terminal_h - menu(1) - footer(1)."""
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            view = app.query_one(MoleculeView)
            # At least 35 rows (40 - menu - footer - status)
            assert view.size.height >= 35


# ===========================================================================
# SpinBox
# ===========================================================================

class TestSliderSizing:
    @pytest.mark.asyncio
    async def test_slider_height_1(self):
        """Custom Slider widget should be height 1."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("V")
            vis = app.query_one(VisualPanel)
            from pdb_wizard.panels import Slider
            sliders = vis.query(Slider)
            assert len(sliders) > 0, "No sliders found in VisualPanel"
            for s in sliders:
                assert s.size.height == 1, f"Slider {s.id} height={s.size.height}, expected 1"

    @pytest.mark.asyncio
    async def test_slider_has_width(self):
        """Sliders should have nonzero width."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("V")
            vis = app.query_one(VisualPanel)
            from pdb_wizard.panels import Slider
            for s in vis.query(Slider):
                assert s.size.width > 0


# ===========================================================================
# Trajectory bar
# ===========================================================================

class TestTrajBarSizing:
    @pytest.mark.asyncio
    async def test_height_3_when_visible(self):
        """Trajectory bar should be height 3."""
        m1, m2 = _make_mol(), _make_mol()
        app = PdbWizardApp(molecule=m1, filepath="t.xyz", frames=[m1, m2])
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#traj-bar")
            assert bar.has_class("visible")
            assert bar.size.height == 3

    @pytest.mark.asyncio
    async def test_docked_bottom(self):
        """Trajectory bar should be docked to bottom."""
        m1, m2 = _make_mol(), _make_mol()
        app = PdbWizardApp(molecule=m1, filepath="t.xyz", frames=[m1, m2])
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#traj-bar")
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            # Traj bar should be just above footer
            assert bar.region.y + bar.region.height <= footer.region.y + 1

    @pytest.mark.asyncio
    async def test_full_width(self):
        m1, m2 = _make_mol(), _make_mol()
        app = PdbWizardApp(molecule=m1, filepath="t.xyz", frames=[m1, m2])
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#traj-bar")
            assert bar.size.width == 120


# ===========================================================================
# Status bar
# ===========================================================================

class TestStatusBarSizing:
    @pytest.mark.asyncio
    async def test_status_bar_is_always_one_row(self):
        """The status bar must occupy exactly 1 row at all times — idle and
        busy. If it toggled between 0 and 1 rows, the MoleculeView would
        reflow every time an analysis ran (visible viewport jolt)."""
        from pdb_wizard.viewer import MoleculeView
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#status-bar")
            assert bar.size.height == 1, (
                f"idle status bar should be 1 row, got {bar.size.height}"
            )
            molview_h_idle = app.query_one(MoleculeView).size.height

            app._show_progress("Computing...", total=100)
            await pilot.pause()
            assert bar.size.height == 1
            molview_h_busy = app.query_one(MoleculeView).size.height
            assert molview_h_busy == molview_h_idle, (
                "MoleculeView height must NOT change when status bar becomes "
                f"busy: idle={molview_h_idle} busy={molview_h_busy}"
            )

            app._hide_progress()
            await pilot.pause()
            assert bar.size.height == 1
            assert app.query_one(MoleculeView).size.height == molview_h_idle

    @pytest.mark.asyncio
    async def test_status_label_clears_on_hide(self):
        """After _hide_progress, the label must be empty — otherwise the
        always-visible idle bar shows a stale 'Computing...' message."""
        from textual.widgets import Label
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            label = app.query_one("#status-label", Label)
            app._show_progress("Computing rdf...", total=100)
            await pilot.pause()
            assert "Computing" in str(label.render())
            app._hide_progress()
            await pilot.pause()
            assert str(label.render()) == "", (
                f"label should be empty after hide; got {str(label.render())!r}"
            )


# ===========================================================================
# Footer
# ===========================================================================

class TestFooterSizing:
    @pytest.mark.asyncio
    async def test_at_bottom(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            assert footer.region.y + footer.region.height == 40

    @pytest.mark.asyncio
    async def test_full_width(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)):
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            assert footer.size.width == 120


# ===========================================================================
# Modals
# ===========================================================================

class TestInputModalSizing:
    @pytest.mark.asyncio
    async def test_centered(self):
        """InputModal should be centered on screen."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(InputModal("Test:", default="val"))
            await pilot.pause()
            screen = app.screen
            # The Vertical container inside should be roughly centered
            verts = screen.query("Vertical")
            if verts:
                v = verts.first()
                # Check it's not at x=0 (it should be centered)
                assert v.region.x > 0, "Modal not horizontally centered"

    @pytest.mark.asyncio
    async def test_has_input_and_buttons(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(InputModal("Test:", default="val"))
            await pilot.pause()
            screen = app.screen
            inputs = screen.query(Input)
            assert len(inputs) >= 1
            buttons = screen.query(Button)
            assert len(buttons) >= 2  # OK + Cancel

    @pytest.mark.asyncio
    async def test_min_width_40(self):
        """InputModal CSS specifies min-width: 40."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(InputModal("Test:", default="val"))
            await pilot.pause()
            verts = app.screen.query("Vertical")
            if verts:
                assert verts.first().size.width >= 40


class TestConfirmModalSizing:
    @pytest.mark.asyncio
    async def test_has_two_buttons(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ConfirmModal("Title", "Are you sure?"))
            await pilot.pause()
            buttons = app.screen.query(Button)
            assert len(buttons) == 2

    @pytest.mark.asyncio
    async def test_yes_no_labels(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ConfirmModal("Title", "Q?", yes_no=True))
            await pilot.pause()
            buttons = app.screen.query(Button)
            labels = {str(b.label).strip() for b in buttons}
            assert "Yes" in labels
            assert "No" in labels

    @pytest.mark.asyncio
    async def test_ok_cancel_labels(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ConfirmModal("Title", "Q?", yes_no=False))
            await pilot.pause()
            buttons = app.screen.query(Button)
            labels = {str(b.label).strip() for b in buttons}
            assert "OK" in labels
            assert "Cancel" in labels

    @pytest.mark.asyncio
    async def test_min_width_36(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ConfirmModal("T", "Q?"))
            await pilot.pause()
            verts = app.screen.query("Vertical")
            if verts:
                assert verts.first().size.width >= 36


class TestInfoModalSizing:
    @pytest.mark.asyncio
    async def test_has_textarea_and_button(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(InfoModal("Title", "Some\ncontent\nhere"))
            await pilot.pause()
            tas = app.screen.query(TextArea)
            assert len(tas) >= 1
            buttons = app.screen.query(Button)
            assert len(buttons) >= 1

    @pytest.mark.asyncio
    async def test_min_width_40(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(InfoModal("T", "C"))
            await pilot.pause()
            verts = app.screen.query("Vertical")
            if verts:
                assert verts.first().size.width >= 40


class TestFileSaveModalSizing:
    @pytest.mark.asyncio
    async def test_has_input_tree_buttons(self):
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(FileSaveModal("Save:", default="out.xyz"))
            await pilot.pause()
            inputs = app.screen.query(Input)
            assert len(inputs) >= 1
            buttons = app.screen.query(Button)
            assert len(buttons) >= 2

    @pytest.mark.asyncio
    async def test_open_button_label(self):
        """button_label='Open' should show Open, not Save."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(FileSaveModal("Open:", default="", button_label="Open"))
            await pilot.pause()
            buttons = app.screen.query(Button)
            labels = {str(b.label).strip() for b in buttons}
            assert "Open" in labels
            assert "Save" not in labels

    @pytest.mark.asyncio
    async def test_min_width_45(self):
        """FileSaveModal CSS specifies min-width: 45."""
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(FileSaveModal("Save:", default="f.xyz"))
            await pilot.pause()
            verts = app.screen.query("Vertical")
            if verts:
                assert verts.first().size.width >= 45


# ===========================================================================
# Resize behavior
# ===========================================================================

class TestResizeBehavior:
    @pytest.mark.asyncio
    async def test_menu_width_follows_resize(self):
        app = _make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.size.width == 80
            await pilot.resize_terminal(160, 48)
            await pilot.pause()
            assert bar.size.width == 160

    @pytest.mark.asyncio
    async def test_view_grows_on_resize(self):
        app = _make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            view = app.query_one(MoleculeView)
            w1 = view.size.width
            h1 = view.size.height
            await pilot.resize_terminal(160, 48)
            await pilot.pause()
            assert view.size.width > w1
            assert view.size.height > h1

    @pytest.mark.asyncio
    async def test_footer_stays_at_bottom(self):
        app = _make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            assert footer.region.y + footer.region.height == 24
            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            assert footer.region.y + footer.region.height == 40


# ===========================================================================
# SpinBox width constraint audit
# ===========================================================================

class TestSpinBoxNotOverwide:
    """Every SpinBox in the app should have a width constraint (not 1fr)."""

    @pytest.mark.asyncio
    async def test_rdf_spinboxes_constrained(self):
        from pdb_wizard.viewer import RdfScreen
        mol = _make_mol()
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(RdfScreen(mol))
            await pilot.pause()
            for sb in app.screen.query(SpinBox):
                assert sb.size.width <= 20, f"SpinBox {sb.id} too wide: {sb.size.width}"

    @pytest.mark.asyncio
    async def test_pxrd_spinboxes_constrained(self):
        from pdb_wizard.viewer import PxrdScreen
        mol = _make_mol()
        app = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(PxrdScreen(mol))
            await pilot.pause()
            for sb in app.screen.query(SpinBox):
                assert sb.size.width <= 20, f"SpinBox {sb.id} too wide: {sb.size.width}"


# Need imports for modal tests
from textual.widgets import Button, Input, TextArea
