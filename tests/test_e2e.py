"""End-to-end tests for the TUI viewer using Textual's run_test() pilot.

Tests cover widget layout/sizing, keyboard bindings, tab lifecycle,
view state isolation, trajectory controls, measurement mode, and
rendering style toggles.
"""


import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.panels import GeometryPanel, OperationsPanel, VisualPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import MenuBar, MoleculeView, PdbWizardApp, TabState

TERM_W, TERM_H = 120, 40


def _make_mol(n_atoms=3, charge=0.0):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n_atoms)]
    for a in atoms:
        a.charge = charge
    pbc = PBC(10, 10, 10, 90, 90, 90)
    mol = Molecule(atoms=atoms, pbc=pbc)
    mol.detect_bonds()
    return mol


def _make_water():
    atoms = [Atom(0, 0, 0, "O"), Atom(0.757, 0.586, 0, "H"), Atom(-0.757, 0.586, 0, "H")]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _make_app(**kwargs):
    mol = _make_mol(3)
    return PdbWizardApp(molecule=mol, filepath="test.pdb", **kwargs)


def _get_tab_labels(app):
    menu_bar = app.query_one("#app-menu", MenuBar)
    return [c for c in menu_bar.children if c.has_class("tab-label")]


def _get_active_tab_labels(app):
    menu_bar = app.query_one("#app-menu", MenuBar)
    return [c for c in menu_bar.children if c.has_class("tab-label") and c.has_class("active")]


def _label_text(label) -> str:
    return label._Static__content


# ===========================================================================
# Layout and sizing
# ===========================================================================

class TestLayout:
    @pytest.mark.asyncio
    async def test_app_fills_terminal(self):
        """App should fill the entire terminal size."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            assert app.screen.size.width == TERM_W
            assert app.screen.size.height == TERM_H

    @pytest.mark.asyncio
    async def test_menu_bar_docked_top(self):
        """Menu bar should be docked to top, full width, height 1."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            bar = app.query_one("#app-menu", MenuBar)
            assert bar.size.height == 1
            assert bar.size.width == TERM_W
            assert bar.region.y == 0  # at very top

    @pytest.mark.asyncio
    async def test_menu_bar_labels_present(self):
        """All 6 menu labels + separator should be present."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            bar = app.query_one("#app-menu", MenuBar)
            ids = {c.id for c in bar.children if c.id}
            for expected in ("menu-file", "menu-edit", "menu-view",
                             "menu-analysis", "menu-undo", "menu-redo"):
                assert expected in ids, f"Missing menu label: {expected}"

    @pytest.mark.asyncio
    async def test_molecule_view_exists_and_visible(self):
        """MoleculeView widget should exist and have nonzero size."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            view = app.query_one(MoleculeView)
            assert view.size.width > 0
            assert view.size.height > 0

    @pytest.mark.asyncio
    async def test_operations_panel_visible_by_default(self):
        """Operations panel should be visible on launch."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            ops = app.query_one(OperationsPanel)
            assert ops.has_class("visible")
            assert ops.size.width > 0

    @pytest.mark.asyncio
    async def test_geometry_panel_hidden_by_default(self):
        """Geometry panel should be hidden on launch."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            geom = app.query_one(GeometryPanel)
            assert not geom.has_class("visible")

    @pytest.mark.asyncio
    async def test_visual_panel_hidden_by_default(self):
        """Visual panel should be hidden on launch."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            vis = app.query_one(VisualPanel)
            assert not vis.has_class("visible")

    @pytest.mark.asyncio
    async def test_status_bar_hidden_by_default(self):
        """Status/progress bar should be hidden on launch."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            bar = app.query_one("#status-bar")
            assert not bar.has_class("visible")

    @pytest.mark.asyncio
    async def test_traj_bar_hidden_by_default(self):
        """Trajectory bar should be hidden for single-frame molecule."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            traj = app.query_one("#traj-bar")
            assert not traj.has_class("visible")

    @pytest.mark.asyncio
    async def test_footer_at_bottom(self):
        """Footer should be at the bottom of the screen."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            assert footer.region.y + footer.region.height == TERM_H


# ===========================================================================
# Tabs
# ===========================================================================

class TestTabs:
    @pytest.mark.asyncio
    async def test_duplicate_names_disambiguated(self):
        """Two tabs with the same basename should show parent directory."""
        mol1 = _make_mol(3)
        app = PdbWizardApp(molecule=mol1, filepath="/sim_A/trajectory.xyz")
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            mol2 = _make_mol(5)
            app._save_current_tab_state()
            app._tabs.append(TabState(mol2, "/sim_B/trajectory.xyz"))
            app._restore_tab_state(1)
            await pilot.pause()
            tabs = _get_tab_labels(app)
            labels = [_label_text(t) for t in tabs]
            assert any("sim_A/trajectory.xyz" in l for l in labels)
            assert any("sim_B/trajectory.xyz" in l for l in labels)

    @pytest.mark.asyncio
    async def test_unique_names_no_disambiguation(self):
        """Tabs with unique names should NOT have parent directory prefix."""
        mol1 = _make_mol(3)
        app = PdbWizardApp(molecule=mol1, filepath="/sim_A/foo.xyz")
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            mol2 = _make_mol(5)
            app._save_current_tab_state()
            app._tabs.append(TabState(mol2, "/sim_B/bar.xyz"))
            app._restore_tab_state(1)
            await pilot.pause()
            tabs = _get_tab_labels(app)
            labels = [_label_text(t) for t in tabs]
            # No path separator should appear in labels (unique names)
            assert not any("/" in l.strip() for l in labels)

    @pytest.mark.asyncio
    async def test_initial_tab(self):
        """One tab on launch, active, showing filename."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            tabs = _get_tab_labels(app)
            assert len(tabs) == 1
            assert tabs[0].has_class("active")
            assert "test.pdb" in _label_text(tabs[0])

    @pytest.mark.asyncio
    async def test_plus_button(self):
        """Plus button present after tab labels."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            bar = app.query_one("#app-menu", MenuBar)
            add_btns = [c for c in bar.children if c.has_class("tab-add")]
            assert len(add_btns) == 1

    @pytest.mark.asyncio
    async def test_add_two_tabs(self):
        """Adding tabs updates count and active state correctly."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            for name in ("a.pdb", "b.pdb"):
                app._save_current_tab_state()
                app._tabs.append(TabState(_make_mol(4), name))
                app._restore_tab_state(len(app._tabs) - 1)
                await pilot.pause()

            tabs = _get_tab_labels(app)
            assert len(tabs) == 3
            active = _get_active_tab_labels(app)
            assert len(active) == 1
            assert "b.pdb" in _label_text(active[0])

    @pytest.mark.asyncio
    async def test_switch_tab_updates_molecule(self):
        """Switching tabs changes app.molecule."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(7), "big.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert len(app.molecule.atoms) == 7
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert len(app.molecule.atoms) == 3

    @pytest.mark.asyncio
    async def test_menu_survives_many_switches(self):
        """Menu labels survive rapid tab switching."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "b.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            for _ in range(5):
                app._save_current_tab_state()
                app._restore_tab_state(0)
                await pilot.pause()
                app._save_current_tab_state()
                app._restore_tab_state(1)
                await pilot.pause()
            bar = app.query_one("#app-menu", MenuBar)
            ids = {c.id for c in bar.children if c.id}
            assert "menu-file" in ids
            assert "menu-edit" in ids
            assert len(_get_tab_labels(app)) == 2

    @pytest.mark.asyncio
    async def test_cant_close_last_tab(self):
        """Closing the only tab should not crash."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            bar = app.query_one("#app-menu", MenuBar)
            bar.post_message(MenuBar.TabClose(0))
            await pilot.pause()
            assert len(app._tabs) == 1

    @pytest.mark.asyncio
    async def test_new_tab_auto_fits_camera(self):
        """New tab should auto-fit camera, not use defaults."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.camera_distance = 1.0
            atoms = [Atom(float(i) * 3.0, 0, 0, "C") for i in range(10)]
            mol2 = Molecule(atoms=atoms, pbc=PBC(50, 50, 50, 90, 90, 90))
            mol2.detect_bonds()
            app._save_current_tab_state()
            app._tabs.append(TabState(mol2, "big.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert view.camera_distance != 1.0
            assert view.camera_distance != 4.0

    @pytest.mark.asyncio
    async def test_switching_back_keeps_camera(self):
        """Saved camera distance survives a tab switch cycle."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.camera_distance = 15.0
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "o.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert abs(view.camera_distance - 15.0) < 0.01


# ===========================================================================
# View state isolation
# ===========================================================================

class TestViewIsolation:
    @pytest.mark.asyncio
    async def test_dark_bg_independent(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.dark_bg = False
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "b.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert view.dark_bg is True  # new tab default
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert view.dark_bg is False  # restored

    @pytest.mark.asyncio
    async def test_show_bonds_independent(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.show_bonds = False
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "b.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert view.show_bonds is True
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert view.show_bonds is False

    @pytest.mark.asyncio
    async def test_color_mode_independent(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.color_mode = "charge"
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "b.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert view.color_mode == "element"
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert view.color_mode == "charge"


# ===========================================================================
# Keyboard bindings
# ===========================================================================

class TestKeybindings:
    @pytest.mark.asyncio
    async def test_zoom_in(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            d0 = view.camera_distance
            await pilot.press("plus")
            assert view.camera_distance < d0

    @pytest.mark.asyncio
    async def test_zoom_out(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            d0 = view.camera_distance
            await pilot.press("minus")
            assert view.camera_distance > d0

    @pytest.mark.asyncio
    async def test_toggle_bonds(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.show_bonds is True
            await pilot.press("b")
            assert view.show_bonds is False
            await pilot.press("b")
            assert view.show_bonds is True

    @pytest.mark.asyncio
    async def test_toggle_hide_water(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.hide_water is False
            await pilot.press("w")
            assert view.hide_water is True
            await pilot.press("w")
            assert view.hide_water is False

    @pytest.mark.asyncio
    async def test_toggle_measure_mode(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.measure_mode is False
            await pilot.press("m")
            assert view.measure_mode is True
            await pilot.press("m")
            assert view.measure_mode is False

    @pytest.mark.asyncio
    async def test_toggle_polyhedra(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.show_polyhedra is False
            await pilot.press("P")
            assert view.show_polyhedra is True

    @pytest.mark.asyncio
    async def test_style_cycle(self):
        """v cycles CPK -> Licorice -> VDW -> Ribbon -> CPK."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            states = []
            for _ in range(5):
                states.append((view.licorice, view.vdw, view.ribbon))
                await pilot.press("v")
            assert states == [
                (False, False, False),  # CPK
                (True, False, False),   # Licorice
                (False, True, False),   # VDW
                (False, False, True),   # Ribbon
                (False, False, False),  # back to CPK
            ]

    @pytest.mark.asyncio
    async def test_color_mode_cycle(self):
        """C cycles through color modes."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            modes = []
            for _ in range(6):
                modes.append(view.color_mode)
                await pilot.press("C")
            assert modes == ["element", "charge", "residue", "chain", "index", "element"]

    @pytest.mark.asyncio
    async def test_toggle_bg(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.dark_bg is True
            await pilot.press("i")
            assert view.dark_bg is False

    @pytest.mark.asyncio
    async def test_pan_mode_toggle(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.pan_mode is False
            await pilot.press("t")
            assert view.pan_mode is True
            await pilot.press("t")
            assert view.pan_mode is False


# ===========================================================================
# Trajectory bar
# ===========================================================================

class TestTrajectoryBar:
    @pytest.mark.asyncio
    async def test_visible_with_frames(self):
        mol1, mol2 = _make_mol(3), _make_mol(3)
        app = PdbWizardApp(molecule=mol1, filepath="t.xyz", frames=[mol1, mol2])
        async with app.run_test(size=(TERM_W, TERM_H)):
            traj = app.query_one("#traj-bar")
            assert traj.has_class("visible")
            assert traj.size.height == 3

    @pytest.mark.asyncio
    async def test_hidden_without_frames(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            assert not app.query_one("#traj-bar").has_class("visible")

    @pytest.mark.asyncio
    async def test_hidden_after_switch_to_static(self):
        mol1, mol2 = _make_mol(3), _make_mol(3)
        app = PdbWizardApp(molecule=mol1, filepath="t.xyz", frames=[mol1, mol2])
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            assert app.query_one("#traj-bar").has_class("visible")
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "s.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert not app.query_one("#traj-bar").has_class("visible")

    @pytest.mark.asyncio
    async def test_reappears_on_switch_back(self):
        mol1, mol2 = _make_mol(3), _make_mol(3)
        app = PdbWizardApp(molecule=mol1, filepath="t.xyz", frames=[mol1, mol2])
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app._save_current_tab_state()
            app._tabs.append(TabState(_make_mol(5), "s.pdb"))
            app._restore_tab_state(1)
            await pilot.pause()
            assert not app.query_one("#traj-bar").has_class("visible")
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert app.query_one("#traj-bar").has_class("visible")


# ===========================================================================
# Atom pick safety
# ===========================================================================

class TestAtomPickSafety:
    @pytest.mark.asyncio
    async def test_empty_molecule(self):
        app = PdbWizardApp(molecule=Molecule(atoms=[]), filepath="e.pdb")
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.post_message(MoleculeView.AtomPicked(atom_index=514, shift=False))
            await pilot.pause()  # no crash

    @pytest.mark.asyncio
    async def test_out_of_range(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            view.post_message(MoleculeView.AtomPicked(atom_index=999, shift=False))
            await pilot.pause()  # no crash

    @pytest.mark.asyncio
    async def test_valid_pick_in_measure_mode(self):
        """Atom pick only works in measure mode."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            # Without measure mode, pick should be ignored by mouse_up
            # But AtomPicked message still works directly (for shift+click legacy)
            await pilot.press("m")  # enable measure mode
            view.post_message(MoleculeView.AtomPicked(atom_index=0, shift=False))
            await pilot.pause()
            # In measure mode, first click starts measurement
            assert len(view._picked_atoms) == 1


# ===========================================================================
# Measurement mode
# ===========================================================================

class TestMeasureMode:
    @pytest.mark.asyncio
    async def test_measure_pins_distance(self):
        """Clicking 2 atoms in measure mode should pin a distance label."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            await pilot.press("m")  # enable measure mode
            assert view.measure_mode is True
            view.post_message(MoleculeView.AtomPicked(atom_index=0, shift=False))
            await pilot.pause()
            view.post_message(MoleculeView.AtomPicked(atom_index=1, shift=False))
            await pilot.pause()
            assert len(view._measurements) == 1
            assert "A" in view._measurements[0][1]  # distance label has "A" (Angstrom)

    @pytest.mark.asyncio
    async def test_measure_resets_after_pin(self):
        """After pinning, picked atoms should reset for next measurement."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            view = app.query_one(MoleculeView)
            await pilot.press("m")
            view.post_message(MoleculeView.AtomPicked(atom_index=0, shift=False))
            await pilot.pause()
            view.post_message(MoleculeView.AtomPicked(atom_index=1, shift=False))
            await pilot.pause()
            assert len(view._picked_atoms) == 0  # reset


# ===========================================================================
# Tab name truncation with ellipsis
# ===========================================================================

class TestTabNameTruncation:
    @pytest.mark.asyncio
    async def test_long_name_truncates_with_ellipsis(self):
        long_path = "/sims/" + ("very_long_filename_" * 3) + ".xyz"
        mol = _make_mol(3)
        app = PdbWizardApp(molecule=mol, filepath=long_path)
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            tabs = _get_tab_labels(app)
            text = _label_text(tabs[0])
            # Ellipsis character should appear, full name should not
            assert "…" in text
            assert "very_long_filename_very_long_filename_very_long_filename_" not in text

    @pytest.mark.asyncio
    async def test_short_name_no_ellipsis(self):
        app = _make_app()  # filepath="test.pdb"
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            tabs = _get_tab_labels(app)
            assert "…" not in _label_text(tabs[0])


# ===========================================================================
# Modal q-key dismiss
# ===========================================================================

class TestModalQDismiss:
    """Modals without focused inputs should accept `q` as a close key."""

    @pytest.mark.asyncio
    async def test_input_modal_escape(self):
        """InputModal has a focused Input — escape only, q types into field."""
        from pdb_wizard.viewer import InputModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(InputModal("Test:", default="x"))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, InputModal)

    @pytest.mark.asyncio
    async def test_file_save_modal_escape(self):
        """FileSaveModal has a focused Input — escape only."""
        from pdb_wizard.viewer import FileSaveModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(FileSaveModal("Save:", default="o.xyz"))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, FileSaveModal)

    @pytest.mark.asyncio
    async def test_force_field_modal_q(self):
        from pdb_wizard.viewer import ForceFieldModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(ForceFieldModal(_make_mol()))
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ForceFieldModal)

    @pytest.mark.asyncio
    async def test_extend_axis_modal_q(self):
        from pdb_wizard.viewer import ExtendAxisModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(ExtendAxisModal(PBC(10, 10, 10, 90, 90, 90)))
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ExtendAxisModal)

    @pytest.mark.asyncio
    async def test_sorbate_modal_q(self):
        from pdb_wizard.viewer import SorbateModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(SorbateModal())
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, SorbateModal)

    @pytest.mark.asyncio
    async def test_charges_file_modal_q(self):
        from pdb_wizard.viewer import ChargesFileModal
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.push_screen(ChargesFileModal())
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ChargesFileModal)


# ===========================================================================
# Panel toggles
# ===========================================================================

class TestPanelToggles:
    @pytest.mark.asyncio
    async def test_geometry_panel_toggle(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            geom = app.query_one(GeometryPanel)
            assert not geom.has_class("visible")
            await pilot.press("g")
            assert geom.has_class("visible")
            assert geom.size.width > 0

    @pytest.mark.asyncio
    async def test_operations_panel_toggle(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            ops = app.query_one(OperationsPanel)
            assert ops.has_class("visible")
            await pilot.press("o")
            assert not ops.has_class("visible")
            await pilot.press("o")
            assert ops.has_class("visible")

    @pytest.mark.asyncio
    async def test_visual_panel_toggle(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            vis = app.query_one(VisualPanel)
            assert not vis.has_class("visible")
            await pilot.press("V")
            assert vis.has_class("visible")
            assert vis.size.width > 0


# ===========================================================================
# MoleculeView state defaults
# ===========================================================================

class TestMoleculeViewDefaults:
    @pytest.mark.asyncio
    async def test_default_state(self):
        """MoleculeView should have sane defaults on launch."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            view = app.query_one(MoleculeView)
            assert view.molecule is not None
            assert len(view.molecule.atoms) == 3
            assert view.dark_bg is True
            assert view.show_bonds is True
            assert view.show_atom_numbers is False
            assert view.color_by_charge is False
            assert view.hide_water is False
            assert view.ribbon is False
            assert view.show_polyhedra is False
            assert view.measure_mode is False
            assert view.color_mode == "element"
            assert view.licorice is False
            assert view.vdw is False
            assert view.pan_mode is False
            assert view.camera_distance > 0
            assert len(view._measurements) == 0
            assert len(view._picked_atoms) == 0
            assert len(view.highlighted_atoms) == 0

    @pytest.mark.asyncio
    async def test_pbc_set(self):
        """PBC should be set from the molecule."""
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)):
            view = app.query_one(MoleculeView)
            assert view.pbc is not None
            assert view.pbc.a == 10.0


# ===========================================================================
# TabState defaults
# ===========================================================================

class TestTabStateDefaults:
    def test_all_defaults(self):
        tab = TabState(_make_mol(), "test.pdb")
        assert tab.dark_bg is True
        assert tab.show_bonds is True
        assert tab.show_atom_numbers is False
        assert tab.color_by_charge is False
        assert tab.hide_water is False
        assert tab.ribbon is False
        assert tab.show_polyhedra is False
        assert tab.color_mode == "element"
        assert tab.licorice is False
        assert tab.vdw is False
        assert tab.camera_initialized is False
        assert tab.frame_idx == 0
        assert len(tab.undo_stack) == 0
        assert len(tab.redo_stack) == 0

    def test_name_from_path(self):
        tab = TabState(_make_mol(), "/some/long/path/structure.pdb")
        assert tab.name == "structure.pdb"

    def test_name_untitled(self):
        tab = TabState(_make_mol(), "")
        assert tab.name == "untitled"

    def test_frames_none_by_default(self):
        tab = TabState(_make_mol(), "f.pdb")
        assert tab.frames is None

    def test_separate_undo_stacks(self):
        t1 = TabState(_make_mol(), "a.pdb")
        t2 = TabState(_make_mol(), "b.pdb")
        t1.undo_stack.append("x")
        assert len(t1.undo_stack) == 1
        assert len(t2.undo_stack) == 0

    def test_separate_molecules(self):
        m1 = _make_mol(3, charge=1.0)
        m2 = _make_mol(5, charge=2.0)
        t1 = TabState(m1, "a.pdb")
        t2 = TabState(m2, "b.pdb")
        assert len(t1.molecule.atoms) == 3
        assert len(t2.molecule.atoms) == 5
        t1.molecule.atoms[0].x = np.array([99, 99, 99])
        assert t2.molecule.atoms[0].x[0] == 0.0


# ===========================================================================
# Resize
# ===========================================================================

class TestResize:
    @pytest.mark.asyncio
    async def test_resize_updates_view(self):
        """Resizing the terminal should update widget sizes."""
        app = _make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            view = app.query_one(MoleculeView)
            old_w = view.size.width
            await pilot.resize_terminal(160, 48)
            await pilot.pause()
            # View should have grown
            assert view.size.width > old_w


# ===========================================================================
# Overlapping-atom removal reporting
# ===========================================================================

class TestOverlapNotification:
    """The TUI has no stdout, so deleted overlapping atoms must surface as a toast."""

    @pytest.mark.asyncio
    async def test_startup_load_messages_become_warning_toast(self):
        app = _make_app(load_messages=["Deleting overlapping atoms\n  C 1 --- C 2"])
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            notes = list(app._notifications)
            texts = [str(n.message) for n in notes]
            assert any("Deleted 1 overlapping atom " in t for t in texts), texts
            assert any(n.severity == "warning" for n in notes)
            # Consumed once; not re-shown later.
            assert app._startup_load_messages == []

    @pytest.mark.asyncio
    async def test_no_toast_without_messages(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            await pilot.pause()
            texts = [str(n.message) for n in app._notifications]
            assert not any("overlapping" in t for t in texts)

    @pytest.mark.asyncio
    async def test_notify_overlaps_removed_pluralises(self):
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app.notify_overlaps_removed(["a", "b", "c"])
            await pilot.pause()
            texts = [str(n.message) for n in app._notifications]
            assert any("Deleted 3 overlapping atoms" in t for t in texts), texts

    @pytest.mark.asyncio
    async def test_open_file_with_overlaps_toasts(self, tmp_path):
        content = """3
10.0 10.0 10.0 90.0 90.0 90.0
C  0.000  0.000  0.000
C  0.010  0.000  0.000
O  5.000  5.000  5.000
"""
        path = tmp_path / "dup.xyz"
        path.write_text(content)
        app = _make_app()
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app._open_new_tab(str(path))
            await pilot.pause()
            assert len(app.molecule.atoms) == 2
            texts = [str(n.message) for n in app._notifications]
            assert any("Deleted 1 overlapping atom" in t for t in texts), texts

    @pytest.mark.asyncio
    async def test_open_into_empty_buffer_toasts(self, tmp_path):
        content = """3
10.0 10.0 10.0 90.0 90.0 90.0
C  0.000  0.000  0.000
C  0.010  0.000  0.000
O  5.000  5.000  5.000
"""
        path = tmp_path / "dup.xyz"
        path.write_text(content)
        app = PdbWizardApp(molecule=Molecule(atoms=[]), filepath="")
        async with app.run_test(size=(TERM_W, TERM_H)) as pilot:
            app._do_open_file(str(path))
            await pilot.pause()
            assert len(app.molecule.atoms) == 2
            texts = [str(n.message) for n in app._notifications]
            assert any("Deleted 1 overlapping atom" in t for t in texts), texts
