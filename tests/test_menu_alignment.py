"""The left-hand Operations panel and the top-bar menus must stay aligned.

If a feature is added to one place but not the other, users searching for it
in the wrong location won't find it. This file pins:

  - Every action in MENU_FILE/EDIT/ANALYSIS (top bar) is exposed in the
    Operations panel — except the items intentionally excluded (Quit,
    panel-toggle entries, single-key view toggles).
  - Every command in OPERATIONS (left panel) is dispatchable.
  - The Isotherm Planner is reachable from BOTH places.
  - The 'gen_mpmc_inp' garbage is removed from BOTH places.
"""

from __future__ import annotations

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.panels import OPERATIONS, OperationsPanel
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    MENU_ANALYSIS,
    MENU_EDIT,
    MENU_FILE,
    MENU_VIEW,
    PdbWizardApp,
)


def _mol():
    atoms = [Atom(float(i), 0, 0, "C") for i in range(4)]
    return Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))


# Items in the top bar that intentionally do NOT belong in the Operations panel.
TOP_BAR_PANEL_EXEMPT = {
    "quit",                    # File: app exit
    "toggle_operations",       # View: toggles this very panel
    "toggle_geometry",         # View: panel toggle
    "toggle_visual",           # View: panel toggle
    "toggle_bonds",            # View: display toggle (key 'b')
    "toggle_style",            # View: display toggle (key 'v')
    "toggle_bg",               # View: display toggle (key 'i')
    "toggle_atom_numbers",     # View: display toggle (key '#')
    "cycle_color_mode",        # View: display toggle (key 'C')
    "toggle_hide_water",       # View: display toggle (key 'w')
    "toggle_measure",          # View: display toggle (key 'm')
    "clear_measurements",      # View: display action
    "toggle_polyhedra",        # View: display toggle (key 'P')
    "reset_view",              # View: viewport reset (key 'r')
}


def _menu_action_ids(menu) -> set[str]:
    """Extract the action IDs from a top-bar menu spec, dropping separators."""
    return {cmd for cmd, _ in menu if cmd != "---"}


# ---------------------------------------------------------------------------
# Top bar ↔ left panel
# ---------------------------------------------------------------------------

class TestTopBarAndPanelAlignment:
    def test_every_top_bar_action_is_in_panel_or_exempt(self):
        """Every item in MENU_FILE / EDIT / ANALYSIS must be reachable from
        the Operations panel. Items in TOP_BAR_PANEL_EXEMPT are intentional
        exceptions (panel toggles, single-key view toggles, app-exit)."""
        panel_cmds = {cmd for cmd, _label, _group in OPERATIONS}
        top_bar_cmds = (
            _menu_action_ids(MENU_FILE)
            | _menu_action_ids(MENU_EDIT)
            | _menu_action_ids(MENU_ANALYSIS)
        )
        missing_from_panel = top_bar_cmds - panel_cmds - TOP_BAR_PANEL_EXEMPT
        assert not missing_from_panel, (
            f"Top-bar menu commands missing from Operations panel: "
            f"{sorted(missing_from_panel)}"
        )

    def test_every_panel_command_appears_in_top_bar(self):
        """Symmetry check: panel commands must show up somewhere in the top
        bar too. No tool should be panel-only — it strands keyboard users
        and people exploring via the menu."""
        panel_cmds = {cmd for cmd, _label, _group in OPERATIONS}
        all_top_bar = (
            _menu_action_ids(MENU_FILE)
            | _menu_action_ids(MENU_EDIT)
            | _menu_action_ids(MENU_ANALYSIS)
            | _menu_action_ids(MENU_VIEW)
        )
        missing_from_top_bar = panel_cmds - all_top_bar
        assert not missing_from_top_bar, (
            f"Panel commands missing from any top-bar menu: "
            f"{sorted(missing_from_top_bar)}"
        )


# ---------------------------------------------------------------------------
# Specific user-visible asks
# ---------------------------------------------------------------------------

class TestSpecificEntries:
    def test_input_generator_in_panel(self):
        cmds = {cmd for cmd, _l, _g in OPERATIONS}
        assert "input_generator" in cmds, (
            "Input Generator missing from Operations panel"
        )

    def test_input_generator_in_top_bar(self):
        assert "input_generator" in _menu_action_ids(MENU_ANALYSIS)

    def test_gen_mpmc_inp_removed_from_panel(self):
        cmds = {cmd for cmd, _l, _g in OPERATIONS}
        assert "gen_mpmc_inp" not in cmds, (
            "gen_mpmc_inp ('garbage') should be gone from Operations panel"
        )

    def test_gen_mpmc_inp_removed_from_top_bar(self):
        all_top = (
            _menu_action_ids(MENU_FILE)
            | _menu_action_ids(MENU_EDIT)
            | _menu_action_ids(MENU_ANALYSIS)
        )
        assert "gen_mpmc_inp" not in all_top

    def test_panel_groups_cover_all_three_categories(self):
        """The OPERATIONS list should still be partitioned into the three
        canonical groups so the panel renders meaningful section headers."""
        groups = {g for _c, _l, g in OPERATIONS}
        assert groups == {"analysis", "modify", "export"}, (
            f"Unexpected group labels in OPERATIONS: {groups}"
        )


# ---------------------------------------------------------------------------
# Live dispatch: every panel command is wired (no silent no-ops)
# ---------------------------------------------------------------------------

# These commands open modals or background tasks; just confirm dispatch is
# wired (no AttributeError or unknown-command fall-through) without driving
# the resulting UI to completion.
SAFE_PANEL_DISPATCH_CHECKS = [
    cmd for cmd, _l, _g in OPERATIONS
    # Skip commands that mutate the molecule destructively or write to disk
    # without a modal.
    if cmd not in {
        "wrap_center", "wrap_forward", "sort", "del_lone",
        "qeq_charges", "write_xyz", "write_pdb", "write_cif",
        "write_poscar", "write_lammps", "write_mpmc",
        "frame_to_tab", "export_png", "export_gif",
        "quit",  # exits the app — tested explicitly below
    }
]


class TestPanelCommandsAreWired:
    @pytest.mark.parametrize("cmd", SAFE_PANEL_DISPATCH_CHECKS)
    @pytest.mark.asyncio
    async def test_dispatch_does_not_crash(self, cmd):
        """Posting a panel command must not raise from the dispatcher.
        The cmd may pop a modal or do nothing visible, but never raise."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            try:
                app.post_message(OperationsPanel.RunCommand(cmd))
                await pilot.pause()
                await pilot.pause()
            except Exception as e:
                pytest.fail(
                    f"Panel cmd {cmd!r} dispatcher crashed: "
                    f"{type(e).__name__}: {e}"
                )


# ---------------------------------------------------------------------------
# Quit from the side panel
# ---------------------------------------------------------------------------

class TestQuitFromPanel:
    def test_quit_row_in_panel_file_group(self):
        """The Operations (side) panel must offer a Quit entry, grouped under
        File (export)."""
        assert any(
            cmd == "quit" and label == "Quit" and group == "export"
            for cmd, label, group in OPERATIONS
        ), "Quit entry missing from the Operations panel File group"

    @pytest.mark.asyncio
    async def test_quit_command_exits_app(self):
        """Selecting Quit in the side panel must exit the app (directly, no
        'press q again' confirmation — that's only for the bare 'q' key)."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            calls = []
            app.exit = lambda *a, **k: calls.append((a, k))  # don't really exit
            app.post_message(OperationsPanel.RunCommand("quit"))
            await pilot.pause()
            assert calls, "quit command must call app.exit()"
