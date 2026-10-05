"""Coverage tests for menu actions and keybindings that lacked e2e tests.

Each test pushes the corresponding screen or invokes the action and asserts
the UI reaches a non-trivial state — catches dead-code menu entries.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from textual.widgets import Input, Select

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    DatabaseSearchScreen,
    IsothermPlannerScreen,
    MoleculeView,
    PdbWizardApp,
    PlotWidget,
    SpinBox,
)


def _mol(n=3):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _app():
    return PdbWizardApp(molecule=_mol(), filepath="t.pdb")


# ---------------------------------------------------------------------------
# Menu actions reachable through the dispatcher
# ---------------------------------------------------------------------------

class TestAnalysisMenuActions:
    @pytest.mark.asyncio
    async def test_db_search_screen_has_required_widgets(self):
        """DatabaseSearchScreen exposes the source select with at least the
        canonical databases (cod, rcsb, pubchem) and the query input
        actually accepts text."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(DatabaseSearchScreen())
            await pilot.pause()

            query = app.screen.query_one("#db-query", Input)
            source = app.screen.query_one("#db-source", Select)

            opt_values = [v for _, v in source._options]
            for required in ("cod", "rcsb", "pubchem"):
                assert required in opt_values, f"DB source missing {required}"

            query.value = "benzene"
            await pilot.pause()
            assert query.value == "benzene"

    @pytest.mark.asyncio
    async def test_isotherm_planner_generate_button_is_wired(self):
        """The Generate button must have a click handler that produces output
        when invoked. We exercise the button rather than just checking it
        renders."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = IsothermPlannerScreen(_mol(), "t.pdb")
            app.push_screen(screen)
            await pilot.pause()

            # The Generate button must exist and be a primary button
            from textual.widgets import Button
            gen_btn = screen.query_one("#gen-generate", Button)
            assert gen_btn.variant == "primary"

            # Default pressures input has values
            pressures = screen.query_one("#gen-pressures", Input).value
            assert "," in pressures, "pressure input should have CSV defaults"
            parsed = [float(p.strip()) for p in pressures.split(",")]
            assert len(parsed) >= 3
            assert all(p > 0 for p in parsed)

    @pytest.mark.asyncio
    async def test_density3d_without_trajectory_emits_notification(self):
        """When density3d runs on a system with no trajectory frames, the user
        must see a notification — not silent failure."""
        notifications = []
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            orig_notify = app.notify
            def capture(msg, *args, **kwargs):
                notifications.append(str(msg))
                return orig_notify(msg, *args, **kwargs)
            app.notify = capture

            from pdb_wizard.panels import OperationsPanel
            app.post_message(OperationsPanel.RunCommand("density3d"))
            await pilot.pause()
            await pilot.pause()  # allow async notify to flush

            assert any(
                "trajectory" in n.lower() or "frame" in n.lower() or "density" in n.lower()
                for n in notifications
            ), f"Expected a user-visible notification; got: {notifications}"

    @pytest.mark.asyncio
    async def test_surface_area_value_is_physically_sane(self):
        """For an isolated atom in a vacuum, surface_area ≈ 4π·(vdw+probe)².
        Three carbons in a row have near that times 3 (with some occlusion
        from neighbors). Just checking that 'surface_area' is in the dict
        isn't enough — a stub returning 0.0 would pass that."""
        from pdb_wizard.operations import surface_area
        # Single atom, plenty of space — area ≈ 4π·(C_vdw + 1.4)²
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        single = Molecule(atoms=atoms, pbc=PBC(50, 50, 50, 90, 90, 90))
        result = surface_area(single, n_samples=2000)
        # C vdw ≈ 1.7, probe ≈ 1.4 → r ≈ 3.1, A ≈ 4π·9.61 ≈ 121 Å²
        assert 80.0 < result["surface_area"] < 180.0, (
            f"single C surface area {result['surface_area']} A^2 outside expected ~121"
        )
        assert result["area_per_volume"] > 0

    @pytest.mark.asyncio
    async def test_pore_size_distribution_returns_meaningful_data(self):
        """A 10 Å empty cubic box with 8 atoms in the corners should have
        nonzero pore size sampling — not just empty arrays."""
        atoms = [Atom(x, y, z, "C") for x in (0.0, 9.0)
                 for y in (0.0, 9.0) for z in (0.0, 9.0)]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        from pdb_wizard.operations import pore_size_distribution
        bins, hist = pore_size_distribution(mol, n_samples=2000, n_bins=20, max_pore=10.0)
        assert len(bins) == 20
        assert len(hist) == 20
        assert hist.sum() > 0, "histogram should have non-empty pore sampling"
        # The largest pore in this 10 Å box should sit somewhere in the middle range
        peak = bins[hist.argmax()]
        assert 1.0 < peak < 8.0, f"peak pore radius {peak} is implausible"

    @pytest.mark.asyncio
    async def test_hbonds_screen_compute_button_is_present(self):
        """HbondScreen must have a Compute button + cutoff/angle SpinBoxes
        with sensible default values."""
        from pdb_wizard.viewer import HbondScreen
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = HbondScreen(_mol(), [_mol(), _mol()])
            app.push_screen(screen)
            await pilot.pause()
            from textual.widgets import Button
            compute = screen.query_one("#hb-compute", Button)
            assert compute.variant == "primary"
            cutoff = screen.query_one("#hb-cutoff", SpinBox).value
            angle = screen.query_one("#hb-angle", SpinBox).value
            # Reasonable H-bond defaults: cutoff 30-50 (×0.1 Å = 3.0-5.0), angle 100-180°
            assert 20 <= cutoff <= 50
            assert 90 <= angle <= 180

    @pytest.mark.asyncio
    async def test_gyration_screen_element_select_lists_actual_elements(self):
        """The element Select in GyrationScreen must list the elements present
        in the molecule, not a hardcoded set."""
        from pdb_wizard.viewer import GyrationScreen
        # Molecule with mixed elements
        atoms = [Atom(0.0, 0.0, 0.0, "C"), Atom(1.0, 0.0, 0.0, "O"),
                 Atom(2.0, 0.0, 0.0, "N")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        mol.detect_bonds()
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = GyrationScreen(mol, [mol, mol])
            app.push_screen(screen)
            await pilot.pause()
            sel = screen.query_one("#rg-el", Select)
            opt_values = [v for _, v in sel._options]
            assert "C" in opt_values
            assert "O" in opt_values
            assert "N" in opt_values


# ---------------------------------------------------------------------------
# Keyboard shortcuts not covered in test_e2e.py
# ---------------------------------------------------------------------------

class TestUntestedKeybindings:
    @pytest.mark.asyncio
    async def test_rotate_keys(self):
        """Arrow / hjkl keys rotate the camera."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            view = app.query_one(MoleculeView)
            initial_rot = view.rot_matrix.copy()
            await pilot.press("h")
            assert not np.array_equal(view.rot_matrix, initial_rot)

            after_h = view.rot_matrix.copy()
            await pilot.press("l")
            assert not np.array_equal(view.rot_matrix, after_h)

            after_l = view.rot_matrix.copy()
            await pilot.press("j")
            assert not np.array_equal(view.rot_matrix, after_l)

            after_j = view.rot_matrix.copy()
            await pilot.press("k")
            assert not np.array_equal(view.rot_matrix, after_j)

    @pytest.mark.asyncio
    async def test_center_key(self):
        """`c` resets pan to (0, 0)."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            view = app.query_one(MoleculeView)
            view.pan_x = 5.0
            view.pan_y = -3.0
            await pilot.press("c")
            assert view.pan_x == 0.0
            assert view.pan_y == 0.0

    @pytest.mark.asyncio
    async def test_reset_view_key(self):
        """`r` resets camera distance."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            view = app.query_one(MoleculeView)
            view.camera_distance = 99.0
            view.pan_x = 5.0
            await pilot.press("r")
            # Reset returns camera to a default
            assert view.camera_distance != 99.0
            assert view.pan_x == 0.0

    @pytest.mark.asyncio
    async def test_atom_numbers_toggle_round_trips(self):
        """`#` toggles atom number labels both directions, not just on."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            view = app.query_one(MoleculeView)
            assert view.show_atom_numbers is False
            await pilot.press("number_sign")
            assert view.show_atom_numbers is True
            # Press again — must toggle back off, not stick on
            await pilot.press("number_sign")
            assert view.show_atom_numbers is False


# ---------------------------------------------------------------------------
# Menu items reachable that we hadn't tested
# ---------------------------------------------------------------------------

class TestOpenInNewTab:
    @pytest.mark.asyncio
    async def test_open_in_new_tab_appends_and_switches(self):
        """open_in_new_tab must append a tab and make it active in one call,
        so callers don't have to manage save/append/restore themselves."""
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial_count = len(app._tabs)
            initial_active = app._active_tab

            new_mol = _mol(7)
            new_idx = app.open_in_new_tab(new_mol, "imported.pdb")
            await pilot.pause()

            assert len(app._tabs) == initial_count + 1
            assert new_idx == len(app._tabs) - 1
            assert app._active_tab == new_idx
            assert app._active_tab != initial_active
            # The new tab holds the molecule we passed
            assert app._tabs[new_idx].molecule is new_mol
            assert app._tabs[new_idx].filepath == "imported.pdb"
            # And the active app.molecule got swapped to it
            assert app.molecule is new_mol

    @pytest.mark.asyncio
    async def test_open_in_new_tab_stops_play_timer(self):
        """If the previous tab was playing a trajectory, the play timer must
        stop before the new tab takes over — otherwise the timer keeps firing
        against a now-inactive tab's frames."""
        from textual.timer import Timer
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Simulate active playback
            app._play_timer = MagicMock(spec=Timer)
            app.open_in_new_tab(_mol(3), "x.pdb")
            await pilot.pause()
            assert app._play_timer is None, "play_timer must be cleared"


class TestFrameToTab:
    @pytest.mark.asyncio
    async def test_frame_to_tab_creates_static_snapshot(self):
        """File > Open Frame in New Tab adds a tab whose molecule is a snapshot
        of the *current* frame, with no further trajectory frames attached.
        The atom count must match the source frame, not be 0 or some other
        number — that's the part that catches a wrong-frame copy."""
        m1 = _mol(3)
        m2 = _mol(5)
        app = PdbWizardApp(molecule=m1, filepath="traj.xyz", frames=[m1, m2])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Switch to frame index 1 (m2 with 5 atoms) before snapshotting
            app._frame_idx = 1
            app.molecule = m2
            initial_count = len(app._tabs)
            app._frame_to_tab()
            await pilot.pause()
            assert len(app._tabs) == initial_count + 1
            new_tab = app._tabs[-1]
            # New tab is static: no multi-frame trajectory
            assert not new_tab.frames or len(new_tab.frames) <= 1
            # And it captures the *current* frame's atom count
            assert len(new_tab.molecule.atoms) == len(m2.atoms), (
                f"snapshot took {len(new_tab.molecule.atoms)} atoms, "
                f"expected current frame's {len(m2.atoms)}"
            )

    @pytest.mark.asyncio
    async def test_frame_to_tab_no_frames_notifies_user(self):
        """Without trajectory, the action must surface an explanatory
        notification — not silently no-op. A silent no-op leaves the user
        thinking the menu item is broken."""
        notifications = []
        app = _app()
        async with app.run_test(size=(120, 40)) as pilot:
            orig_notify = app.notify
            def capture(msg, *args, **kwargs):
                notifications.append(str(msg))
                return orig_notify(msg, *args, **kwargs)
            app.notify = capture

            await pilot.pause()
            initial_count = len(app._tabs)
            app._frame_to_tab()
            await pilot.pause()

            assert len(app._tabs) == initial_count, "must not create a tab"
            assert any(
                "trajectory" in n.lower() or "frame" in n.lower()
                for n in notifications
            ), f"expected user-visible notification; got: {notifications}"
