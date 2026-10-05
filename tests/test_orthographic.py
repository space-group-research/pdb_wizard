"""Tests for orthographic projection mode."""

from __future__ import annotations

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.renderer import (
    ImageRenderer,
    render_scene,
    rotation_matrix,
)
from pdb_wizard.viewer import (
    MENU_VIEW,
    MoleculeView,
    PdbWizardApp,
)


def _mol():
    """A small molecule with atoms at known z-distances."""
    atoms = [
        Atom(0.0, 0.0, 0.0, "C"),
        Atom(2.0, 0.0, 0.0, "C"),
        Atom(0.0, 2.0, 0.0, "C"),
        Atom(0.0, 0.0, 2.0, "C"),
    ]
    mol = Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))
    mol.detect_bonds()
    return mol


# ---------------------------------------------------------------------------
# Renderer-level: orthographic projection math
# ---------------------------------------------------------------------------

class TestOrthographicProjection:
    def test_default_is_perspective(self):
        r = ImageRenderer(80, 40)
        assert r.orthographic is False

    def test_orthographic_x_independent_of_z(self):
        """In orthographic, projecting (1, 0, z) for varying z must give
        the same screen-space x. In perspective, x scales with 1/z."""
        r = ImageRenderer(80, 40)
        r.orthographic = True
        r.ortho_scale = 0.1
        x_far, _, _ = r._project(np.array([1.0, 0.0, 5.0]))
        x_near, _, _ = r._project(np.array([1.0, 0.0, 50.0]))
        assert pytest.approx(x_far) == x_near

    def test_perspective_x_scales_with_inverse_z(self):
        r = ImageRenderer(80, 40)
        # Default perspective
        x_far, _, _ = r._project(np.array([1.0, 0.0, 50.0]))
        x_near, _, _ = r._project(np.array([1.0, 0.0, 5.0]))
        # Closer atom should project further from center
        center_x = 80 / 2
        assert abs(x_near - center_x) > abs(x_far - center_x)

    def test_orthographic_projected_radius_independent_of_z(self):
        """A sphere in ortho keeps its on-screen size at any depth.
        Render two scenes with the same sphere at different z, check
        that the painted bounding box has the same size."""
        from pdb_wizard.renderer import ImageRenderer
        r = ImageRenderer(80, 40)
        r.orthographic = True
        r.ortho_scale = 0.05

        # Two atoms identical except z
        r.clear()
        r.render_sphere(np.array([0.0, 0.0, 5.0]), radius=1.0, color=(255, 0, 0))
        hit_near = (r.z_buf < float("inf")).sum()

        r.clear()
        r.render_sphere(np.array([0.0, 0.0, 25.0]), radius=1.0, color=(255, 0, 0))
        hit_far = (r.z_buf < float("inf")).sum()

        # Allow small difference from sub-pixel rounding
        assert abs(hit_near - hit_far) < 5, (
            f"orthographic sphere should look the same regardless of z; "
            f"near hit {hit_near}, far hit {hit_far}"
        )

    def test_perspective_sphere_shrinks_with_distance(self):
        """Sanity check: in perspective mode, the same sphere covers fewer
        pixels at greater z."""
        r = ImageRenderer(80, 40)
        # default: orthographic = False

        r.clear()
        r.render_sphere(np.array([0.0, 0.0, 5.0]), radius=1.0, color=(255, 0, 0))
        near = (r.z_buf < float("inf")).sum()

        r.clear()
        r.render_sphere(np.array([0.0, 0.0, 25.0]), radius=1.0, color=(255, 0, 0))
        far = (r.z_buf < float("inf")).sum()

        assert near > far * 1.5, (
            f"perspective sphere should shrink with distance; "
            f"near={near}, far={far}"
        )


# ---------------------------------------------------------------------------
# render_scene plumbs the flag
# ---------------------------------------------------------------------------

class TestRenderScenePlumbing:
    def test_orthographic_kwarg_changes_output(self):
        """Rendering the same scene perspective vs ortho must produce
        different pixel buffers — proves the flag actually wires through."""
        mol = _mol()
        rot = rotation_matrix(0.3, 0.5, 0.0)
        cam = max(15.0, mol.radius() * 3)

        persp_pixels, _ = render_scene(
            120, 60, mol, rot, cam, ssaa=1, orthographic=False,
        )
        ortho_pixels, _ = render_scene(
            120, 60, mol, rot, cam, ssaa=1, orthographic=True,
        )
        # If the flag wasn't plumbed, both buffers would be identical
        assert not np.array_equal(persp_pixels, ortho_pixels), (
            "orthographic=True must produce different output than perspective"
        )


# ---------------------------------------------------------------------------
# UI: keybinding + menu entry + tab persistence
# ---------------------------------------------------------------------------

class TestOrthographicUi:
    def test_menu_view_has_entry(self):
        cmds = [c for c, _l in MENU_VIEW if c != "---"]
        assert "toggle_orthographic" in cmds, (
            "View menu must list 'toggle_orthographic' for discoverability"
        )

    @pytest.mark.asyncio
    async def test_O_key_toggles_orthographic(self):
        """Pressing 'O' toggles the projection mode."""
        app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            view = app.query_one(MoleculeView)
            assert view.orthographic is False

            await pilot.press("O")
            await pilot.pause()
            assert view.orthographic is True, "O should turn ortho on"

            await pilot.press("O")
            await pilot.pause()
            assert view.orthographic is False, "second O should turn ortho off"

    @pytest.mark.asyncio
    async def test_orthographic_state_persists_across_tab_switches(self):
        """Toggling ortho on tab 0, switching to tab 1, switching back
        must preserve ortho on tab 0 (and not leak it to tab 1)."""
        app = PdbWizardApp(molecule=_mol(), filepath="a.pdb")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            view = app.query_one(MoleculeView)

            # Tab 0: turn ortho ON
            await pilot.press("O")
            await pilot.pause()
            assert view.orthographic is True

            # Open a second tab
            from pdb_wizard.molecule import Molecule
            app.open_in_new_tab(Molecule(atoms=[]), "b.pdb")
            await pilot.pause()
            assert view.orthographic is False, (
                "fresh tab should start in perspective"
            )

            # Switch back to tab 0
            app._save_current_tab_state()
            app._restore_tab_state(0)
            await pilot.pause()
            assert view.orthographic is True, (
                "tab 0's ortho state must be preserved across switch"
            )
