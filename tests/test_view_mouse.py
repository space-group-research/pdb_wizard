"""Mouse handling of the terminal viewer's 3D view.

Regression: a button release that was not preceded by a press on the view (pressed on
another widget, or before the app was ready, or a terminal still reporting the mouse from an
earlier program) crashed the viewer with
``AttributeError: 'MoleculeView' object has no attribute '_drag_moved'``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("textual")

from pdb_wizard.io import read_file  # noqa: E402
from pdb_wizard.viewer import MoleculeView, PdbWizardApp  # noqa: E402

HKUST = Path(__file__).resolve().parent.parent / "HKUST-1.pdb"


def mouse(x, y, shift=False):
    return SimpleNamespace(x=x, y=y, shift=shift)


def record_picks(view) -> list:
    """Record the AtomPicked messages the view posts, still delivering every message
    (replacing post_message outright stalls the app's shutdown)."""
    picked: list = []
    original = view.post_message

    def spy(message):
        if isinstance(message, MoleculeView.AtomPicked):
            picked.append(message)
        return original(message)

    view.post_message = spy
    return picked


def with_view(fn):
    async def go():
        app = PdbWizardApp(molecule=read_file(str(HKUST)), filepath=str(HKUST))
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            return await fn(pilot, app, app.query_one(MoleculeView))
    return asyncio.run(go())


def test_a_fresh_view_has_its_drag_state():
    view = MoleculeView()
    assert view._dragging is False and view._drag_moved is False


def test_release_without_a_press_is_ignored():
    async def run(pilot, app, view):
        picked = record_picks(view)
        before = view.rot_matrix.copy()
        view.on_mouse_up(mouse(62, 19))  # the event from the bug report: no press first
        view.measure_mode = True
        view.on_mouse_up(mouse(62, 19))  # and it must not count as a click in measure mode
        await pilot.pause()
        return before, view.rot_matrix.copy(), picked, view._dragging

    before, after, picked, dragging = with_view(run)
    np.testing.assert_array_equal(before, after)
    assert picked == [] and dragging is False


def test_move_without_a_press_does_nothing():
    async def run(pilot, app, view):
        before = view.rot_matrix.copy()
        view.on_mouse_move(mouse(70, 25))
        return before, view.rot_matrix.copy()

    before, after = with_view(run)
    np.testing.assert_array_equal(before, after)


def test_drag_still_rotates_and_can_be_repeated():
    async def run(pilot, app, view):
        start = view.rot_matrix.copy()
        for _ in range(3):  # "click and drag the view around too much"
            view.on_mouse_down(mouse(40, 20))
            for step in range(1, 6):
                view.on_mouse_move(mouse(40 + 4 * step, 20 + 2 * step))
            view.on_mouse_up(mouse(60, 30))
            view.on_mouse_up(mouse(60, 30))  # a duplicate release is harmless
            await pilot.pause()
        return start, view.rot_matrix.copy(), view._dragging, view._drag_moved

    start, end, dragging, moved = with_view(run)
    assert not np.allclose(start, end)
    assert np.allclose(end @ end.T, np.eye(3), atol=1e-9)  # still a rotation
    assert dragging is False and moved is False


def test_real_mouse_events_from_the_pilot():
    """Press on one widget and release over the view, through textual's own event path."""
    from textual.events import MouseDown, MouseMove, MouseUp

    async def run(pilot, app, view):
        before = view.rot_matrix.copy()
        await pilot._post_mouse_events([MouseUp], widget=view, offset=(20, 10))
        await pilot.pause()
        untouched = np.array_equal(before, view.rot_matrix)
        await pilot._post_mouse_events([MouseDown], widget=view, offset=(20, 10))
        await pilot._post_mouse_events([MouseMove], widget=view, offset=(32, 16))
        await pilot._post_mouse_events([MouseUp], widget=view, offset=(32, 16))
        await pilot.pause()
        return untouched, not np.allclose(before, view.rot_matrix), app.is_running

    untouched, rotated, running = with_view(run)
    assert untouched and rotated and running


def test_click_in_measure_mode_still_picks_an_atom():
    async def run(pilot, app, view):
        view.measure_mode = True
        picked = record_picks(view)
        view._pick_atom_at = lambda x, y: 7
        view.on_mouse_down(mouse(50, 20))
        view.on_mouse_up(mouse(50, 20))
        view.on_mouse_down(mouse(50, 20))
        view.on_mouse_move(mouse(60, 26))  # a drag is not a pick
        view.on_mouse_up(mouse(60, 26))
        await pilot.pause()
        return picked

    picked = with_view(run)
    assert len(picked) == 1 and isinstance(picked[0], MoleculeView.AtomPicked)
