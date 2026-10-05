"""Tests that long-running async tasks are cancelled when their modal is closed.

Without cancellation, tasks keep running after the user dismisses the modal,
wasting CPU and (worse) trying to update widgets that no longer exist.

These tests construct screens directly without mounting through pilot — they
verify the task-tracking contract of `_spawn_task` and `action_close`.
The dismiss() call inside action_close raises (no app), but the cancellation
runs first; we wrap dismiss in try/except by catching at the test level.
"""

import asyncio

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    CoordinationScreen,
    DatabaseSearchScreen,
    DensityScreen,
    GyrationScreen,
    HbondScreen,
    MsdScreen,
    PxrdScreen,
    RdfScreen,
    RmsdScreen,
)


def _mol(n=5):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n)]
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _close_screen(screen):
    """Call action_close while ignoring 'no app' from dismiss()."""
    try:
        screen.action_close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_every_analysis_screen_cancels_on_close():
    """For every analysis screen subclass, action_close cancels in-flight task.
    This exercises each subclass — not just RdfScreen — so a regression in
    one subclass's cancellation path is caught."""
    cases = [
        (RdfScreen, lambda: RdfScreen(_mol(), [_mol(), _mol()])),
        (MsdScreen, lambda: MsdScreen(_mol(), [_mol(), _mol()])),
        (CoordinationScreen, lambda: CoordinationScreen(_mol(), [_mol(), _mol()])),
        (RmsdScreen, lambda: RmsdScreen(_mol(), [_mol(), _mol()])),
        (HbondScreen, lambda: HbondScreen(_mol(), [_mol(), _mol()])),
        (GyrationScreen, lambda: GyrationScreen(_mol(), [_mol(), _mol()])),
        (DensityScreen, lambda: DensityScreen(_mol(), [_mol(), _mol()])),
        (PxrdScreen, lambda: PxrdScreen(_mol())),
        (DatabaseSearchScreen, lambda: DatabaseSearchScreen()),
    ]
    for cls, ctor in cases:
        screen = ctor()
        assert screen._compute_task is None, f"{cls.__name__} _task should start None"

        async def slow():
            await asyncio.sleep(60)

        task = screen._spawn_task(slow())
        assert screen._compute_task is task, f"{cls.__name__} _spawn_task did not assign"
        assert not task.done()

        _close_screen(screen)
        await asyncio.sleep(0)
        assert task.cancelled(), (
            f"{cls.__name__}.action_close did not cancel its in-flight task"
        )


@pytest.mark.asyncio
async def test_spawn_task_assigns_to_self_task():
    screen = RdfScreen(_mol(), [_mol(), _mol()])

    async def slow():
        await asyncio.sleep(60)

    t = screen._spawn_task(slow())
    try:
        assert screen._compute_task is t
        assert not t.done()
    finally:
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t


@pytest.mark.asyncio
async def test_spawn_task_replaces_prior_task():
    """Calling _spawn_task again cancels the previous in-flight task."""
    screen = RdfScreen(_mol(), [_mol(), _mol()])

    async def slow():
        await asyncio.sleep(60)

    t1 = screen._spawn_task(slow())
    t2 = screen._spawn_task(slow())
    try:
        await asyncio.sleep(0)  # let cancellation propagate
        assert t1.cancelled() or t1.done()
        assert screen._compute_task is t2
        assert not t2.done()
    finally:
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2


@pytest.mark.asyncio
async def test_action_close_cancels_running_task():
    """Closing an analysis screen mid-compute cancels the task."""
    screen = RdfScreen(_mol(), [_mol(), _mol()])

    async def slow():
        await asyncio.sleep(60)

    task = screen._spawn_task(slow())
    assert not task.done()

    _close_screen(screen)
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_database_search_action_close_cancels_task():
    screen = DatabaseSearchScreen()

    async def slow():
        await asyncio.sleep(60)

    task = screen._spawn_task(slow())
    assert not task.done()

    _close_screen(screen)
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_action_close_with_no_task_leaves_task_none():
    """action_close with no task in flight must leave _task=None and not raise
    inside the cancel logic. (The dismiss() may raise because there's no app —
    that's expected and ignored by _close_screen.)"""
    screen = RdfScreen(_mol(), [_mol(), _mol()])
    assert screen._compute_task is None
    _close_screen(screen)
    assert screen._compute_task is None  # unchanged


@pytest.mark.asyncio
async def test_action_close_after_task_completed_does_not_cancel_result():
    """If the task already completed normally, action_close must not error
    AND must not turn the completed result into a cancellation."""
    screen = RdfScreen(_mol(), [_mol(), _mol()])

    async def quick():
        return 42

    task = screen._spawn_task(quick())
    await task
    assert task.done()
    assert not task.cancelled()
    assert task.result() == 42

    _close_screen(screen)
    # Cancellation must not have been called on a done task
    assert not task.cancelled()
    assert task.result() == 42


@pytest.mark.asyncio
async def test_app_cancels_pending_bg_tasks_on_unmount():
    """PdbWizardApp must cancel still-pending tasks when unmounting.
    A 60s sleep clearly cannot complete inside the test, so the only way for
    `task.done()` to be true after exit is via cancellation. Asserting
    `task.cancelled()` (not `cancelled() or done()`) catches the case where
    the unmount handler is silently dropped."""
    from pdb_wizard.viewer import PdbWizardApp
    app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app._bg_tasks, set)

        async def slow():
            await asyncio.sleep(60)

        task = app._track_task(slow())
        assert task in app._bg_tasks
        assert not task.done()
    # Exit triggers on_unmount
    await asyncio.sleep(0)
    assert task.cancelled(), (
        "Pending bg task was not cancelled on app unmount — "
        "tasks will leak with 'Task was destroyed' warnings."
    )


@pytest.mark.asyncio
async def test_app_done_callback_removes_task_from_set():
    """When a task completes normally, the done_callback must drop it from
    _bg_tasks so the set doesn't grow unboundedly across the app lifetime."""
    from pdb_wizard.viewer import PdbWizardApp
    app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        async def quick():
            return "ok"

        task = app._track_task(quick())
        assert task in app._bg_tasks
        await task
        await asyncio.sleep(0)  # let done_callback fire
        assert task not in app._bg_tasks, (
            "Completed tasks must be removed from _bg_tasks; otherwise the "
            "set grows for every analysis the user runs."
        )


@pytest.mark.asyncio
async def test_spawn_task_does_not_destroy_screen_widgets():
    """Regression: assigning the spawned task to `self._task` (Textual's
    private MessagePump attribute name) overwrote the screen's own message-
    pump task. Textual's loop lost its handle, the screen's message loop
    died, and all child widgets got pruned — the user saw a frozen UI.

    This test catches the issue by spawning a task that yields long enough
    for the loop to process events, then asserting the screen still has
    its widgets. The earlier test only checked task assignment within a
    single tick, where the destruction wouldn't have surfaced yet."""
    from pdb_wizard.panels import OperationsPanel
    from pdb_wizard.viewer import PdbWizardApp
    app = PdbWizardApp(molecule=_mol(), filepath="t.pdb")
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        app.post_message(OperationsPanel.RunCommand("rdf"))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RdfScreen)
        n_before = len(list(screen.query("*")))
        assert n_before > 10, "RDF screen should compose at least the Selects + buttons + plot"

        async def benign():
            await asyncio.sleep(0.05)

        screen._spawn_task(benign())
        # Let the loop tick long enough that Prune messages would fire if
        # the message pump task had been overwritten.
        await asyncio.sleep(0.1)

        n_after = len(list(screen.query("*")))
        assert n_after == n_before, (
            f"Spawning a compute task destroyed screen widgets: "
            f"{n_before} -> {n_after}. This is the 'pressed Compute and "
            f"the screen froze' bug — name collision with MessagePump._task."
        )
        assert screen.is_attached, (
            "Screen detached from DOM after _spawn_task — message pump died."
        )


@pytest.mark.asyncio
async def test_rdf_compute_button_actually_produces_results():
    """End-to-end: open the RDF modal, fire Compute, wait for it to finish,
    assert the status label reports a peak. No prior test pressed Compute,
    so a freeze in the compute path went undetected."""
    from pdb_wizard.panels import OperationsPanel
    from pdb_wizard.viewer import PdbWizardApp
    from textual.widgets import Label
    # Need a real molecule with multiple element types so RDF has pairs to count
    atoms = [Atom(float(i), 0, 0, "C") for i in range(5)] + \
            [Atom(float(i) + 1.5, 0.5, 0, "O") for i in range(5)]
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()

    app = PdbWizardApp(molecule=mol, filepath="t.pdb")
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        app.post_message(OperationsPanel.RunCommand("rdf"))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RdfScreen)

        # Click Compute via the button
        await pilot.click("#rdf-compute")

        # Wait up to 3s for the compute to finish (status changes to "max g(r)..." or "No ... pairs found")
        for _ in range(30):
            await asyncio.sleep(0.1)
            try:
                status_text = str(screen.query_one("#rdf-status", Label).render())
            except Exception:
                pytest.fail("Screen got destroyed mid-compute — message pump died")
            if "max g(r)" in status_text or "pairs found" in status_text:
                break
        else:
            pytest.fail(f"RDF compute never finished. Last status: {status_text!r}")

        # Compute task should have completed without exception
        assert screen._compute_task is not None
        assert screen._compute_task.done()
        assert not screen._compute_task.cancelled()
        # Re-raises if the task crashed
        screen._compute_task.result()


@pytest.mark.asyncio
async def test_main_app_create_task_sites_use_track_task():
    """grep-driven: viewer.py main app must use _track_task, not raw create_task,
    for orphaned tasks. Only valid raw call_sites are inside _spawn_task helpers
    and _track_task itself."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "src/pdb_wizard/viewer.py"
    text = src.read_text()
    for line_no, line in enumerate(text.splitlines(), 1):
        if "asyncio.create_task" not in line:
            continue
        # Allowed: inside _spawn_task helper (assigns to self._task) or
        # inside _track_task itself (assigns to local `task`).
        if "self._task = asyncio.create_task" in line:
            continue
        if "task = asyncio.create_task" in line:
            continue
        raise AssertionError(
            f"viewer.py:{line_no} uses raw asyncio.create_task — should be self._track_task: {line.strip()!r}"
        )


@pytest.mark.asyncio
async def test_all_compute_call_sites_use_spawn_task():
    """grep-driven test — viewer.py compute buttons must route through _spawn_task,
    not raw asyncio.create_task. This catches regressions when adding new analysis
    screens."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "src/pdb_wizard/viewer.py"
    text = src.read_text()
    # These compute sites are inside _AnalysisScreen / DatabaseSearchScreen
    # and must use _spawn_task.
    forbidden_patterns = [
        "asyncio.create_task(self._compute_rdf_async()",
        "asyncio.create_task(self._run_msd()",
        "asyncio.create_task(self._run_coord()",
        "asyncio.create_task(self._run_rmsd()",
        "asyncio.create_task(self._run_density()",
        "asyncio.create_task(self._run_pxrd()",
        "asyncio.create_task(self._do_search()",
        "asyncio.create_task(self._do_fetch(",
    ]
    for pat in forbidden_patterns:
        assert pat not in text, (
            f"Found {pat!r} — should be self._spawn_task(...) for cancellation"
        )
