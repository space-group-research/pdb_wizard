"""End-to-end tests for every analysis screen's Compute button.

For each analysis modal we:
  1. Push the screen with a realistic molecule (and trajectory frames where
     the analysis requires them).
  2. Click the Compute button via pilot.
  3. Wait up to 5 s for the status label to report a completion state
     (either a success line like "max g(r) = ..." or a recognized
     "no data" message).
  4. Verify the spawned task completed without exception AND the screen
     widgets weren't pruned (catches the MessagePump._task name-collision
     class of bug).

A previous round of tests only opened the modals — none pressed Compute,
so any compute-time bug shipped silently. This file is the gap-closer.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Label

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import (
    CoordinationScreen,
    DensityScreen,
    GyrationScreen,
    HbondScreen,
    MsdScreen,
    PdbWizardApp,
    PxrdScreen,
    RdfScreen,
    RmsdScreen,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _mixed_mol() -> Molecule:
    """Cubic cell with C, O, and H atoms — covers RDF, coordination, density."""
    atoms = []
    rng = np.random.default_rng(42)
    for el in ("C", "O", "N", "H"):
        for _ in range(8):
            x, y, z = rng.uniform(0, 14, 3)
            atoms.append(Atom(float(x), float(y), float(z), el))
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()
    return mol


def _trajectory(n_frames: int = 8) -> tuple[Molecule, list[Molecule]]:
    """Build a small trajectory with consistent atom ordering.
    O atoms drift along x so MSD has signal; small jitter for everything else."""
    rng = np.random.default_rng(7)
    n_atoms = 16
    elements = ["O"] * 8 + ["H"] * 8

    # Initial positions
    coords0 = rng.uniform(2, 13, (n_atoms, 3))
    pbc = PBC(15, 15, 15, 90, 90, 90)
    frames = []
    for fi in range(n_frames):
        atoms = []
        for ai, el in enumerate(elements):
            x = coords0[ai, 0] + (0.3 * fi if el == "O" else rng.normal(0, 0.05))
            y = coords0[ai, 1] + rng.normal(0, 0.05)
            z = coords0[ai, 2] + rng.normal(0, 0.05)
            atoms.append(Atom(float(x), float(y), float(z), el))
        m = Molecule(atoms=atoms, pbc=pbc)
        m.detect_bonds()
        frames.append(m)
    return frames[0], frames


def _hkust1() -> Molecule | None:
    """Real-data canonical test case: HKUST-1 unit cell."""
    path = Path(__file__).parent.parent / "HKUST-1.pdb"
    if not path.exists():
        return None
    from pdb_wizard.io import read_file
    return read_file(str(path))


# ---------------------------------------------------------------------------
# Helper: drive a screen through Compute, wait for status, validate
# ---------------------------------------------------------------------------

DONE_PATTERNS = [
    # Successful completion (regex; first group is the metric)
    re.compile(r"max g\(r\) ="),
    re.compile(r"D = .* cm\^2/s"),
    re.compile(r"Avg coordination at .* A:"),
    re.compile(r"RMSD avg .* ="),
    re.compile(r"Avg: .* H-bonds/frame"),
    re.compile(r"Avg Rg = "),
    re.compile(r"max density ="),
    re.compile(r"\d+ peaks above"),
    re.compile(r"^Done$"),
    # No-data outcomes (also a "compute finished" terminus)
    re.compile(r"No .* pairs found"),
    re.compile(r"No .* atoms found"),
    re.compile(r"No matching atoms"),
    re.compile(r"No frames"),
    re.compile(r"No data$"),
    re.compile(r"No reflections found"),
]


def _is_done(status: str) -> bool:
    return any(p.search(status) for p in DONE_PATTERNS)


async def _drive_compute(
    pilot, screen, button_id: str, status_id: str, timeout_s: float = 5.0,
) -> str:
    """Click the compute button, return the final status string.
    Fails the test if the screen got destroyed mid-compute or the task crashed."""
    n_widgets_before = len(list(screen.query("*")))
    await pilot.click(f"#{button_id}")

    deadline = asyncio.get_event_loop().time() + timeout_s
    last_status = ""
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        # Catch the message-pump-died class of bug
        if not screen.is_attached:
            pytest.fail(
                f"{type(screen).__name__} got detached during compute — "
                f"likely a self._task name collision with MessagePump._task."
            )
        try:
            last_status = str(screen.query_one(f"#{status_id}", Label).render())
        except Exception as e:
            pytest.fail(
                f"{type(screen).__name__}: status label vanished during compute "
                f"(screen children went 51→0 type bug). Error: {e}"
            )
        if _is_done(last_status):
            break
    else:
        pytest.fail(
            f"{type(screen).__name__} compute timed out after {timeout_s}s; "
            f"last status = {last_status!r}"
        )

    # Screen wasn't destroyed
    n_widgets_after = len(list(screen.query("*")))
    assert n_widgets_after > 0.5 * n_widgets_before, (
        f"{type(screen).__name__} lost most of its widgets during compute "
        f"({n_widgets_before} → {n_widgets_after})"
    )

    # Task completed without exception
    task = screen._compute_task
    assert task is not None, f"{type(screen).__name__}: no compute task was tracked"
    assert task.done(), f"{type(screen).__name__}: task still running after status reported done"
    assert not task.cancelled(), f"{type(screen).__name__}: task was cancelled"
    # Re-raise if compute had an exception
    task.result()
    return last_status


# ---------------------------------------------------------------------------
# Each analysis screen: open → click Compute → assert result
# ---------------------------------------------------------------------------

class TestRdfCompute:
    @pytest.mark.asyncio
    async def test_rdf_finishes_on_simple_mol(self):
        mol = _mixed_mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RdfScreen(mol))
            await pilot.pause()
            screen = app.screen
            status = await _drive_compute(pilot, screen, "rdf-compute", "rdf-status")
            assert ("max g(r)" in status or "pairs found" in status), status

    @pytest.mark.asyncio
    async def test_rdf_finishes_on_hkust1(self):
        mol = _hkust1()
        if mol is None:
            pytest.skip("HKUST-1.pdb not available")
        app = PdbWizardApp(molecule=mol, filepath="HKUST-1.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RdfScreen(mol))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "rdf-compute", "rdf-status")
            # Real data should produce a peak, not "no pairs"
            assert "max g(r)" in status, f"Real MOF RDF should find pairs; got {status!r}"


class TestMsdCompute:
    @pytest.mark.asyncio
    async def test_msd_on_drift_trajectory(self):
        mol, frames = _trajectory(n_frames=10)
        app = PdbWizardApp(molecule=mol, filepath="t.pdb", frames=frames)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(MsdScreen(mol, frames))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "msd-compute", "msd-status")
            # Either "D = ..." (success) or "No O atoms found" (filter mismatch). Drifting O should produce D.
            assert "cm^2/s" in status or "Done" in status or "No" in status, status


class TestCoordinationCompute:
    @pytest.mark.asyncio
    async def test_coordination_finds_neighbors(self):
        mol = _mixed_mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(CoordinationScreen(mol))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "coord-compute", "coord-status")
            assert "Avg coordination" in status or "No" in status, status


class TestRmsdCompute:
    @pytest.mark.asyncio
    async def test_rmsd_on_trajectory(self):
        mol, frames = _trajectory(n_frames=12)
        app = PdbWizardApp(molecule=mol, filepath="t.pdb", frames=frames)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(RmsdScreen(mol, frames))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "rmsd-compute", "rmsd-status")
            assert "RMSD avg" in status or "No" in status, status


class TestHbondCompute:
    @pytest.mark.asyncio
    async def test_hbonds_on_water_like_trajectory(self):
        # Build a tiny H2O cluster trajectory (O–H pairs at hbond distance)
        rng = np.random.default_rng(11)
        n_water = 4
        frames = []
        for _ in range(6):
            atoms = []
            for w in range(n_water):
                ox = rng.uniform(2, 13, 3)
                atoms.append(Atom(float(ox[0]), float(ox[1]), float(ox[2]), "O"))
                # Two H atoms ~0.96 A from O
                for hi in range(2):
                    hx = ox + rng.normal(0, 0.1, 3)
                    hx = ox + (hx - ox) / np.linalg.norm(hx - ox) * 0.96
                    atoms.append(Atom(float(hx[0]), float(hx[1]), float(hx[2]), "H"))
            m = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
            m.detect_bonds()
            frames.append(m)
        mol = frames[0]
        app = PdbWizardApp(molecule=mol, filepath="t.pdb", frames=frames)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(HbondScreen(mol, frames))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "hb-compute", "hb-status")
            assert "H-bonds/frame" in status or "No" in status, status


class TestGyrationCompute:
    @pytest.mark.asyncio
    async def test_gyration_on_trajectory(self):
        mol, frames = _trajectory(n_frames=10)
        app = PdbWizardApp(molecule=mol, filepath="t.pdb", frames=frames)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(GyrationScreen(mol, frames))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "rg-compute", "rg-status")
            assert "Avg Rg" in status or "No" in status, status


class TestDensityCompute:
    @pytest.mark.asyncio
    async def test_density_profile(self):
        mol = _mixed_mol()
        app = PdbWizardApp(molecule=mol, filepath="t.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(DensityScreen(mol))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "dens-compute", "dens-status")
            assert "max density" in status or "No" in status, status


class TestPxrdCompute:
    @pytest.mark.asyncio
    async def test_pxrd_finds_peaks_on_hkust1(self):
        mol = _hkust1()
        if mol is None:
            pytest.skip("HKUST-1.pdb not available")
        app = PdbWizardApp(molecule=mol, filepath="HKUST-1.pdb")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            app.push_screen(PxrdScreen(mol))
            await pilot.pause()
            status = await _drive_compute(pilot, app.screen, "pxrd-compute", "pxrd-status",
                                           timeout_s=15.0)
            assert "peaks above" in status, (
                f"Real HKUST-1 should produce diffraction peaks; got {status!r}"
            )


# ---------------------------------------------------------------------------
# Cross-cutting: the screen survives compute on every analysis
# ---------------------------------------------------------------------------

class TestNoFreezeAcrossAllAnalyses:
    """If any subclass of _AnalysisScreen ever picks up a `_task` attribute
    name (or otherwise breaks the message pump) again, this catches it
    in one shot for every screen at once."""

    @pytest.mark.asyncio
    async def test_compute_task_attribute_not_self_task(self):
        """Verify class doesn't define `_task` as an instance attribute name —
        Textual's MessagePump owns that name."""
        for cls in (RdfScreen, MsdScreen, CoordinationScreen, RmsdScreen,
                    HbondScreen, GyrationScreen, DensityScreen, PxrdScreen):
            # An empty molecule is enough — we just need __init__ to run
            try:
                inst = cls(_mixed_mol(), [_mixed_mol(), _mixed_mol()])
            except TypeError:
                inst = cls(_mixed_mol())
            # _task must be the inherited message-pump attribute (None at this point),
            # not our compute-task slot. _compute_task is the project's own slot.
            assert hasattr(inst, "_compute_task")
            assert inst._compute_task is None
            # _task may exist (from MessagePump) but must not be re-typed by us
            # to any value other than None / a Textual Task.
            t = getattr(inst, "_task", None)
            if t is not None:
                # Pre-mount: should be either None or a Task running the screen's pump.
                # We can't easily check it's the right Task without mounting; checking
                # type lets us catch silly assignments like a list or random object.
                import asyncio as _aio
                assert isinstance(t, _aio.Task), (
                    f"{cls.__name__}._task is {type(t).__name__}, "
                    f"expected None or asyncio.Task — name collision with MessagePump"
                )
