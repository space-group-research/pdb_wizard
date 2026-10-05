"""Full-workflow e2e test.

Loads HKUST-1, runs every analysis screen with multiple parameter
combinations per window, then doubles the a-axis and runs them all again,
then opens a new tab and runs them all again.

This is the integration test that catches:
  - Re-running an analysis after parameter change (closes nothing, just
    pressing Compute again with the same screen open).
  - State leaking between analyses (e.g. one screen mutating the molecule
    in a way that breaks the next).
  - The extend operation refreshing the active tab's molecule correctly.
  - Tabs holding their own independent state (analyses on tab 2 still work
    after compute on tab 1).

Slow — runs the whole pipeline twice on increasing system size + once on a
fresh tab. Total cost ~30 s on the canonical HKUST-1 (1× and 2×1×1 cells).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Label, Select

from pdb_wizard.io import read_file
from pdb_wizard.viewer import (
    CoordinationScreen,
    DensityScreen,
    HbondScreen,
    PdbWizardApp,
    PxrdScreen,
    RdfScreen,
    SpinBox,
)

HKUST = Path(__file__).parent.parent / "HKUST-1.pdb"
pytestmark = pytest.mark.skipif(not HKUST.exists(), reason="HKUST-1.pdb not found")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DONE_PATTERNS = [
    re.compile(r"max g\(r\) ="),
    re.compile(r"D = .* cm\^2/s"),
    re.compile(r"Avg coordination at .* A:"),
    re.compile(r"RMSD avg .* ="),
    re.compile(r"Avg: .* H-bonds/frame"),
    re.compile(r"Avg Rg = "),
    re.compile(r"max density ="),
    re.compile(r"\d+ peaks above"),
    re.compile(r"^Done$"),
    re.compile(r"No .* pairs found"),
    re.compile(r"No .* atoms found"),
    re.compile(r"No matching atoms"),
    re.compile(r"No frames"),
    re.compile(r"No data$"),
    re.compile(r"No reflections found"),
]


def _is_done(status: str) -> bool:
    return any(p.search(status) for p in DONE_PATTERNS)


async def _wait_compute(pilot, screen, status_id: str,
                        timeout_s: float = 20.0, label: str = "") -> str:
    """Poll the status label until a recognized done-pattern appears."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last = ""
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        if not screen.is_attached:
            pytest.fail(
                f"{label}: screen detached during compute (message-pump death)"
            )
        try:
            last = str(screen.query_one(f"#{status_id}", Label).render())
        except Exception as e:
            pytest.fail(f"{label}: status query failed mid-compute: {e}")
        if _is_done(last):
            return last
    pytest.fail(f"{label}: compute timed out after {timeout_s}s. last = {last!r}")


async def _click_compute(pilot, screen, button_id: str, status_id: str,
                          *, label: str, timeout_s: float = 20.0) -> str:
    """Press Compute, wait for completion, return final status text."""
    await pilot.click(f"#{button_id}")
    return await _wait_compute(pilot, screen, status_id, timeout_s, label)


# ---------------------------------------------------------------------------
# Per-screen variation runners — each opens a screen, runs N parameter sweeps,
# then closes it. Returns the list of status strings observed.
# ---------------------------------------------------------------------------

async def _rdf_variations(app, pilot, *, label: str) -> list[str]:
    """Run RDF with three different element-pair selections."""
    mol = app.molecule
    elements = sorted({a.element.symbol for a in mol.atoms})
    app.push_screen(RdfScreen(mol))
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, RdfScreen)

    # Pick three different (el1, el2) combos available in this molecule
    pairs: list[tuple[str, str]] = []
    for a in elements:
        for b in elements:
            if (a, b) not in pairs and (b, a) not in pairs:
                pairs.append((a, b))
            if len(pairs) >= 3:
                break
        if len(pairs) >= 3:
            break

    results = []
    for i, (el1, el2) in enumerate(pairs):
        screen.query_one("#rdf-el1", Select).value = el1
        screen.query_one("#rdf-el2", Select).value = el2
        # Vary x_max and bins between iterations
        screen.query_one("#rdf-xmax", SpinBox)._update(8 + 2 * i)
        screen.query_one("#rdf-bins", SpinBox)._update(50 + 50 * i)
        await pilot.pause()
        status = await _click_compute(
            pilot, screen, "rdf-compute", "rdf-status",
            label=f"{label} RDF[{el1}-{el2}]",
        )
        results.append(status)

    # Close
    await pilot.click("#rdf-close")
    await pilot.pause()
    return results


async def _coordination_variations(app, pilot, *, label: str) -> list[str]:
    """Coordination at two different cutoffs and element pairs."""
    mol = app.molecule
    elements = sorted({a.element.symbol for a in mol.atoms})
    if len(elements) < 2:
        return []
    app.push_screen(CoordinationScreen(mol))
    await pilot.pause()
    screen = app.screen

    cases = [
        (elements[0], elements[1], 35),  # 3.5 A (SpinBox stores integer × 0.1)
        (elements[1], elements[0], 30),
    ]
    if len(elements) >= 3:
        cases.append((elements[0], elements[2], 40))

    results = []
    for el1, el2, cutoff_int in cases:
        screen.query_one("#coord-el1", Select).value = el1
        screen.query_one("#coord-el2", Select).value = el2
        # CoordinationScreen has a cutoff SpinBox — find it dynamically
        try:
            cutoff = screen.query_one("#coord-cutoff", SpinBox)
            cutoff._update(cutoff_int)
        except Exception:
            pass
        await pilot.pause()
        status = await _click_compute(
            pilot, screen, "coord-compute", "coord-status",
            label=f"{label} Coord[{el1}-{el2}]",
        )
        results.append(status)

    await pilot.click("#coord-close")
    await pilot.pause()
    return results


async def _density_variations(app, pilot, *, label: str) -> list[str]:
    """Density profile along all 3 axes."""
    mol = app.molecule
    elements = sorted({a.element.symbol for a in mol.atoms})
    if not elements:
        return []
    app.push_screen(DensityScreen(mol))
    await pilot.pause()
    screen = app.screen

    results = []
    # Try first element along each axis
    for axis_label in ("x", "y", "z"):
        screen.query_one("#dens-el", Select).value = elements[0]
        try:
            screen.query_one("#dens-axis", Select).value = axis_label
        except Exception:
            pass
        await pilot.pause()
        status = await _click_compute(
            pilot, screen, "dens-compute", "dens-status",
            label=f"{label} Density[{elements[0]},{axis_label}]",
        )
        results.append(status)

    await pilot.click("#dens-close")
    await pilot.pause()
    return results


async def _pxrd_variations(app, pilot, *, label: str,
                            timeout_s: float = 30.0) -> list[str]:
    """PXRD with two different wavelengths."""
    mol = app.molecule
    app.push_screen(PxrdScreen(mol))
    await pilot.pause()
    screen = app.screen

    # Try Cu (default) and Mo (shorter wavelength)
    results = []
    for wl_value in ("1.5406", "0.7107"):
        try:
            screen.query_one("#pxrd-wl", Select).value = wl_value
        except Exception:
            pass
        # Reduce 2theta range to keep test fast on larger systems
        try:
            screen.query_one("#pxrd-ttmax", SpinBox)._update(30)
        except Exception:
            pass
        await pilot.pause()
        status = await _click_compute(
            pilot, screen, "pxrd-compute", "pxrd-status",
            label=f"{label} PXRD[{wl_value}]", timeout_s=timeout_s,
        )
        results.append(status)

    await pilot.click("#pxrd-close")
    await pilot.pause()
    return results


async def _hbond_single_frame(app, pilot, *, label: str) -> str:
    """H-bond detection on a single frame (HKUST-1 has none, but the path runs)."""
    mol = app.molecule
    # HbondScreen requires frames; pass [mol] as a single-frame trajectory
    app.push_screen(HbondScreen(mol, [mol]))
    await pilot.pause()
    screen = app.screen
    # Default cutoff/angle should be fine
    await pilot.pause()
    status = await _click_compute(
        pilot, screen, "hb-compute", "hb-status",
        label=f"{label} H-bonds",
    )
    await pilot.click("#hb-close")
    await pilot.pause()
    return status


# ---------------------------------------------------------------------------
# Run-everything helper
# ---------------------------------------------------------------------------

async def _run_all_analyses(app, pilot, *, label: str,
                             include_pxrd: bool = True,
                             pxrd_timeout: float = 30.0) -> dict:
    """Run RDF + coord + density + PXRD + Hbond on the active tab.
    Returns dict[name -> list of status strings]."""
    out = {}
    out["rdf"] = await _rdf_variations(app, pilot, label=label)
    out["coord"] = await _coordination_variations(app, pilot, label=label)
    out["density"] = await _density_variations(app, pilot, label=label)
    out["hbonds"] = [await _hbond_single_frame(app, pilot, label=label)]
    if include_pxrd:
        out["pxrd"] = await _pxrd_variations(app, pilot, label=label, timeout_s=pxrd_timeout)
    return out


# ---------------------------------------------------------------------------
# THE BIG TEST
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_workflow_hkust1_extend_and_new_tab():
    """Open HKUST-1 → run every analysis with multiple option sweeps →
    extend a-axis 2× → run them all again → open new tab → run them again."""
    mol = read_file(str(HKUST))
    n_atoms_initial = len(mol.atoms)

    app = PdbWizardApp(molecule=mol, filepath=str(HKUST))
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        await pilot.pause()

        # ----- Phase 1: HKUST-1 1×1×1 -----
        phase1 = await _run_all_analyses(app, pilot, label="phase1_1x1x1")

        # Sanity: RDF on real MOF should find peaks (not "no pairs")
        rdf_results = phase1["rdf"]
        assert any("max g(r)" in s for s in rdf_results), (
            f"phase1 RDF found no peaks at all on HKUST-1: {rdf_results}"
        )
        # PXRD on real crystal must produce reflections
        assert any("peaks above" in s for s in phase1["pxrd"]), (
            f"phase1 PXRD found no reflections: {phase1['pxrd']}"
        )

        # ----- Phase 2: extend a-axis 2× via the production code path -----
        # _do_extend mirrors what ExtendAxisModal's callback does.
        app._do_extend((2, 1, 1))
        # _do_extend kicks off a background task; wait for it to finish
        for _ in range(100):
            await asyncio.sleep(0.05)
            if len(app.molecule.atoms) == n_atoms_initial * 2:
                break
        else:
            pytest.fail(
                f"a-axis extend never finished. "
                f"atoms = {len(app.molecule.atoms)}, expected {n_atoms_initial * 2}"
            )
        await pilot.pause()
        await pilot.pause()

        # ----- Phase 3: run all again on doubled cell -----
        # PXRD on 2× cell is slower; bump timeout
        phase2 = await _run_all_analyses(
            app, pilot, label="phase2_2x1x1", pxrd_timeout=60.0,
        )
        # RDF on the doubled cell should still find peaks
        assert any("max g(r)" in s for s in phase2["rdf"]), (
            f"phase2 (2×1×1) RDF found no peaks: {phase2['rdf']}"
        )

        # ----- Phase 4: open a NEW tab with a fresh HKUST-1 -----
        fresh = read_file(str(HKUST))
        n_tabs_before = len(app._tabs)
        active_before = app._active_tab
        new_idx = app.open_in_new_tab(fresh, str(HKUST))
        await pilot.pause()
        await pilot.pause()
        assert len(app._tabs) == n_tabs_before + 1
        assert app._active_tab == new_idx
        assert app._active_tab != active_before
        # The fresh tab must have the original 1× atom count, not the 2× from tab 1
        assert len(app.molecule.atoms) == n_atoms_initial, (
            f"new tab inherited extended state: {len(app.molecule.atoms)} atoms"
        )

        # ----- Phase 5: run all analyses on the new tab -----
        phase3 = await _run_all_analyses(app, pilot, label="phase3_newtab")
        assert any("max g(r)" in s for s in phase3["rdf"]), (
            f"phase3 (new tab) RDF found no peaks: {phase3['rdf']}"
        )

        # ----- Phase 6: switch back to tab 0 — its state must be preserved -----
        app._save_current_tab_state()
        app._restore_tab_state(active_before)
        await pilot.pause()
        # Tab 0 still has the 2× cell from phase 2
        assert len(app.molecule.atoms) == n_atoms_initial * 2, (
            f"tab 0 lost its extended state: {len(app.molecule.atoms)} atoms"
        )
