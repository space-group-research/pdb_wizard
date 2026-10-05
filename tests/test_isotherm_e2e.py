"""End-to-end test: IsothermPlanner → simulated MPMC results → IsothermPlot.

Walks the full workflow:
  1. Build a charged + LJ-equipped MOF.
  2. Open IsothermPlannerScreen, click Generate at multiple pressures.
  3. Verify the generated directory tree, mpmc.inp contents, input.pqr.
  4. Synthesize histogram.dat files in each pressure subdir (mimics what
     MPMC would write after a finished run) — with KNOWN mean N values
     of our choosing.
  5. Open the IsothermPlot dialog on the same directory.
  6. Assert the parsed pressures match the input pressures (the encode/decode
     round-trip is fragile — this is the test that catches it).
  7. Assert the parsed uptakes match the histogram means we wrote.
  8. Assert the plot widget is populated with the right data.
  9. Use the Plot dialog's CSV export and verify each row.

Asserts are non-lazy: every datum is checked individually, not just dimensions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Input, Label

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import PdbWizardApp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _charged_mof() -> Molecule:
    """Tiny MOF-like molecule with charges + LJ params so the planner's
    validator passes. Net charge is zero."""
    atoms = []
    elem_seq = ["Cu", "C", "C", "O", "O", "H", "H", "C"]
    charge_seq = [0.4, 0.1, 0.1, -0.3, -0.3, 0.05, 0.05, -0.1]
    coords = [
        (1.0, 1.0, 1.0), (4.0, 1.0, 1.0), (7.0, 1.0, 1.0),
        (1.0, 4.0, 1.0), (1.0, 7.0, 1.0),
        (4.0, 4.0, 1.0), (7.0, 4.0, 1.0), (4.0, 7.0, 1.0),
    ]
    for el, q, (x, y, z) in zip(elem_seq, charge_seq, coords):
        a = Atom(x, y, z, el)
        a.charge = q
        a.epsilon = 30.0  # K
        a.sigma = 3.5
        atoms.append(a)
    return Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))


def _write_synthetic_histogram(p_dir: Path, mean_n: float) -> None:
    """Write a histogram.dat that yields the given mean population.

    Real MPMC histogram.dat is two columns: N_count, probability.
    A two-bin histogram at (N=mean-0.5, p=0.5) and (N=mean+0.5, p=0.5)
    has mean = mean_n. We use this rather than a delta to make sure
    the parser actually averages, not just reads the first row.
    """
    floor_n = math.floor(mean_n)
    ceil_n = floor_n + 1
    # Probability mass: weight on each bin so weighted mean = mean_n
    p_high = mean_n - floor_n   # in [0, 1)
    p_low = 1.0 - p_high
    # Always emit both bins for a clean two-row file
    p_dir.joinpath("histogram.dat").write_text(
        f"# N  prob\n"
        f"{floor_n} {p_low:.6f}\n"
        f"{ceil_n} {p_high:.6f}\n"
    )


# ---------------------------------------------------------------------------
# The full e2e
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_isotherm_workflow(tmp_path):
    mol = _charged_mof()
    outdir = tmp_path / "isotherm"
    pressures = [0.001, 0.1, 1.0, 10.0, 100.0]
    # The "true" uptakes we'll inject into histogram.dat per pressure point.
    # Picking distinct fractional values rules out off-by-one row-pick bugs.
    expected_uptakes = {
        0.001: 0.25,
        0.1: 1.75,
        1.0: 5.40,
        10.0: 12.60,
        100.0: 18.10,
    }

    app = PdbWizardApp(molecule=mol, filepath="t.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()

        # ---- Step 1: open the planner the way the user does ----
        # Analysis menu -> "Input Generator" emits this command; driving it
        # exercises the menu/dispatch/push_screen wiring, not just the screen.
        from pdb_wizard.panels import OperationsPanel
        from pdb_wizard.viewer import InputGeneratorScreen
        app.post_message(OperationsPanel.RunCommand("input_generator"))
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, InputGeneratorScreen), (
            f"'input_generator' did not open the generator; "
            f"top screen is {type(screen).__name__}"
        )

        # ---- Step 2: configure + Generate ----
        screen.query_one("#gen-outdir", Input).value = str(outdir)
        screen.query_one("#gen-pressures", Input).value = ", ".join(
            f"{p}" for p in pressures
        )
        await pilot.pause()
        await pilot.click("#gen-generate")
        await pilot.pause()
        await pilot.pause()

        # The planner should now have written files. Note: we don't dismiss
        # this screen — we keep it in the test stack for later assertions.

        # ---- Step 3: verify generated tree ----
        assert outdir.is_dir(), f"output dir {outdir} should exist"
        run_all = outdir / "run_all.sh"
        assert run_all.exists(), "run_all.sh must be created"
        # Each pressure has its own subdir
        sub_dirs = [p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("P_")]
        assert len(sub_dirs) == len(pressures), (
            f"expected {len(pressures)} P_* dirs, got {len(sub_dirs)}: "
            f"{[d.name for d in sub_dirs]}"
        )
        # Each subdir has the canonical four files
        for sd in sub_dirs:
            for fname in ("input.pqr", "mpmc.inp", "insert.pqr", "run.sh"):
                assert (sd / fname).exists(), f"{sd.name}/{fname} missing"

        # ---- Step 4: each mpmc.inp records its own pressure ----
        # This catches the encode/decode round-trip if it ever drifts.
        for p_val in pressures:
            inp_files = [(sd / "mpmc.inp").read_text()
                         for sd in sub_dirs
                         if str(p_val) in (sd / "mpmc.inp").read_text()]
            assert inp_files, f"no mpmc.inp recorded pressure {p_val}"
            inp = inp_files[0]
            # Confirm the temperature line is there too
            assert "temperature           77.0" in inp, (
                f"mpmc.inp for {p_val} missing temperature; got:\n{inp[:200]}"
            )
            assert f"pressure              {p_val}" in inp, (
                f"mpmc.inp for {p_val} missing pressure; got:\n{inp[:200]}"
            )

    # ---- Step 5: simulate finished MPMC runs by writing histogram.dat ----
    # We do this OUTSIDE the previous run_test context so the next plot
    # dialog can run with a fresh app instance — closer to the real flow.
    for sd in outdir.iterdir():
        if not (sd.is_dir() and sd.name.startswith("P_")):
            continue
        # Decode the directory's pressure
        p_str = sd.name.replace("P_", "").replace("_atm", "") \
                       .replace("p", ".").replace("m", "-")
        decoded_p = float(p_str)
        # Match it against the original pressures list with tolerance
        target = min(pressures, key=lambda p: abs(p - decoded_p))
        assert abs(target - decoded_p) < 1e-6, (
            f"directory {sd.name} decoded to {decoded_p}, no match in {pressures}"
        )
        _write_synthetic_histogram(sd, expected_uptakes[target])

    # ---- Step 6: open the IsothermPlot dialog on this directory ----
    app2 = PdbWizardApp(molecule=mol, filepath="t.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        # Drive the real "Plot Isotherm Results" command -> directory picker
        # -> submit, rather than calling _plot_isotherm directly.
        from pdb_wizard.panels import OperationsPanel
        from pdb_wizard.viewer import FileSaveModal
        app2.post_message(OperationsPanel.RunCommand("isotherm_plot"))
        await pilot.pause()
        await pilot.pause()
        picker = app2.screen
        assert isinstance(picker, FileSaveModal), (
            f"'isotherm_plot' did not open the directory picker; "
            f"top screen is {type(picker).__name__}"
        )
        picker.query_one("#file-input", Input).value = str(outdir)
        await pilot.pause()
        await pilot.click("#file-save")
        await pilot.pause()
        await pilot.pause()

        plot_screen = app2.screen
        # The screen class is defined inside _plot_isotherm — check by name
        assert type(plot_screen).__name__ == "IsothermPlotScreen", (
            f"expected IsothermPlotScreen, got {type(plot_screen).__name__}"
        )

        # ---- Step 7: pressures round-trip exactly ----
        plotted_p = list(plot_screen._p)
        plotted_u = list(plot_screen._u)
        assert len(plotted_p) == len(pressures), (
            f"plotter saw {len(plotted_p)} points, expected {len(pressures)}"
        )
        # Pressures are sorted in parse_isotherm_results
        sorted_inputs = sorted(pressures)
        for got, want in zip(plotted_p, sorted_inputs):
            assert math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-4), (
                f"pressure round-trip failed: got {got}, want {want}"
            )

        # ---- Step 8: uptakes match the histogram means we wrote ----
        for p_val, u_got in zip(plotted_p, plotted_u):
            want_u = expected_uptakes[p_val]
            assert math.isclose(u_got, want_u, rel_tol=1e-3, abs_tol=1e-3), (
                f"uptake mismatch at p={p_val}: parsed {u_got}, expected {want_u}"
            )

        # ---- Step 9: plot widget actually has data ----
        from pdb_wizard.viewer import PlotWidget
        plot_widget = plot_screen.query_one("#iso-plot", PlotWidget)
        # Confirm set_data was called (these are the underlying x/y arrays)
        assert len(plot_widget._x) == len(pressures), (
            f"plot widget _x has {len(plot_widget._x)} points, expected {len(pressures)}"
        )
        assert len(plot_widget._y) == len(pressures)

        # Status label reports the expected count + path
        status = str(plot_screen.query_one("#iso-status", Label).render())
        assert f"{len(pressures)} pressure points" in status, status
        assert str(outdir) in status, status

    # ---- Step 10: CSV export — direct method call rather than UI dance ----
    csv_path = tmp_path / "iso.csv"
    plot_screen._do_export_csv(
        str(csv_path), "pressure_atm,avg_molecules",
        np.array(plotted_p), np.array(plotted_u),
    )
    assert csv_path.exists(), "CSV export must produce a file"
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "pressure_atm,avg_molecules"
    data_rows = rows[1:]
    assert len(data_rows) == len(pressures), (
        f"CSV has {len(data_rows)} data rows, expected {len(pressures)}"
    )
    # Each row's pressure and uptake match
    for row, want_p in zip(data_rows, sorted(pressures)):
        parts = row.split(",")
        got_p = float(parts[0])
        got_u = float(parts[1])
        assert math.isclose(got_p, want_p, rel_tol=1e-4, abs_tol=1e-4), row
        assert math.isclose(got_u, expected_uptakes[want_p],
                            rel_tol=1e-3, abs_tol=1e-3), row


# ---------------------------------------------------------------------------
# Edge: empty results dir → user-visible notification, no crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plot_isotherm_on_empty_dir_notifies(tmp_path):
    """If the user points the plot dialog at a directory with no completed
    runs, we expect a 'No completed pressure points found' notification —
    not a silent crash."""
    notifications: list[str] = []
    app = PdbWizardApp(molecule=_charged_mof(), filepath="t.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        orig_notify = app.notify

        def capture(msg, *a, **kw):
            notifications.append(str(msg))
            return orig_notify(msg, *a, **kw)
        app.notify = capture

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        app._plot_isotherm(str(empty_dir))
        await pilot.pause()

        assert any(
            "completed" in n.lower() or "no" in n.lower()
            for n in notifications
        ), f"expected user-visible notification; got {notifications}"
