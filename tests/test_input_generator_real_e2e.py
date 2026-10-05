"""Full round-trip e2e: InputGeneratorScreen → real simulator → load back into pdb_wizard.

For each engine where a simulator binary is available, we drive the real user
path end to end:
  1. Open the generator via the Analysis menu command (not a direct push)
  2. Switch to the target engine via the Select dropdown
  3. Tweak form values for a fast tiny-system run
  4. Set output dir to tmp_path
  5. Click Generate and assert the status line reports success (not an error)
  6. Verify the rendered files exist and have no leftover {{{placeholders}}}
  7. Run the simulator binary as a subprocess
  8. Open the resulting trajectory via the File "Open File..." command +
     picker modal (not a direct _do_open_file call)
  9. Assert tab count, frame count, atom/element fidelity

OpenMM tests need an openmm-equipped Python; CP2K tests need a working cp2k
binary; MPMC tests need a built mpmc binary (see github.com/mpmccode/mpmc).
Tests skip if the relevant binary isn't available.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from textual.widgets import Input, Select

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import InputGeneratorScreen, PdbWizardApp, SpinBox

# ---------------------------------------------------------------------------
# Simulator discovery
# ---------------------------------------------------------------------------

_OPENMM_CANDIDATES = [
    "/home/user/.conda-envs/smee/bin/python",
    "/home/user/.conda-envs/smirnoff-plugins/bin/python",
    "/home/user/.conda-envs/tholedipoleplugin/bin/python",
    "/home/user/.conda-envs/phast_sim/bin/python",
]


def _find_openmm_python() -> str | None:
    for p in _OPENMM_CANDIDATES:
        if not os.path.isfile(p):
            continue
        try:
            r = subprocess.run([p, "-c", "import openmm"],
                               capture_output=True, timeout=20)
            if r.returncode == 0:
                return p
        except Exception:
            continue
    return None


_CP2K_CANDIDATES = [
    "/home/user/.conda-envs/cp2k/bin/cp2k.ssmp",
    "/home/user/.conda-envs/cp2k/bin/cp2k.popt",
]
_CP2K_DATA = "/home/user/.conda-envs/cp2k/share/cp2k/data"
_CP2K_LIBS = "/home/user/.conda-envs/cp2k/lib"


def _find_cp2k() -> str | None:
    for p in _CP2K_CANDIDATES:
        if not os.path.isfile(p):
            continue
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _CP2K_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
        try:
            r = subprocess.run([p, "--version"], capture_output=True,
                               timeout=20, env=env)
            if r.returncode == 0 and b"CP2K version" in r.stdout:
                return p
        except Exception:
            continue
    return None


_MPMC_CANDIDATES = [
    "/home/user/mpmc/build/mpmc",
    "/home/user/mpmc/bin/mpmc",
]


def _find_mpmc() -> str | None:
    import shutil
    cands = list(_MPMC_CANDIDATES)
    on_path = shutil.which("mpmc")
    if on_path:
        cands.append(on_path)
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            # No-arg mpmc prints a usage line and exits 0.
            r = subprocess.run([p], capture_output=True, timeout=20, text=True)
            if "usage" in (r.stdout + r.stderr).lower():
                return p
        except Exception:
            continue
    return None


OPENMM_PY = _find_openmm_python()
CP2K_BIN = _find_cp2k()
MPMC_BIN = _find_mpmc()

openmm_required = pytest.mark.skipif(
    OPENMM_PY is None, reason="No openmm-equipped Python found",
)
cp2k_required = pytest.mark.skipif(
    CP2K_BIN is None, reason="No working CP2K binary found",
)
mpmc_required = pytest.mark.skipif(
    MPMC_BIN is None, reason="No working MPMC binary found",
)


def _run_cp2k(workdir: Path, timeout: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["CP2K_DATA_DIR"] = _CP2K_DATA
    env["LD_LIBRARY_PATH"] = _CP2K_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [CP2K_BIN, "-i", "input.inp", "-o", "cp2k.out"],
        cwd=workdir, capture_output=True, text=True,
        timeout=timeout, env=env,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_water_box() -> Molecule:
    """Two TIP3P-shaped waters in a 20 Å cubic box."""
    atoms = [
        Atom(0.5, 0.5, 0.5, "O"),
        Atom(1.457, 0.5, 0.5, "H"),
        Atom(0.260, 1.428, 0.5, "H"),
        Atom(4.5, 4.5, 4.0, "O"),
        Atom(5.457, 4.5, 4.0, "H"),
        Atom(4.260, 5.428, 4.0, "H"),
    ]
    return Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))


def _single_water() -> Molecule:
    atoms = [
        Atom(6.0, 6.0, 6.0, "O"),
        Atom(6.957, 6.0, 6.0, "H"),
        Atom(5.760, 6.928, 6.0, "H"),
    ]
    return Molecule(atoms=atoms, pbc=PBC(12, 12, 12, 90, 90, 90))


def _charged_mof() -> Molecule:
    """Tiny MOF analogue with charges + LJ — passes MPMC validator."""
    atoms = []
    seq = [
        ("Cu", 0.4), ("C", 0.1), ("C", 0.1), ("O", -0.3),
        ("O", -0.3), ("H", 0.05), ("H", 0.05), ("C", -0.1),
    ]
    coords = [
        (1.0, 1.0, 1.0), (4.0, 1.0, 1.0), (7.0, 1.0, 1.0),
        (1.0, 4.0, 1.0), (1.0, 7.0, 1.0),
        (4.0, 4.0, 1.0), (7.0, 4.0, 1.0), (4.0, 7.0, 1.0),
    ]
    for (el, q), (x, y, z) in zip(seq, coords):
        a = Atom(x, y, z, el)
        a.charge = q
        a.epsilon = 30.0
        a.sigma = 3.5
        atoms.append(a)
    return Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))


def _porous_framework() -> Molecule:
    """A self-contained, *physical* framework MPMC can actually simulate.

    Carbon-like atoms on a 6 Å simple-cubic lattice in a 24 Å box: spacing
    (6 Å) is well clear of sigma (3.4 Å) so there are no overlaps, the large
    voids adsorb H2, and alternating ±0.2 e charges give a net-neutral,
    electrostatically active host. Unlike _charged_mof (atoms 3 Å apart with
    sigma 3.5 → overlap → NaN energies → MPMC segfault) this runs to
    completion and yields a finite uptake.
    """
    L, spacing = 24.0, 6.0
    n = int(L / spacing)  # 4 per axis -> 64 atoms
    atoms = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = Atom(i * spacing + 1.0, j * spacing + 1.0, k * spacing + 1.0, "C")
                a.charge = 0.2 if (i + j + k) % 2 == 0 else -0.2
                a.epsilon = 50.0   # K
                a.sigma = 3.4      # Å
                atoms.append(a)
    return Molecule(atoms=atoms, pbc=PBC(L, L, L, 90, 90, 90))


def _run_mpmc(workdir: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run MPMC in `workdir` on its mpmc.inp. MPMC locates its model_dir
    relative to the binary (../models), so no env setup is needed."""
    return subprocess.run(
        [MPMC_BIN, "mpmc.inp"],
        cwd=workdir, capture_output=True, text=True, timeout=timeout,
    )


def _write_synthetic_energy_dat(p_dir: Path, mean_n: float) -> None:
    """Write an energy.dat in the real MPMC column layout whose last-quarter
    mean of the N column equals `mean_n`.

    This is the file format real MPMC actually writes and that
    parse_isotherm_results actually reads — MPMC's histogram.dat is an OpenDX
    3D grid, not a two-column N/prob table, so the parser falls through to
    energy.dat. The earlier version of this test synthesized a fake two-column
    histogram.dat and exercised a parser branch that never fires on real
    output; this mirrors reality instead.
    """
    header = ("#step #energy #coulombic #rd #polar #vdw #kinetic #kin_temp "
              "#N #spin_ratio #volume #core_temp\n")
    # 8 rows: parse averages the last quarter (rows 6,7). Earlier rows differ
    # so the test proves it averages the tail, not just reads a single row.
    n_seq = [0.0, mean_n * 0.2, mean_n * 0.5, mean_n * 2.0,
             mean_n * 1.5, mean_n * 0.7, mean_n, mean_n]
    rows = []
    for i, n in enumerate(n_seq):
        rows.append(
            f"{i * 100} -12.0 1.5 -13.5 0.0 0.0 0.0 0.0 "
            f"{n:.6f} 100.0 1000.0 298.0"
        )
    (p_dir / "energy.dat").write_text(header + "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

async def _switch_engine(screen: InputGeneratorScreen, pilot, engine_key: str):
    """Drive the engine Select; wait for the form to rebuild."""
    sel = screen.query_one("#gen-engine", Select)
    sel.value = engine_key
    # Setting Select.value posts a Changed message; pump the loop until the
    # screen's on_select_changed handler runs and _build_form completes.
    await pilot.pause()
    await pilot.pause()
    assert screen._engine_key == engine_key, (
        f"engine didn't switch: still {screen._engine_key}"
    )


def _set_form(screen: InputGeneratorScreen, **values):
    """Set form field values by VarSpec.name."""
    for name, value in values.items():
        widget = screen._form_widgets.get(name)
        assert widget is not None, f"no form widget for {name!r}"
        if isinstance(widget, SpinBox):
            widget._update(int(value))
        elif isinstance(widget, Select):
            widget.value = str(value)
        else:
            widget.value = str(value)


def _status_text(screen: InputGeneratorScreen) -> str:
    """Read the user-visible status line back as plain text."""
    from textual.widgets import Label
    return str(screen.query_one("#gen-status", Label).render())


async def _open_generator_via_menu(app, pilot):
    """Open the Input Generator the way a user does: the Analysis panel command.

    Posts the same OperationsPanel.RunCommand the panel emits when the user
    clicks 'Input Generator', so this exercises the menu->dispatch->push_screen
    wiring rather than constructing the screen directly.
    """
    from pdb_wizard.panels import OperationsPanel
    app.post_message(OperationsPanel.RunCommand("input_generator"))
    await pilot.pause()
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, InputGeneratorScreen), (
        f"'input_generator' command did not open the generator; "
        f"top screen is {type(screen).__name__}"
    )
    return screen


async def _generate(screen, pilot):
    """Click Generate and assert the status line reports success, not an error.

    A user only knows generation worked from this status line — _do_generate
    swallows exceptions into it, so a green file-existence check alone would
    miss a generator that reports failure to the user."""
    await pilot.click("#gen-generate")
    await pilot.pause()
    await pilot.pause()
    status = _status_text(screen)
    low = status.lower()
    assert not (low.startswith("error") or low.startswith("invalid")
                or low.startswith("refusing") or "cannot generate" in low), (
        f"Generate reported failure to the user: {status!r}"
    )
    return status


async def _open_file_via_menu(app, pilot, filepath) -> int:
    """Open a file the way a user does: the File 'Open File...' command, which
    pushes FileSaveModal; type the path and click Open. Returns the tab count
    before opening so callers can assert a new tab appeared.

    This drives the real modal submit path instead of calling _do_open_file.
    """
    from pdb_wizard.panels import OperationsPanel
    from pdb_wizard.viewer import FileSaveModal
    n_before = len(app._tabs)
    app.post_message(OperationsPanel.RunCommand("open_file"))
    await pilot.pause()
    await pilot.pause()
    modal = app.screen
    assert isinstance(modal, FileSaveModal), (
        f"'open_file' did not open the picker; top screen is {type(modal).__name__}"
    )
    modal.query_one("#file-input", Input).value = str(filepath)
    await pilot.pause()
    await pilot.click("#file-save")
    await pilot.pause()
    await pilot.pause()
    return n_before


# ---------------------------------------------------------------------------
# OpenMM NVT — full round trip
# ---------------------------------------------------------------------------

@openmm_required
@pytest.mark.asyncio
async def test_openmm_nvt_ui_full_round_trip(tmp_path):
    """UI flow: switch to OpenMM NVT, tweak step count, Generate, run,
    open the DCD back as a new tab and verify it has the expected frames."""
    mol = _two_water_box()
    app = PdbWizardApp(molecule=mol, filepath="water.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()

        # --- Open via the Analysis menu, then switch engine ---
        screen = await _open_generator_via_menu(app, pilot)
        await _switch_engine(screen, pilot, "openmm_nvt")

        # --- Fill form for a tiny-fast run ---
        _set_form(
            screen,
            forcefield="amber14/tip3pfb.xml",
            temperature_k="300",
            timestep_fs="1.0",
            n_steps=200,
            report_interval=20,
            nonbonded_cutoff_nm="0.5",
        )
        screen.query_one("#gen-outdir", Input).value = str(tmp_path)
        await pilot.pause()
        await _generate(screen, pilot)

        # --- Verify the rendered script + companion PDB are present ---
        run_py = tmp_path / "run.py"
        sys_pdb = tmp_path / "system.pdb"
        assert run_py.exists(), f"run.py not written; contents: {list(tmp_path.iterdir())}"
        assert sys_pdb.exists(), "system.pdb not written"
        text = run_py.read_text()
        assert "{{{" not in text, "rendered run.py still has placeholders"
        # The placeholders we set MUST appear in the rendered file
        assert "200" in text, "n_steps=200 not in rendered script"
        assert "300" in text, "temperature_k=300 not in rendered script"
        assert "amber14/tip3pfb.xml" in text, "forcefield not in rendered script"

        # Close the generator screen so we can drive the app cleanly afterwards
        screen.dismiss(None)
        await pilot.pause()

    # --- Run the real simulator outside the pilot context ---
    proc = subprocess.run(
        [OPENMM_PY, str(run_py)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"OpenMM run failed (rc={proc.returncode}).\n"
        f"STDERR tail:\n{proc.stderr[-2000:]}"
    )
    dcd = tmp_path / "trajectory.dcd"
    assert dcd.exists() and dcd.stat().st_size > 200, (
        "trajectory.dcd missing/empty after sim run"
    )

    # --- Round-trip: open the DCD in a fresh pdb_wizard app ---
    app2 = PdbWizardApp(molecule=mol, filepath="water.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        n_tabs_before = await _open_file_via_menu(app2, pilot, dcd)
        # A new tab should have opened with the DCD trajectory
        assert len(app2._tabs) == n_tabs_before + 1, (
            f"opening DCD didn't create a new tab "
            f"({n_tabs_before} → {len(app2._tabs)})"
        )
        active = app2._tabs[app2._active_tab]
        # 200 steps / 20 = 10 reported frames
        assert active.frames is not None, "DCD tab has no frames list"
        assert len(active.frames) == 10, (
            f"expected 10 DCD frames in the tab, got {len(active.frames)}"
        )
        # Elements survived the round trip via the sibling system.pdb topology
        loaded = [a.element.symbol for a in active.frames[0].atoms]
        assert sorted(loaded) == sorted(["O", "H", "H"] * 2), (
            f"DCD-loaded elements wrong: {loaded}"
        )
        # The trajectory actually evolved. OpenMM's DCDReporter writes
        # wrapped coordinates, so a molecule that crosses the 20 Å box
        # boundary would show a ~20 Å jump; measure under minimum image.
        first = np.array([a.x for a in active.frames[0].atoms])
        last = np.array([a.x for a in active.frames[-1].atoms])
        pbc = active.frames[0].pbc
        if pbc is not None:
            disps = [pbc.min_image(d) for d in (last - first)]
        else:
            disps = list(np.linalg.norm(last - first, axis=1))
        max_disp = float(max(disps))
        assert 0.01 < max_disp < 10.0, (
            f"trajectory frozen / exploded: max disp {max_disp:.4f} Å"
        )


# ---------------------------------------------------------------------------
# CP2K cell_opt — full round trip
# ---------------------------------------------------------------------------

@cp2k_required
@pytest.mark.asyncio
async def test_cp2k_cell_opt_ui_full_round_trip(tmp_path):
    """UI flow → cp2k cell_opt → load *-pos-1.xyz back as a trajectory tab."""
    mol = _single_water()
    app = PdbWizardApp(molecule=mol, filepath="h2o.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _open_generator_via_menu(app, pilot)
        await _switch_engine(screen, pilot, "cp2k_cell_opt")

        # Tiny: 2 opt steps; default cutoffs are fine
        _set_form(
            screen,
            project="ce2e",
            n_opt_steps=2,
        )
        screen.query_one("#gen-outdir", Input).value = str(tmp_path)
        await pilot.pause()
        await _generate(screen, pilot)

        # Files must exist and be fully rendered
        inp = tmp_path / "input.inp"
        xyz = tmp_path / "system.xyz"
        assert inp.exists(), f"input.inp missing; {list(tmp_path.iterdir())}"
        assert xyz.exists(), "system.xyz missing"
        text = inp.read_text()
        assert "{{{" not in text, "rendered input.inp still has placeholders"
        # The user-supplied project name and step count must be present
        assert "PROJECT ce2e" in text
        assert "MAX_ITER 2" in text
        assert "STRESS_TENSOR" in text, "cell_opt without stress tensor will abort"

        screen.dismiss(None)
        await pilot.pause()

    # --- Run CP2K ---
    proc = _run_cp2k(tmp_path, timeout=300)
    assert proc.returncode == 0, (
        f"CP2K cell_opt failed (rc={proc.returncode}).\n"
        f"STDERR tail:\n{proc.stderr[-2000:]}"
    )
    pos = list(tmp_path.glob("*-pos-1.xyz"))
    assert len(pos) == 1, (
        f"expected one *-pos-1.xyz, got {[p.name for p in pos]}"
    )

    # --- Round trip the trajectory back through PdbWizard ---
    app2 = PdbWizardApp(molecule=mol, filepath="h2o.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        n_tabs_before = await _open_file_via_menu(app2, pilot, pos[0])
        assert len(app2._tabs) == n_tabs_before + 1, (
            "opening cell_opt trajectory should create a new tab"
        )
        active = app2._tabs[app2._active_tab]
        assert active.frames is not None and len(active.frames) >= 2, (
            f"expected ≥2 frames in CP2K cell_opt trajectory, "
            f"got {len(active.frames) if active.frames else 'None'}"
        )
        # Each frame is still water
        for f in active.frames:
            els = sorted(a.element.symbol for a in f.atoms)
            assert els == ["H", "H", "O"], f"frame elements wrong: {els}"


# ---------------------------------------------------------------------------
# CP2K DFT-MD — full round trip
# ---------------------------------------------------------------------------

@cp2k_required
@pytest.mark.asyncio
async def test_cp2k_dft_md_ui_full_round_trip(tmp_path):
    """UI flow → cp2k DFT-MD (4 steps) → load trajectory.xyz as a trajectory tab."""
    mol = _single_water()
    app = PdbWizardApp(molecule=mol, filepath="h2o.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _open_generator_via_menu(app, pilot)
        await _switch_engine(screen, pilot, "cp2k_dft_md")

        # 4 steps with traj_interval=1 → ≥2 frames
        _set_form(
            screen,
            project="me2e",
            temperature="300",
            timestep_fs="0.5",
            n_md_steps=4,
            md_traj_interval=1,
        )
        screen.query_one("#gen-outdir", Input).value = str(tmp_path)
        await pilot.pause()
        await _generate(screen, pilot)

        inp = tmp_path / "input.inp"
        assert inp.exists()
        text = inp.read_text()
        assert "{{{" not in text
        assert "PROJECT me2e" in text
        assert "STEPS 4" in text
        assert "TIMESTEP 0.5" in text
        assert "TEMPERATURE 300" in text

        screen.dismiss(None)
        await pilot.pause()

    # --- Run CP2K MD ---
    proc = _run_cp2k(tmp_path, timeout=600)
    assert proc.returncode == 0, (
        f"CP2K DFT-MD failed (rc={proc.returncode}).\n"
        f"STDERR tail:\n{proc.stderr[-2000:]}"
    )
    traj = tmp_path / "trajectory.xyz"
    assert traj.exists(), (
        f"trajectory.xyz missing; files: {[p.name for p in tmp_path.iterdir()]}"
    )

    # --- Open the trajectory back ---
    app2 = PdbWizardApp(molecule=mol, filepath="h2o.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        n_tabs_before = await _open_file_via_menu(app2, pilot, traj)
        assert len(app2._tabs) == n_tabs_before + 1
        active = app2._tabs[app2._active_tab]
        assert active.frames is not None and len(active.frames) >= 2, (
            f"DFT-MD trajectory tab missing frames "
            f"(got {len(active.frames) if active.frames else 'None'})"
        )


async def _open_plot_via_menu(app, pilot, outdir):
    """Drive the real 'Plot Isotherm Results' command -> directory picker ->
    submit, and return the resulting plot screen."""
    from pdb_wizard.panels import OperationsPanel
    from pdb_wizard.viewer import FileSaveModal
    app.post_message(OperationsPanel.RunCommand("isotherm_plot"))
    await pilot.pause()
    await pilot.pause()
    picker = app.screen
    assert isinstance(picker, FileSaveModal), (
        f"'isotherm_plot' did not open the directory picker; "
        f"top screen is {type(picker).__name__}"
    )
    picker.query_one("#file-input", Input).value = str(outdir)
    await pilot.pause()
    await pilot.click("#file-save")
    await pilot.pause()
    await pilot.pause()
    plot = app.screen
    assert type(plot).__name__ == "IsothermPlotScreen", (
        f"submitting the results dir did not open the plot; "
        f"top screen is {type(plot).__name__}"
    )
    return plot


# ---------------------------------------------------------------------------
# MPMC isotherm — real run, full round trip (needs the MPMC binary)
# ---------------------------------------------------------------------------

@mpmc_required
@pytest.mark.asyncio
async def test_mpmc_isotherm_real_run_round_trip(tmp_path):
    """UI -> generate MPMC inputs -> run real MPMC at two pressures ->
    parse the real energy.dat -> load a real trajectory.pqr back as a tab ->
    plot. This is the genuine 'run the program and round-trip' path."""
    mol = _porous_framework()
    pressures = [10.0, 100.0]
    outdir = tmp_path / "iso"

    app = PdbWizardApp(molecule=mol, filepath="framework.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _open_generator_via_menu(app, pilot)
        assert screen._engine_key == "mpmc_isotherm"
        _set_form(
            screen,
            sorbate="h2_bss",
            temperature="298.0",
            pressures=", ".join(str(p) for p in pressures),
            numsteps=2000,
            corrtime=100,
            ensemble="uvt",
            insert_probability="0.667",
            move_factor="0.01",
            rot_factor="0.01",
        )
        screen.query_one("#gen-outdir", Input).value = str(outdir)
        await pilot.pause()
        await _generate(screen, pilot)

        sub_dirs = sorted(
            p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("P_")
        )
        assert len(sub_dirs) == len(pressures)
        for sd in sub_dirs:
            inp = (sd / "mpmc.inp").read_text()
            assert "{{{" not in inp, "mpmc.inp has unfilled placeholders"
            # The sorbate is embedded in input.pqr; a separate insert_input pool
            # triggers a double-free crash in MPMC's cleanup, so it must be absent.
            assert "\ninsert_input " not in inp, (
                "mpmc.inp references insert_input — MPMC double-frees on exit"
            )
        screen.dismiss(None)
        await pilot.pause()

    # --- Run real MPMC at each pressure point ---
    for sd in sub_dirs:
        proc = _run_mpmc(sd, timeout=300)
        assert proc.returncode == 0, (
            f"MPMC failed in {sd.name} (rc={proc.returncode}).\n"
            f"STDOUT tail:\n{proc.stdout[-1500:]}\nSTDERR tail:\n{proc.stderr[-1500:]}"
        )
        assert "simulation exiting successfully" in proc.stdout, (
            f"MPMC did not report a clean exit in {sd.name}; "
            f"tail:\n{proc.stdout[-800:]}"
        )
        assert (sd / "energy.dat").exists(), f"{sd.name}/energy.dat missing"
        assert (sd / "trajectory.pqr").exists(), f"{sd.name}/trajectory.pqr missing"

    # --- Parse the REAL energy.dat via pdb_wizard (the path real output hits) ---
    from pdb_wizard.isotherm import parse_isotherm_results
    results = parse_isotherm_results(str(outdir))
    assert len(results) == len(pressures), f"parsed {results}"
    assert [p for p, _ in results] == sorted(pressures), (
        f"pressures decoded wrong: {[p for p, _ in results]}"
    )
    for p, u in results:
        assert math.isfinite(u) and 0.0 <= u < 1000.0, (
            f"uptake at {p} atm is not physically sane: {u}"
        )
    assert any(u > 0.0 for _, u in results), (
        f"MPMC adsorbed nothing at any pressure — suspicious: {results}"
    )

    # --- Round-trip a real MPMC trajectory.pqr back into pdb_wizard ---
    traj = sub_dirs[-1] / "trajectory.pqr"
    app2 = PdbWizardApp(molecule=mol, filepath="framework.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        n_before = await _open_file_via_menu(app2, pilot, traj)
        assert len(app2._tabs) == n_before + 1, (
            "opening the MPMC trajectory did not create a new tab"
        )
        active = app2._tabs[app2._active_tab]
        assert active.frames is not None and len(active.frames) >= 2, (
            f"MPMC trajectory tab missing frames "
            f"(got {len(active.frames) if active.frames else 'None'})"
        )
        # μVT frames have variable atom counts (sorbates inserted/deleted);
        # the framework carbons persist and adsorbed H2 shows up as hydrogen.
        els = {a.element.symbol for a in active.frames[-1].atoms}
        assert "C" in els, f"framework carbon missing from trajectory frame: {els}"

    # --- And the plot dialog round-trips the real parsed results ---
    app3 = PdbWizardApp(molecule=mol, filepath="framework.pdb")
    async with app3.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        plot = await _open_plot_via_menu(app3, pilot, outdir)
        for got, want in zip(list(plot._p), sorted(pressures)):
            assert math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-4), (
                f"pressure encode/decode broken: {got} vs {want}"
            )
        # The plotted uptakes equal what the parser read from energy.dat
        parsed_u = {p: u for p, u in results}
        for p_val, u_got in zip(list(plot._p), list(plot._u)):
            assert math.isclose(u_got, parsed_u[p_val], rel_tol=1e-3, abs_tol=1e-3), (
                f"plot uptake {u_got} != parsed {parsed_u[p_val]} at p={p_val}"
            )


# ---------------------------------------------------------------------------
# MPMC isotherm — generation + plot round trip (no MPMC binary needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mpmc_isotherm_ui_to_plot_round_trip(tmp_path):
    """UI flow: switch to MPMC, Generate, then synthesize a real-format
    energy.dat per pressure and open the IsothermPlot dialog — verify the
    parsed pressures + uptakes match what we wrote.

    Runs without an MPMC binary, so it stops short of MC sampling (that's
    covered by test_mpmc_isotherm_real_run_round_trip when MPMC is present).
    It synthesizes energy.dat in MPMC's real column layout — the file the
    parser actually reads — not a fabricated histogram.dat."""
    mol = _charged_mof()
    pressures = [0.01, 0.1, 1.0, 10.0]
    expected = {0.01: 0.5, 0.1: 1.5, 1.0: 4.5, 10.0: 9.5}
    outdir = tmp_path / "iso"

    app = PdbWizardApp(molecule=mol, filepath="m.pdb")
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _open_generator_via_menu(app, pilot)
        # The Analysis menu opens the generator on the MPMC engine by default
        assert screen._engine_key == "mpmc_isotherm"

        # Override pressures CSV + outdir + shorter run
        _set_form(
            screen,
            sorbate="h2_bss",
            temperature="77.0",
            pressures=", ".join(str(p) for p in pressures),
            numsteps=10000,
            corrtime=100,
            ensemble="uvt",
            insert_probability="0.667",
            move_factor="0.01",
            rot_factor="0.01",
        )
        screen.query_one("#gen-outdir", Input).value = str(outdir)
        await pilot.pause()
        await _generate(screen, pilot)

        # Each pressure point gets its own dir + mpmc.inp + input.pqr
        sub_dirs = [p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("P_")]
        assert len(sub_dirs) == len(pressures)
        for sd in sub_dirs:
            for fname in ("mpmc.inp", "input.pqr", "insert.pqr", "run.sh"):
                assert (sd / fname).exists(), f"{sd.name}/{fname} missing"
            inp_text = (sd / "mpmc.inp").read_text()
            assert "{{{" not in inp_text, "mpmc.inp has unfilled placeholders"
            # Check the per-pressure value really made it in
            assert "ensemble              uvt" in inp_text
            assert "numsteps              10000" in inp_text

    # --- Synthesize a real-format energy.dat (what finished MPMC runs write) ---
    for sd in outdir.iterdir():
        if not sd.is_dir() or not sd.name.startswith("P_"):
            continue
        # Decode pressure from directory name
        p_str = sd.name.replace("P_", "").replace("_atm", "") \
                       .replace("p", ".").replace("m", "-")
        decoded = float(p_str)
        target = min(pressures, key=lambda p: abs(p - decoded))
        assert math.isclose(target, decoded, rel_tol=1e-6, abs_tol=1e-6)
        _write_synthetic_energy_dat(sd, expected[target])

    # --- Now drive the IsothermPlot dialog and assert it round-trips ---
    app2 = PdbWizardApp(molecule=mol, filepath="m.pdb")
    async with app2.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        plot = await _open_plot_via_menu(app2, pilot, outdir)
        # Pressures round-trip exactly
        for got, want in zip(list(plot._p), sorted(pressures)):
            assert math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-4), (
                f"pressure encode/decode broken: {got} vs {want}"
            )
        # Uptakes match the energy.dat tails we synthesized
        for p_val, u_got in zip(list(plot._p), list(plot._u)):
            want_u = expected[p_val]
            assert math.isclose(u_got, want_u, rel_tol=1e-3, abs_tol=1e-3), (
                f"uptake mismatch at p={p_val}: parsed {u_got}, expected {want_u}"
            )
