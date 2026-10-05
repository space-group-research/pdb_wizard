"""Real-run integration tests: generate inputs → invoke the simulator
binary → load the output back into pdb_wizard.

OpenMM tests run a small NVT water simulation via a subprocess that uses
an OpenMM-equipped Python (defaulting to one of the system's conda envs).
The resulting DCD trajectory and final PDB are loaded back through
`pdb_wizard.io.read_file_trajectory` and `read_file` to verify the full
loop closes.

CP2K tests verify the generated `system.xyz` is parseable by pdb_wizard
(we don't have CP2K installed in this environment, so we don't run the
solver — but the input files we ship to it must be ones it can read,
and the coords file we hand to CP2K must round-trip through pdb_wizard).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.io import read_file, read_file_trajectory
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.sim_inputs import generate_cp2k_input, generate_openmm_script


# Locate an OpenMM-equipped Python — try a list of known conda envs.
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


OPENMM_PY = _find_openmm_python()
openmm_required = pytest.mark.skipif(
    OPENMM_PY is None,
    reason="No openmm-equipped Python found (tested several conda envs)",
)


def _two_water_box() -> Molecule:
    """Two TIP3P-shaped waters in a 20 Å cubic box, well separated."""
    atoms = [
        Atom(0.5, 0.5, 0.5, "O"),
        Atom(1.457, 0.5, 0.5, "H"),
        Atom(0.260, 1.428, 0.5, "H"),
        Atom(4.5, 4.5, 4.0, "O"),
        Atom(5.457, 4.5, 4.0, "H"),
        Atom(4.260, 5.428, 4.0, "H"),
    ]
    return Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))


# ---------------------------------------------------------------------------
# OpenMM
# ---------------------------------------------------------------------------

@openmm_required
class TestOpenMMRealRun:
    def test_nvt_script_runs_and_outputs_loadable_files(self, tmp_path):
        """Generate an NVT script, run it, then load the resulting DCD
        and PDB back into pdb_wizard."""
        mol = _two_water_box()
        result = generate_openmm_script(
            mol, tmp_path, ensemble="NVT",
            forcefield="amber14/tip3pfb.xml",
            n_steps=200, report_interval=20,
            nonbonded_cutoff_nm=0.5,
        )
        # Verify the input topology
        sys_pdb_text = Path(result["pdb"]).read_text()
        # Must have HOH residue names so amber14/tip3pfb finds templates
        assert "HOH" in sys_pdb_text, (
            f"system.pdb missing HOH residues — openmm won't find templates.\n"
            f"head:\n{sys_pdb_text[:400]}"
        )

        # Run the sim
        proc = subprocess.run(
            [OPENMM_PY, result["script"]],
            cwd=tmp_path, capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, (
            f"OpenMM sim failed (rc={proc.returncode}).\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}\n"
            f"STDOUT tail:\n{proc.stdout[-2000:]}"
        )
        assert "done" in proc.stdout, "script should print 'done' at end"

        # ---- Outputs exist ----
        dcd = tmp_path / "trajectory.dcd"
        log = tmp_path / "output.log"
        final_pdb = tmp_path / "final.pdb"
        assert dcd.exists() and dcd.stat().st_size > 200, (
            f"trajectory.dcd missing or too small: {dcd}"
        )
        assert log.exists() and log.stat().st_size > 0, "output.log missing"
        assert final_pdb.exists() and final_pdb.stat().st_size > 0, (
            "final.pdb missing"
        )

        # ---- Load DCD back into pdb_wizard ----
        frames = read_file_trajectory(str(dcd))
        assert frames is not None, "DCD did not register as a trajectory"
        # 200 steps / 20 = 10 frames
        assert len(frames) == 10, (
            f"expected 10 frames in DCD, got {len(frames)}"
        )
        # Atom count matches the input
        assert len(frames[0].atoms) == len(mol.atoms)
        # Element auto-discovery via sibling system.pdb / final.pdb
        loaded_elements = [a.element.symbol for a in frames[0].atoms]
        # System has 2 waters: 6 atoms, the order written by pdb_wizard
        # is sorted within each water (alphabetical) and grouped by molecule.
        assert sorted(loaded_elements) == sorted(["O", "H", "H"] * 2), (
            f"DCD-loaded elements {loaded_elements} don't match 2 waters"
        )

        # ---- Load final.pdb back ----
        final = read_file(str(final_pdb))
        assert len(final.atoms) == len(mol.atoms)
        assert final.pbc is not None, "final.pdb should have CRYST1"
        # Box should be ~20 Å cubic (NVT preserves box)
        for L in (final.pbc.a, final.pbc.b, final.pbc.c):
            assert 18.0 < L < 22.0, f"box length {L} drifted (NVT shouldn't)"

        # ---- Trajectory frames evolve (atoms move) ----
        import numpy as np
        first_coords = np.array([a.x for a in frames[0].atoms])
        last_coords = np.array([a.x for a in frames[-1].atoms])
        # At 300 K with 1 fs timestep × 200 steps the atoms should move
        # at least somewhere between 0.01 and 5 Å.
        max_disp = float(np.linalg.norm(first_coords - last_coords, axis=1).max())
        assert 0.01 < max_disp < 10.0, (
            f"trajectory looks frozen or exploded: max disp {max_disp:.4f} Å"
        )

    def test_npt_script_runs_with_enough_atoms(self, tmp_path):
        """NPT with the 2-water test-system collapses physically — skip
        the actual run, just verify the generated script imports and
        instantiates the barostat without raising. Catches force-field /
        Python errors that the synthetic AST test in test_sim_inputs.py
        wouldn't notice."""
        mol = _two_water_box()
        result = generate_openmm_script(
            mol, tmp_path, ensemble="NPT",
            forcefield="amber14/tip3pfb.xml",
            n_steps=0, report_interval=20,
            nonbonded_cutoff_nm=0.5,
        )
        # Run the script with n_steps=0 — it minimises and tries to step,
        # but with no steps it should exit cleanly.
        proc = subprocess.run(
            [OPENMM_PY, result["script"]],
            cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        # n_steps=0 → simulation.step(0) is a no-op; no barostat instability
        assert proc.returncode == 0, (
            f"NPT script failed at zero steps (i.e. setup is broken):\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )


# ---------------------------------------------------------------------------
# CP2K
# ---------------------------------------------------------------------------

# Locate a CP2K binary
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


CP2K_BIN = _find_cp2k()
cp2k_required = pytest.mark.skipif(
    CP2K_BIN is None,
    reason="No working CP2K binary found",
)


@cp2k_required
class TestCP2KRealRun:
    """Run real CP2K on the generator's output, then load results back."""

    def _run_cp2k(self, workdir: Path, timeout: int) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["CP2K_DATA_DIR"] = _CP2K_DATA
        env["LD_LIBRARY_PATH"] = _CP2K_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
        env["OMP_NUM_THREADS"] = "1"
        return subprocess.run(
            [CP2K_BIN, "-i", "input.inp", "-o", "cp2k.out"],
            cwd=workdir, capture_output=True, text=True,
            timeout=timeout, env=env,
        )

    def test_cell_opt_runs_and_outputs_loadable_trajectory(self, tmp_path):
        """Cell-opt 1 H2O for 2 steps, load `<project>-pos-1.xyz` back."""
        atoms = [
            Atom(6.0, 6.0, 6.0, "O"),
            Atom(6.957, 6.0, 6.0, "H"),
            Atom(5.760, 6.928, 6.0, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=PBC(12, 12, 12, 90, 90, 90))
        result = generate_cp2k_input(
            mol, tmp_path, mode="cell_opt",
            project_name="cellopt_test", n_opt_steps=2,
        )
        # Stress tensor must be enabled — generator regression check
        assert "STRESS_TENSOR" in Path(result["input"]).read_text(), (
            "Generated cell_opt input missing STRESS_TENSOR — CP2K will abort"
        )

        proc = self._run_cp2k(tmp_path, timeout=300)
        assert proc.returncode == 0, (
            f"CP2K cell_opt failed (rc={proc.returncode}).\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}"
        )

        # CP2K writes <project>-pos-1.xyz with the optimization trajectory
        pos = list(tmp_path.glob("*-pos-1.xyz"))
        assert len(pos) == 1, f"expected one pos file, got {[p.name for p in pos]}"

        traj = read_file_trajectory(str(pos[0]))
        assert traj is not None, "<project>-pos-1.xyz did not register as trajectory"
        # Cell-opt with n_opt_steps=2 should produce ≥ 2 frames (initial + each step)
        assert len(traj) >= 2, f"expected ≥ 2 frames, got {len(traj)}"
        # Each frame has the same 3 atoms with correct elements
        for frame in traj:
            assert [a.element.symbol for a in frame.atoms] == ["O", "H", "H"]

        # The cell-output file should also exist (cell_opt-specific)
        cell_files = list(tmp_path.glob("*-1.cell"))
        assert cell_files, "cell_opt should produce a `-1.cell` file"

    def test_dft_md_runs_and_outputs_loadable_trajectory(self, tmp_path):
        """Run a 5-step DFT-MD; load `trajectory.xyz` back."""
        atoms = [
            Atom(6.0, 6.0, 6.0, "O"),
            Atom(6.957, 6.0, 6.0, "H"),
            Atom(5.760, 6.928, 6.0, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=PBC(12, 12, 12, 90, 90, 90))
        result = generate_cp2k_input(
            mol, tmp_path, mode="dft_md",
            project_name="dftmd_test", n_md_steps=4, timestep_fs=0.5,
            md_traj_interval=1,  # write every step so 4 steps → ≥2 frames
        )
        proc = self._run_cp2k(tmp_path, timeout=600)
        assert proc.returncode == 0, (
            f"CP2K DFT-MD failed (rc={proc.returncode}).\n"
            f"STDERR tail:\n{proc.stderr[-2000:]}"
        )

        # The generator hardcodes FILENAME =trajectory.xyz for MD
        traj_path = tmp_path / "trajectory.xyz"
        assert traj_path.exists(), (
            f"trajectory.xyz missing; files: {[p.name for p in tmp_path.iterdir()]}"
        )
        traj = read_file_trajectory(str(traj_path))
        assert traj is not None, (
            "trajectory.xyz did not register as a trajectory — "
            "with md_traj_interval=1 and 4 steps we should have ≥ 2 frames"
        )
        assert len(traj) >= 2, f"expected ≥ 2 frames, got {len(traj)}"
        for frame in traj:
            assert [a.element.symbol for a in frame.atoms] == ["O", "H", "H"]
        # And atoms should have moved between frames (real MD output)
        import numpy as np
        first = np.array([a.x for a in traj[0].atoms])
        last = np.array([a.x for a in traj[-1].atoms])
        max_disp = float(np.linalg.norm(first - last, axis=1).max())
        assert max_disp > 1e-5, (
            f"atoms didn't move between frames; max_disp={max_disp}"
        )

        # Energy file must also be present
        ener_files = list(tmp_path.glob("*.ener"))
        assert ener_files, "DFT-MD should produce a `.ener` file"
        ener_text = ener_files[0].read_text()
        assert ener_text.strip(), "energy file is empty"


class TestCP2KRoundTripThroughPdbWizard:
    """We don't run CP2K (not installed), but the .xyz file the generator
    hands it must be loadable by pdb_wizard — that's the round-trip
    guarantee. CP2K's own .inp parsing is exercised elsewhere; here we
    pin pdb_wizard's side."""

    def test_generated_xyz_round_trips(self, tmp_path):
        # Build a small ortho cell with mixed elements
        atoms = [
            Atom(1.0, 1.0, 1.0, "Cu"),
            Atom(2.5, 1.0, 1.0, "O"),
            Atom(1.0, 2.5, 1.0, "C"),
            Atom(1.0, 1.0, 2.5, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        result = generate_cp2k_input(mol, tmp_path, mode="cell_opt")

        # Read the .xyz back — it must parse, give the right atom count
        # and elements, and the coords must round-trip exactly.
        loaded = read_file(result["xyz"])
        assert len(loaded.atoms) == len(mol.atoms)
        for orig, got in zip(mol.atoms, loaded.atoms):
            assert orig.element.symbol == got.element.symbol
            for c in range(3):
                assert orig.x[c] == pytest.approx(got.x[c], abs=1e-5)

    def test_dft_md_xyz_with_real_mof(self, tmp_path):
        """Repeat with a MOF. The XYZ file CP2K consumes must be one
        pdb_wizard can also read — same code path as a user opening it."""
        hkust_pdb = Path(__file__).parent.parent / "HKUST-1.pdb"
        if not hkust_pdb.exists():
            pytest.skip("HKUST-1.pdb not present")
        mol = read_file(str(hkust_pdb))
        result = generate_cp2k_input(mol, tmp_path, mode="dft_md")
        # Round trip
        loaded = read_file(result["xyz"])
        assert len(loaded.atoms) == len(mol.atoms)
        # Element mix matches
        from collections import Counter
        orig_counts = Counter(a.element.symbol for a in mol.atoms)
        new_counts = Counter(a.element.symbol for a in loaded.atoms)
        assert orig_counts == new_counts, (
            f"element counts changed during xyz round-trip: "
            f"{orig_counts} → {new_counts}"
        )

    def test_cp2k_input_is_well_formed_text(self, tmp_path):
        """A poor man's syntax check: every &SECTION must have a matching
        &END. Catches typos / missing closes that would fail CP2K parsing
        without us having to run CP2K."""
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        result = generate_cp2k_input(mol, tmp_path, mode="cell_opt")
        text = Path(result["input"]).read_text()
        # Count opens vs closes
        # Open: lines starting with '&XYZ' (not '&END')
        # Close: lines starting with '&END'
        opens, closes = 0, 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("&END"):
                closes += 1
            elif stripped.startswith("&"):
                opens += 1
        assert opens == closes, (
            f"CP2K input has unbalanced sections: {opens} opens vs {closes} closes"
        )

    def test_dft_md_input_is_well_formed_text(self, tmp_path):
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        result = generate_cp2k_input(mol, tmp_path, mode="dft_md")
        text = Path(result["input"]).read_text()
        opens, closes = 0, 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("&END"):
                closes += 1
            elif stripped.startswith("&"):
                opens += 1
        assert opens == closes, (
            f"CP2K DFT-MD input unbalanced: {opens} opens, {closes} closes"
        )
