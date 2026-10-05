"""Performance regression tests.

Render budget: trajectory playback uses _play_interval = 0.15s, so a single
frame render must take < 0.5 × 150ms = 75ms median. Bond detection should
not exceed ~150ms even for large systems.

Run with `-v -s` to see the actual times.
"""

import os
import time

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.renderer import render_scene, rotation_matrix

# Budgets in ms — fail if median exceeds these.
RENDER_BUDGET_MS = 75
BOND_DETECT_BUDGET_MS_PER_1K_ATOMS = 50  # 50ms per 1000 atoms


def _build_random_molecule(n_atoms: int, box_size: float = 30.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    coords = rng.random((n_atoms, 3)) * box_size
    elements = ["C", "O", "N", "H"]
    atoms = []
    for i in range(n_atoms):
        el = elements[i % len(elements)]
        atoms.append(Atom(float(coords[i, 0]), float(coords[i, 1]),
                          float(coords[i, 2]), el))
    pbc = PBC(box_size, box_size, box_size, 90, 90, 90)
    mol = Molecule(atoms=atoms, pbc=pbc)
    mol.detect_bonds()
    return mol


def _median_render_time_ms(mol, w=160, h=80, n_runs=5) -> float:
    rot = rotation_matrix(0, 0, 0)
    cam = max(20.0, mol.radius() * 3)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        render_scene(w, h, mol, rot, cam, pbc=mol.pbc, ssaa=1)
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[n_runs // 2]


# ---------------------------------------------------------------------------
# Render budget
# ---------------------------------------------------------------------------

class TestRenderBudget:
    @pytest.mark.asyncio
    async def test_render_500_atoms(self):
        mol = _build_random_molecule(500)
        t = _median_render_time_ms(mol)
        print(f"\n  500 atoms render: {t:.1f} ms")
        assert t < RENDER_BUDGET_MS, (
            f"500-atom render took {t:.1f} ms (budget {RENDER_BUDGET_MS} ms)"
        )

    @pytest.mark.asyncio
    async def test_render_1500_atoms(self):
        mol = _build_random_molecule(1500)
        t = _median_render_time_ms(mol)
        print(f"\n  1500 atoms render: {t:.1f} ms")
        # Allow 2x budget at 1500 atoms (still acceptable for non-playback)
        assert t < RENDER_BUDGET_MS * 2, (
            f"1500-atom render took {t:.1f} ms (budget {RENDER_BUDGET_MS * 2} ms)"
        )

    @pytest.mark.asyncio
    async def test_render_with_bonds_disabled_is_fast(self):
        """Rendering large systems with show_bonds=False should be fast."""
        mol = _build_random_molecule(3000)
        # Strip bonds to simulate playback-mode "no bonds" rendering
        mol_no_bonds = Molecule(atoms=mol.atoms, bonds=[], pbc=mol.pbc)
        t = _median_render_time_ms(mol_no_bonds)
        print(f"\n  3000 atoms render (no bonds): {t:.1f} ms")
        assert t < RENDER_BUDGET_MS * 3, (
            f"3000-atom no-bond render took {t:.1f} ms"
        )

    def test_render_hkust1_within_budget(self):
        """HKUST-1 is the canonical test case — must stay under budget."""
        from pdb_wizard.io import read_file
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        if not os.path.exists(path):
            pytest.skip("HKUST-1.pdb not found")
        mol = read_file(path)
        t = _median_render_time_ms(mol)
        print(f"\n  HKUST-1 (624 atoms) render: {t:.1f} ms")
        assert t < RENDER_BUDGET_MS, (
            f"HKUST-1 render took {t:.1f} ms (budget {RENDER_BUDGET_MS} ms)"
        )


# ---------------------------------------------------------------------------
# Bond detection budget
# ---------------------------------------------------------------------------

class TestBondDetectBudget:
    def test_bond_detect_1000_atoms(self):
        mol = _build_random_molecule(1000)
        mol.bonds = []
        t0 = time.perf_counter()
        mol.detect_bonds()
        t = (time.perf_counter() - t0) * 1000
        print(f"\n  1000 atoms bond detect: {t:.1f} ms")
        assert t < BOND_DETECT_BUDGET_MS_PER_1K_ATOMS * 2, (
            f"1000-atom bond detect took {t:.1f} ms"
        )

    def test_bond_detect_3000_atoms(self):
        mol = _build_random_molecule(3000)
        mol.bonds = []
        t0 = time.perf_counter()
        mol.detect_bonds()
        t = (time.perf_counter() - t0) * 1000
        print(f"\n  3000 atoms bond detect: {t:.1f} ms")
        # Spatial hash should keep this under 200ms even for random distributions
        assert t < 300, f"3000-atom bond detect took {t:.1f} ms (limit 300 ms)"

    def test_hkust1_bond_detect(self):
        from pdb_wizard.io import read_file
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        if not os.path.exists(path):
            pytest.skip("HKUST-1.pdb not found")
        mol = read_file(path)
        mol.bonds = []
        t0 = time.perf_counter()
        mol.detect_bonds()
        t = (time.perf_counter() - t0) * 1000
        print(f"\n  HKUST-1 bond detect: {t:.1f} ms")
        assert t < 50, f"HKUST-1 bond detect took {t:.1f} ms (limit 50 ms)"


# ---------------------------------------------------------------------------
# Trajectory load + frame switch
# ---------------------------------------------------------------------------

class TestTrajectoryLoadBudget:
    def test_xyz_500_frames_load(self):
        import io as _io

        from pdb_wizard.io import read_xyz_trajectory
        path = os.path.join(
            os.path.dirname(__file__), "..", "simulations",
            "openmm_md_water", "trajectory.xyz",
        )
        if not os.path.exists(path):
            pytest.skip("Water trajectory not found")
        with open(path) as f:
            content = f.read()
        # Element cache cold — clear to measure cold-start
        from pdb_wizard.atom import Atom
        Atom._element_cache.clear()
        t0 = time.perf_counter()
        buf = _io.StringIO(content)
        frames, _ = read_xyz_trajectory(buf)
        t = (time.perf_counter() - t0) * 1000
        print(f"\n  XYZ load {len(frames)} frames: {t:.0f} ms")
        # Budget: 500 frames of 774 atoms in < 2 seconds
        assert t < 2000, f"XYZ trajectory load took {t:.0f} ms (limit 2000)"

    def test_pdb_trajectory_load(self):
        import io as _io

        from pdb_wizard.io import read_pdb_trajectory
        path = os.path.join(
            os.path.dirname(__file__), "..", "simulations",
            "uvt_co2_hkust1", "trajectory.pqr",
        )
        if not os.path.exists(path):
            pytest.skip("MPMC trajectory not found")
        with open(path) as f:
            content = f.read()
        t0 = time.perf_counter()
        buf = _io.StringIO(content)
        frames, _ = read_pdb_trajectory(buf)
        t = (time.perf_counter() - t0) * 1000
        print(f"\n  PDB load {len(frames)} frames: {t:.0f} ms")
        # PDB lines are slower to parse than XYZ — wider budget
        assert t < 3500, f"PDB trajectory load took {t:.0f} ms (limit 3500)"
