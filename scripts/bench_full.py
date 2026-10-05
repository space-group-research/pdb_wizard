"""Full-system performance benchmark — every public op, with cProfile.

Builds a large MOF system (HKUST-1 supercell) and runs every analysis/IO/op
function on it, reporting wall-clock time. Outputs a function-level cProfile
dump for the top hot paths.

Run:  PYTHONPATH=src python scripts/bench_full.py
"""

import cProfile
import io as _io
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pdb_wizard.atom import Atom  # noqa: E402
from pdb_wizard.coordination import compute_coordination  # noqa: E402
from pdb_wizard.density3d import compute_density_3d  # noqa: E402
from pdb_wizard.density_profile import compute_density_profile  # noqa: E402
from pdb_wizard.geometry import (  # noqa: E402
    get_angles_list,
    get_bonds_list,
    get_close_contacts,
)
from pdb_wizard.gyration import compute_gyration_radius, compute_gyration_trajectory  # noqa: E402
from pdb_wizard.hbonds import detect_hbonds  # noqa: E402
from pdb_wizard.io import (  # noqa: E402
    read_file,
    read_pdb_trajectory,
    read_xyz_trajectory,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.molecule import Molecule  # noqa: E402
from pdb_wizard.msd import compute_msd  # noqa: E402
from pdb_wizard.operations import (  # noqa: E402
    extend_axis,
    pore_size_distribution,
    sort_system,
    surface_area,
    wrap_atoms,
)
from pdb_wizard.pxrd import compute_pxrd  # noqa: E402
from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory  # noqa: E402
from pdb_wizard.reduce_cell import reduce_to_primitive  # noqa: E402
from pdb_wizard.renderer import render_scene, rotation_matrix  # noqa: E402
from pdb_wizard.rmsd import compute_rmsd  # noqa: E402

HKUST = ROOT / "HKUST-1.pdb"
WATER_TRAJ = ROOT / "simulations/openmm_md_water/trajectory.xyz"
CO2_TRAJ = ROOT / "simulations/uvt_co2_hkust1/trajectory.pqr"


def banner(name):
    print(f"\n{'=' * 60}\n {name}\n{'=' * 60}")


def time_it(label, fn, *args, **kw):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kw)
        ok = True
    except Exception as e:
        result = e
        ok = False
    elapsed_ms = (time.perf_counter() - t0) * 1000
    status = "ok" if ok else "FAIL"
    print(f"  {label:<48s} {elapsed_ms:>10.1f} ms  [{status}]")
    if not ok:
        print(f"    error: {result}")
    return result, elapsed_ms


def make_supercell(mol, na, nb, nc):
    if na > 1:
        extend_axis(mol, 0, na - 1)
    if nb > 1:
        extend_axis(mol, 1, nb - 1)
    if nc > 1:
        extend_axis(mol, 2, nc - 1)
    return mol


# ---------------------------------------------------------------------------
# Phase 1: Build large systems
# ---------------------------------------------------------------------------

banner("Phase 1: Building large systems")
print("Loading HKUST-1 base unit cell...")
hkust = read_file(str(HKUST))
print(f"  base: {len(hkust.atoms)} atoms, "
      f"PBC {hkust.pbc.a:.2f}x{hkust.pbc.b:.2f}x{hkust.pbc.c:.2f}")

print("Building 2x2x2 supercell (5,000 atoms target)...")
mol_med = read_file(str(HKUST))
t0 = time.perf_counter()
make_supercell(mol_med, 2, 2, 2)
print(f"  med:  {len(mol_med.atoms)} atoms in {(time.perf_counter() - t0)*1000:.0f} ms")

print("Building 3x3x3 supercell (16,000 atoms target)...")
mol_big = read_file(str(HKUST))
t0 = time.perf_counter()
make_supercell(mol_big, 3, 3, 3)
print(f"  big:  {len(mol_big.atoms)} atoms in {(time.perf_counter() - t0)*1000:.0f} ms")

# ---------------------------------------------------------------------------
# Phase 2: Bond detection
# ---------------------------------------------------------------------------

banner("Phase 2: Bond detection")
for label, m in [("HKUST 1x1x1 (624 atoms)", hkust),
                 ("HKUST 2x2x2 (~5K atoms)", mol_med),
                 ("HKUST 3x3x3 (~17K atoms)", mol_big)]:
    m.bonds = []
    time_it(f"detect_bonds: {label}", m.detect_bonds)

# ---------------------------------------------------------------------------
# Phase 3: Renderer
# ---------------------------------------------------------------------------

banner("Phase 3: Renderer (160x80 cells)")
for label, m in [("HKUST 1x1x1", hkust),
                 ("HKUST 2x2x2", mol_med),
                 ("HKUST 3x3x3", mol_big)]:
    rot = rotation_matrix(0.3, 0.5, 0.0)
    cam = max(20.0, m.radius() * 3)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        render_scene(160, 80, m, rot, cam, pbc=m.pbc, ssaa=1)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  render_scene: {label:<25s} median {sorted(times)[1]:>8.1f} ms  "
          f"(min {min(times):.1f}, max {max(times):.1f})")

# Render with bonds disabled (playback mode)
mol_no_bonds = Molecule(atoms=mol_big.atoms, bonds=[], pbc=mol_big.pbc)
times = []
for _ in range(3):
    t0 = time.perf_counter()
    render_scene(160, 80, mol_no_bonds, rotation_matrix(0, 0, 0),
                 mol_big.radius() * 3, pbc=mol_big.pbc, ssaa=1)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  render_scene: HKUST 3x3x3 (no bonds)   median {sorted(times)[1]:>8.1f} ms")

# ---------------------------------------------------------------------------
# Phase 4: Operations
# ---------------------------------------------------------------------------

banner("Phase 4: Operations (on 17K-atom system)")
import copy

m_op = copy.deepcopy(mol_big)
time_it("wrap_atoms (centered)", wrap_atoms, m_op)
m_op2 = copy.deepcopy(mol_big)
time_it("wrap_atoms (forward)", wrap_atoms, m_op2, True)
m_op3 = copy.deepcopy(mol_big)
time_it("sort_system", sort_system, m_op3, False)
time_it("surface_area (2K probes)", surface_area, m_op, 2000)
time_it("pore_size_distribution (1K samples)",
        pore_size_distribution, m_op, 1000, 30, 15.0)
time_it("reduce_to_primitive (HKUST 3x3x3 → 1x1x1)", reduce_to_primitive, m_op, 3, 3, 3)
time_it("find_molecules (17K atoms)", m_op.find_molecules)

# ---------------------------------------------------------------------------
# Phase 5: Geometry tables
# ---------------------------------------------------------------------------

banner("Phase 5: Geometry tables")
# get_angles_list is O(N^3) per submolecule — completely impractical for 17K atoms.
# Use the small HKUST 1x1x1 to demonstrate timing.
time_it("get_bonds_list (HKUST 1x1x1, 624 atoms)", get_bonds_list, hkust)
time_it("get_angles_list (HKUST 1x1x1, 624 atoms)", get_angles_list, hkust)
time_it("get_close_contacts (HKUST 1x1x1, 624 atoms)", get_close_contacts, hkust)
time_it("get_bonds_list (HKUST 2x2x2, 5K atoms)", get_bonds_list, mol_med)
time_it("get_close_contacts (HKUST 2x2x2, 5K atoms)", get_close_contacts, mol_med)

# ---------------------------------------------------------------------------
# Phase 6: Analyses (single frame)
# ---------------------------------------------------------------------------

banner("Phase 6: Analyses on 17K-atom static frame")
time_it("compute_rdf (Cu-Cu, 100 bins, 10 A)",
        compute_rdf, mol_big, "Cu", "Cu", 100, 10.0)
time_it("compute_coordination (Cu-O, 3.5 A)",
        compute_coordination, mol_big, "Cu", "O", 3.5)
time_it("compute_gyration_radius (single frame)",
        compute_gyration_radius, mol_big)
time_it("compute_density_profile (z-axis, 100 bins)",
        compute_density_profile, mol_big, 2, 100, "C")
time_it("compute_density_3d ([60,60,60] grid, single frame)",
        compute_density_3d, [mol_big], "C", 60)
time_it("compute_pxrd (HKUST 1x1x1, Cu Ka, 50 max 2theta)",
        compute_pxrd, hkust, 1.5406, 5.0, 50.0)
time_it("compute_pxrd (HKUST 2x2x2, Cu Ka, 50 max 2theta)",
        compute_pxrd, mol_med, 1.5406, 5.0, 50.0)
time_it("detect_hbonds (HKUST 1x1x1)", detect_hbonds, hkust, 3.5, 120.0)
time_it("detect_hbonds (HKUST 3x3x3, 17K atoms)",
        detect_hbonds, mol_big, 3.5, 120.0)

# ---------------------------------------------------------------------------
# Phase 7: Trajectory I/O & multi-frame analyses
# ---------------------------------------------------------------------------

banner("Phase 7: Trajectory I/O")
frames_xyz = None
if WATER_TRAJ.exists():
    with open(WATER_TRAJ) as f:
        text = f.read()
    Atom._element_cache.clear()
    t0 = time.perf_counter()
    frames_xyz, _ = read_xyz_trajectory(_io.StringIO(text))
    t = (time.perf_counter() - t0) * 1000
    print(f"  read_xyz_trajectory: {len(frames_xyz)} frames of "
          f"{len(frames_xyz[0].atoms)} atoms in {t:.0f} ms")
else:
    print("  water trajectory not found")

frames_pqr = None
if CO2_TRAJ.exists():
    with open(CO2_TRAJ) as f:
        text = f.read()
    t0 = time.perf_counter()
    frames_pqr, _ = read_pdb_trajectory(_io.StringIO(text))
    t = (time.perf_counter() - t0) * 1000
    print(f"  read_pdb_trajectory: {len(frames_pqr)} frames of "
          f"{len(frames_pqr[0].atoms)} atoms in {t:.0f} ms")
else:
    print("  CO2 trajectory not found")

if frames_xyz:
    banner("Phase 7b: Multi-frame analyses (water trajectory, 50 frames)")
    sub = frames_xyz[:50]
    print(f"  using {len(sub)} frames of {len(sub[0].atoms)} atoms")
    time_it("compute_msd (O atoms, 50 frames, dt=1ps)",
            compute_msd, sub, "O", None, 1.0)
    time_it("compute_rdf_trajectory (O-O, 50 frames)",
            compute_rdf_trajectory, sub, "O", "O", None, None, 100, 8.0)
    time_it("compute_rmsd (50 frames, all atoms)",
            compute_rmsd, sub, None, None)
    time_it("compute_gyration_trajectory (50 frames)",
            compute_gyration_trajectory, sub)

# ---------------------------------------------------------------------------
# Phase 8: Writers (round-trip on 17K atoms)
# ---------------------------------------------------------------------------

banner("Phase 8: Writers (17K atoms)")
import tempfile

tmp = tempfile.mkdtemp()
time_it("write_xyz", write_xyz, mol_big, os.path.join(tmp, "big.xyz"))
time_it("write_standard_pdb", write_standard_pdb, mol_big, os.path.join(tmp, "big.pdb"))
time_it("write_mpmc_pdb", write_mpmc_pdb, mol_big, os.path.join(tmp, "big_mpmc.pdb"))

# ---------------------------------------------------------------------------
# Phase 9: cProfile of the slowest paths
# ---------------------------------------------------------------------------

banner("Phase 9: cProfile of detect_bonds + render_scene + rdf on 17K atoms")

profiler = cProfile.Profile()
profiler.enable()
mol_big.bonds = []
mol_big.detect_bonds()
render_scene(160, 80, mol_big, rotation_matrix(0.3, 0.5, 0.0),
             mol_big.radius() * 3, pbc=mol_big.pbc, ssaa=1)
compute_rdf(mol_big, "Cu", "Cu", 100, 10.0)
profiler.disable()

s = pstats.Stats(profiler).sort_stats("cumulative")
print("\nTop 25 cumulative functions:")
s.print_stats(25)

print("\nTop 20 own-time functions:")
s.sort_stats("tottime").print_stats(20)

print("\nDONE")
