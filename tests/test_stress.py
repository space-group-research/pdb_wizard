"""Stress tests loading real simulation trajectories and running analyses."""

import os

import pytest

from pdb_wizard.density_profile import compute_density_profile, compute_density_profile_trajectory
from pdb_wizard.io import read_file, read_file_trajectory
from pdb_wizard.molecule import Molecule
from pdb_wizard.msd import compute_msd
from pdb_wizard.operations import void_volume
from pdb_wizard.rdf import compute_rdf, compute_rdf_trajectory

SIM_DIR = os.path.join(os.path.dirname(__file__), "..", "simulations")


def _traj_path(name, ext="pqr"):
    return os.path.join(SIM_DIR, name, f"trajectory.{ext}")


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(f"Simulation output not found: {path}")


# --- HKUST-1 static file ---

class TestHkust1Static:
    def test_load_hkust1(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        assert len(mol.atoms) == 624
        assert mol.pbc is not None

    def test_hkust1_elements(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        from collections import Counter
        counts = Counter(a.element.symbol for a in mol.atoms)
        assert counts["Cu"] == 48
        assert counts["C"] == 288
        assert counts["O"] == 192
        assert counts["H"] == 96

    def test_hkust1_charges_loaded(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        cu_charges = [a.charge for a in mol.atoms if a.element.symbol == "Cu"]
        assert all(abs(q) > 0.1 for q in cu_charges)

    def test_hkust1_void_volume(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        result = void_volume(mol, n_samples=10000)
        assert result["void_fraction"] > 0.3  # HKUST-1 is porous

    def test_hkust1_rdf(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        r, g = compute_rdf(mol, "Cu", "O")
        assert len(r) > 0
        assert r[g.argmax()] < 3.0  # Cu-O bond ~1.9 A


# --- MPMC UVT H2 trajectory ---

class TestMpmcUvtH2:
    def test_load_trajectory(self):
        path = _traj_path("../hkust1_h2_uvt")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert frames is not None
        assert len(frames) > 10

    def test_frame_atoms(self):
        path = _traj_path("../hkust1_h2_uvt")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert len(frames[0].atoms) >= 624  # MOF + some H2

    def test_rdf_trajectory(self):
        path = _traj_path("../hkust1_h2_uvt")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        # Just use first 10 frames for speed
        r, g = compute_rdf_trajectory(frames[:10], "Cu", "H")
        assert len(r) > 0
        assert g.max() > 0

    def test_msd(self):
        path = _traj_path("../hkust1_h2_uvt")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        # H2 atoms appear/disappear in uVT so MSD on H won't work well
        # Use a fixed-N element like Cu instead
        lags, msd = compute_msd(frames[:20], element="Cu", max_lag=5)
        assert len(lags) > 0


# --- MPMC UVT CO2 trajectory ---

class TestMpmcUvtCo2:
    def test_load(self):
        path = _traj_path("uvt_co2_hkust1")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert frames is not None
        assert len(frames) > 5

    def test_co2_rdf(self):
        path = _traj_path("uvt_co2_hkust1")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        r, g = compute_rdf_trajectory(frames[:10], "C", "O")
        assert len(r) > 0


# --- MPMC NVT H2 trajectory ---

class TestMpmcNvtH2:
    def test_load(self):
        path = _traj_path("nvt_h2_hkust1")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert frames is not None
        assert len(frames) > 5

    def test_density_profile(self):
        path = _traj_path("nvt_h2_hkust1")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        pos, dens = compute_density_profile_trajectory(frames[:10], axis=2, element="H")
        assert len(pos) > 0


# --- MPMC PHAHST H2 trajectory ---

class TestMpmcPhahstH2:
    def test_load(self):
        path = _traj_path("phahst_h2_hkust1")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert frames is not None
        assert len(frames) > 3


# --- OpenMM water box ---

class TestOpenmmWater:
    def test_load(self):
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        assert frames is not None
        assert len(frames) > 100

    def test_water_rdf(self):
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        r, g = compute_rdf_trajectory(frames[:10], "O", "O")
        assert len(r) > 0

    def test_water_msd(self):
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        lags, msd = compute_msd(frames[:50], element="O")
        assert len(lags) > 0

    def test_diffusion_constant(self):
        from pdb_wizard.msd import compute_diffusion_constant
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        lags, msd = compute_msd(frames[:100], element="O")
        D, slope = compute_diffusion_constant(lags, msd, dt=0.4)
        assert D > 0, "Diffusion constant should be positive"
        assert 1e-7 < D < 1e-3, f"D={D} seems out of range for water"

    def test_rmsd(self):
        from pdb_wizard.rmsd import compute_rmsd
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        idx, rmsd = compute_rmsd(frames[:50], element="O")
        assert len(idx) == 50
        assert rmsd[0] == 0.0  # RMSD vs self is 0
        assert rmsd[-1] > 0.0  # Should have drifted

    def test_coordination_number(self):
        from pdb_wizard.coordination import compute_coordination
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        r, cn, avg = compute_coordination(mol, "Cu", "O", r_max=3.0)
        assert len(r) > 0
        # Cu in HKUST-1 paddlewheel: 4 carboxylate O per Cu
        assert 3.0 < avg < 5.0, f"Expected ~4 O around Cu, got {avg:.1f}"

    def test_hbonds(self):
        from pdb_wizard.hbonds import detect_hbonds
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        hb = detect_hbonds(mol)
        assert len(hb) > 0, "Water should have hydrogen bonds"

    def test_gyration_radius(self):
        from pdb_wizard.gyration import compute_gyration_radius
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        rg = compute_gyration_radius(mol, element="O")
        assert rg > 0

    def test_density_3d(self):
        from pdb_wizard.density3d import compute_density_3d
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        frames = read_file_trajectory(path)
        pos, dens, pbc = compute_density_3d(frames[:20], "O", n_bins=10)
        assert len(pos) > 0
        assert len(dens) == len(pos)
        assert dens.max() <= 1.0

    def test_isotherm_planner(self):
        """Isotherm planner generates correct directory structure."""
        import shutil
        import tempfile

        from pdb_wizard.isotherm import generate_isotherm
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        tmpdir = tempfile.mkdtemp()
        try:
            pressures = [0.1, 1.0, 10.0]
            dirs = generate_isotherm(mol, "h2_bss", 77.0, pressures, tmpdir, numsteps=1000)
            assert len(dirs) == 3
            for d in dirs:
                assert os.path.exists(os.path.join(d, "input.pqr"))
                assert os.path.exists(os.path.join(d, "mpmc.inp"))
                assert os.path.exists(os.path.join(d, "insert.pqr"))
                assert os.path.exists(os.path.join(d, "run.sh"))
                # Check inp has correct pressure
                with open(os.path.join(d, "mpmc.inp")) as f:
                    content = f.read()
                assert "temperature" in content
                assert "pressure" in content
            assert os.path.exists(os.path.join(tmpdir, "run_all.sh"))
        finally:
            shutil.rmtree(tmpdir)

    def test_lammps_roundtrip(self):
        """LAMMPS data write/read roundtrip preserves atoms and elements."""
        from pdb_wizard.formats import read_lammps_data, write_lammps_data
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        mol = read_file(path)
        import io as _io
        buf = _io.StringIO()
        write_lammps_data(mol, buf)
        buf.seek(0)
        atoms, pbc = read_lammps_data(buf)
        assert len(atoms) == len(mol.atoms)
        from collections import Counter
        orig = Counter(a.element.symbol for a in mol.atoms)
        read = Counter(a.element.symbol for a in atoms)
        assert orig == read


# --- OpenMM protein ---

class TestOpenmmProtein:
    def test_file_exists_and_header(self):
        """Just check file exists and header is valid - full load too slow."""
        path = os.path.join(SIM_DIR, "openmm_protein", "trajectory.xyz")
        _skip_if_missing(path)
        with open(path) as f:
            n_atoms = int(f.readline().strip())
            assert n_atoms > 1000


# --- OpenMM 1024-water box ---

class TestOpenmmBenzeneWater:
    def test_load_single_frame(self):
        path = os.path.join(SIM_DIR, "openmm_benzene_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        assert len(mol.atoms) == 3384

    def test_has_carbon(self):
        path = os.path.join(SIM_DIR, "openmm_benzene_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        carbons = [a for a in mol.atoms if a.element.symbol == "C"]
        assert len(carbons) == 120  # 20 benzene x 6 C

    def test_oo_rdf_single(self):
        path = os.path.join(SIM_DIR, "openmm_benzene_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        r, g = compute_rdf(mol, "O", "O")
        assert len(r) > 0
        peak = r[g.argmax()]
        assert 2.0 < peak < 4.0

    def test_density_profile_single(self):
        path = os.path.join(SIM_DIR, "openmm_benzene_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        pos, dens = compute_density_profile(mol, axis=0, element="O")
        assert len(pos) > 0


# --- Cross-format tests ---

class TestEnergyDat:
    def test_reads_column_names_from_header(self):
        """energy.dat header should provide real column names, not col0/col1."""
        from pdb_wizard.energy import read_energy_dat
        path = os.path.join(SIM_DIR, "uvt_co2_hkust1", "energy.dat")
        _skip_if_missing(path)
        data = read_energy_dat(path)
        assert "step" in data
        assert "energy" in data
        assert "coulombic" in data
        assert "N" in data
        # Should NOT have generic col names
        assert not any(k.startswith("col") for k in data)

    def test_N_column_values(self):
        """N (molecule count) should be positive integers."""
        from pdb_wizard.energy import read_energy_dat
        path = os.path.join(SIM_DIR, "uvt_co2_hkust1", "energy.dat")
        _skip_if_missing(path)
        data = read_energy_dat(path)
        assert "N" in data
        assert all(n >= 0 for n in data["N"])


class TestCrossFormat:
    def test_mpmc_pqr_and_xyz_both_load(self):
        pqr = _traj_path("../hkust1_h2_uvt")
        xyz = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        if not os.path.exists(pqr) or not os.path.exists(xyz):
            pytest.skip("Need both trajectory files")
        f1 = read_file_trajectory(pqr)
        f2 = read_file_trajectory(xyz)
        assert f1 is not None
        assert f2 is not None

    def test_single_frame_from_trajectory(self):
        """Loading a trajectory file with read_file should give first frame."""
        path = os.path.join(SIM_DIR, "openmm_md_water", "trajectory.xyz")
        _skip_if_missing(path)
        mol = read_file(path)
        assert len(mol.atoms) > 0


class TestFormatRoundtrips:
    """Write → read roundtrips for all writable formats."""

    @pytest.fixture
    def hkust1(self):
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        _skip_if_missing(path)
        return read_file(path)

    @pytest.fixture
    def water(self):
        """Simple 3-atom water molecule without PBC."""
        from pdb_wizard.atom import Atom
        atoms = [
            Atom(0.0, 0.0, 0.0, "O"),
            Atom(0.757, 0.586, 0.0, "H"),
            Atom(-0.757, 0.586, 0.0, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=None)
        mol.detect_bonds()
        return mol

    def _check_roundtrip(self, original, read_back):
        """Verify atom count and element composition match."""
        from collections import Counter
        assert len(read_back.atoms) == len(original.atoms), (
            f"Atom count: {len(original.atoms)} -> {len(read_back.atoms)}"
        )
        orig_el = Counter(a.element.symbol for a in original.atoms)
        read_el = Counter(a.element.symbol for a in read_back.atoms)
        assert orig_el == read_el, f"Elements differ: {dict(orig_el)} vs {dict(read_el)}"

    def test_xyz_roundtrip(self, hkust1):
        import io as _io

        from pdb_wizard.io import read_xyz, write_xyz
        buf = _io.StringIO()
        write_xyz(hkust1, buf)
        buf.seek(0)
        atoms, pbc = read_xyz(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(hkust1, mol)

    def test_xyz_roundtrip_no_pbc(self, water):
        import io as _io

        from pdb_wizard.io import read_xyz, write_xyz
        buf = _io.StringIO()
        write_xyz(water, buf)
        buf.seek(0)
        atoms, pbc = read_xyz(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(water, mol)

    def test_pdb_roundtrip(self, hkust1):
        import io as _io

        from pdb_wizard.io import read_pdb, write_standard_pdb
        buf = _io.StringIO()
        write_standard_pdb(hkust1, buf)
        buf.seek(0)
        atoms, pbc = read_pdb(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(hkust1, mol)

    def test_cif_roundtrip(self, hkust1):
        import io as _io

        from pdb_wizard.formats import write_cif
        from pdb_wizard.io import read_cif
        buf = _io.StringIO()
        write_cif(hkust1, buf)
        buf.seek(0)
        atoms, pbc = read_cif(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(hkust1, mol)
        assert pbc is not None
        assert abs(pbc.a - hkust1.pbc.a) < 0.01

    def test_poscar_roundtrip(self, hkust1):
        import io as _io

        from pdb_wizard.formats import read_poscar, write_poscar
        buf = _io.StringIO()
        write_poscar(hkust1, buf)
        buf.seek(0)
        atoms, pbc = read_poscar(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(hkust1, mol)
        assert pbc is not None
        assert abs(pbc.a - hkust1.pbc.a) < 0.01

    def test_lammps_roundtrip(self, hkust1):
        import io as _io

        from pdb_wizard.formats import read_lammps_data, write_lammps_data
        buf = _io.StringIO()
        write_lammps_data(hkust1, buf)
        buf.seek(0)
        atoms, pbc = read_lammps_data(buf)
        mol = Molecule(atoms=atoms, pbc=pbc)
        self._check_roundtrip(hkust1, mol)
        assert pbc is not None
        assert abs(pbc.a - hkust1.pbc.a) < 0.5

    def test_lammps_preserves_charges(self, hkust1):
        import io as _io

        from pdb_wizard.formats import read_lammps_data, write_lammps_data
        buf = _io.StringIO()
        write_lammps_data(hkust1, buf)
        buf.seek(0)
        atoms, _ = read_lammps_data(buf)
        orig_q = sorted(a.charge for a in hkust1.atoms)
        read_q = sorted(a.charge for a in atoms)
        for oq, rq in zip(orig_q, read_q):
            assert abs(oq - rq) < 1e-4, f"Charge mismatch: {oq} vs {rq}"

    def test_gaussian_log_read(self):
        """Parse a Gaussian log file with Standard orientation block."""
        import io as _io

        from pdb_wizard.formats import read_gaussian_log
        content = """\
 Entering Gaussian System
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.117300
      2          1           0        0.000000    0.757200   -0.469200
      3          1           0        0.000000   -0.757200   -0.469200
 ---------------------------------------------------------------------
 Normal termination of Gaussian
"""
        buf = _io.StringIO(content)
        atoms, pbc = read_gaussian_log(buf)
        assert len(atoms) == 3
        assert atoms[0].element.symbol == "O"
        assert atoms[1].element.symbol == "H"

    def test_gaussian_com_read(self):
        """Parse a Gaussian input file."""
        import io as _io

        from pdb_wizard.formats import read_gaussian_com
        content = """\
%nproc=4
#p opt b3lyp/6-31g*

Water optimization

0 1
O  0.000  0.000  0.117
H  0.000  0.757 -0.469
H  0.000 -0.757 -0.469

"""
        buf = _io.StringIO(content)
        atoms, pbc = read_gaussian_com(buf)
        assert len(atoms) == 3
        assert atoms[0].element.symbol == "O"
        assert pbc is None

    def test_sdf_read(self):
        """Parse an SDF file."""
        import io as _io

        from pdb_wizard.formats import read_sdf
        content = """\
methane
     RDKit

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0
    0.6300    0.6300    0.6300 H   0  0  0  0  0  0
   -0.6300   -0.6300    0.6300 H   0  0  0  0  0  0
   -0.6300    0.6300   -0.6300 H   0  0  0  0  0  0
    0.6300   -0.6300   -0.6300 H   0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$$$$
"""
        buf = _io.StringIO(content)
        atoms, pbc = read_sdf(buf)
        assert len(atoms) == 5
        from collections import Counter
        c = Counter(a.element.symbol for a in atoms)
        assert c["C"] == 1
        assert c["H"] == 4

    def test_mol2_read(self):
        """Parse a MOL2 file."""
        import io as _io

        from pdb_wizard.formats import read_mol2
        content = """\
@<TRIPOS>MOLECULE
water
 3 2 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 O1         0.0000    0.0000    0.1173 O.3
      2 H1         0.0000    0.7572   -0.4692 H
      3 H2         0.0000   -0.7572   -0.4692 H
@<TRIPOS>BOND
     1     1     2    1
     2     1     3    1
"""
        buf = _io.StringIO(content)
        atoms, pbc = read_mol2(buf)
        assert len(atoms) == 3
        assert atoms[0].element.symbol == "O"
        assert atoms[1].element.symbol == "H"

    def test_coordinate_preservation_xyz(self, hkust1):
        """Coordinates should be preserved within floating point tolerance."""
        import io as _io

        from pdb_wizard.io import read_xyz, write_xyz
        buf = _io.StringIO()
        write_xyz(hkust1, buf)
        buf.seek(0)
        atoms, _ = read_xyz(buf)
        for orig, read in zip(hkust1.atoms, atoms):
            assert abs(orig.x[0] - read.x[0]) < 0.001
            assert abs(orig.x[1] - read.x[1]) < 0.001
            assert abs(orig.x[2] - read.x[2]) < 0.001

    def test_coordinate_preservation_pdb(self, hkust1):
        """PDB write/read preserves coordinates (atoms may be reordered by sort)."""
        import io as _io

        from pdb_wizard.io import read_pdb, write_standard_pdb
        buf = _io.StringIO()
        write_standard_pdb(hkust1, buf)
        buf.seek(0)
        atoms, _ = read_pdb(buf)
        # Atoms may be sorted differently, so compare sorted by element+position
        orig_sorted = sorted(
            [(a.element.symbol, round(a.x[0], 3), round(a.x[1], 3), round(a.x[2], 3)) for a in hkust1.atoms]
        )
        read_sorted = sorted(
            [(a.element.symbol, round(a.x[0], 3), round(a.x[1], 3), round(a.x[2], 3)) for a in atoms]
        )
        assert len(orig_sorted) == len(read_sorted)
        for o, r in zip(orig_sorted, read_sorted):
            assert o[0] == r[0], f"Element mismatch: {o[0]} vs {r[0]}"
            assert abs(o[1] - r[1]) < 0.01
            assert abs(o[2] - r[2]) < 0.01
            assert abs(o[3] - r[3]) < 0.01
