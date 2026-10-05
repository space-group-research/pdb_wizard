"""Tests for batch CLI mode in pdb_wizard.__main__."""

from __future__ import annotations

import csv
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mol(n_atoms=8, element="O", box=10.0):
    """Create a simple molecule with PBC for testing."""
    rng = np.random.default_rng(42)
    atoms = [
        Atom(*(rng.uniform(0, box, 3).tolist()), element) for _ in range(n_atoms)
    ]
    pbc = PBC(box, box, box, 90.0, 90.0, 90.0)
    mol = Molecule(atoms=atoms, pbc=pbc)
    return mol


@pytest.fixture
def water_xyz(tmp_path):
    content = textwrap.dedent("""\
        3
        10.0 10.0 10.0 90.0 90.0 90.0
        O  0.000  0.000  0.000
        H  0.960  0.000  0.000
        H -0.240  0.930  0.000
    """)
    p = tmp_path / "water.xyz"
    p.write_text(content)
    return p


@pytest.fixture
def water_pdb(tmp_path):
    content = textwrap.dedent("""\
        MODEL        1
        CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
        HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
        HETATM    2  H1  HOH A   1       0.960   0.000   0.000  1.00  0.00           H
        HETATM    3  H2  HOH A   1      -0.240   0.930   0.000  1.00  0.00           H
        END
    """)
    p = tmp_path / "water.pdb"
    p.write_text(content)
    return p


@pytest.fixture
def oxygen_box_xyz(tmp_path):
    """8 oxygen atoms in a 10 A box — useful for RDF and surface tests."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 10, (8, 3))
    lines = ["8", "10.0 10.0 10.0 90.0 90.0 90.0"]
    for c in coords:
        lines.append(f"O  {c[0]:.4f}  {c[1]:.4f}  {c[2]:.4f}")
    content = "\n".join(lines) + "\n"
    p = tmp_path / "oxygen_box.xyz"
    p.write_text(content)
    return p


@pytest.fixture
def trajectory_xyz(tmp_path):
    """Two-frame trajectory of 4 O atoms."""
    rng = np.random.default_rng(99)
    frames = []
    for _ in range(2):
        coords = rng.uniform(0, 10, (4, 3))
        lines = ["4", "10.0 10.0 10.0 90.0 90.0 90.0"]
        for c in coords:
            lines.append(f"O  {c[0]:.4f}  {c[1]:.4f}  {c[2]:.4f}")
        frames.append("\n".join(lines))
    content = "\n".join(frames) + "\n"
    p = tmp_path / "traj.xyz"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------

class TestParserAcceptsBatchFlags:
    """Ensure argparse accepts all new flags without error."""

    def test_info_flag(self, water_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--info", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_surface_flag(self, water_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--surface", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Surface area" in result.stdout

    def test_psd_flag(self, water_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--psd", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "pore_radius" in result.stdout

    def test_rdf_requires_elements(self, water_xyz):
        """--rdf without -e1/-e2 should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--rdf", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_convert_requires_output(self, water_xyz):
        """--convert without -o should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--convert", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# --info
# ---------------------------------------------------------------------------

class TestBatchInfo:
    def test_info_prints_filename(self, water_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--info", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "water.xyz" in result.stdout

    def test_info_prints_atom_count(self, water_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--info", str(water_xyz)],
            capture_output=True, text=True,
        )
        assert "O" in result.stdout
        assert "H" in result.stdout

    def test_info_pdb(self, water_pdb):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--info", str(water_pdb)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "water.pdb" in result.stdout


# ---------------------------------------------------------------------------
# --rdf
# ---------------------------------------------------------------------------

class TestBatchRdf:
    def test_rdf_stdout(self, oxygen_box_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--rdf", str(oxygen_box_xyz),
             "-e1", "O", "-e2", "O"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Output should have lines of "r  g(r)"
        lines = result.stdout.strip().splitlines()
        assert len(lines) > 10

    def test_rdf_csv(self, oxygen_box_xyz, tmp_path):
        csv_out = tmp_path / "rdf.csv"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--rdf", str(oxygen_box_xyz),
             "-e1", "O", "-e2", "O", "--csv", str(csv_out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert csv_out.exists()
        with open(csv_out) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "r(A)" in header[0]
            rows = list(reader)
            assert len(rows) > 10

    def test_rdf_bad_element(self, water_xyz):
        """Requesting RDF for a non-existent element should error."""
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--rdf", str(water_xyz),
             "-e1", "Zn", "-e2", "Zn"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# --msd
# ---------------------------------------------------------------------------

class TestBatchMsd:
    def test_msd_stdout(self, trajectory_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--msd", str(trajectory_xyz),
             "-e", "O"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().splitlines()
        assert len(lines) >= 1

    def test_msd_csv(self, trajectory_xyz, tmp_path):
        csv_out = tmp_path / "msd.csv"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--msd", str(trajectory_xyz),
             "-e", "O", "--csv", str(csv_out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert csv_out.exists()
        with open(csv_out) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "lag" in header[0]

    def test_msd_single_frame_fails(self, water_xyz):
        """MSD needs a trajectory; single frame should fail."""
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--msd", str(water_xyz),
             "-e", "O"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_msd_no_element(self, trajectory_xyz):
        """MSD without -e should still work (tracks all atoms)."""
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--msd", str(trajectory_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# --convert
# ---------------------------------------------------------------------------

class TestBatchConvert:
    def test_xyz_to_pdb(self, water_xyz, tmp_path):
        out_pdb = tmp_path / "out.pdb"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--convert", str(water_xyz),
             "-o", str(out_pdb)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out_pdb.exists()
        text = out_pdb.read_text()
        assert "HETATM" in text or "ATOM" in text

    def test_pdb_to_xyz(self, water_pdb, tmp_path):
        out_xyz = tmp_path / "out.xyz"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--convert", str(water_pdb),
             "-o", str(out_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out_xyz.exists()
        text = out_xyz.read_text()
        assert "O" in text

    def test_xyz_to_cif(self, water_xyz, tmp_path):
        out_cif = tmp_path / "out.cif"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--convert", str(water_xyz),
             "-o", str(out_cif)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out_cif.exists()
        text = out_cif.read_text()
        assert "data_pdb_wizard" in text

    def test_unsupported_output_format(self, water_xyz, tmp_path):
        out_bad = tmp_path / "out.blargh"
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--convert", str(water_xyz),
             "-o", str(out_bad)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# --surface
# ---------------------------------------------------------------------------

class TestBatchSurface:
    def test_surface_output(self, oxygen_box_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--surface", str(oxygen_box_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Surface area" in result.stdout
        assert "A^2" in result.stdout

    def test_surface_area_positive(self, oxygen_box_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--surface", str(oxygen_box_xyz)],
            capture_output=True, text=True,
        )
        # Parse the numeric value
        for line in result.stdout.splitlines():
            if "Surface area" in line:
                val = float(line.split(":")[1].strip().split()[0])
                assert val > 0


# ---------------------------------------------------------------------------
# --psd
# ---------------------------------------------------------------------------

class TestBatchPsd:
    def test_psd_output(self, oxygen_box_xyz):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--psd", str(oxygen_box_xyz)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "pore_radius" in result.stdout
        lines = result.stdout.strip().splitlines()
        # Header + data rows
        assert len(lines) > 5


# ---------------------------------------------------------------------------
# Existing flags still work
# ---------------------------------------------------------------------------

class TestExistingFlagsPreserved:
    def test_classic_flag_accepted(self, water_xyz):
        """--classic should be accepted (we just check it doesn't crash on parse)."""
        # We can't fully test classic mode (it needs stdin), but at least verify
        # the flag is still recognised by checking the help output.
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--classic" in result.stdout

    def test_help_shows_batch_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--info" in result.stdout
        assert "--rdf" in result.stdout
        assert "--msd" in result.stdout
        assert "--convert" in result.stdout
        assert "--surface" in result.stdout
        assert "--psd" in result.stdout

    def test_file_positional_still_required_for_batch(self):
        """A batch flag without a file should fail at argument parsing.

        A bare ``python -m pdb_wizard`` (no file, no flags) is a supported
        launch that opens the TUI with an empty buffer, so it must not be
        used here: with textual installed it would block forever.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pdb_wizard", "--info"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0
        assert "require a file" in result.stderr


# ---------------------------------------------------------------------------
# Batch mode exits before importing textual
# ---------------------------------------------------------------------------

class TestBatchNoTextualImport:
    def test_info_does_not_import_textual(self, water_xyz):
        """Batch --info should work even if textual is not installed."""
        # We verify by checking that textual is never imported in the process
        code = (
            "import sys; "
            "sys.argv = ['pdb_wizard', '--info', '{fpath}']; "
            "from pdb_wizard.__main__ import main; main(); "
            "assert 'textual' not in sys.modules, 'textual was imported in batch mode'"
        ).format(fpath=str(water_xyz))
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
