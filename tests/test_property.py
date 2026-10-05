"""Property-based tests using Hypothesis.

These generate random inputs to find edge cases that hand-written tests miss.
Covers: Atom construction, PBC wrapping, file format roundtrips, rendering
invariants, analysis functions, and viewer state consistency.
"""

from __future__ import annotations

import io as _io
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pdb_wizard.atom import Atom
from pdb_wizard.constants import ELEMENTS
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

elements = st.sampled_from(list(ELEMENTS.keys())[:30])  # first 30 elements
coords = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
cell_lengths = st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
angles = st.floats(min_value=60.0, max_value=120.0, allow_nan=False, allow_infinity=False)
small_int = st.integers(min_value=1, max_value=20)


@st.composite
def atoms(draw, n=None):
    """Generate a list of Atoms with random positions and elements."""
    count = n if n is not None else draw(st.integers(min_value=1, max_value=50))
    result = []
    for _ in range(count):
        x = draw(coords)
        y = draw(coords)
        z = draw(coords)
        el = draw(elements)
        result.append(Atom(x, y, z, el))
    return result


@st.composite
def molecules(draw, with_pbc=True):
    """Generate a Molecule with random atoms and optional PBC."""
    atom_list = draw(atoms())
    if with_pbc:
        a = draw(cell_lengths)
        b = draw(cell_lengths)
        c = draw(cell_lengths)
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
    else:
        pbc = None
    return Molecule(atoms=atom_list, pbc=pbc)


# ---------------------------------------------------------------------------
# Atom properties
# ---------------------------------------------------------------------------

class TestAtomProperties:
    @given(x=coords, y=coords, z=coords, el=elements)
    def test_position_roundtrip(self, x, y, z, el):
        """Atom position should be exactly what was set."""
        a = Atom(x, y, z, el)
        assert a.x[0] == pytest.approx(x)
        assert a.x[1] == pytest.approx(y)
        assert a.x[2] == pytest.approx(z)

    @given(el=elements)
    def test_element_lookup(self, el):
        """Every valid element should resolve to a known Element."""
        a = Atom(0, 0, 0, el)
        assert a.element.symbol == el
        assert a.element.mass > 0
        assert a.element.atomic_number > 0

    @given(x=coords, y=coords, z=coords, el=elements)
    def test_position_alias(self, x, y, z, el):
        """atom.position should be the same array as atom.x."""
        a = Atom(x, y, z, el)
        assert np.array_equal(a.position, a.x)


# ---------------------------------------------------------------------------
# PBC
# ---------------------------------------------------------------------------

class TestPBCProperties:
    @given(a=cell_lengths, b=cell_lengths, c=cell_lengths)
    def test_volume_positive(self, a, b, c):
        """Cell volume should always be positive for valid cell lengths."""
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
        assert pbc.volume > 0

    @given(a=cell_lengths, b=cell_lengths, c=cell_lengths)
    def test_orthorhombic_volume(self, a, b, c):
        """Orthorhombic cell volume = a*b*c."""
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
        assert pbc.volume == pytest.approx(a * b * c, rel=1e-6)

    @given(a=cell_lengths, b=cell_lengths, c=cell_lengths)
    def test_corners_count(self, a, b, c):
        """Unit cell should always have 8 corners and 12 edges."""
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
        assert len(pbc.corners()) == 8
        assert len(pbc.edges()) == 12

    @given(a=cell_lengths, b=cell_lengths, c=cell_lengths,
           x=coords, y=coords, z=coords)
    def test_min_image_bounded(self, a, b, c, x, y, z):
        """Min image distance should be <= half the cell diagonal."""
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
        vec = np.array([x, y, z])
        d = pbc.min_image(vec)
        half_diag = 0.5 * math.sqrt(a**2 + b**2 + c**2)
        assert d <= half_diag + 1e-6

    @given(a=cell_lengths, b=cell_lengths, c=cell_lengths)
    def test_reciprocal_inverse(self, a, b, c):
        """basis @ reciprocal should be close to identity."""
        pbc = PBC(a, b, c, 90.0, 90.0, 90.0)
        product = pbc.basis_matrix @ pbc.reciprocal_basis_matrix
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# XYZ roundtrip
# ---------------------------------------------------------------------------

class TestXyzRoundtrip:
    @given(data=atoms(n=None))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_atom_count_preserved(self, data):
        """XYZ write -> read should preserve atom count."""
        from pdb_wizard.io import read_xyz, write_xyz
        mol = Molecule(atoms=data, pbc=PBC(50, 50, 50, 90, 90, 90))
        buf = _io.StringIO()
        write_xyz(mol, buf)
        buf.seek(0)
        atoms_back, _ = read_xyz(buf)
        assert len(atoms_back) == len(data)

    @given(data=atoms(n=None))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_elements_preserved(self, data):
        """XYZ roundtrip should preserve element symbols."""
        from collections import Counter

        from pdb_wizard.io import read_xyz, write_xyz
        mol = Molecule(atoms=data, pbc=PBC(50, 50, 50, 90, 90, 90))
        buf = _io.StringIO()
        write_xyz(mol, buf)
        buf.seek(0)
        atoms_back, _ = read_xyz(buf)
        orig = Counter(a.element.symbol for a in data)
        back = Counter(a.element.symbol for a in atoms_back)
        assert orig == back

    @given(data=atoms(n=None))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_coordinates_preserved(self, data):
        """XYZ roundtrip should preserve coordinates to 4 decimal places."""
        from pdb_wizard.io import read_xyz, write_xyz
        mol = Molecule(atoms=data, pbc=PBC(50, 50, 50, 90, 90, 90))
        buf = _io.StringIO()
        write_xyz(mol, buf)
        buf.seek(0)
        atoms_back, _ = read_xyz(buf)
        for orig, back in zip(data, atoms_back):
            np.testing.assert_allclose(orig.x, back.x, atol=0.001)


# ---------------------------------------------------------------------------
# LAMMPS roundtrip
# ---------------------------------------------------------------------------

class TestLammpsRoundtrip:
    @given(data=atoms(n=None))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_atom_count_preserved(self, data):
        """LAMMPS write -> read should preserve atom count."""
        from pdb_wizard.formats import read_lammps_data, write_lammps_data
        mol = Molecule(atoms=data, pbc=PBC(50, 50, 50, 90, 90, 90))
        mol.detect_bonds()
        buf = _io.StringIO()
        write_lammps_data(mol, buf)
        buf.seek(0)
        atoms_back, _ = read_lammps_data(buf)
        assert len(atoms_back) == len(data)

    @given(data=atoms(n=None))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_elements_preserved(self, data):
        """LAMMPS roundtrip should preserve element composition."""
        from collections import Counter

        from pdb_wizard.formats import read_lammps_data, write_lammps_data
        mol = Molecule(atoms=data, pbc=PBC(50, 50, 50, 90, 90, 90))
        mol.detect_bonds()
        buf = _io.StringIO()
        write_lammps_data(mol, buf)
        buf.seek(0)
        atoms_back, _ = read_lammps_data(buf)
        orig = Counter(a.element.symbol for a in data)
        back = Counter(a.element.symbol for a in atoms_back)
        assert orig == back


# ---------------------------------------------------------------------------
# Molecule invariants
# ---------------------------------------------------------------------------

class TestMoleculeInvariants:
    @given(mol=molecules(with_pbc=True))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_center_finite(self, mol):
        """Molecule center should always be finite."""
        c = mol.center()
        assert np.all(np.isfinite(c))

    @given(mol=molecules(with_pbc=True))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_radius_non_negative(self, mol):
        """Molecule radius should never be negative."""
        assert mol.radius() >= 0.0

    @given(mol=molecules(with_pbc=False))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_detect_bonds_no_crash(self, mol):
        """Bond detection should never crash, with or without PBC."""
        mol.detect_bonds()
        assert isinstance(mol.bonds, list)
        for a, b in mol.bonds:
            assert 0 <= a < len(mol.atoms)
            assert 0 <= b < len(mol.atoms)
            assert a != b


# ---------------------------------------------------------------------------
# Renderer invariants
# ---------------------------------------------------------------------------

class TestRendererInvariants:
    @given(rx=st.floats(-3.14, 3.14), ry=st.floats(-3.14, 3.14), rz=st.floats(-3.14, 3.14))
    def test_rotation_matrix_orthogonal(self, rx, ry, rz):
        """Rotation matrix should always be orthogonal (R @ R.T = I)."""
        from pdb_wizard.renderer import rotation_matrix
        R = rotation_matrix(rx, ry, rz)
        product = R @ R.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    @given(rx=st.floats(-3.14, 3.14), ry=st.floats(-3.14, 3.14), rz=st.floats(-3.14, 3.14))
    def test_rotation_matrix_det_one(self, rx, ry, rz):
        """Rotation matrix determinant should be 1 (proper rotation)."""
        from pdb_wizard.renderer import rotation_matrix
        R = rotation_matrix(rx, ry, rz)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    @given(mol=molecules(with_pbc=True))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_render_scene_shapes(self, mol):
        """render_scene should always return correct array shapes."""
        from pdb_wizard.renderer import render_scene, rotation_matrix
        w, h = 20, 16
        rot = rotation_matrix(0, 0, 0)
        pixels, hit = render_scene(w, h, mol, rot, 5.0, ssaa=1)
        assert pixels.shape == (h, w, 3)
        assert hit.shape == (h, w)
        assert pixels.dtype == np.uint8


# ---------------------------------------------------------------------------
# Analysis invariants
# ---------------------------------------------------------------------------

class TestAnalysisInvariants:
    def test_rdf_symmetric(self):
        """g(r) for A-B should equal g(r) for B-A."""
        from pdb_wizard.rdf import compute_rdf
        atoms = [Atom(0, 0, 0, "O"), Atom(1, 0, 0, "H"), Atom(0, 1, 0, "H"),
                 Atom(3, 0, 0, "O"), Atom(4, 0, 0, "H")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        r1, g1 = compute_rdf(mol, "O", "H")
        r2, g2 = compute_rdf(mol, "H", "O")
        np.testing.assert_allclose(g1, g2, atol=0.01)

    def test_rmsd_self_is_zero(self):
        """RMSD of a frame vs itself should be 0."""
        from pdb_wizard.rmsd import compute_rmsd
        mol = Molecule(atoms=[Atom(i, 0, 0, "C") for i in range(5)])
        idx, rmsd = compute_rmsd([mol, mol, mol])
        assert rmsd[0] == 0.0

    def test_coordination_self_element(self):
        """Coordination of A around A should count self-neighbors."""
        from pdb_wizard.coordination import compute_coordination
        atoms = [Atom(0, 0, 0, "Cu"), Atom(2.0, 0, 0, "Cu"), Atom(50, 50, 50, "Cu")]
        mol = Molecule(atoms=atoms, pbc=PBC(100, 100, 100, 90, 90, 90))
        r, cn, avg = compute_coordination(mol, "Cu", "Cu", r_max=3.0)
        # At r=3, each Cu sees 1 neighbor (the one 2A away), except the isolated one
        # avg = (1 + 1 + 0) / 3 = 0.67
        assert avg > 0

    def test_diffusion_constant_positive(self):
        """Diffusion constant from random walk MSD should be positive."""
        from pdb_wizard.msd import compute_diffusion_constant
        # Create a linearly increasing MSD (ideal diffusion)
        lags = np.arange(1, 100, dtype=float)
        msd = 6.0 * lags  # D = 1.0 A^2/ps
        D, slope = compute_diffusion_constant(lags, msd, dt=1.0)
        assert D > 0
        assert slope == pytest.approx(6.0, rel=0.1)

    def test_hbond_water_has_hbonds(self):
        """A cluster of water molecules should have hydrogen bonds."""
        from pdb_wizard.hbonds import detect_hbonds
        # Two waters in H-bond geometry
        atoms = [
            Atom(0.0, 0.0, 0.0, "O"),
            Atom(0.757, 0.586, 0.0, "H"),
            Atom(-0.757, 0.586, 0.0, "H"),
            Atom(2.8, 0.0, 0.0, "O"),
            Atom(2.8 + 0.757, 0.586, 0.0, "H"),
            Atom(2.8 - 0.757, 0.586, 0.0, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))
        mol.detect_bonds()
        hb = detect_hbonds(mol, d_cutoff=3.5, angle_cutoff=90.0)
        assert len(hb) > 0

    def test_pore_size_normalized(self):
        """Pore size histogram should sum to ~1."""
        from pdb_wizard.operations import pore_size_distribution
        atoms = [Atom(5, 5, 5, "Cu")]
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        bins, hist = pore_size_distribution(mol, n_samples=10000)
        if len(hist) > 0:
            assert abs(hist.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Regression: specific bugs caught in code review
# ---------------------------------------------------------------------------

class TestRegressions:
    def test_detect_filetype_log_is_gaussian(self):
        from pdb_wizard.io import detect_filetype
        assert detect_filetype("opt.log") == "gaussian_log"

    def test_detect_filetype_out_is_not_gaussian(self):
        """Generic .out files should NOT be detected as Gaussian."""
        from pdb_wizard.io import detect_filetype
        with pytest.raises(ValueError):
            detect_filetype("output.out")

    def test_detect_filetype_sdf(self):
        from pdb_wizard.io import detect_filetype
        assert detect_filetype("mol.sdf") == "sdf"

    def test_lammps_mol_ids_separate_molecules(self):
        """Disconnected atoms should get different mol_ids."""
        from pdb_wizard.formats import write_lammps_data
        atoms = [Atom(0, 0, 0, "C"), Atom(50, 50, 50, "C")]
        mol = Molecule(atoms=atoms, pbc=PBC(100, 100, 100, 90, 90, 90))
        # No bonds → each atom is its own molecule
        buf = _io.StringIO()
        write_lammps_data(mol, buf)
        content = buf.getvalue()
        # Find the two atom lines
        in_atoms = False
        mol_ids = []
        for line in content.split("\n"):
            if line.strip().startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms and line.strip():
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        mol_ids.append(int(tokens[1]))
                    except ValueError:
                        in_atoms = False
        assert len(set(mol_ids)) == 2, f"Expected 2 mol_ids, got {mol_ids}"

    def test_lammps_bond_types_by_element_pair(self):
        """Different element pairs should get different bond types."""
        from pdb_wizard.formats import write_lammps_data
        atoms = [Atom(0, 0, 0, "C"), Atom(1.0, 0, 0, "C"),
                 Atom(3, 0, 0, "C"), Atom(4.0, 0, 0, "O")]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (2, 3)], pbc=PBC(10, 10, 10, 90, 90, 90))
        buf = _io.StringIO()
        write_lammps_data(mol, buf)
        content = buf.getvalue()
        assert "2 bond types" in content

    def test_reduce_cell_exact_multiplicity(self):
        """Framework detection should require EXACT multiplicity."""
        from pdb_wizard.reduce_cell import identify_framework_and_sorbates
        # 2x1x1 cell: atom at (2,2,2) has copy at (12,2,2) -> exactly 2 copies
        pbc = PBC(20, 10, 10, 90, 90, 90)
        atoms = [
            Atom(2.0, 2.0, 2.0, "Cu"),
            Atom(12.0, 2.0, 2.0, "Cu"),
            Atom(7.0, 5.0, 5.0, "H"),  # sorbate, only 1 copy
        ]
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol.detect_bonds()
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        assert len(fw) == 1  # Cu is framework
        sorbate_count = sum(len(s) for s in sorbs)
        assert sorbate_count == 1  # H is sorbate

    def test_batch_cli_missing_file(self):
        """Batch CLI should exit cleanly for missing files."""
        from pdb_wizard.__main__ import _check_file
        with pytest.raises(SystemExit):
            _check_file("/nonexistent/file.pdb")

    def test_find_molecules_bond_remapping(self):
        """find_molecules should remap bond indices to local molecule."""
        from pdb_wizard.molecule import Molecule
        # Two separate molecules: atoms 0-1 bonded, atoms 2-3 bonded
        atoms = [Atom(0, 0, 0, "C"), Atom(1, 0, 0, "C"),
                 Atom(10, 0, 0, "O"), Atom(11, 0, 0, "O")]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (2, 3)])
        mols = mol.find_molecules()
        assert len(mols) == 2
        for m in mols:
            for a, b in m.bonds:
                assert 0 <= a < len(m.atoms), f"Bond index {a} out of range for {len(m.atoms)} atoms"
                assert 0 <= b < len(m.atoms), f"Bond index {b} out of range for {len(m.atoms)} atoms"

    def test_rdf_same_type_normalized(self):
        """Same-type RDF should converge to ~1 at large r for uniform distribution."""
        from pdb_wizard.rdf import compute_rdf
        # Create a uniform distribution of O atoms in a large box
        rng = np.random.default_rng(42)
        n = 200
        positions = rng.random((n, 3)) * 20.0
        atoms = [Atom(float(p[0]), float(p[1]), float(p[2]), "O") for p in positions]
        mol = Molecule(atoms=atoms, pbc=PBC(20, 20, 20, 90, 90, 90))
        r, g = compute_rdf(mol, "O", "O", r_max=8.0, n_bins=100)
        # At large r, g(r) should approach 1 for uniform distribution
        far = g[len(g)//2:]  # second half
        assert 0.5 < far.mean() < 1.5, f"g(r) mean at large r = {far.mean():.2f}, expected ~1.0"

    def test_delete_lone_atoms_updates_bonds(self):
        """delete_lone_atoms should remap bond indices after removal."""
        from pdb_wizard.geometry import delete_lone_atoms
        atoms = [
            Atom(0, 0, 0, "C"), Atom(1, 0, 0, "C"),   # bonded pair
            Atom(50, 50, 50, "H"),                      # lone atom
            Atom(3, 0, 0, "C"), Atom(4, 0, 0, "C"),   # another bonded pair
        ]
        mol = Molecule(atoms=atoms, bonds=[(0, 1), (3, 4)], pbc=PBC(100, 100, 100, 90, 90, 90))
        result = delete_lone_atoms(mol)
        # H at index 2 removed -> atoms [C,C,C,C], bonds should be [(0,1),(2,3)]
        assert len(result.atoms) == 4
        for a, b in result.bonds:
            assert 0 <= a < len(result.atoms), f"Bond {a} out of range"
            assert 0 <= b < len(result.atoms), f"Bond {b} out of range"

    def test_pbc_sqrt_safe(self):
        """PBC should handle near-degenerate cell parameters without NaN."""
        # Angles that could cause negative under sqrt due to rounding
        pbc = PBC(10, 10, 10, 60.0, 60.0, 60.0)
        assert np.all(np.isfinite(pbc.basis_matrix))
        assert pbc.volume > 0

    def test_poscar_roundtrip_with_comment(self):
        """write_poscar should produce valid VASP file with comment line."""
        import io as _io

        from pdb_wizard.formats import read_poscar, write_poscar
        atoms = [Atom(0, 0, 0, "Si"), Atom(2.7, 2.7, 2.7, "Si")]
        mol = Molecule(atoms=atoms, pbc=PBC(5.43, 5.43, 5.43, 90, 90, 90))
        buf = _io.StringIO()
        write_poscar(mol, buf)
        content = buf.getvalue()
        lines = content.strip().split("\n")
        # Line 0 should be comment, not element names
        assert "Generated" in lines[0] or not lines[0][0].isupper()
        # Line 1 should be scale factor
        assert float(lines[1].strip()) == 1.0
        # Read back
        buf.seek(0)
        atoms_back, pbc_back = read_poscar(buf)
        assert len(atoms_back) == 2

    def test_atom_pure_digit_name(self):
        """Atom with pure digit name should get default element, not crash."""
        a = Atom(0, 0, 0, "123")
        assert a.element.symbol == "X"  # default element

    def test_msd_varying_atom_count(self):
        """MSD should handle frames with different atom counts without zero-bias."""
        from pdb_wizard.msd import compute_msd
        # Frame 0: 3 atoms, Frame 1: 2 atoms (one lost)
        m1 = Molecule(atoms=[Atom(0, 0, 0, "H"), Atom(1, 0, 0, "H"), Atom(2, 0, 0, "H")])
        m2 = Molecule(atoms=[Atom(0.1, 0.1, 0.1, "H"), Atom(1.1, 0.1, 0.1, "H")])
        m3 = Molecule(atoms=[Atom(0.2, 0.2, 0.2, "H"), Atom(1.2, 0.2, 0.2, "H")])
        lags, msd = compute_msd([m1, m2, m3], element="H")
        # MSD should be > 0 (atoms moved)
        assert len(msd) > 0
        assert msd[0] > 0
