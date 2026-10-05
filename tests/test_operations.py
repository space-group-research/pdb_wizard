"""Tests for pdb_wizard.operations."""

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.operations import (
    extend_axis,
    formula_unit,
    formula_unit_reduced,
    gcd_list,
    pore_size_distribution,
    sort_system,
    surface_area,
    system_info,
    wrap_atoms,
)
from pdb_wizard.pbc import PBC


class TestGcdList:
    def test_basic(self):
        assert gcd_list([4, 6, 8]) == 2

    def test_coprime(self):
        assert gcd_list([7, 13]) == 1

    def test_single(self):
        assert gcd_list([5]) == 5


class TestFormulaUnit:
    def test_water(self):
        atoms = [Atom(0, 0, 0, "O"), Atom(1, 0, 0, "H"), Atom(-1, 0, 0, "H")]
        mol = Molecule(atoms=atoms)
        fu = formula_unit(mol)
        assert fu == {"O": 1, "H": 2}

    def test_reduced(self):
        atoms = [
            Atom(0, 0, 0, "O"), Atom(1, 0, 0, "H"), Atom(-1, 0, 0, "H"),
            Atom(5, 0, 0, "O"), Atom(6, 0, 0, "H"), Atom(4, 0, 0, "H"),
        ]
        mol = Molecule(atoms=atoms)
        r = formula_unit_reduced(mol)
        assert r == {"O": 1, "H": 2}


class TestWrapAtoms:
    def test_wrap_centered(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(15.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        wrap_atoms(mol)
        assert mol.atoms[0].x[0] == pytest.approx(-5.0, abs=0.1)

    def test_wrap_forward(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(-3.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        wrap_atoms(mol, forward=True)
        assert mol.atoms[0].x[0] == pytest.approx(7.0, abs=0.1)

    def test_no_pbc(self):
        atoms = [Atom(15.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms)
        wrap_atoms(mol)
        assert mol.atoms[0].x[0] == pytest.approx(15.0)


class TestSortSystem:
    def test_sorts_by_element(self):
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(0, 0, 0, "H"), Atom(1, 0, 0, "O"),
            Atom(50, 50, 50, "C"),
        ]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol, submols = sort_system(mol, output=False)
        assert len(submols) >= 1


class TestExtendAxis:
    def test_doubles_a(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(1.0, 1.0, 1.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol = extend_axis(mol, axis=0, times=1)
        assert len(mol.atoms) == 2
        assert mol.pbc.a == pytest.approx(20.0)

    def test_triples_b(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(1, 1, 1, "C"), Atom(3, 3, 3, "O")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol = extend_axis(mol, axis=1, times=2)
        assert len(mol.atoms) == 6
        assert mol.pbc.b == pytest.approx(30.0)


class TestSystemInfo:
    def test_returns_dict(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(0, 0, 0, "C"), Atom(1, 0, 0, "O")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        info = system_info(mol, "test.pdb")
        assert info["filename"] == "test.pdb"
        assert info["n_atoms"] == 2
        assert "volume" in info
        assert "density" in info
        assert "formula" in info


class TestSurfaceArea:
    def test_no_pbc_returns_zeros(self):
        atoms = [Atom(0, 0, 0, "C")]
        mol = Molecule(atoms=atoms)
        result = surface_area(mol, n_samples=100)
        assert result["surface_area"] == 0.0
        assert result["area_per_volume"] == 0.0

    def test_empty_mol_returns_zeros(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        mol = Molecule(atoms=[], pbc=pbc)
        result = surface_area(mol, n_samples=100)
        assert result["surface_area"] == 0.0

    def test_single_atom_full_sphere(self):
        """A single atom in a large box should have SASA ~ 4*pi*(vdw+probe)^2."""
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        atoms = [Atom(50.0, 50.0, 50.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        probe = 1.4
        vdw = atoms[0].vdw
        expected = 4.0 * np.pi * (vdw + probe) ** 2
        result = surface_area(mol, n_samples=200000, probe_radius=probe)
        # Monte Carlo — allow 5% tolerance
        assert result["surface_area"] == pytest.approx(expected, rel=0.05)

    def test_area_per_volume_units(self):
        """Check that area_per_volume is in m^2/g with correct conversion."""
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        atoms = [Atom(50.0, 50.0, 50.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        result = surface_area(mol, n_samples=100000, probe_radius=1.4)
        assert result["area_per_volume"] > 0.0
        # Manually check: sa_A2 * 1e-20 / (mass_amu * 1.66054e-24)
        sa = result["surface_area"]
        mass_g = atoms[0].mass * 1.66054e-24
        expected_apv = sa * 1e-20 / mass_g
        assert result["area_per_volume"] == pytest.approx(expected_apv, rel=1e-6)

    def test_two_distant_atoms(self):
        """Two atoms far apart should have SASA ~ sum of individual spheres."""
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        probe = 1.4
        atoms = [Atom(20.0, 50.0, 50.0, "C"), Atom(80.0, 50.0, 50.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        vdw = atoms[0].vdw
        single_sa = 4.0 * np.pi * (vdw + probe) ** 2
        result = surface_area(mol, n_samples=200000, probe_radius=probe)
        assert result["surface_area"] == pytest.approx(2.0 * single_sa, rel=0.05)

    def test_overlapping_atoms_less_than_sum(self):
        """Two overlapping atoms should have less total SASA than two separate ones."""
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        probe = 1.4
        vdw = Atom(0, 0, 0, "C").vdw
        # Place atoms closer than 2*(vdw+probe) so spheres overlap
        sep = (vdw + probe) * 0.8
        atoms = [Atom(50.0, 50.0, 50.0, "C"), Atom(50.0 + sep, 50.0, 50.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        single_sa = 4.0 * np.pi * (vdw + probe) ** 2
        result = surface_area(mol, n_samples=200000, probe_radius=probe)
        assert result["surface_area"] < 2.0 * single_sa * 0.99


class TestPoreSizeDistribution:
    def test_no_pbc_returns_zeros(self):
        atoms = [Atom(0, 0, 0, "C")]
        mol = Molecule(atoms=atoms)
        centers, hist = pore_size_distribution(mol, n_samples=100, n_bins=10)
        assert len(centers) == 10
        assert len(hist) == 10
        assert hist.sum() == 0.0

    def test_empty_mol_returns_zeros(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        mol = Molecule(atoms=[], pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=100, n_bins=10)
        assert hist.sum() == 0.0

    def test_normalized_histogram(self):
        """Histogram should sum to 1 for a valid system."""
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [Atom(10.0, 10.0, 10.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=10000, n_bins=50)
        assert hist.sum() == pytest.approx(1.0, abs=1e-10)

    def test_bin_centers_shape(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [Atom(10.0, 10.0, 10.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=100, n_bins=25, max_pore=15.0)
        assert len(centers) == 25
        assert len(hist) == 25
        assert centers[0] == pytest.approx(0.3, abs=0.01)  # 15/25/2
        assert centers[-1] == pytest.approx(15.0 - 0.3, abs=0.01)

    def test_dense_system_small_pores(self):
        """A densely packed system should have most pore radii near zero."""
        pbc = PBC(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        # Place atoms at all corners of the box
        atoms = []
        for x in [0.0, 2.5]:
            for y in [0.0, 2.5]:
                for z in [0.0, 2.5]:
                    atoms.append(Atom(x, y, z, "C"))
        mol = Molecule(atoms=atoms, pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=20000, n_bins=50, max_pore=10.0)
        # Most of the distribution should be in the lower bins
        lower_half = hist[:25].sum()
        assert lower_half > 0.9

    def test_returns_numpy_arrays(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(5.0, 5.0, 5.0, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=100, n_bins=20)
        assert isinstance(centers, np.ndarray)
        assert isinstance(hist, np.ndarray)

    def test_no_negative_pore_radii(self):
        """Points inside atoms get clamped to 0, so no negatives in histogram."""
        pbc = PBC(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        atoms = [Atom(2.5, 2.5, 2.5, "C")]
        mol = Molecule(atoms=atoms, pbc=pbc)
        centers, hist = pore_size_distribution(mol, n_samples=5000, n_bins=50, max_pore=10.0)
        # All bin centers are >= 0
        assert np.all(centers >= 0)
        # No probability in negative range (there are no negative bins anyway)
        assert np.all(hist >= 0)
