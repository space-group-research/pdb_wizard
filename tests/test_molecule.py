"""Tests for pdb_wizard.molecule."""

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC


def _water_mol():
    o = Atom(0.0, 0.0, 0.0, "O")
    h1 = Atom(0.96, 0.0, 0.0, "H")
    h2 = Atom(-0.24, 0.93, 0.0, "H")
    return Molecule(atoms=[o, h1, h2], bonds=[(0, 1), (0, 2)])


class TestMoleculeBasics:
    def test_center(self):
        mol = _water_mol()
        c = mol.center()
        assert c.shape == (3,)
        assert c[0] == pytest.approx(0.24, abs=0.01)

    def test_radius(self):
        mol = _water_mol()
        r = mol.radius()
        assert r > 0

    def test_empty_molecule(self):
        mol = Molecule(atoms=[], bonds=[])
        np.testing.assert_array_equal(mol.center(), [0, 0, 0])
        assert mol.radius() == 1.0


class TestBondLengths:
    def test_water_bond_lengths(self):
        mol = _water_mol()
        lengths = mol.get_bond_lengths()
        assert len(lengths) == 2
        for i, j, d in lengths:
            assert d == pytest.approx(0.96, abs=0.02)


class TestAngles:
    def test_water_angle(self):
        mol = _water_mol()
        angles = mol.get_angles()
        assert len(angles) > 0
        for i, j, k, angle in angles:
            assert 100 < angle < 115


class TestDihedrals:
    def test_ethane_like(self):
        atoms = [
            Atom(0, 0, 0, "C"),
            Atom(1.54, 0, 0, "C"),
            Atom(-0.5, 0.9, 0, "H"),
            Atom(2.04, -0.9, 0, "H"),
        ]
        mol = Molecule(
            atoms=atoms,
            bonds=[(0, 1), (0, 2), (1, 3)],
        )
        dihedrals = mol.get_dihedrals()
        assert len(dihedrals) > 0


class TestDetectBonds:
    def test_without_pbc(self):
        o = Atom(0.0, 0.0, 0.0, "O")
        h1 = Atom(0.96, 0.0, 0.0, "H")
        h2 = Atom(5.0, 5.0, 5.0, "H")
        mol = Molecule(atoms=[o, h1, h2])
        mol.detect_bonds()
        assert (0, 1) in mol.bonds
        assert (0, 2) not in mol.bonds

    def test_with_pbc(self):
        pbc = PBC(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        a1 = Atom(0.1, 0.0, 0.0, "C")
        a2 = Atom(4.5, 0.0, 0.0, "C")
        mol = Molecule(atoms=[a1, a2], pbc=pbc)
        mol.detect_bonds()
        assert len(mol.bonds) > 0


class TestFindMolecules:
    def test_two_separate_molecules(self):
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        a1 = Atom(0.0, 0.0, 0.0, "O")
        a2 = Atom(0.96, 0.0, 0.0, "H")
        a3 = Atom(50.0, 50.0, 50.0, "O")
        a4 = Atom(50.96, 50.0, 50.0, "H")
        mol = Molecule(atoms=[a1, a2, a3, a4], pbc=pbc)
        submols = mol.find_molecules()
        assert len(submols) == 2
        assert len(submols[0].atoms) == 2
        assert len(submols[1].atoms) == 2

    def test_without_pbc(self):
        a1 = Atom(0.0, 0.0, 0.0, "O")
        a2 = Atom(0.96, 0.0, 0.0, "H")
        a3 = Atom(50.0, 50.0, 50.0, "N")
        mol = Molecule(atoms=[a1, a2, a3])
        submols = mol.find_molecules()
        assert len(submols) >= 2


class TestSetAtomIds:
    def test_sequential_ids(self):
        atoms = [Atom(0, 0, 0, "C"), Atom(1, 0, 0, "H")]
        set_atom_ids(atoms)
        assert atoms[0].id == 1
        assert atoms[1].id == 2


class TestPBCAwareBondLengths:
    def test_pbc_bond_length(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        a1 = Atom(0.5, 0.0, 0.0, "C")
        a2 = Atom(9.5, 0.0, 0.0, "C")
        mol = Molecule(atoms=[a1, a2], bonds=[(0, 1)], pbc=pbc)
        lengths = mol.get_bond_lengths()
        assert len(lengths) == 1
        assert lengths[0][2] == pytest.approx(1.0, rel=1e-6)
