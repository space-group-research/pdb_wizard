"""Tests for supercell reduction."""

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.reduce_cell import identify_framework_and_sorbates, reduce_to_primitive


def _make_supercell_with_sorbate():
    """Create a 2x1x1 supercell of 2 framework atoms + 1 sorbate."""
    pbc = PBC(20, 10, 10, 90, 90, 90)  # 2x1x1 of a 10x10x10 cell

    # Framework: atom at (2,2,2) and its periodic copy at (12,2,2)
    atoms = [
        Atom(2.0, 2.0, 2.0, "Cu"),   # framework site 1, cell (0,0,0)
        Atom(12.0, 2.0, 2.0, "Cu"),  # framework site 1, cell (1,0,0)
        Atom(5.0, 5.0, 5.0, "Cu"),   # framework site 2, cell (0,0,0)
        Atom(15.0, 5.0, 5.0, "Cu"),  # framework site 2, cell (1,0,0)
        # Sorbate: H2 molecule at (7, 3, 3) — only in one cell, no copy
        Atom(7.0, 3.0, 3.0, "H"),
        Atom(7.7, 3.0, 3.0, "H"),
    ]
    mol = Molecule(atoms=atoms, pbc=pbc)
    mol.detect_bonds()
    return mol


class TestIdentifyFrameworkSorbates:
    def test_framework_detection(self):
        mol = _make_supercell_with_sorbate()
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        # 2 unique framework sites
        assert len(fw) == 2

    def test_sorbate_detection(self):
        mol = _make_supercell_with_sorbate()
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        # The two H atoms should be grouped as sorbate
        sorbate_atoms = []
        for s in sorbs:
            sorbate_atoms.extend(s)
        h_atoms = [i for i in sorbate_atoms if mol.atoms[i].element.symbol == "H"]
        assert len(h_atoms) == 2

    def test_no_pbc_returns_empty(self):
        mol = Molecule(atoms=[Atom(0, 0, 0, "C")], pbc=None)
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        assert fw == []
        assert sorbs == []

    def test_pure_framework(self):
        """All atoms are framework — no sorbates."""
        pbc = PBC(20, 10, 10, 90, 90, 90)
        atoms = [
            Atom(2.0, 2.0, 2.0, "Cu"),
            Atom(12.0, 2.0, 2.0, "Cu"),
        ]
        mol = Molecule(atoms=atoms, pbc=pbc)
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        assert len(fw) == 1
        assert len(sorbs) == 0


class TestReduceToPrimitive:
    def test_reduces_atom_count(self):
        mol = _make_supercell_with_sorbate()
        result = reduce_to_primitive(mol, 2, 1, 1)
        # 2 framework + 2 sorbate H = 4 atoms (from 6)
        assert len(result.atoms) == 4

    def test_cell_parameters_halved(self):
        mol = _make_supercell_with_sorbate()
        result = reduce_to_primitive(mol, 2, 1, 1)
        assert abs(result.pbc.a - 10.0) < 0.01
        assert abs(result.pbc.b - 10.0) < 0.01

    def test_keep_no_sorbates(self):
        mol = _make_supercell_with_sorbate()
        result = reduce_to_primitive(mol, 2, 1, 1, keep_sorbate_indices=[])
        # Only framework atoms
        assert len(result.atoms) == 2
        assert all(a.element.symbol == "Cu" for a in result.atoms)

    def test_keep_selected_sorbates(self):
        mol = _make_supercell_with_sorbate()
        fw, sorbs = identify_framework_and_sorbates(mol, 2, 1, 1)
        # Keep only the first sorbate molecule
        result = reduce_to_primitive(mol, 2, 1, 1, keep_sorbate_indices=[0])
        h_count = sum(1 for a in result.atoms if a.element.symbol == "H")
        assert h_count > 0

    def test_atoms_inside_primitive_cell(self):
        mol = _make_supercell_with_sorbate()
        result = reduce_to_primitive(mol, 2, 1, 1)
        # All fractional coords should be in [0, 1)
        for atom in result.atoms:
            frac = atom.x @ result.pbc.reciprocal_basis_matrix
            assert all(0 <= f < 1.01 for f in frac), f"Atom outside cell: frac={frac}"


class TestReduceHKUST1:
    def test_extend_then_reduce_roundtrip(self):
        """Extend a cell then reduce it — should get back original atom count."""
        import os

        from pdb_wizard.io import read_file
        from pdb_wizard.operations import extend_axis
        path = os.path.join(os.path.dirname(__file__), "..", "HKUST-1.pdb")
        if not os.path.exists(path):
            pytest.skip("HKUST-1.pdb not found")
        mol = read_file(path)
        original_n = len(mol.atoms)
        # Extend to 3x along a (times=2 adds 2 copies: 1+2=3x)
        import copy
        mol_copy = copy.deepcopy(mol)
        extended = extend_axis(mol_copy, axis=0, times=2)
        assert len(extended.atoms) == original_n * 3
        # Reduce back
        reduced = reduce_to_primitive(extended, 3, 1, 1, keep_sorbate_indices=[])
        # Should be back to original count (pure framework)
        assert len(reduced.atoms) == original_n
