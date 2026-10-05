"""Tests for tab system — ensures no cross-communication between tabs."""

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.viewer import TabState


def _make_mol(n_atoms=3, charge=0.0):
    atoms = [Atom(float(i), 0, 0, "C") for i in range(n_atoms)]
    for a in atoms:
        a.charge = charge
    pbc = PBC(10, 10, 10, 90, 90, 90)
    mol = Molecule(atoms=atoms, pbc=pbc)
    mol.detect_bonds()
    return mol


class TestTabStateIsolation:
    def test_separate_molecules(self):
        mol1 = _make_mol(3, charge=1.0)
        mol2 = _make_mol(5, charge=2.0)
        tab1 = TabState(mol1, "file1.pdb")
        tab2 = TabState(mol2, "file2.pdb")
        assert len(tab1.molecule.atoms) == 3
        assert len(tab2.molecule.atoms) == 5
        assert tab1.molecule.atoms[0].charge == 1.0
        assert tab2.molecule.atoms[0].charge == 2.0

    def test_separate_undo_stacks(self):
        tab1 = TabState(_make_mol(3), "a.pdb")
        tab2 = TabState(_make_mol(5), "b.pdb")
        tab1.undo_stack.append("state1")
        assert len(tab1.undo_stack) == 1
        assert len(tab2.undo_stack) == 0

    def test_separate_redo_stacks(self):
        tab1 = TabState(_make_mol(), "a.pdb")
        tab2 = TabState(_make_mol(), "b.pdb")
        tab1.redo_stack.append("r1")
        tab2.redo_stack.append("r2")
        tab2.redo_stack.append("r3")
        assert len(tab1.redo_stack) == 1
        assert len(tab2.redo_stack) == 2

    def test_separate_view_state(self):
        tab1 = TabState(_make_mol(), "a.pdb")
        tab2 = TabState(_make_mol(), "b.pdb")
        tab1.camera_distance = 10.0
        tab2.camera_distance = 50.0
        tab1.pan_x = 5.0
        tab2.pan_x = -5.0
        assert tab1.camera_distance == 10.0
        assert tab2.camera_distance == 50.0
        assert tab1.pan_x == 5.0
        assert tab2.pan_x == -5.0

    def test_separate_frames(self):
        tab1 = TabState(_make_mol(), "a.pdb", frames=[_make_mol(), _make_mol()])
        tab2 = TabState(_make_mol(), "b.pdb")
        assert tab1.frames is not None
        assert len(tab1.frames) == 2
        assert tab2.frames is None

    def test_name(self):
        tab = TabState(_make_mol(), "/path/to/structure.pdb")
        assert tab.name == "structure.pdb"

    def test_default_view_state(self):
        tab = TabState(_make_mol(), "test.pdb")
        assert tab.dark_bg is True
        assert tab.show_bonds is True
        assert tab.show_atom_numbers is False
        assert tab.color_by_charge is False
        assert tab.hide_water is False
        assert tab.ribbon is False
        assert tab.licorice is False
        assert tab.vdw is False

    def test_modifying_one_tab_doesnt_affect_other(self):
        mol1 = _make_mol(3)
        mol2 = _make_mol(3)
        tab1 = TabState(mol1, "a.pdb")
        tab2 = TabState(mol2, "b.pdb")

        # Modify tab1's molecule
        tab1.molecule.atoms[0].x = np.array([99.0, 99.0, 99.0])
        tab1.molecule.atoms[0].charge = 999.0

        # tab2 should be unaffected
        assert tab2.molecule.atoms[0].x[0] == 0.0
        assert tab2.molecule.atoms[0].charge == 0.0

    def test_frame_idx_independent(self):
        tab1 = TabState(_make_mol(), "a.pdb")
        tab2 = TabState(_make_mol(), "b.pdb")
        tab1.frame_idx = 42
        tab2.frame_idx = 7
        assert tab1.frame_idx == 42
        assert tab2.frame_idx == 7
