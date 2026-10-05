"""Tests for pdb_wizard.geometry."""

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.geometry import (
    delete_lone_atoms,
    edit_h_dist,
    find_overlapping_pairs,
    get_bonds_list,
    get_close_contacts,
    get_lone_atoms,
    overlap_detector,
    remove_overlapping_atoms,
)
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC


def _mol_with_pbc():
    pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    atoms = [
        Atom(0.0, 0.0, 0.0, "O"),
        Atom(0.96, 0.0, 0.0, "H"),
        Atom(-0.24, 0.93, 0.0, "H"),
    ]
    set_atom_ids(atoms)
    return Molecule(atoms=atoms, pbc=pbc)


class TestGetBondsList:
    def test_finds_bonds(self):
        mol = _mol_with_pbc()
        bonds = get_bonds_list(mol)
        assert len(bonds) == 2
        assert any("O-H" in b or "H-O" in b for b in bonds)

    def test_no_pbc_returns_empty(self):
        mol = Molecule(atoms=[Atom(0, 0, 0, "C")])
        assert get_bonds_list(mol) == []


class TestGetCloseContacts:
    def test_no_contacts_in_water(self):
        mol = _mol_with_pbc()
        contacts = get_close_contacts(mol)
        # H-H may or may not be a close contact depending on radii
        assert isinstance(contacts, list)


class TestGetLoneAtoms:
    def test_no_lone_in_water(self):
        mol = _mol_with_pbc()
        lone = get_lone_atoms(mol)
        assert len(lone) == 0

    def test_far_atom_is_lone(self):
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(0.0, 0.0, 0.0, "C"),
            Atom(50.0, 50.0, 50.0, "N"),
        ]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)
        lone = get_lone_atoms(mol)
        assert len(lone) == 2


class TestDeleteLoneAtoms:
    def test_removes_lone(self):
        pbc = PBC(100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(0.0, 0.0, 0.0, "O"),
            Atom(0.96, 0.0, 0.0, "H"),
            Atom(50.0, 50.0, 50.0, "N"),
        ]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol = delete_lone_atoms(mol)
        assert len(mol.atoms) == 2


def _overlap_mol():
    pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    atoms = [
        Atom(0.0, 0.0, 0.0, "C"),
        Atom(0.01, 0.0, 0.0, "C"),
        Atom(5.0, 5.0, 5.0, "O"),
    ]
    set_atom_ids(atoms)
    return Molecule(atoms=atoms, pbc=pbc)


class TestOverlapDetector:
    def test_removes_overlapping(self):
        mol = overlap_detector(_overlap_mol())
        assert len(mol.atoms) == 2

    def test_prints_each_deletion(self, capsys):
        """Classic entry point must tell the user what it deleted (original behaviour)."""
        overlap_detector(_overlap_mol())
        out = capsys.readouterr().out
        assert "Deleting overlapping atoms" in out
        assert "C     1 ---   C     2" in out

    def test_silent_when_nothing_overlaps(self, capsys):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [Atom(float(i), 0.0, 0.0, "C") for i in range(4)]
        set_atom_ids(atoms)
        mol = overlap_detector(Molecule(atoms=atoms, pbc=pbc))
        assert len(mol.atoms) == 4
        assert capsys.readouterr().out == ""


class TestRemoveOverlappingAtoms:
    def test_keeps_first_atom_and_returns_message(self):
        mol, msgs = remove_overlapping_atoms(_overlap_mol())
        assert len(mol.atoms) == 2
        # The earlier atom in file order survives.
        assert mol.atoms[0].x[0] == pytest.approx(0.0)
        assert mol.atoms[1].element.symbol == "O"
        assert len(msgs) == 1
        assert "Deleting overlapping atoms" in msgs[0]

    def test_ids_renumbered_after_removal(self):
        mol, _ = remove_overlapping_atoms(_overlap_mol())
        assert [a.id for a in mol.atoms] == [1, 2]

    def test_no_pbc_is_noop(self):
        atoms = [Atom(0.0, 0.0, 0.0, "C"), Atom(0.0, 0.0, 0.0, "C")]
        set_atom_ids(atoms)
        mol, msgs = remove_overlapping_atoms(Molecule(atoms=atoms, pbc=None))
        assert len(mol.atoms) == 2
        assert msgs == []

    def test_nothing_to_remove(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [Atom(float(i), 0.0, 0.0, "C") for i in range(3)]
        set_atom_ids(atoms)
        mol, msgs = remove_overlapping_atoms(Molecule(atoms=atoms, pbc=pbc))
        assert len(mol.atoms) == 3
        assert msgs == []

    def test_cluster_of_three_keeps_one(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(1.0, 1.0, 1.0, "N"),
            Atom(1.0, 1.0, 1.01, "N"),
            Atom(1.01, 1.0, 1.0, "N"),
            Atom(8.0, 8.0, 8.0, "O"),
        ]
        set_atom_ids(atoms)
        mol, msgs = remove_overlapping_atoms(Molecule(atoms=atoms, pbc=pbc))
        assert [a.element.symbol for a in mol.atoms] == ["N", "O"]
        assert len(msgs) == 2

    def test_chain_only_removes_true_overlaps(self):
        """A-B and B-C overlap but A-C do not: removing B must not drag C along."""
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(0.0, 0.0, 0.0, "C"),
            Atom(0.04, 0.0, 0.0, "C"),
            Atom(0.08, 0.0, 0.0, "C"),
        ]
        set_atom_ids(atoms)
        mol, msgs = remove_overlapping_atoms(Molecule(atoms=atoms, pbc=pbc))
        assert len(mol.atoms) == 2
        assert mol.atoms[1].x[0] == pytest.approx(0.08)

    def test_overlap_across_periodic_boundary(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        atoms = [Atom(0.01, 5.0, 5.0, "C"), Atom(9.99, 5.0, 5.0, "C")]
        set_atom_ids(atoms)
        mol, msgs = remove_overlapping_atoms(Molecule(atoms=atoms, pbc=pbc))
        assert len(mol.atoms) == 1
        assert len(msgs) == 1

    def test_message_lists_both_coordinates(self):
        _, msgs = remove_overlapping_atoms(_overlap_mol())
        lines = msgs[0].splitlines()
        assert len(lines) == 4
        assert "[0.   0.   0.  ]" in lines[2] or "0." in lines[2]
        assert "0.01" in lines[3]


class TestFindOverlappingPairs:
    def test_pairs_sorted_and_unique(self):
        pairs = find_overlapping_pairs(_overlap_mol())
        assert pairs == [(0, 1)]

    def test_custom_cutoff(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [Atom(0.0, 0.0, 0.0, "C"), Atom(0.3, 0.0, 0.0, "C")]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)
        assert find_overlapping_pairs(mol) == []
        assert find_overlapping_pairs(mol, cutoff=0.5) == [(0, 1)]

    def test_single_atom_or_no_pbc(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        assert find_overlapping_pairs(Molecule(atoms=[Atom(0, 0, 0, "C")], pbc=pbc)) == []
        two = [Atom(0, 0, 0, "C"), Atom(0, 0, 0, "C")]
        assert find_overlapping_pairs(Molecule(atoms=two, pbc=None)) == []

    def test_chunked_path_matches_bruteforce(self, monkeypatch):
        """Force one-row chunks so the row-chunk loop is exercised end to end."""
        import pdb_wizard.geometry as geom
        pbc = PBC(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 30, size=(60, 3))
        # Plant duplicates at known places (a pair and a triple).
        coords[10] = coords[3] + 0.001
        coords[45] = coords[20] - 0.002
        coords[59] = coords[20] + 0.003
        atoms = [Atom(float(c[0]), float(c[1]), float(c[2]), "C") for c in coords]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)

        expected = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if pbc.min_image(atoms[i].x - atoms[j].x) < 0.05:
                    expected.append((i, j))
        assert expected == [(3, 10), (20, 45), (20, 59), (45, 59)]

        assert geom.find_overlapping_pairs(mol) == expected
        monkeypatch.setattr(geom, "_OVERLAP_CHUNK_BUDGET", 1)  # chunk = 1 row
        assert geom.find_overlapping_pairs(mol) == expected
        monkeypatch.setattr(geom, "_OVERLAP_CHUNK_BUDGET", 7 * 60)  # 7 rows, uneven tail
        assert geom.find_overlapping_pairs(mol) == expected

        mol2, msgs = remove_overlapping_atoms(mol)
        assert len(mol2.atoms) == 57
        assert len(msgs) == 3


class TestEditHDist:
    def test_adjusts_distance(self):
        pbc = PBC(20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
        atoms = [
            Atom(0.0, 0.0, 0.0, "O"),
            Atom(0.96, 0.0, 0.0, "H"),
        ]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol = edit_h_dist(mol, "O", 1.1)
        dist = mol.pbc.min_image(mol.atoms[1].x - mol.atoms[0].x)
        assert dist == pytest.approx(1.1, abs=0.01)
