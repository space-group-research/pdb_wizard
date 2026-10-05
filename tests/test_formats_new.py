"""Tests for new format readers/writers in pdb_wizard.formats."""

import io
import textwrap

import numpy as np

from pdb_wizard.atom import Atom
from pdb_wizard.formats import (
    read_gaussian_com,
    read_gaussian_log,
    read_mol2,
    read_sdf,
    write_lammps_data,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

# ---------------------------------------------------------------------------
# Gaussian log reader
# ---------------------------------------------------------------------------

GAUSSIAN_LOG = textwrap.dedent("""\
    Entering Gaussian System
    ...
                             Standard orientation:
     ---------------------------------------------------------------------
     Center     Atomic      Atomic             Coordinates (Angstroms)
     Number     Number       Type             X           Y           Z
     ---------------------------------------------------------------------
          1          8           0        0.000000    0.000000    0.117300
          2          1           0        0.000000    0.757200   -0.469200
          3          1           0        0.000000   -0.757200   -0.469200
     ---------------------------------------------------------------------
     Rotational constants (GHZ):    795.399     401.557     266.899
    ...
                             Standard orientation:
     ---------------------------------------------------------------------
     Center     Atomic      Atomic             Coordinates (Angstroms)
     Number     Number       Type             X           Y           Z
     ---------------------------------------------------------------------
          1          8           0        0.000000    0.000000    0.119300
          2          1           0        0.000000    0.762200   -0.477100
          3          1           0        0.000000   -0.762200   -0.477100
     ---------------------------------------------------------------------
     Normal termination of Gaussian.
""")


class TestReadGaussianLog:
    def test_reads_last_orientation_block(self):
        f = io.StringIO(GAUSSIAN_LOG)
        atoms, pbc = read_gaussian_log(f)
        assert len(atoms) == 3
        assert pbc is None
        # Should be the LAST block's coordinates
        assert atoms[0].element.symbol == "O"
        assert atoms[1].element.symbol == "H"
        assert atoms[2].element.symbol == "H"
        np.testing.assert_allclose(atoms[0].x, [0.0, 0.0, 0.119300], atol=1e-6)
        np.testing.assert_allclose(atoms[1].x, [0.0, 0.762200, -0.477100], atol=1e-6)

    def test_reads_input_orientation(self):
        log = textwrap.dedent("""\
                                 Input orientation:
             ---------------------------------------------------------------------
             Center     Atomic      Atomic             Coordinates (Angstroms)
             Number     Number       Type             X           Y           Z
             ---------------------------------------------------------------------
                  1          6           0        0.000000    0.000000    0.000000
                  2          1           0        1.089000    0.000000    0.000000
             ---------------------------------------------------------------------
        """)
        atoms, pbc = read_gaussian_log(io.StringIO(log))
        assert len(atoms) == 2
        assert atoms[0].element.symbol == "C"
        assert atoms[1].element.symbol == "H"

    def test_empty_file(self):
        atoms, pbc = read_gaussian_log(io.StringIO(""))
        assert atoms == []
        assert pbc is None

    def test_no_orientation_block(self):
        atoms, pbc = read_gaussian_log(io.StringIO("No orientation here\n"))
        assert atoms == []

    def test_atom_ids_are_set(self):
        f = io.StringIO(GAUSSIAN_LOG)
        atoms, _ = read_gaussian_log(f)
        # set_atom_ids assigns sequential ids starting from 1
        ids = [a.id for a in atoms]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Gaussian com/gjf reader
# ---------------------------------------------------------------------------

GAUSSIAN_COM = textwrap.dedent("""\
    %chk=water.chk
    %mem=2gb
    #p b3lyp/6-31g(d) opt

    Water optimization

    0 1
    O  0.000000  0.000000  0.117300
    H  0.000000  0.757200 -0.469200
    H  0.000000 -0.757200 -0.469200

""")


class TestReadGaussianCom:
    def test_reads_atoms(self):
        atoms, pbc = read_gaussian_com(io.StringIO(GAUSSIAN_COM))
        assert len(atoms) == 3
        assert pbc is None
        assert atoms[0].element.symbol == "O"
        assert atoms[1].element.symbol == "H"
        assert atoms[2].element.symbol == "H"
        np.testing.assert_allclose(atoms[0].x, [0.0, 0.0, 0.117300], atol=1e-6)

    def test_no_link0(self):
        com = textwrap.dedent("""\
            #p hf/sto-3g

            Title

            0 1
            C  0.0  0.0  0.0
            H  1.0  0.0  0.0
        """)
        atoms, pbc = read_gaussian_com(io.StringIO(com))
        assert len(atoms) == 2
        assert atoms[0].element.symbol == "C"
        assert atoms[1].element.symbol == "H"

    def test_multiline_route(self):
        com = textwrap.dedent("""\
            %chk=test.chk
            #p b3lyp/6-311+g(d,p)
               opt freq

            Title card

            0 1
            N  0.0  0.0  0.0
        """)
        atoms, _ = read_gaussian_com(io.StringIO(com))
        assert len(atoms) == 1
        assert atoms[0].element.symbol == "N"

    def test_atom_ids_are_set(self):
        atoms, _ = read_gaussian_com(io.StringIO(GAUSSIAN_COM))
        assert [a.id for a in atoms] == [1, 2, 3]


# ---------------------------------------------------------------------------
# SDF/MOL reader
# ---------------------------------------------------------------------------

SDF_CONTENT = textwrap.dedent("""\
    methane
         RDKit          3D

      5  4  0  0  0  0  0  0  0  0999 V2000
        0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.6294    0.6294    0.6294 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.6294   -0.6294    0.6294 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.6294    0.6294   -0.6294 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.6294   -0.6294   -0.6294 H   0  0  0  0  0  0  0  0  0  0  0  0
      1  2  1  0
      1  3  1  0
      1  4  1  0
      1  5  1  0
    M  END
""")


class TestReadSdf:
    def test_reads_atoms(self):
        atoms, pbc = read_sdf(io.StringIO(SDF_CONTENT))
        assert len(atoms) == 5
        assert pbc is None
        assert atoms[0].element.symbol == "C"
        for i in range(1, 5):
            assert atoms[i].element.symbol == "H"

    def test_coordinates(self):
        atoms, _ = read_sdf(io.StringIO(SDF_CONTENT))
        np.testing.assert_allclose(atoms[0].x, [0.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(atoms[1].x, [0.6294, 0.6294, 0.6294], atol=1e-4)

    def test_too_few_lines(self):
        atoms, pbc = read_sdf(io.StringIO("line1\nline2\nline3\n"))
        assert atoms == []
        assert pbc is None

    def test_atom_ids_are_set(self):
        atoms, _ = read_sdf(io.StringIO(SDF_CONTENT))
        assert [a.id for a in atoms] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# MOL2 reader
# ---------------------------------------------------------------------------

MOL2_CONTENT = textwrap.dedent("""\
    @<TRIPOS>MOLECULE
    water
     3 2 0 0 0
    SMALL
    GASTEIGER

    @<TRIPOS>ATOM
          1 O1         0.0000    0.0000    0.1173 O.3     1  LIG1       -0.4116
          2 H2         0.0000    0.7572   -0.4692 H       1  LIG1        0.2058
          3 H3         0.0000   -0.7572   -0.4692 H       1  LIG1        0.2058
    @<TRIPOS>BOND
         1     1     2    1
         2     1     3    1
""")


class TestReadMol2:
    def test_reads_atoms(self):
        atoms, pbc = read_mol2(io.StringIO(MOL2_CONTENT))
        assert len(atoms) == 3
        assert pbc is None
        assert atoms[0].element.symbol == "O"
        assert atoms[1].element.symbol == "H"
        assert atoms[2].element.symbol == "H"

    def test_coordinates(self):
        atoms, _ = read_mol2(io.StringIO(MOL2_CONTENT))
        np.testing.assert_allclose(atoms[0].x, [0.0, 0.0, 0.1173], atol=1e-4)
        np.testing.assert_allclose(atoms[1].x, [0.0, 0.7572, -0.4692], atol=1e-4)

    def test_element_from_atom_type_with_dot(self):
        """Atom type like 'C.3' should yield element 'C'."""
        mol2 = textwrap.dedent("""\
            @<TRIPOS>ATOM
                  1 C1         1.0000    2.0000    3.0000 C.3     1  RES        0.0000
                  2 N1         4.0000    5.0000    6.0000 N.am    1  RES        0.0000
            @<TRIPOS>BOND
        """)
        atoms, _ = read_mol2(io.StringIO(mol2))
        assert len(atoms) == 2
        assert atoms[0].element.symbol == "C"
        assert atoms[1].element.symbol == "N"

    def test_no_atom_section(self):
        mol2 = "@<TRIPOS>MOLECULE\ntest\n"
        atoms, pbc = read_mol2(io.StringIO(mol2))
        assert atoms == []
        assert pbc is None

    def test_atom_ids_are_set(self):
        atoms, _ = read_mol2(io.StringIO(MOL2_CONTENT))
        assert [a.id for a in atoms] == [1, 2, 3]

    def test_atom_section_at_eof(self):
        """ATOM section that ends at EOF (no trailing @<TRIPOS> section)."""
        mol2 = textwrap.dedent("""\
            @<TRIPOS>ATOM
                  1 O1         0.0  0.0  0.0 O.3     1  LIG  -0.4
        """)
        atoms, _ = read_mol2(io.StringIO(mol2))
        assert len(atoms) == 1
        assert atoms[0].element.symbol == "O"


# ---------------------------------------------------------------------------
# LAMMPS data file writer
# ---------------------------------------------------------------------------


class TestWriteLammpsData:
    def _make_water(self, pbc=None):
        atoms = [
            Atom(0.0, 0.0, 0.117, "O"),
            Atom(0.0, 0.757, -0.469, "H"),
            Atom(0.0, -0.757, -0.469, "H"),
        ]
        mol = Molecule(atoms=atoms, pbc=pbc)
        mol.bonds = [(0, 1), (0, 2)]
        return mol

    def test_header_counts(self):
        mol = self._make_water()
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        assert "3 atoms" in text
        assert "2 bonds" in text
        assert "2 atom types" in text

    def test_masses_section(self):
        mol = self._make_water()
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        assert "Masses" in text
        # O mass ~ 15.9994
        assert "15.9994" in text
        # H mass ~ 1.0080
        assert "1.0080" in text

    def test_atoms_section_full_style(self):
        mol = self._make_water()
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        # Find the Atoms section
        lines = text.split("\n")
        atom_start = None
        for i, line in enumerate(lines):
            if line.strip() == "Atoms":
                atom_start = i + 2  # skip blank line after header
                break
        assert atom_start is not None
        # First atom: id=1, mol_id=1, type=1 (O), charge=0.0, x y z
        atom_line = lines[atom_start]
        tokens = atom_line.split()
        assert tokens[0] == "1"   # id
        assert tokens[1] == "1"   # mol-id
        assert tokens[2] == "1"   # type (O is first)

    def test_bonds_section(self):
        mol = self._make_water()
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        assert "Bonds" in text
        # Bond 1: atoms 1-2 (1-indexed)
        lines = text.split("\n")
        bond_start = None
        for i, line in enumerate(lines):
            if line.strip() == "Bonds":
                bond_start = i + 2
                break
        assert bond_start is not None
        tokens = lines[bond_start].split()
        assert tokens[0] == "1"   # bond id
        assert tokens[1] == "1"   # bond type
        assert tokens[2] == "1"   # atom1 (0+1)
        assert tokens[3] == "2"   # atom2 (1+1)

    def test_with_pbc(self):
        pbc = PBC(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        mol = self._make_water(pbc=pbc)
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        assert "0.000000 10.000000 xlo xhi" in text
        assert "0.000000 10.000000 ylo yhi" in text
        assert "0.000000 10.000000 zlo zhi" in text

    def test_without_pbc_auto_box(self):
        mol = self._make_water(pbc=None)
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        # Should contain xlo xhi with auto-computed margins
        assert "xlo xhi" in text
        assert "ylo yhi" in text
        assert "zlo zhi" in text

    def test_no_bonds(self):
        atoms = [Atom(0.0, 0.0, 0.0, "C")]
        mol = Molecule(atoms=atoms)
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        assert "1 atoms" in text
        assert "bonds" not in text.lower().split("atoms")[0]  # no "N bonds" in header
        assert "Bonds" not in text

    def test_charge_written(self):
        atoms = [Atom(0.0, 0.0, 0.0, "Na")]
        atoms[0].charge = 1.0
        mol = Molecule(atoms=atoms)
        out = io.StringIO()
        write_lammps_data(mol, out)
        text = out.getvalue()
        # The atom line should contain charge 1.000000
        lines = text.split("\n")
        for line in lines:
            if line.strip().startswith("1 1 1"):
                assert "1.000000" in line
                break
