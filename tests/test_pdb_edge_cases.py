"""Edge cases for the structure readers and writers.

Lines are built at exact PDB columns with ``_std_line`` so each test states
precisely which column convention it exercises.
"""

from __future__ import annotations

import io
import logging

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.io import (
    _parse_pdb_atom,
    check_pdb_trajectory,
    check_xyz_trajectory,
    is_fallback_cell,
    read_file,
    read_file_trajectory,
    read_pdb,
    read_pdb_trajectory,
    read_xyz,
    write_mpmc_pdb,
    write_standard_pdb,
)
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC


def _std_line(
    name="C", resname="MOL", chain="A", resseq=1, x=1.0, y=2.0, z=3.0,
    occ=1.0, bfac=0.0, segid="", element="C", charge="", record="ATOM", serial=1,
    icode=" ", altloc=" ",
) -> str:
    """A standard PDB ATOM line with every field at its specification column."""
    name_field = f" {name:<3}" if len(name) < 4 else name
    return (
        f"{record:<6}{serial:>5} {name_field}{altloc}{resname:>3} {chain:1}{resseq:>4}{icode}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfac:>6.2f}      {segid:<4}{element:>2}{charge:<2}"
    )


def _mpmc_line(name="Cu", mol="MOF", flag="F", molid=1, x=1.0, y=2.0, z=3.0, mass=63.546,
               charge=1.2, alpha=2.11, eps=2.5161, sigma=3.11369, extra="") -> str:
    return (f"ATOM      1 {name:<4} {mol:<3} {flag}    {molid}    {x:>7.3f} {y:>7.3f} {z:>7.3f} "
            f"{mass:>9.5f} {charge:>9.5f} {alpha:>9.5f} {eps:>9.5f} {sigma:>9.5f}{extra}")


def _parse(line: str, **kw):
    return _parse_pdb_atom(line, **kw)


# ---------------------------------------------------------------------------
# standard PDB columns
# ---------------------------------------------------------------------------

class TestStandardColumns:
    def test_spec_layout_sanity(self):
        line = _std_line()
        assert line[30:38] == "   1.000" and line[76:78] == " C"

    def test_formal_charge_in_cols_79_80(self):
        a = _parse(_std_line(name="ZN", element="ZN", charge="2+"))
        assert a.element.symbol == "Zn" and a.charge == 2.0

    def test_segid_element_and_charge_make_14_tokens_but_stay_standard(self):
        line = _std_line(name="N", resname="MET", segid="PROT", element="N", charge="1+",
                         bfac=23.45) + " X"
        assert len(line.split()) == 14
        a = _parse(line)
        assert list(a.x) == [1.0, 2.0, 3.0]
        assert a.charge == 1.0, "formal charge, not the B-factor"

    def test_b_factor_never_becomes_charge(self):
        a = _parse(_std_line(bfac=99.99, charge=""))
        assert a.charge == 0.0

    def test_altloc_and_four_char_name(self):
        a = _parse(_std_line(name="HD21", altloc="A", resname="ASN", element="H"))
        assert a.name == "HD21" and a.element.symbol == "H"

    def test_insertion_code(self):
        a = _parse(_std_line(resseq=52, icode="B", x=-7.5))
        assert a.x[0] == pytest.approx(-7.5)

    def test_negative_zero(self):
        a = _parse(_std_line(x=-0.0, y=0.0, z=-0.0))
        assert list(a.x) == [0.0, 0.0, 0.0]

    def test_hybrid36_serial_does_not_matter(self):
        line = _std_line().replace("    1 ", "A0000 ", 1)
        a = _parse(line)
        assert list(a.x) == [1.0, 2.0, 3.0]

    def test_deuterium_maps_to_hydrogen(self):
        a = _parse(_std_line(name="D", element="D"))
        assert a.element.symbol == "H"

    def test_unknown_element_column_falls_back_to_name(self):
        a = _parse(_std_line(name="CU", element="Qq"))
        assert a.element.symbol == "Cu"

    def test_element_column_lowercase(self):
        a = _parse(_std_line(name="CU", element="cu"))
        assert a.element.symbol == "Cu"

    def test_line_truncated_after_coordinates(self):
        a = _parse(_std_line(name="CU", element="")[:54])
        assert list(a.x) == [1.0, 2.0, 3.0]
        assert a.element.symbol == "Cu"

    def test_crlf_line_endings(self):
        a = _parse(_std_line(element="C") + "\r\n")
        assert list(a.x) == [1.0, 2.0, 3.0] and a.element.symbol == "C"


class TestWideAndPackedCoordinates:
    def test_packed_two_wide_values(self):
        a = _parse("ATOM      1  C   MOL A   1   -1000.000-1000.000 -999.000  1.00  0.00"
                   "           C")
        assert a.x == pytest.approx([-1000.0, -1000.0, -999.0])

    def test_packed_three_wide_values(self):
        a = _parse("ATOM      1  C   MOL A   1   -1000.000-1000.000-1000.000  1.00  0.00"
                   "           C")
        assert a.x == pytest.approx([-1000.0, -1000.0, -1000.0])

    def test_free_form_packed(self):
        a = _parse("ATOM 1 C1 UNK 1 -1000.000-1000.000 -999.000 1.00 0.00 C")
        assert a.x == pytest.approx([-1000.0, -1000.0, -999.0])

    def test_serial_overflow_with_packed(self):
        a = _parse("HETATM99999  O   MOF A   1    100.123-100.456 200.789  1.00  0.00           O")
        assert a.x == pytest.approx([100.123, -100.456, 200.789])

    def test_sign_flip_overflow_left_is_caught(self):
        """x = -1234.500 (9 chars) starts in column 30; slice 31-38 would read +1234.5."""
        line = "ATOM      1  C   MOL A   1   -1234.500 2345.250-3456.125  1.00  0.00           C"
        a = _parse(line)
        assert a.x == pytest.approx([-1234.5, 2345.25, -3456.125])

    def test_writer_uses_fixed_columns(self):
        atoms = [Atom(-12.3156, 0.0, 100.0, "C")]
        set_atom_ids(atoms)
        buf = io.StringIO()
        write_standard_pdb(Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90)), buf,
                           skip_mols_step=True)
        line = next(ln for ln in buf.getvalue().splitlines() if ln.startswith("HETATM"))
        assert line[30:54] == " -12.316   0.000 100.000"

    def test_write_then_read_wide_box(self, tmp_path):
        atoms = [Atom(-1234.5, 2345.25, -3456.125, "C"), Atom(0, 0, 0, "O")]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC(5000, 5000, 5000, 90, 90, 90))
        out = tmp_path / "wide.pdb"
        write_standard_pdb(mol, str(out), skip_mols_step=True)
        back = read_file(str(out), remove_overlaps=False)
        assert back.atoms[0].x == pytest.approx([-1234.5, 2345.25, -3456.125])


class TestDegenerateLines:
    @pytest.mark.parametrize("line", ["ATOM", "ATOM      1", "ATOM      1  C1  UNK     1",
                                      "ATOM 1 C1 UNK 1 1 2 3"])
    def test_lines_without_coordinates_are_skipped_and_reported(self, line, caplog):
        text = line + "\n" + _std_line() + "\n"
        with caplog.at_level(logging.WARNING, logger="pdb_wizard.io"):
            atoms, _ = read_pdb(io.StringIO(text), "t.pdb")
        assert len(atoms) == 1
        assert any("Skipped 1 unparsable" in r.getMessage() for r in caplog.records)

    def test_blank_and_short_lines_ignored(self):
        text = "\n\n  \nAT\n" + _std_line() + "\nEND\n"
        atoms, _ = read_pdb(io.StringIO(text))
        assert len(atoms) == 1

    def test_header_only_file(self):
        text = "HEADER    TEST\nCRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1\nEND\n"
        atoms, pbc = read_pdb(io.StringIO(text))
        assert atoms == [] and pbc is not None and pbc.a == 10.0

    def test_malformed_cryst1_gives_no_cell(self):
        text = "CRYST1   abc   10.000   10.000  90.00  90.00  90.00 P 1\n" + _std_line() + "\n"
        atoms, pbc = read_pdb(io.StringIO(text))
        assert len(atoms) == 1 and pbc is None


# ---------------------------------------------------------------------------
# MPMC dialect
# ---------------------------------------------------------------------------

class TestMpmcEdgeCases:
    def test_full_record(self):
        a = _parse(_mpmc_line())
        assert a.element.symbol == "Cu" and a.charge == pytest.approx(1.2)
        assert a.sigma == pytest.approx(3.11369)

    def test_long_atom_type_shifts_columns_harmlessly(self):
        a = _parse(_mpmc_line(name="Cu_paddle", x=-12.3))
        assert a.name == "Cu_paddle" and a.x[0] == pytest.approx(-12.3)
        assert a.charge == pytest.approx(1.2)

    @pytest.mark.parametrize("flag", ["F", "M", "A", "S", "T"])
    def test_all_mpmc_flags(self, flag):
        a = _parse(_mpmc_line(flag=flag))
        assert a.charge == pytest.approx(1.2)

    def test_crlf(self):
        a = _parse(_mpmc_line() + "\r\n")
        assert a.charge == pytest.approx(1.2)

    def test_short_records_in_a_file_with_box_markers(self):
        text = (
            "ATOM      1 H2G  H2  M    1      1.000   2.000   3.000   2.01600  -0.74640\n"
            "ATOM      2 H2E  H2  M    1      1.370   2.000   3.000   0.00000   0.37320\n"
            "ATOM      3 X    BOX F    2      0.000   0.000   0.000 0.0 0.0 0.0 0.0 0.0\n"
        )
        atoms, _ = read_pdb(io.StringIO(text), "h2.pqr")
        assert [(a.name, a.charge) for a in atoms] == [("H2G", -0.7464), ("H2E", 0.3732)]

    def test_short_records_with_basis_remark(self):
        text = (
            "REMARK BOX BASIS[0]  10.0 0.0 0.0\n"
            "ATOM      1 H2G  H2  M    1      1.000   2.000   3.000   2.01600  -0.74640\n"
        )
        atoms, _ = read_pdb(io.StringIO(text), "h2.pdb")
        assert atoms[0].charge == pytest.approx(-0.7464)

    def test_one_full_record_marks_whole_file(self):
        text = _mpmc_line() + "\n" + \
            "ATOM      2 H2G  H2  M    1      1.000   2.000   3.000   2.01600  -0.74640\n"
        atoms, _ = read_pdb(io.StringIO(text), "mix.pdb")
        assert atoms[1].charge == pytest.approx(-0.7464)

    def test_short_record_without_markers_is_standard(self):
        """Without any MPMC marker an 11-token line is a standard PDB line."""
        text = "ATOM      1  C   MOL A   1       1.000   2.000   3.000  1.00 23.45\n"
        atoms, _ = read_pdb(io.StringIO(text), "plain.pdb")
        assert atoms[0].charge == 0.0

    def test_missing_params_default_to_zero(self):
        text = "ATOM      1 X    BOX F    2      0.000   0.000   0.000 0.0 0.0 0.0 0.0 0.0\n" \
               "ATOM      2 C    MOF F    1      1.000   2.000   3.000  12.011 0.5\n"
        atoms, _ = read_pdb(io.StringIO(text), "t.pqr")
        assert len(atoms) == 1
        a = atoms[0]
        assert a.charge == 0.5 and a.alpha == a.epsilon == a.sigma == 0.0

    def test_box_corners_dropped_regardless_of_flag(self):
        text = _mpmc_line() + "\n" + \
               "ATOM      2 X    BOX F    2      0.000   0.000   0.000 0.0 0.0 0.0 0.0 0.0\n" + \
               "ATOM      3 X    BOX M    2      1.000   0.000   0.000 0.0 0.0 0.0 0.0 0.0\n"
        atoms, _ = read_pdb(io.StringIO(text))
        assert len(atoms) == 1

    def test_write_long_names_and_read_back(self, tmp_path):
        atoms = [Atom(1, 1, 1, "Cu_pad"), Atom(3, 1, 1, "O")]
        atoms[0].charge = 1.5
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
        out = tmp_path / "long.pqr"
        write_mpmc_pdb(mol, str(out))
        back = read_file(str(out), remove_overlaps=False)
        assert len(back.atoms) == 2
        assert back.atoms[0].charge == pytest.approx(1.5)

    def test_many_atoms_ids_stay_distinct(self, tmp_path):
        """Serial field is 6 wide in the MPMC writer; ids past 99999 must not merge."""
        n = 100_005
        import numpy as np
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 500, size=(n, 3))
        atoms = [Atom(float(c[0]), float(c[1]), float(c[2]), "C") for c in coords]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC(500, 500, 500, 90, 90, 90))
        buf = io.StringIO()
        write_mpmc_pdb(mol, buf)
        atom_lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("ATOM")]
        serials = [int(ln.split()[1]) for ln in atom_lines]
        assert serials == list(range(1, n + 8 + 1))


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

_CRYST = "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"


def _frame(x, cryst=False):
    head = _CRYST if cryst else ""
    return (head + _std_line(x=x) + "\n"
            + _std_line(name="O", element="O", x=x + 1.2, serial=2) + "\n")


class TestTrajectoryEdgeCases:
    def test_crlf_end_separated(self, tmp_path):
        p = tmp_path / "crlf.pdb"
        p.write_bytes((_frame(0) + "END\n" + _frame(1) + "END\n").replace("\n", "\r\n").encode())
        assert check_pdb_trajectory(str(p)) is True
        frames = read_file_trajectory(str(p))
        assert len(frames) == 2

    def test_single_frame_with_conect_and_master_after_end(self, tmp_path):
        p = tmp_path / "one.pdb"
        p.write_text(_frame(0) + "TER\nEND\nCONECT    1    2\nMASTER\nEND\n")
        assert check_pdb_trajectory(str(p)) is False

    def test_cryst1_only_in_first_frame_is_inherited(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text(_frame(0, cryst=True) + "END\n" + _frame(1) + "END\n" + _frame(2) + "END\n")
        with open(p) as f:
            mols, pbcs = read_pdb_trajectory(f)
        assert len(mols) == 3
        assert all(m.pbc is not None and m.pbc.a == 10.0 for m in mols)
        assert not any(is_fallback_cell(m.pbc) for m in mols)

    def test_endmdl_without_model_records(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text(_frame(0) + "ENDMDL\n" + _frame(1) + "ENDMDL\n")
        assert check_pdb_trajectory(str(p)) is True
        assert len(read_file_trajectory(str(p))) == 2

    def test_model_records_without_endmdl(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text("MODEL 1\n" + _frame(0) + "MODEL 2\n" + _frame(1) + "END\n")
        assert len(read_file_trajectory(str(p))) == 2

    def test_empty_model_blocks_do_not_create_frames(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text("MODEL 1\nENDMDL\nMODEL 2\n" + _frame(0) + "ENDMDL\nMODEL 3\n"
                     + _frame(1) + "ENDMDL\n")
        assert len(read_file_trajectory(str(p))) == 2

    def test_frames_with_different_atom_counts_are_all_read(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text(_frame(0) + "END\n" + _std_line(x=5) + "\nEND\n")
        frames = read_file_trajectory(str(p))
        assert [len(f.atoms) for f in frames] == [2, 1]

    def test_single_structure_read_file_takes_first_frame_only(self, tmp_path):
        p = tmp_path / "t.pdb"
        p.write_text(_frame(0) + "END\n" + _frame(1) + "END\n")
        mol = read_file(str(p), remove_overlaps=False)
        assert len(mol.atoms) == 2 and mol.atoms[0].x[0] == 0.0


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------

class TestXyzEdgeCases:
    def test_blank_comment_line(self):
        atoms, pbc = read_xyz(io.StringIO("2\n\nC 0 0 0\nO 1.2 0 0\n"))
        assert len(atoms) == 2 and pbc is None

    def test_text_comment_line(self):
        atoms, pbc = read_xyz(io.StringIO("2\nwater dimer step 5\nC 0 0 0\nO 1.2 0 0\n"))
        assert len(atoms) == 2 and pbc is None

    def test_cell_line_with_six_numbers(self):
        _, pbc = read_xyz(io.StringIO("1\n10 11 12 90 90 120\nC 0 0 0\n"))
        assert pbc is not None and pbc.gamma == 120.0

    def test_extended_xyz_lattice_comment_is_not_a_cell(self):
        _, pbc = read_xyz(io.StringIO('1\nLattice="10 0 0 0 10 0 0 0 10"\nC 0 0 0\n'))
        assert pbc is None  # unsupported syntax must not produce a bogus cell

    def test_fifth_column_read_as_charge(self):
        atoms, _ = read_xyz(io.StringIO("1\n\nC 0 0 0 -0.25\n"))
        assert atoms[0].charge == pytest.approx(-0.25)

    def test_crlf(self):
        atoms, pbc = read_xyz(io.StringIO("1\r\n10 10 10 90 90 90\r\nC 0 0 0\r\n"))
        assert len(atoms) == 1 and pbc is not None

    def test_trailing_blank_lines_are_not_a_second_frame(self, tmp_path):
        p = tmp_path / "t.xyz"
        p.write_text("1\n\nC 0 0 0\n\n\n")
        assert check_xyz_trajectory(str(p)) is False

    def test_atomic_number_symbols(self):
        atoms, _ = read_xyz(io.StringIO("2\n\n6 0 0 0\n8 1.2 0 0\n"))
        assert [a.element.symbol for a in atoms] == ["C", "O"]


# ---------------------------------------------------------------------------
# real MPMC samples survive a trip through the standard PDB writer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["HKUST-1.pdb", "HKUST-1_out_mpmc.pdb", "Ni-MOF-74.pqr"])
def test_sample_files_round_trip_through_standard_pdb(name, tmp_path):
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / name
    if not src.exists():
        pytest.skip(f"{name} not present")
    mol = read_file(str(src), remove_overlaps=False)
    if mol.pbc is None:
        mol.pbc = PBC(50, 50, 50, 90, 90, 90)
    out = tmp_path / "std.pdb"
    write_standard_pdb(mol, str(out), skip_mols_step=True)
    back = read_file(str(out), remove_overlaps=False)
    assert len(back.atoms) == len(mol.atoms)
    assert [a.element.symbol for a in back.atoms] == [a.element.symbol for a in mol.atoms]
    for a, b in zip(mol.atoms, back.atoms):
        assert b.x == pytest.approx(a.x, abs=5e-4)
