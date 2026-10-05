"""PDB reader dialect detection: standard PDB, pdb2pqr PQR, and MPMC PDB/PQR.

The original single-file tool used fixed columns; the port used whitespace
tokens so MPMC files (which ignore PDB columns) would load. Each alone
misreads the other dialect. These tests pin the per-line detection that
serves both, using the project's real MPMC sample files as the guard.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from pdb_wizard.io import (
    MPMC_MIN_TOKENS,
    _parse_pdb_atom,
    read_file,
    read_pdb,
    write_mpmc_pdb,
    write_standard_pdb,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ["HKUST-1.pdb", "HKUST-1_out_mpmc.pdb", "Ni-MOF-74.pqr", "WOFOUR-1-Ni.pqr"]


def _atom(line: str, pqr: bool = False):
    return _parse_pdb_atom(line, pqr=pqr)


# ---------------------------------------------------------------------------
# standard PDB
# ---------------------------------------------------------------------------

class TestStandardPdb:
    def test_element_column_shifted_by_one_still_read(self):
        """Many writers put the symbol one column early; must not become 'U'."""
        a = _atom("HETATM    1  Cu  MOF A   1     -7.257  -7.257  -7.257  1.00  0.00"
                  "          Cu")
        assert a.element.symbol == "Cu"
        b = _atom("HETATM    1  C   MOF A   1     -7.257  -7.257  -7.257  1.00  0.00"
                  "            C")
        assert b.element.symbol == "C"

    def test_with_chain_id(self):
        a = _atom("ATOM      1  C1  UNK A   1       1.000   2.000   3.000  1.00  0.00           C")
        assert list(a.x) == [1.0, 2.0, 3.0]
        assert a.element.symbol == "C"

    def test_without_chain_id_is_not_shifted(self):
        """Used to come back as (y, z, occupancy)."""
        a = _atom("HETATM    1  C1  UNK     1       1.000   2.000   3.000  1.00  0.00           C")
        assert list(a.x) == [1.0, 2.0, 3.0]

    def test_resseq_fused_to_chain(self):
        a = _atom("ATOM   1234  O   HOH A1234       4.500  -5.250   6.125  1.00 20.00           O")
        assert list(a.x) == [4.5, -5.25, 6.125]

    def test_packed_negative_coordinates(self):
        """Columns touch when values are wide: '-100.123-100.456' must still parse."""
        a = _atom("ATOM      1  C   MOL A   1    -100.123-100.456 -99.000  1.00  0.00           C")
        assert a.x == pytest.approx([-100.123, -100.456, -99.0])

    def test_b_factor_is_not_a_charge(self):
        a = _atom("ATOM      1  C   MOL A   1       1.000   2.000   3.000  1.00 23.45           C")
        assert a.charge == 0.0

    def test_formal_charge_columns(self):
        a = _atom("HETATM    1 ZN    ZN A   1       1.000   2.000   3.000  1.00  0.00"
                  "          ZN2+")
        assert a.element.symbol == "Zn"
        assert a.charge == 2.0
        b = _atom("HETATM    1 CL    CL A   1       1.000   2.000   3.000  1.00  0.00"
                  "          CL1-")
        assert b.charge == -1.0

    def test_element_column_beats_name_inference(self):
        ca = _atom("ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00"
                   "           C")
        ho = _atom("ATOM      3  HO  SER A   1       1.000   2.000   3.000  1.00  0.00"
                   "           H")
        assert ca.element.symbol == "C", "alpha carbon, not calcium"
        assert ho.element.symbol == "H", "hydroxyl H, not holmium"
        assert ca.name == "CA" and ho.name == "HO"

    def test_missing_element_column_falls_back_to_name(self):
        a = _atom("ATOM      1  CU  MOL A   1       1.000   2.000   3.000")
        assert a.element.symbol == "Cu"

    def test_misaligned_columns_fall_back_to_tokens(self):
        a = _atom("ATOM 1 C1 UNK 1 1.000 2.000 3.000 1.00 0.00 C")
        assert list(a.x) == [1.0, 2.0, 3.0]
        assert a.element.symbol == "C"

    def test_short_line_without_coordinates_raises(self):
        with pytest.raises((ValueError, IndexError)):
            _atom("ATOM      1  C1  UNK     1")


# ---------------------------------------------------------------------------
# pdb2pqr
# ---------------------------------------------------------------------------

class TestPdb2PqrDialect:
    LINE = "ATOM      1  N   MET A   1      27.340  24.430   2.614 -0.3000 1.8240"

    def test_charge_from_token_after_z(self):
        a = _atom(self.LINE, pqr=True)
        assert a.x == pytest.approx([27.34, 24.43, 2.614])
        assert a.charge == pytest.approx(-0.3)

    def test_radius_is_not_charge(self):
        a = _atom(self.LINE, pqr=True)
        assert a.charge != pytest.approx(1.824)

    def test_read_pdb_passes_pqr_flag_by_suffix(self):
        atoms, _ = read_pdb(io.StringIO(self.LINE + "\n"), "protein.pqr")
        assert atoms[0].charge == pytest.approx(-0.3)
        atoms, _ = read_pdb(io.StringIO(self.LINE + "\n"), "protein.pdb")
        assert atoms[0].charge == 0.0


# ---------------------------------------------------------------------------
# MPMC
# ---------------------------------------------------------------------------

class TestMpmcDialect:
    FREEFORM = ("ATOM   1 C1 MOF F 1 -10.504740   8.631261  -4.539737  12.011000   "
                "0.916790   1.288600  52.840000   3.431000")
    SIXTEEN = ("ATOM      1 NI   MOF F    1      3.062 -10.583  -9.317 58.69340  1.50150  "
               "2.94650  7.55330  2.52480  0.00000  0.00000")
    NINETEEN = ("ATOM      1 Cu   MOF F    1    -12.315   6.469  -4.844  63.54600   1.24091   "
                "2.11000   2.51610   3.11369 0.0 0.0      6.97      262.8    13951.497")

    def test_min_tokens_constant(self):
        assert MPMC_MIN_TOKENS == 14
        assert len(self.FREEFORM.split()) == 14

    def test_free_form_fourteen_tokens(self):
        a = _atom(self.FREEFORM)
        assert a.x == pytest.approx([-10.50474, 8.631261, -4.539737])
        assert a.element.symbol == "C" and a.name == "C1"
        assert a.charge == pytest.approx(0.91679)
        assert (a.alpha, a.epsilon, a.sigma) == pytest.approx((1.2886, 52.84, 3.431))

    def test_sixteen_tokens(self):
        a = _atom(self.SIXTEEN)
        assert a.element.symbol == "Ni"
        assert a.charge == pytest.approx(1.5015)
        assert a.sigma == pytest.approx(2.5248)

    def test_nineteen_tokens_with_dispersion(self):
        a = _atom(self.NINETEEN)
        assert a.charge == pytest.approx(1.24091)
        assert (a.c6, a.c8, a.c10) == pytest.approx((6.97, 262.8, 13951.497))

    def test_pqr_suffix_does_not_change_mpmc_parsing(self):
        atoms, _ = read_pdb(io.StringIO(self.SIXTEEN + "\n"), "mof.pqr")
        assert atoms[0].charge == pytest.approx(1.5015)

    def test_box_corners_dropped_but_other_unknowns_kept(self, caplog):
        text = (
            "ATOM      1 C    MOF F    1      1.000   1.000   1.000  12.01100   0.10000   "
            "1.00000   1.00000   3.00000 0.0 0.0 0.0 0.0 0.0\n"
            "ATOM      2 X    BOX F    2      0.000   0.000   0.000 0.0 0.0 0.0 0.0 0.0\n"
            "ATOM      3 EP   MOL A   1       2.000   2.000   2.000  1.00  0.00\n"
        )
        with caplog.at_level(logging.WARNING, logger="pdb_wizard.io"):
            atoms, _ = read_pdb(io.StringIO(text), "t.pdb")
        assert [a.name for a in atoms] == ["C", "EP"]
        assert atoms[1].element.symbol == "X"
        assert any("Unrecognised element" in r.getMessage() and "EP" in r.getMessage()
                   for r in caplog.records)

    def test_unparsable_line_is_reported_not_silent(self, caplog):
        text = ("ATOM      1  C1  UNK     1\n"
                "ATOM      2  C   MOL A   1       1.000   2.000   3.000\n")
        with caplog.at_level(logging.WARNING, logger="pdb_wizard.io"):
            atoms, _ = read_pdb(io.StringIO(text), "t.pdb")
        assert len(atoms) == 1
        assert any("Skipped 1 unparsable atom line" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# the project's real MPMC files are the regression guard
# ---------------------------------------------------------------------------

def _expected_from_tokens(path: Path):
    """Independent token-based reading of an MPMC file for cross-checking."""
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        t = line.split()
        if t[3] == "BOX":
            continue
        atoms.append((t[2], float(t[6]), float(t[7]), float(t[8]), float(t[10])))
    return atoms


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_mpmc_files_load_exactly(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    expected = _expected_from_tokens(path)
    mol = read_file(str(path), remove_overlaps=False)
    assert len(mol.atoms) == len(expected)
    for a, (nm, x, y, z, q) in zip(mol.atoms, expected):
        assert a.name == nm
        assert a.x == pytest.approx([x, y, z])
        assert a.charge == pytest.approx(q)


def test_standard_pdb_round_trip_wide_coordinates(tmp_path):
    """write_standard_pdb output (fixed columns, possibly packed) reads back exactly."""
    from pdb_wizard.atom import Atom
    from pdb_wizard.molecule import Molecule, set_atom_ids
    from pdb_wizard.pbc import PBC
    atoms = [Atom(-100.123, -100.456, 99.5, "C"), Atom(0.5, 0.5, 0.5, "O"),
             Atom(1.0, 1.0, 1.0, "H")]
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC(300, 300, 300, 90, 90, 90))
    out = tmp_path / "wide.pdb"
    write_standard_pdb(mol, str(out), skip_mols_step=True)
    back = read_file(str(out), remove_overlaps=False)
    assert len(back.atoms) == 3
    assert back.atoms[0].x == pytest.approx([-100.123, -100.456, 99.5])
    assert [a.element.symbol for a in back.atoms] == ["C", "O", "H"]


def test_mpmc_round_trip_still_exact(tmp_path):
    from pdb_wizard.atom import Atom
    from pdb_wizard.molecule import Molecule, set_atom_ids
    from pdb_wizard.pbc import PBC
    atoms = [Atom(1, 1, 1, "Cu"), Atom(3, 1, 1, "O")]
    atoms[0].charge, atoms[1].charge = 1.2, -0.6
    atoms[0].sigma = 2.5
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC(10, 10, 10, 90, 90, 90))
    out = tmp_path / "m.pqr"
    write_mpmc_pdb(mol, str(out))
    back = read_file(str(out), remove_overlaps=False)
    assert [(a.element.symbol, a.charge) for a in back.atoms] == [("Cu", 1.2), ("O", -0.6)]
    assert back.atoms[0].sigma == pytest.approx(2.5)
