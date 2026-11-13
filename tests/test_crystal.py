"""CRYSTAL23 input/output and the symmetry ranking.

The input conventions are the CRYSTAL manual's ("Geometry input for crystalline
compounds"): IFLAG IFHR IFSO, the space group number, the minimal set of cell parameters
for the crystal system, then the asymmetric unit as atomic number and fractional
coordinates of the centred conventional cell. Two of the manual's own examples are
reproduced here (MgO, and silicon in its first origin setting).

Needs spglib for the symmetry parts (skipped without it); P1 writing and the reader do not.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from pdb_wizard import crystal as cr
from pdb_wizard.analyses import symmetry_text
from pdb_wizard.atom import Atom
from pdb_wizard.io import detect_filetype, next_mpmc_mol_id, read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC
from pdb_wizard.sorbates import SORBATE_MODELS, insert_sorbates, sorbate_atoms

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
needs_spglib = pytest.mark.skipif(not cr.has_spglib(), reason="needs spglib")
FCC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]


def crystal_of(cell, sites) -> Molecule:
    """sites: (symbol, fractional x, y, z)."""
    pbc = PBC(*cell)
    atoms = [Atom(*(np.array(f) @ pbc.basis_matrix), sym) for sym, *f in sites]
    return Molecule(atoms=atoms, pbc=pbc)


def mgo() -> Molecule:
    sites = [("Mg", *f) for f in FCC] + [("O", *(np.array(f) + 0.5)) for f in FCC]
    return crystal_of((4.21, 4.21, 4.21, 90, 90, 90), sites)


def silicon(shift=0.0) -> Molecule:
    sites = [("Si", *(np.array(b) + np.array(f) + shift))
             for b in ((0, 0, 0), (0.25, 0.25, 0.25)) for f in FCC]
    return crystal_of((5.43, 5.43, 5.43, 90, 90, 90), sites)


def block(text: str) -> list:
    return text.splitlines()


def atoms_of(text: str) -> list:
    lines = block(text)
    return [(int(t[0]), *map(float, t[1:])) for t in (ln.split() for ln in lines[5:])]


# ---- the lattice line -------------------------------------------------------------------

CELL = (5.0, 6.0, 7.0, 80.0, 95.0, 100.0)


@pytest.mark.parametrize("number,expected", [
    (1, "5.000000 6.000000 7.000000 80.000000 95.000000 100.000000"),
    (2, "5.000000 6.000000 7.000000 80.000000 95.000000 100.000000"),
    (3, "5.000000 6.000000 7.000000 95.000000"), (15, "5.000000 6.000000 7.000000 95.000000"),
    (16, "5.000000 6.000000 7.000000"), (74, "5.000000 6.000000 7.000000"),
    (75, "5.000000 7.000000"), (142, "5.000000 7.000000"),
    (143, "5.000000 7.000000"), (167, "5.000000 7.000000"), (194, "5.000000 7.000000"),
    (195, "5.000000"), (230, "5.000000"),
])
def test_minimal_cell_parameters_per_crystal_system(number, expected):
    assert cr.crystal_lattice_line(number, CELL) == expected


# ---- P1 (no spglib needed) ----------------------------------------------------------------


def test_p1_geometry_lists_every_atom():
    mol = read_file(str(HKUST))
    lines = block(cr.crystal_geometry(mol))
    assert lines[:3] == ["CRYSTAL", "0 0 0", "1"]
    assert lines[3] == "26.343000 26.343000 26.343000 90.000000 90.000000 90.000000"
    assert lines[4] == "624" and len(lines) == 5 + 624
    atoms = atoms_of(cr.crystal_geometry(mol))
    assert sorted({z for z, *_ in atoms}) == [1, 6, 8, 29]
    frac = np.array([a[1:] for a in atoms])
    assert frac.min() >= 0 and frac.max() < 1
    back = frac @ mol.pbc.basis_matrix - np.array([a.x for a in mol.atoms])
    back = back @ mol.pbc.reciprocal_basis_matrix
    assert np.abs(back - np.round(back)).max() < 1e-8  # same atoms, modulo the lattice


def test_p1_leaves_out_massless_model_sites():
    mol = read_file(str(HKUST))
    placed = insert_sorbates(mol, SORBATE_MODELS["h2_bss"], 3, seed=1)
    mol.atoms.extend(sorbate_atoms(SORBATE_MODELS["h2_bss"], placed, next_mpmc_mol_id(mol)))
    lines = block(cr.crystal_geometry(mol))
    assert lines[4] == str(624 + 3 * 2)  # the two H nuclei of each H2, not its 3 dummy sites


def test_sorbate_sites_are_written_as_their_real_elements():
    """MPMC names CO2's sites COG/COE/CON; they are C, O, O and two massless sites, not Co."""
    mol = read_file(str(HKUST))
    for model, expected in (("co2_phast", [6, 8, 8]), ("ch4_9site", [1, 1, 1, 1, 6]),
                            ("h2o_tip4p", [1, 1, 8]), ("n2_polar", [7, 7])):
        packed = read_file(str(HKUST))
        placed = insert_sorbates(packed, SORBATE_MODELS[model], 1, seed=2)
        packed.atoms.extend(sorbate_atoms(SORBATE_MODELS[model], placed, 2))
        atoms = atoms_of(cr.crystal_geometry(packed))
        assert sorted(z for z, *_ in atoms[624:]) == expected, model
    assert len(atoms_of(cr.crystal_geometry(mol))) == 624


def test_p1_needs_a_cell_and_atoms():
    with pytest.raises(ValueError, match="unit cell"):
        cr.crystal_geometry(Molecule(atoms=[Atom(0, 0, 0, "C")]))
    with pytest.raises(ValueError, match="No atoms"):
        cr.crystal_geometry(Molecule(atoms=[], pbc=PBC(5, 5, 5, 90, 90, 90)))


def test_full_d12_is_the_group_template():
    out = io.StringIO()
    cr.write_crystal_d12(mgo(), out)
    text = out.getvalue()
    lines = text.splitlines()
    assert "cell optimization, HSE06/POB-TZVP-REV2 (P1)" in lines[0]
    assert lines[1:4] == ["CRYSTAL", "0 0 0", "1"]
    assert text.endswith(cr.CRYSTAL_D12_TAIL) and lines[-1] == "END"
    tail = cr.CRYSTAL_D12_TAIL.split()
    assert tail[:3] == ["OPTGEOM", "FULLOPTG", "ENDOPT"]
    assert "POB-TZVP-REV2" in tail and "HSE06" in tail
    out = io.StringIO()
    cr.write_crystal_d12(mgo(), out, title="my job\nsecond line")
    assert out.getvalue().splitlines()[0] == "my job"


# ---- the manual's examples ------------------------------------------------------------------


@needs_spglib
def test_mgo_matches_the_manual_input():
    """Manual: CRYSTAL / 0 0 0 / 225 / 4.21 / 2 / 12 0. 0. 0. / 8 0.5 0.5 0.5"""
    red = cr.reduction_for_group(mgo(), 225)
    lines = block(cr.crystal_geometry(mgo(), red))
    assert lines[:5] == ["CRYSTAL", "0 0 0", "225", "4.210000", "2"]
    (z1, *mg), (z2, *o) = atoms_of("\n".join(lines))
    assert (z1, mg) == (12, [0.0, 0.0, 0.0]) and z2 == 8
    # (0.5, 0, 0) is (0.5, 0.5, 0.5) plus a face-centring translation
    shifted = (np.array(o) - np.array([0.5, 0.5, 0.5])) % 1.0
    assert any(np.allclose(shifted, f) for f in FCC)
    assert red.n_conventional == 8 and red.origin_flag == 0


@needs_spglib
@pytest.mark.parametrize("shift", [0.0, 0.125, 0.3])
def test_silicon_is_written_in_the_first_origin_setting(shift):
    """Manual, group 227: Si is at (0, 0, 0) in the 1st setting and (1/8, 1/8, 1/8) in the
    2nd, which is CRYSTAL's default. spglib standardises to the 1st, so IFSO must be 1,
    wherever the atoms sat in the file."""
    mol = silicon(shift)
    red = cr.reduction_for_group(mol, 227)
    lines = block(cr.crystal_geometry(mol, red))
    assert lines[:5] == ["CRYSTAL", "0 0 1", "227", "5.430000", "1"]
    (z, *xyz), = atoms_of("\n".join(lines))
    assert z == 14 and any(np.allclose(xyz, f) for f in FCC)  # (0,0,0) up to F-centring


@needs_spglib
def test_origin_flag_rules():
    assert cr._origin_flag(227, "1") == 1 and cr._origin_flag(227, "2") == 0
    assert cr._origin_flag(225, "") == 0 and cr._origin_flag(166, "H") == 0
    assert cr._origin_flag(14, "b1") == 0
    with pytest.raises(ValueError, match="unique-axis-b"):
        cr._origin_flag(14, "c1")
    with pytest.raises(ValueError, match="Rhombohedral axes"):
        cr._origin_flag(166, "R")


# ---- every crystal system -------------------------------------------------------------------


def expand(red) -> dict:
    """The conventional cell rebuilt from the asymmetric unit with the group's operators
    (from spglib's database, in the setting that was written). {atomic number: count}"""
    import spglib
    hall = next(h for h in range(1, 531)
                if spglib.get_spacegroup_type(h).number == red.number
                and (spglib.get_spacegroup_type(h).choice == "2") == (
                    red.origin_flag == 0 and _has_two_origins(red.number)))
    ops = spglib.get_symmetry_from_database(hall)
    counts: dict = {}
    for z, x in zip(red.numbers, red.positions):
        images = (x @ np.transpose(ops["rotations"], (0, 2, 1)) + ops["translations"]) % 1.0
        unique = np.unique(np.round(images % 1.0, 5) % 1.0, axis=0)
        counts[z] = counts.get(z, 0) + len(unique)
    return counts


def _has_two_origins(number: int) -> bool:
    import spglib
    return any(spglib.get_spacegroup_type(h).number == number
               and spglib.get_spacegroup_type(h).choice == "2" for h in range(1, 531))


SYSTEMS = [
    ("triclinic", (5.0, 6.0, 7.0, 80.0, 95.0, 100.0), [("C", 0.1, 0.2, 0.3)], 2, 6),
    ("monoclinic", (5.0, 6.0, 7.0, 90.0, 105.0, 90.0), [("C", 0, 0, 0)], 10, 4),
    ("monoclinic, unique axis c in the file", (5.0, 6.0, 7.0, 90.0, 90.0, 105.0),
     [("C", 0, 0, 0)], 10, 4),
    ("orthorhombic", (5.0, 6.0, 7.0, 90.0, 90.0, 90.0), [("C", 0, 0, 0)], 47, 3),
    ("tetragonal", (5.0, 5.0, 7.0, 90.0, 90.0, 90.0), [("C", 0, 0, 0)], 123, 2),
    ("hexagonal", (5.0, 5.0, 7.0, 90.0, 90.0, 120.0), [("C", 0, 0, 0)], 191, 2),
    ("rhombohedral", (5.0, 5.0, 5.0, 70.0, 70.0, 70.0), [("C", 0, 0, 0)], 166, 2),
    ("cubic", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), [("C", 0, 0, 0)], 221, 1),
]


@needs_spglib
@pytest.mark.parametrize("name,cell,sites,number,n_params", SYSTEMS, ids=[s[0] for s in SYSTEMS])
def test_each_crystal_system(name, cell, sites, number, n_params):
    mol = crystal_of(cell, sites)
    found = cr.find_symmetry(mol)
    assert found[-1].number == number
    red = cr.reduce_symmetry(mol, found[-1].tolerance)
    lines = block(cr.crystal_geometry(mol, red))
    assert lines[2] == str(number) and len(lines[3].split()) == n_params
    assert expand(red) == {6: red.n_conventional}
    volume = PBC(*red.cell).volume if hasattr(PBC(*red.cell), "volume") else None
    primitive = abs(np.linalg.det(mol.pbc.basis_matrix))
    conventional = abs(np.linalg.det(PBC(*red.cell).basis_matrix))
    assert conventional / primitive == pytest.approx(red.n_conventional / len(sites), rel=1e-6)
    assert volume is None or volume == pytest.approx(conventional)


@needs_spglib
def test_monoclinic_angle_is_beta_whatever_axis_the_file_used():
    for cell in ((5.0, 6.0, 7.0, 90.0, 105.0, 90.0), (5.0, 6.0, 7.0, 90.0, 90.0, 105.0),
                 (5.0, 6.0, 7.0, 105.0, 90.0, 90.0)):
        mol = crystal_of(cell, [("C", 0, 0, 0)])
        red = cr.reduction_for_group(mol, 10)
        params = [float(v) for v in cr.crystal_lattice_line(10, red.cell).split()]
        assert len(params) == 4 and params[3] == pytest.approx(105.0)
        assert red.cell[3] == pytest.approx(90.0) and red.cell[5] == pytest.approx(90.0)


@needs_spglib
def test_rhombohedral_group_is_written_on_hexagonal_axes():
    mol = crystal_of((5.0, 5.0, 5.0, 70.0, 70.0, 70.0), [("C", 0, 0, 0)])
    red = cr.reduction_for_group(mol, 166)
    a, b, c, alpha, beta, gamma = red.cell
    assert a == pytest.approx(b) and (alpha, beta, gamma) == pytest.approx((90, 90, 120))
    assert red.n_conventional == 3 and len(red.numbers) == 1


# ---- HKUST-1 and the ranking ------------------------------------------------------------------


@needs_spglib
def test_hkust_is_fm3m_with_six_atoms():
    mol = read_file(str(HKUST))
    found = cr.find_symmetry(mol)
    best = next(c for c in found if c.suggested)
    assert (best.number, best.symbol, best.n_asymmetric) == (225, "Fm-3m", 6)
    assert best.max_drift < 0.01 and best.rms_drift <= best.max_drift
    assert sum(c.suggested for c in found) == 1
    red = cr.reduction_for_group(mol, 225)
    assert sorted(red.numbers) == [1, 6, 6, 6, 8, 29] and red.n_conventional == 624
    assert red.cell == pytest.approx((26.343, 26.343, 26.343, 90, 90, 90), abs=1e-6)
    assert expand(red) == {1: 96, 6: 288, 8: 192, 29: 48}  # the whole cell comes back
    lines = block(cr.crystal_geometry(mol, red))
    assert lines[:5] == ["CRYSTAL", "0 0 0", "225", "26.343000", "6"]


@needs_spglib
def test_ranking_lists_each_group_once_with_growing_tolerance():
    found = cr.find_symmetry(read_file(str(HKUST)))
    assert len({c.number for c in found}) == len(found)
    assert [c.tolerance for c in found] == sorted(c.tolerance for c in found)
    assert [c.n_operations for c in found] == sorted(c.n_operations for c in found)


@needs_spglib
def test_drift_measures_how_far_atoms_are_from_the_ideal_structure():
    mol = read_file(str(HKUST))
    rng = np.random.default_rng(0)
    for a in mol.atoms:
        a.x = a.x + rng.normal(0, 0.01, 3)
    found = cr.find_symmetry(mol)
    assert found[0].number == 1 and found[0].max_drift < 1e-6  # as given, it is only P1
    cubic = next(c for c in found if c.number == 225)
    assert cubic.tolerance >= 0.03  # not found until the tolerance covers the noise
    assert 0.01 < cubic.max_drift < 0.08 and 0.005 < cubic.rms_drift < 0.03
    assert cubic.suggested
    red = cr.reduce_symmetry(mol, cubic.tolerance)
    assert len(red.numbers) == 6 and red.max_drift == pytest.approx(cubic.max_drift)


@needs_spglib
def test_a_guest_breaks_the_symmetry_and_the_report_says_how_much():
    mol = read_file(str(HKUST))
    mol.atoms.append(Atom(3.1, 1.2, 0.4, "Ar"))
    found = cr.find_symmetry(mol)
    assert all(c.number != 225 for c in found)
    with pytest.raises(ValueError, match="was not found for this structure"):
        cr.reduction_for_group(mol, 225)


@needs_spglib
def test_report_text():
    text = symmetry_text(read_file(str(HKUST)))
    assert text.startswith("Space groups found for 624 atoms")
    row = next(ln for ln in text.splitlines() if ln.strip().startswith("225"))
    assert "Fm-3m" in row and "<- suggested" in row and row.split()[-3] == "6"
    assert symmetry_text(None) == "No atoms loaded"
    assert "needs a unit cell" in symmetry_text(Molecule(atoms=[Atom(0, 0, 0, "C")]))


# ---- choosing a group, and number 231 ---------------------------------------------------------


@needs_spglib
def test_space_group_choice():
    mol = read_file(str(HKUST))
    for p1 in ("", "1", " P1 ", "p 1"):
        assert cr.crystal_reduction_choice(mol, p1) is None
    assert cr.crystal_reduction_choice(mol, "225").number == 225
    for bad, message in (("abc", "Enter a space group number"), ("0", "numbered 1 to 230"),
                         ("999", "numbered 1 to 230"), ("14", "was not found")):
        with pytest.raises(ValueError, match=message):
            cr.crystal_reduction_choice(mol, bad)


def test_space_group_231_is_the_space_group():
    mol = read_file(str(HKUST))
    with pytest.raises(ValueError) as e:
        cr.crystal_reduction_choice(mol, "231")
    assert str(e.value) == cr.SPACE_GROUP_231
    assert "only 230" in cr.SPACE_GROUP_231 and "Space group at NC State" in cr.SPACE_GROUP_231
    assert cr.space_group_name(231) == cr.SPACE_GROUP_231
    with pytest.raises(ValueError, match="numbered 1 to 230"):
        cr.space_group_name(232)


@needs_spglib
def test_space_group_names():
    assert cr.space_group_name(225) == "Fm-3m" and cr.space_group_name(1) == "P1"
    with pytest.raises(ValueError, match="1 to 230"):
        cr.space_group_name(0)


def test_without_spglib_p1_still_works(monkeypatch):
    monkeypatch.setattr(cr, "_spglib", None)
    mol = read_file(str(HKUST))
    assert not cr.has_spglib() and symmetry_text(mol) == cr.SPGLIB_HINT
    assert cr.crystal_reduction_choice(mol, "1") is None
    with pytest.raises(ValueError, match="pip install spglib"):
        cr.crystal_reduction_choice(mol, "225")
    with pytest.raises(cr.SymmetryUnavailable):
        cr.find_symmetry(mol)
    with pytest.raises(cr.SymmetryUnavailable):
        cr.space_group_name(225)
    out = io.StringIO()
    cr.write_crystal_d12(mol, out)
    assert out.getvalue().splitlines()[3] == "1"


# ---- reading a CRYSTAL output -----------------------------------------------------------------

# The block CRYSTAL prints (format from the manual's MgO supercell example).
OUT_FIRST = """ PRIMITIVE CELL - CENTRING CODE 1/0 VOLUME=    18.654615 - DENSITY  3.559 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     2.97691955     2.97691955     2.97691955    60.000000  60.000000  60.000000
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2
     ATOM                 X/A                 Y/B                 Z/C
 *******************************************************************************
      1 T  12 MG    0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
      2 T   8 O    -5.000000000000E-01 -5.000000000000E-01 -5.000000000000E-01

 TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC CELL
"""
OUT_FINAL = """ FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3
 (NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500)
 *******************************************************************************
 LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM
 PRIMITIVE CELL - CENTRING CODE 1/0 VOLUME=    19.000000 - DENSITY  3.494 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     2.99520000     2.99520000     2.99520000    60.000000  60.000000  60.000000
 *******************************************************************************
 ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    2
     ATOM                 X/A                 Y/B                 Z/C
 *******************************************************************************
      1 T 212 MG    0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
      2 T   8 O    -5.000000000000E-01 -5.000000000000E-01 -5.000000000000E-01

 T = ATOM BELONGING TO THE ASYMMETRIC UNIT
"""


def test_reader_takes_the_last_geometry(tmp_path):
    atoms, pbc = cr.read_crystal_output(io.StringIO("header\n" + OUT_FIRST + "...\n" + OUT_FINAL))
    assert (pbc.a, pbc.alpha) == (2.9952, 60.0)  # the optimised cell, not the first one
    assert [a.element.symbol for a in atoms] == ["Mg", "O"]  # 212 is Mg with a basis tag
    assert np.allclose(atoms[0].x, 0)
    frac = atoms[1].x @ pbc.reciprocal_basis_matrix
    assert np.allclose(frac, [-0.5, -0.5, -0.5])
    path = tmp_path / "mgo.out"
    path.write_text(OUT_FIRST + OUT_FINAL)
    assert detect_filetype(str(path)) == "crystal_out"
    mol = read_file(str(path))
    assert len(mol.atoms) == 2 and mol.pbc.a == pytest.approx(2.9952)


def test_reader_single_point_and_errors():
    atoms, pbc = cr.read_crystal_output(io.StringIO(OUT_FIRST))
    assert len(atoms) == 2 and pbc.a == pytest.approx(2.97691955)
    with pytest.raises(ValueError, match="No CRYSTAL geometry found"):
        cr.read_crystal_output(io.StringIO("nothing here\n"))
    headless = OUT_FIRST.split(" ATOMS IN THE ASYMMETRIC UNIT")[0]
    with pytest.raises(ValueError, match="No CRYSTAL geometry found"):
        cr.read_crystal_output(io.StringIO(headless))


@needs_spglib
def test_optimised_primitive_cell_goes_back_to_a_reduced_input():
    """Read an output, find its symmetry again, write the next input."""
    atoms, pbc = cr.read_crystal_output(io.StringIO(OUT_FINAL))
    mol = Molecule(atoms=atoms, pbc=pbc)
    red = cr.reduction_for_group(mol, 225)
    lines = block(cr.crystal_geometry(mol, red))
    assert lines[2] == "225" and lines[4] == "2"
    assert float(lines[3]) == pytest.approx(2.9952 * np.sqrt(2), abs=1e-5)


# ---- the three front ends ---------------------------------------------------------------------


def classic(path: str, answers: list) -> str:
    import pdb_wizard.menu as menu
    printed: list = []
    it = iter(answers)

    def fake_input(prompt: str = "") -> str:
        printed.append(prompt)
        return next(it)

    with mock.patch("builtins.input", fake_input), \
            mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
        menu.run_classic(path)
    assert next(it, None) is None
    return "\n".join(printed)


@needs_spglib
def test_classic_menu_writes_reduced_and_p1_inputs(tmp_path):
    reduced, p1 = tmp_path / "r.d12", tmp_path / "p1.d12"
    text = classic(str(HKUST), ["13", "7", "231", "abc", "225", str(reduced),
                                "7", "", str(p1), "6", "9"])
    assert "<- suggested" in text and cr.SPACE_GROUP_231 in text
    assert "Enter a space group number" in text
    assert f"wrote {reduced} in space group 225 (Fm-3m), 6 atoms" in text
    assert f"wrote {p1} in P1" in text
    assert reduced.read_text().splitlines()[1:6] == ["CRYSTAL", "0 0 0", "225", "26.343000", "6"]
    assert p1.read_text().splitlines()[5] == "624"


@needs_spglib
def test_all_three_front_ends_write_the_same_input(tmp_path, request):
    pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pytestqt")
    pytest.importorskip("textual")
    qtbot = request.getfixturevalue("qtbot")
    from textual.widgets import Input, TextArea

    from pdb_wizard import gui
    from pdb_wizard.viewer import FileSaveModal, InfoModal, InputModal, PdbWizardApp

    out = {name: tmp_path / f"{name}.d12" for name in ("classic", "tui", "gui")}
    report = symmetry_text(read_file(str(HKUST)))

    text = classic(str(HKUST), ["13", "7", "225", str(out["classic"]), "6", "9"])
    assert report in text

    async def run_tui():
        app = PdbWizardApp(molecule=read_file(str(HKUST)), filepath=str(HKUST))
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()

            async def wait(cond):
                for _ in range(200):
                    await pilot.pause(0.05)
                    if cond():
                        return
                raise AssertionError("timed out")

            app.on_operations_panel_run_command(SimpleNamespace(command="symmetry"))
            await wait(lambda: isinstance(app.screen, InfoModal))
            shown = app.screen.query_one("#info-text", TextArea).text
            await pilot.click("#info-ok")
            await wait(lambda: not isinstance(app.screen, InfoModal))
            for answer in ("231", "225"):
                app.on_operations_panel_run_command(SimpleNamespace(command="write_crystal"))
                await wait(lambda: isinstance(app.screen, InputModal))
                app.screen.query_one("#modal-input", Input).value = answer
                await pilot.pause()
                await pilot.click("#input-ok")
                await pilot.pause()
            await wait(lambda: isinstance(app.screen, FileSaveModal))
            app.screen.query_one("#file-input", Input).value = str(out["tui"])
            await pilot.pause()
            await pilot.click("#file-save")
            await wait(out["tui"].exists)
            return shown, "\n".join(str(n.message) for n in app._notifications)

    shown, notes = asyncio.run(run_tui())
    assert shown == report and cr.SPACE_GROUP_231 in notes

    w = gui.MainWindow()
    w.warnings = []
    w._warn = w.warnings.append
    qtbot.addWidget(w)
    assert w.open_path(str(HKUST))
    assert w.open_tool("symmetry").text.toPlainText() == report
    assert not w.save_crystal(str(tmp_path / "no.d12"), "231")
    assert cr.SPACE_GROUP_231 in w.warnings[-1] and not (tmp_path / "no.d12").exists()
    assert w.save_crystal(str(out["gui"]), "225")
    assert "space group 225 (Fm-3m), 6 atoms" in w.statusBar().currentMessage()
    p1 = tmp_path / "gui_p1.d12"
    assert w.save_path(str(p1)) and p1.read_text().splitlines()[3] == "1"
    for d in w._live_windows():
        d.close()

    bodies = {name: path.read_text() for name, path in out.items()}
    assert bodies["classic"] == bodies["tui"] == bodies["gui"]
    assert bodies["gui"].splitlines()[1:6] == ["CRYSTAL", "0 0 0", "225", "26.343000", "6"]
