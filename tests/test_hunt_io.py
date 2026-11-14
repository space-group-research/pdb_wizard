"""Reading and writing structure files: the bugs found by the io bug hunt.

Every test checks the tool against something written here from the format's own
definition (a fixed-column PDB parser, a POSCAR parser, the LAMMPS box definition, MPMC's
sscanf record, a CHARMM DCD writer, textbook lattice formulas), never against the tool's
own reader or writer alone.
"""

import hashlib
import io
import json
import logging
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.formats import (
    read_gaussian_com,
    read_lammps_data,
    read_mol2,
    read_poscar,
    read_sdf,
    write_cif,
    write_lammps_data,
    write_poscar,
)
from pdb_wizard.io import (
    read_charges_file,
    read_cif,
    read_file,
    read_pdb,
    read_xyz,
    read_xyz_trajectory,
    write_mpmc_pdb,
    write_standard_pdb,
    write_xyz,
)
from pdb_wizard.molecule import Molecule
from pdb_wizard.pbc import PBC

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"
HKUST_MPMC = ROOT / "HKUST-1_out_mpmc.pdb"
NI_MOF = ROOT / "Ni-MOF-74.pqr"


# ----------------------------------------------------------------------
# Independent references
# ----------------------------------------------------------------------

def ref_basis(a, b, c, al, be, ga):
    """Lattice rows a, b, c with a along x and b in the xy plane (International Tables)."""
    al, be, ga = (math.radians(v) for v in (al, be, ga))
    bx, by = b * math.cos(ga), b * math.sin(ga)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(c * c - cx * cx - cy * cy)
    return np.array([[a, 0, 0], [bx, by, 0], [cx, cy, cz]])


def cellpar(m):
    m = np.asarray(m, float)
    lengths = [np.linalg.norm(v) for v in m]

    def ang(u, v):
        return math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))

    return (lengths[0], lengths[1], lengths[2], ang(m[1], m[2]), ang(m[0], m[2]), ang(m[0], m[1]))


def cell_of(pbc):
    return (pbc.a, pbc.b, pbc.c, pbc.alpha, pbc.beta, pbc.gamma)


def frac_of(atoms, pbc):
    """Fractional coordinates from the returned positions and the returned cell."""
    return np.array([a.x for a in atoms]) @ np.linalg.inv(ref_basis(*cell_of(pbc)))


def frac_error(got, expected, basis):
    """Largest distance (A) between fractional positions, modulo lattice vectors."""
    d = np.asarray(got) - np.asarray(expected)
    d -= np.round(d)
    return float(np.linalg.norm(d @ basis, axis=1).max())


def mk(elements, coords, cell=None, names=None):
    atoms = [Atom(x[0], x[1], x[2], names[i] if names else el)
             for i, (el, x) in enumerate(zip(elements, coords))]
    for i, a in enumerate(atoms):
        a.id = i + 1
    return Molecule(atoms=atoms, pbc=PBC(*cell) if cell else None)


def text_of(writer, mol, *args, **kw):
    buf = io.StringIO()
    writer(mol, buf, *args, **kw)
    return buf.getvalue()


def vasp_parse(text):
    """POSCAR per the VASP wiki: the first N1 positions belong to species 1, and so on."""
    lines = text.splitlines()
    scale = float(lines[1])
    lat = np.array([[float(t) for t in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    species, counts = lines[5].split(), [int(t) for t in lines[6].split()]
    cartesian = lines[7].strip()[0].lower() in "ck"
    out, k = [], 8
    for sp, n in zip(species, counts):
        for _ in range(n):
            v = np.array([float(t) for t in lines[k].split()[:3]])
            out.append((sp, v * scale if cartesian else v @ lat))
            k += 1
    return lat, out


def lammps_box(text):
    """Box vectors of a LAMMPS data file (Howto triclinic): a = (lx, 0, 0),
    b = (xy, ly, 0), c = (xz, yz, lz); no tilt line means an orthogonal box."""
    size, tilt = {}, (0.0, 0.0, 0.0)
    for line in text.splitlines():
        t = line.split()
        for ax in "xyz":
            if len(t) >= 4 and t[2] == ax + "lo" and t[3] == ax + "hi":
                size[ax] = float(t[1]) - float(t[0])
        if len(t) >= 6 and t[3:6] == ["xy", "xz", "yz"]:
            tilt = tuple(float(v) for v in t[:3])
    return np.array([[size["x"], 0, 0], [tilt[0], size["y"], 0], [tilt[1], tilt[2], size["z"]]])


MPMC_COLS = ["mass", "charge", "alpha", "epsilon", "sigma", "omega", "gwp_alpha", "c6", "c8",
             "c10", "c9"]


def mpmc_parse(text):
    """MPMC's own record (src/io/read_pqr.c): ATOM id type molecule flag molid x y z mass
    charge alpha epsilon sigma omega gwp_alpha c6 c8 c10 c9, whitespace separated."""
    out = []
    for line in text.splitlines():
        t = line.split()
        if not t or t[0] != "ATOM" or t[3] == "BOX":
            continue
        vals = [float(v) for v in t[9:20]]
        row = dict(zip(MPMC_COLS, vals + [0.0] * (11 - len(vals))))
        row.update(name=t[2], flag=t[4], x=np.array([float(v) for v in t[6:9]]), n_tokens=len(t))
        out.append(row)
    return out


def mpmc_basis(text):
    rows = {}
    for line in text.splitlines():
        if line.startswith("REMARK BOX BASIS["):
            t = line.split()
            rows[int(t[2][6])] = [float(v) for v in t[-3:]]
    return np.array([rows[0], rows[1], rows[2]])


def strict_pdb_fields(line):
    """wwPDB v3.3 ATOM/HETATM columns."""
    return dict(serial=line[6:11], name=line[12:16], resseq=line[22:26],
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                element=line[76:78].strip())


def write_dcd(path, frames, boxes, nset=None):
    """A CHARMM/NAMD DCD written from the format description (little endian)."""
    def rec(payload):
        n = struct.pack("<i", len(payload))
        return n + payload + n

    ic = [0] * 20
    ic[0] = len(frames) if nset is None else nset
    ic[2], ic[3], ic[10], ic[19] = 1, len(frames), 1, 24
    out = rec(b"CORD" + struct.pack("<20i", *ic))
    out += rec(struct.pack("<i", 1) + b"REMARKS test".ljust(80))
    out += rec(struct.pack("<i", len(frames[0])))
    for f, (a, b, c, al, be, ga) in zip(frames, boxes):
        out += rec(struct.pack("<6d", a, ga, b, be, al, c))
        f = np.asarray(f, dtype="<f4")
        for ax in range(3):
            out += rec(f[:, ax].tobytes())
    Path(path).write_bytes(out)


CO2_IN_FRAMEWORK = """CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1 ZN   MOF F    1      2.000   2.000   2.000 65.38000  1.20000  1.98870 62.39930  2.46200  0.25000  1.50000  100.50000  2000.25000  45000.12500  77.00000
ATOM      2 C1   MOF F    1      5.000   2.000   2.000 12.01100 -0.10000  1.28860 52.83810  3.43100  0.00000  0.00000   30.50000   700.25000  20000.12500
ATOM      3 COG  CO2 M    2     10.000  10.000  10.000 12.01070  0.77106  0.00000  8.52238  3.05549  0.00000  0.00000  0.00000  0.00000  0.00000
ATOM      4 COE  CO2 M    2     11.162  10.000  10.000 15.99940 -0.38553  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000
ATOM      5 COE  CO2 M    2      8.838  10.000  10.000 15.99940 -0.38553  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000
ATOM      6 CON  CO2 M    2     11.091  10.000  10.000  0.00000  0.00000  0.00000 76.76607  2.94473  0.00000  0.00000  0.00000  0.00000  0.00000
ATOM      7 CON  CO2 M    2      8.909  10.000  10.000  0.00000  0.00000  0.00000 76.76607  2.94473  0.00000  0.00000  0.00000  0.00000  0.00000
END
"""  # noqa: E501


@pytest.fixture
def co2_mol(tmp_path):
    p = tmp_path / "co2.pqr"
    p.write_text(CO2_IN_FRAMEWORK)
    return read_file(str(p), remove_overlaps=False)


# ----------------------------------------------------------------------
# 1. POSCAR writer: positions grouped like the species counts
# ----------------------------------------------------------------------

def test_poscar_positions_follow_the_species_counts():
    xyz = [[1, 1, 1], [1.96, 1, 1], [0.76, 1.93, 1], [5, 5, 5], [5.96, 5, 5], [4.76, 5.93, 5]]
    mol = mk(["O", "H", "H", "O", "H", "H"], xyz, (10.0, 10.0, 10.0, 90, 90, 90))
    _, atoms = vasp_parse(text_of(write_poscar, mol))
    assert len(atoms) == 6
    for a in mol.atoms:  # the element VASP finds at each atom's position
        hit = [sp for sp, x in atoms if np.linalg.norm(x - a.x) < 1e-6]
        assert hit == [a.element.symbol]


def test_poscar_of_shipped_mpmc_output_keeps_every_element():
    mol = read_file(str(HKUST_MPMC), remove_overlaps=False)
    _, atoms = vasp_parse(text_of(write_poscar, mol))
    assert Counter(sp for sp, _ in atoms) == {"Cu": 48, "C": 289, "H": 96, "O": 194}
    cu = np.array([a.x for a in mol.atoms if a.element.symbol == "Cu"])
    got = np.array([x for sp, x in atoms if sp == "Cu"])
    assert np.abs(np.sort(cu, axis=0) - np.sort(got, axis=0)).max() < 1e-6


# ----------------------------------------------------------------------
# 2. LAMMPS writer: triclinic cells
# ----------------------------------------------------------------------

def test_lammps_box_of_a_hexagonal_cell_has_its_tilt():
    mol = read_file(str(NI_MOF), remove_overlaps=False)
    box = lammps_box(text_of(write_lammps_data, mol))
    true = ref_basis(*cell_of(mol.pbc))
    assert cellpar(box) == pytest.approx(cell_of(mol.pbc), abs=1e-4)
    assert abs(np.linalg.det(box)) == pytest.approx(abs(np.linalg.det(true)), rel=1e-6)


@pytest.mark.parametrize("cell", [(7.3, 9.1, 11.7, 65.0, 72.0, 58.0),
                                  (7.3, 9.1, 11.7, 115.0, 108.0, 122.0)])
def test_lammps_triclinic_atoms_keep_their_fractional_positions(cell):
    frac = np.array([[0.1, 0.2, 0.3], [0.6, 0.15, 0.7], [0.35, 0.8, 0.45]])
    mol = mk(["C", "O", "N"], frac @ ref_basis(*cell), cell)
    text = text_of(write_lammps_data, mol)
    box = lammps_box(text)
    rows = text.split("Atoms")[1].strip().splitlines()
    xyz = np.array([[float(v) for v in r.split()[4:7]] for r in rows[:3]])
    assert frac_error(xyz @ np.linalg.inv(box), frac, box) < 1e-4
    # and the reader takes the tilt line back
    atoms, pbc = read_lammps_data(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx(cell, abs=1e-4)
    assert frac_error(frac_of(atoms, pbc), frac, ref_basis(*cell)) < 1e-4


def test_lammps_orthogonal_box_has_no_tilt_line():
    mol = mk(["C"], [[1, 2, 3]], (10.0, 11.0, 12.0, 90, 90, 90))
    text = text_of(write_lammps_data, mol)
    assert "xy xz yz" not in text
    assert "0.000000 10.000000 xlo xhi" in text and "0.000000 12.000000 zlo zhi" in text


# ----------------------------------------------------------------------
# 3. POSCAR reader
# ----------------------------------------------------------------------

def _poscar(scale, lattice, mode, positions, species="H", counts=None):
    counts = counts or str(len(positions))
    return ("t\n%s\n" % scale + "".join("%s %s %s\n" % tuple(v) for v in lattice)
            + species + "\n" + counts + "\n" + mode + "\n"
            + "".join("%s %s %s\n" % tuple(p) for p in positions))


def test_poscar_fcc_primitive_cell_atoms_agree_with_the_cell():
    text = _poscar("5.64", [(0, .5, .5), (.5, 0, .5), (.5, .5, 0)], "Direct",
                   [(0, 0, 0), (.5, .5, .5)], "Na Cl", "1 1")
    atoms, pbc = read_poscar(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx((3.98808, 3.98808, 3.98808, 60, 60, 60), abs=1e-4)
    assert frac_error(frac_of(atoms, pbc), [[0, 0, 0], [.5, .5, .5]],
                      ref_basis(*cell_of(pbc))) < 1e-6


def test_poscar_hexagonal_cell_in_another_orientation():
    text = _poscar("1.0", [(1.5, -2.598076211, 0), (1.5, 2.598076211, 0), (0, 0, 5.0)],
                   "Direct", [(0.333333, 0.666667, 0.25), (0.1, 0.2, 0.3)], "C")
    atoms, pbc = read_poscar(io.StringIO(text))
    assert frac_error(frac_of(atoms, pbc), [[0.333333, 0.666667, 0.25], [0.1, 0.2, 0.3]],
                      ref_basis(*cell_of(pbc))) < 1e-6


def test_poscar_scale_applies_to_cartesian_positions():
    text = _poscar("2.0", [(1, 0, 0), (0, 1, 0), (0, 0, 1)], "Cartesian", [(.5, .5, .5)])
    atoms, pbc = read_poscar(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx((2, 2, 2, 90, 90, 90))
    assert atoms[0].x == pytest.approx([1.0, 1.0, 1.0])


def test_poscar_negative_scale_is_the_volume():
    text = _poscar("-64.0", [(1, 0, 0), (0, 1, 0), (0, 0, 1)], "Direct", [(.25, .25, .25)])
    atoms, pbc = read_poscar(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx((4, 4, 4, 90, 90, 90))
    assert atoms[0].x == pytest.approx([1.0, 1.0, 1.0])


def test_poscar_three_scale_factors():
    text = _poscar("1.0 2.0 3.0", [(4, 0, 0), (0, 4, 0), (0, 0, 4)], "Direct",
                   [(.25, .25, .25)])
    atoms, pbc = read_poscar(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx((4, 8, 12, 90, 90, 90))
    assert atoms[0].x == pytest.approx([1.0, 2.0, 3.0])


def test_poscar_in_standard_orientation_reads_as_before():
    cell = (7.3, 9.1, 11.7, 65.0, 110.0, 75.0)
    frac = np.array([[0.1, 0.2, 0.3], [0.6, 0.15, 0.7]])
    mol = mk(["C", "C"], frac @ ref_basis(*cell), cell)
    atoms, pbc = read_poscar(io.StringIO(text_of(write_poscar, mol)))
    assert cell_of(pbc) == pytest.approx(cell, abs=1e-8)
    assert np.abs(np.array([a.x for a in atoms]) - frac @ ref_basis(*cell)).max() < 1e-8


# 20. a truncated POSCAR is a clean error, in the reader and on the command line
@pytest.mark.parametrize("text", ["c\n1.0\n1 0 0\n0 1 0\n0 0 1\n",
                                  "c\n1.0\n4 0 0\n0 4 0\n0 0 4\nH\n3\nDirect\n0 0 0\n"])
def test_truncated_poscar_is_a_value_error(text, tmp_path, capsys):
    with pytest.raises(ValueError, match="POSCAR ends"):
        read_poscar(io.StringIO(text))
    from pdb_wizard.__main__ import _read_or_exit
    p = tmp_path / "trunc.vasp"
    p.write_text(text)
    with pytest.raises(SystemExit) as e:
        _read_or_exit(str(p))
    assert e.value.code == 1 and "POSCAR ends" in capsys.readouterr().err


# ----------------------------------------------------------------------
# 4, 9, 18. CIF reader
# ----------------------------------------------------------------------

CIF_CELL = ("data_t\n_cell_length_a 10.0\n_cell_length_b 10.0\n_cell_length_c 10.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n")
CIF_ATOMS = ("loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
             "_atom_site_fract_y\n_atom_site_fract_z\n")


def _cif(text, notes=None):
    atoms, pbc = read_cif(io.StringIO(text), notes=notes)
    return [(a.element.symbol, tuple(round(float(v), 4) for v in a.x)) for a in atoms], pbc


def test_cif_upper_case_symmetry_operators():
    text = (CIF_CELL + "loop_\n_symmetry_equiv_pos_as_xyz\n'X,Y,Z'\n'-X,-Y,-Z'\n"
            + CIF_ATOMS + "O1 O 0.1 0.2 0.3\n")
    atoms, _ = _cif(text)
    assert atoms == [("O", (1.0, 2.0, 3.0)), ("O", (9.0, 8.0, 7.0))]


def test_cif_element_is_the_leading_letters_of_the_label():
    labels = ["C1A", "C2B", "N1A", "C1O", "C1L", "C1U", "C1D", "O1S", "N1B", "H1A", "Cu1",
              "CL1", "O1W"]
    text = (CIF_CELL.replace("10.0", "20.0") + "loop_\n_atom_site_label\n_atom_site_fract_x\n"
            "_atom_site_fract_y\n_atom_site_fract_z\n"
            + "".join("%s %.2f 0.1 0.1\n" % (lab, 0.05 * i) for i, lab in enumerate(labels)))
    atoms, _ = _cif(text)
    assert [el for el, _ in atoms] == ["C", "C", "N", "C", "C", "C", "C", "O", "N", "H", "Cu",
                                       "Cl", "O"]


def test_cif_atom_row_wrapped_over_two_lines():
    text = (CIF_CELL + CIF_ATOMS + "_atom_site_U_iso_or_equiv\n"
            "Cu1 Cu 0.25 0.25\n   0.25 0.012\nO1 O 0.5 0.5\n   0.5 0.020\n")
    atoms, _ = _cif(text)
    assert atoms == [("Cu", (2.5, 2.5, 2.5)), ("O", (5.0, 5.0, 5.0))]


def test_cif_value_on_the_line_after_its_tag():
    text = ("data_t\n_cell_length_a\n  10.0\n_cell_length_b\n  10.0\n_cell_length_c\n  10.0\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            + CIF_ATOMS + "O1 O 0.1 0.2 0.3\n")
    atoms, pbc = _cif(text)
    assert pbc is not None and cell_of(pbc) == (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    assert atoms == [("O", (1.0, 2.0, 3.0))]


def test_cif_with_two_data_blocks_reads_the_first_and_says_so():
    text = (CIF_CELL + CIF_ATOMS + "O1 O 0.1 0.2 0.3\n"
            + CIF_CELL.replace("data_t", "data_second").replace("10.0", "20.0")
            + CIF_ATOMS + "N1 N 0.5 0.5 0.5\n")
    notes = []
    atoms, pbc = _cif(text, notes)
    assert atoms == [("O", (1.0, 2.0, 3.0))] and pbc.a == 10.0
    assert len(notes) == 1 and "more than one data block" in notes[0]


def test_cif_sites_without_coordinates_are_left_out_with_a_note():
    notes = []
    atoms, _ = _cif(CIF_CELL + CIF_ATOMS + "O1 O 0.1 0.2 0.3\nH1 H ? ? ?\nH2 H . . .\n", notes)
    assert atoms == [("O", (1.0, 2.0, 3.0))]
    assert len(notes) == 1 and notes[0].startswith("2 sites")


def test_mmcif_is_refused_with_a_clear_message():
    text = ("data_1ABC\n_cell.length_a 10.0\nloop_\n_atom_site.group_PDB\n_atom_site.id\n"
            "_atom_site.type_symbol\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\nATOM 1 N 1.0 2.0 3.0\n")
    with pytest.raises(ValueError, match="mmCIF"):
        read_cif(io.StringIO(text))


def test_cif_expansion_in_a_large_cell_keeps_bonded_atoms():
    # C, an H 0.95 A from it and an O 1.7 A from it in a 120 A cell; inversion doubles them
    text = (CIF_CELL.replace("10.0", "120.0")
            + "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,-y,-z'\n" + CIF_ATOMS
            + "C1 C 0.10000 0.20000 0.30000\nH1 H 0.10792 0.20000 0.30000\n"
              "O1 O 0.11000 0.21000 0.30000\n")
    atoms, _ = _cif(text)
    assert Counter(el for el, _ in atoms) == {"C": 2, "H": 2, "O": 2}


def test_cif_special_position_images_still_merge():
    # an atom on the inversion centre, written slightly off: one atom, as before
    text = (CIF_CELL + "loop_\n_symmetry_equiv_pos_as_xyz\n'x,y,z'\n'-x,-y,-z'\n" + CIF_ATOMS
            + "Cu1 Cu 0.5001 0.5 0.5\nO1 O 0.1 0.2 0.3\n")
    atoms, _ = _cif(text)
    assert Counter(el for el, _ in atoms) == {"Cu": 1, "O": 2}


# ----------------------------------------------------------------------
# 5. MPMC sorbate sites in the other writers
# ----------------------------------------------------------------------

def test_xyz_export_of_co2_is_c_and_o(co2_mol):
    lines = text_of(write_xyz, co2_mol).splitlines()
    assert lines[0] == "5"
    assert [ln.split()[0] for ln in lines[2:]] == ["Zn", "C", "C", "O", "O"]


def test_standard_pdb_export_of_co2_is_c_and_o(co2_mol):
    for kw in ({"skip_mols_step": True}, {}):
        text = text_of(write_standard_pdb, co2_mol, **kw)
        fields = [strict_pdb_fields(ln) for ln in text.splitlines() if ln.startswith("HETATM")]
        assert Counter(f["element"] for f in fields) == {"Zn": 1, "C": 2, "O": 2}
    assert [a.element.symbol for a in co2_mol.atoms][2:] == ["Co"] * 5  # the structure is kept


def test_cif_export_of_co2_is_c_and_o(co2_mol):
    rows = [ln.split() for ln in text_of(write_cif, co2_mol).splitlines()
            if len(ln.split()) == 5]
    assert [r[1] for r in rows] == ["Zn", "C", "C", "O", "O"]
    assert [float(v) for v in rows[3][2:]] == pytest.approx([0.5581, 0.5, 0.5])


def test_poscar_export_of_co2_is_c_and_o(co2_mol):
    _, atoms = vasp_parse(text_of(write_poscar, co2_mol))
    assert Counter(sp for sp, _ in atoms) == {"Zn": 1, "C": 2, "O": 2}


def test_lammps_export_of_co2_is_c_and_o(co2_mol):
    co2_mol.detect_bonds()
    text = text_of(write_lammps_data, co2_mol)
    masses = text.split("Masses")[1].split("Atoms")[0].split()
    assert [m for m in masses if m in ("Zn", "C", "O", "Co")] == ["Zn", "C", "O"]
    assert text.splitlines()[2] == "5 atoms"
    n_atoms = 5
    if "Bonds" in text:  # bond indices refer to the atoms written
        for row in text.split("Bonds")[1].strip().splitlines():
            assert all(1 <= int(v) <= n_atoms for v in row.split()[2:4])


def test_shipped_mpmc_output_exports_no_cobalt():
    mol = read_file(str(HKUST_MPMC), remove_overlaps=False)
    symbols = Counter(ln.split()[0] for ln in text_of(write_xyz, mol).splitlines()[2:])
    assert symbols == {"Cu": 48, "C": 289, "H": 96, "O": 194}  # by the mass column


# ----------------------------------------------------------------------
# 6. MPMC REMARK BOX BASIS in another orientation
# ----------------------------------------------------------------------

def _mpmc_file(basis, frac):
    a, b, c, al, be, ga = cellpar(basis)
    out = ["CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f\n" % (a, b, c, al, be, ga)]
    for i, f in enumerate(frac):
        x = np.array(f) @ basis
        out.append("ATOM  %5d C%d   MOF F    1    %8.3f%8.3f%8.3f 12.01100  0.00000  0.00000"
                   "  0.00000  0.00000\n" % (i + 1, i + 1, x[0], x[1], x[2]))
    for k in range(3):
        out.append("REMARK BOX BASIS[%d] = %20.14f %20.14f %20.14f\n" % ((k,) + tuple(basis[k])))
    return "".join(out) + "END\n"


@pytest.mark.parametrize("basis", [
    np.array([[0, 6.0, 6.0], [6.0, 0, 6.0], [6.0, 6.0, 0]]),
    np.array([[8.660254, -5.0, 0], [0, 10.0, 0], [0, 0, 12.0]]),
])
def test_mpmc_basis_in_another_orientation_keeps_the_crystal(basis, tmp_path):
    frac = [[0.10, 0.20, 0.30], [0.60, 0.15, 0.70], [0.35, 0.80, 0.45]]
    p = tmp_path / "rot.pqr"
    p.write_text(_mpmc_file(basis, frac))
    mol = read_file(str(p), remove_overlaps=False)
    # what was read: cell and atoms agree
    assert frac_error(frac_of(mol.atoms, mol.pbc), frac, basis) < 2e-3
    # what is written back: the same fractional positions in the new BOX BASIS
    text = text_of(write_mpmc_pdb, mol)
    new_basis = mpmc_basis(text)
    assert cellpar(new_basis) == pytest.approx(cellpar(basis), abs=1e-6)
    got = np.array([r["x"] for r in mpmc_parse(text)]) @ np.linalg.inv(new_basis)
    by_name = {r["name"]: g for r, g in zip(mpmc_parse(text), got)}
    assert frac_error([by_name["C%d" % (i + 1)] for i in range(3)], frac, basis) < 2e-3


def test_mpmc_basis_in_standard_orientation_reads_exactly_as_before(tmp_path):
    basis = ref_basis(10, 10, 10, 59.4567, 59.4567, 59.4567)
    frac = [[0.10, 0.20, 0.30], [0.60, 0.15, 0.70]]
    text = _mpmc_file(basis, frac)
    atoms, pbc = read_pdb(io.StringIO(text), "std.pqr")
    assert cell_of(pbc) == (10.0, 10.0, 10.0, 59.46, 59.46, 59.46)  # the CRYST1 record
    for a, f in zip(atoms, frac):  # positions exactly as printed in the file
        assert list(a.x) == [float("%.3f" % v) for v in np.array(f) @ basis]


# ----------------------------------------------------------------------
# 7. The placeholder CRYST1 of a non-crystal PDB
# ----------------------------------------------------------------------

def test_placeholder_cryst1_is_not_a_one_angstrom_cell(tmp_path):
    rng = np.random.default_rng(0)
    pts = [np.zeros(3)]
    while len(pts) < 120:
        step = rng.normal(size=3)
        p = pts[-1] + step * 1.5 / np.linalg.norm(step)
        if np.min(np.linalg.norm(np.array(pts) - p, axis=1)) > 1.45:
            pts.append(p)
    lines = ["CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"]
    lines += ["ATOM  %5d  C   LIG A   1    %8.3f%8.3f%8.3f  1.00  0.00           C  \n"
              % (i + 1, p[0], p[1], p[2]) for i, p in enumerate(pts)]
    path = tmp_path / "model.pdb"
    path.write_text("".join(lines) + "END\n")
    messages = []
    mol = read_file(str(path), on_message=messages.append)
    assert len(mol.atoms) == 120 and messages == []
    assert mol.pbc is None
    assert 119 <= len(mol.bonds) < 360  # a chain, not everything bonded to everything


def test_real_one_by_one_cells_that_are_not_the_placeholder_are_kept():
    _, pbc = read_pdb(io.StringIO("CRYST1    1.000    1.000    2.000  90.00  90.00  90.00\n"))
    assert pbc is not None and pbc.c == 2.0


# ----------------------------------------------------------------------
# 8. LAMMPS reader
# ----------------------------------------------------------------------

LMP_HEAD = ("LAMMPS data file\n\n3 atoms\n2 atom types\n\n0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n"
            "0.0 10.0 zlo zhi\n\nMasses\n\n1 15.9994\n2 1.008\n\n")
LMP_WATER = [("O", (1.0, 1.0, 1.0)), ("H", (1.96, 1.0, 1.0)), ("H", (0.76, 1.93, 1.0))]
LMP_BODIES = {
    "atomic + image flags": ("Atoms # atomic\n\n1 1 1.0 1.0 1.0 0 0 0\n2 2 1.96 1.0 1.0 0 0 0\n"
                             "3 2 0.76 1.93 1.0 0 0 0\n", [0.0, 0.0, 0.0]),
    "atomic + comments": ("Atoms # atomic\n\n1 1 1.0 1.0 1.0 # O\n2 2 1.96 1.0 1.0 # H\n"
                          "3 2 0.76 1.93 1.0 # H\n", [0.0, 0.0, 0.0]),
    "charge + image flags": ("Atoms # charge\n\n1 1 -0.8 1.0 1.0 1.0 0 0 0\n"
                             "2 2 0.4 1.96 1.0 1.0 0 0 0\n3 2 0.4 0.76 1.93 1.0 0 0 0\n",
                             [-0.8, 0.4, 0.4]),
    "charge + image flags, no hint": ("Atoms\n\n1 1 -0.8 1.0 1.0 1.0 0 0 0\n"
                                      "2 2 0.4 1.96 1.0 1.0 0 0 0\n3 2 0.4 0.76 1.93 1.0 0 0 0\n",
                                      [-0.8, 0.4, 0.4]),
    "molecular": ("Atoms # molecular\n\n1 1 1 1.0 1.0 1.0\n2 1 2 1.96 1.0 1.0\n"
                  "3 1 2 0.76 1.93 1.0\n", [0.0, 0.0, 0.0]),
    "full": ("Atoms # full\n\n1 1 1 -0.8 1.0 1.0 1.0\n2 1 2 0.4 1.96 1.0 1.0\n"
             "3 1 2 0.4 0.76 1.93 1.0\n", [-0.8, 0.4, 0.4]),
    "full, Angle Coeffs after Masses": ("Angle Coeffs\n\n1 55.0 104.52\n\nAtoms # full\n\n"
                                        "1 1 1 -0.8 1.0 1.0 1.0\n2 1 2 0.4 1.96 1.0 1.0\n"
                                        "3 1 2 0.4 0.76 1.93 1.0\n", [-0.8, 0.4, 0.4]),
}


@pytest.mark.parametrize("case", sorted(LMP_BODIES))
def test_lammps_atom_styles(case):
    body, charges = LMP_BODIES[case]
    atoms, _ = read_lammps_data(io.StringIO(LMP_HEAD + body))
    assert [(a.element.symbol, tuple(a.x)) for a in atoms] == LMP_WATER
    assert [a.charge for a in atoms] == charges


def test_lammps_reader_takes_the_tilt_line():
    text = LMP_HEAD.replace("0.0 10.0 ylo yhi", "0.0 8.660254 ylo yhi").replace(
        "0.0 10.0 zlo zhi", "0.0 12.0 zlo zhi\n-5.0 0.0 0.0 xy xz yz") + LMP_BODIES["full"][0]
    _, pbc = read_lammps_data(io.StringIO(text))
    assert cell_of(pbc) == pytest.approx((10, 10, 12, 90, 90, 120), abs=1e-5)


def test_lammps_mass_comment_names_the_element():
    text = LMP_HEAD.replace("1 15.9994", "1 58.80 # Co") + LMP_BODIES["full"][0]
    atoms, _ = read_lammps_data(io.StringIO(text))
    assert atoms[0].element.symbol == "Co"  # 58.80 alone is nearer to Ni (58.69)


# 25. unknown elements
def test_lammps_writer_refuses_atoms_without_a_mass():
    mol = mk(["X", "C"], [[0, 0, 0], [2, 0, 0]], (10.0, 10.0, 10.0, 90, 90, 90),
             names=["Q1", "C"])
    with pytest.raises(ValueError, match="Q1"):
        write_lammps_data(mol, io.StringIO())


# ----------------------------------------------------------------------
# 10, 11. PDB columns
# ----------------------------------------------------------------------

def test_pdb_serial_wraps_instead_of_shifting_the_columns():
    n = 100002
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.arange(n) % 100 * 2.0 + 0.123
    xyz[:, 1] = np.arange(n) // 100 % 100 * 2.0 + 0.456
    xyz[:, 2] = np.arange(n) // 10000 * 2.0 + 0.789
    mol = mk(["Ar"] * n, xyz, (200.0, 200.0, 200.0, 90, 90, 90))
    recs = [ln for ln in text_of(write_standard_pdb, mol, skip_mols_step=True).splitlines()
            if ln.startswith("HETATM")]
    for k in (99998, 99999, 100001):
        f = strict_pdb_fields(recs[k])
        assert (f["x"], f["y"], f["z"]) == pytest.approx(tuple(xyz[k]), abs=1e-3)
        assert f["element"] == "Ar" and int(f["serial"]) == (k + 1) % 100000
        assert len(recs[k]) == len(recs[0])


def test_pdb_residue_number_wraps_instead_of_shifting_the_columns():
    n = 10001
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.arange(n) % 30 * 4.0 + 0.123
    xyz[:, 1] = np.arange(n) // 30 % 30 * 4.0 + 0.456
    xyz[:, 2] = np.arange(n) // 900 * 4.0 + 0.789
    mol = mk(["Ar"] * n, xyz, (120.0, 120.0, 120.0, 90, 90, 90))
    recs = [ln for ln in text_of(write_standard_pdb, mol).splitlines()
            if ln.startswith("HETATM")]
    assert len(recs) == n and {len(r) for r in recs} == {len(recs[0])}
    f = strict_pdb_fields(recs[9999])
    assert f["element"] == "Ar" and int(f["resseq"]) == 0
    written = sorted((strict_pdb_fields(r)["x"], strict_pdb_fields(r)["y"],
                      strict_pdb_fields(r)["z"]) for r in recs)
    assert np.abs(np.array(written) - np.array(sorted(map(tuple, xyz)))).max() < 1e-3


@pytest.mark.parametrize("writer", [write_standard_pdb, write_mpmc_pdb])
def test_cryst1_keeps_its_columns_for_a_cell_over_1000_angstrom(writer):
    mol = mk(["C"], [[1, 2, 3]], (1234.567, 26.343, 1000.5, 90.0, 101.25, 90.0))
    line = [ln for ln in text_of(writer, mol).splitlines() if ln.startswith("CRYST1")][0]
    # wwPDB: a 7-15, b 16-24, c 25-33, alpha 34-40, beta 41-47, gamma 48-54
    assert float(line[6:15]) == 1234.567 and float(line[15:24]) == 26.343
    assert float(line[24:33]) == 1000.5
    assert (float(line[33:40]), float(line[40:47]), float(line[47:54])) == (90.0, 101.25, 90.0)


def test_cryst1_text_is_unchanged_for_ordinary_cells():
    mol = mk(["C"], [[1, 2, 3]], (26.343, 26.343, 26.343, 90.0, 90.0, 90.0))
    for writer in (write_standard_pdb, write_mpmc_pdb):
        assert ("CRYST1   26.343   26.343   26.343   90.0   90.0   90.0 P 1           1\n"
                in text_of(writer, mol))


def test_hetatm_with_five_digit_serial_uses_the_element_and_charge_columns():
    text = ("HETATM 9999  CA  LIG A   1      11.639   6.071  -5.147  1.00  0.00           C  \n"
            "HETATM10000  CA  LIG A   1      12.639   6.071  -5.147  1.00  0.00           C  \n"
            "HETATM10001  HO  LIG A   1      13.639   6.071  -5.147  1.00  0.00           H  \n"
            "HETATM10002 ZN    ZN A   2       1.639   6.071  -5.147  1.00  0.00          ZN2+\n")
    atoms, _ = read_pdb(io.StringIO(text))
    assert [a.element.symbol for a in atoms] == ["C", "C", "H", "Zn"]
    assert [a.name for a in atoms] == ["CA", "CA", "HO", "ZN"]
    assert atoms[3].charge == 2.0 and atoms[1].x[0] == 12.639


# ----------------------------------------------------------------------
# 13. MPMC columns omega, gwp_alpha, c9
# ----------------------------------------------------------------------

def test_mpmc_omega_gwp_alpha_and_c9_survive_read_and_write(co2_mol):
    before = {r["name"]: r for r in mpmc_parse(CO2_IN_FRAMEWORK)}
    after = {r["name"]: r for r in mpmc_parse(text_of(write_mpmc_pdb, co2_mol))}
    for name in ("ZN", "C1", "COG"):
        for col in MPMC_COLS:
            assert after[name][col] == pytest.approx(before[name][col], abs=1e-4), (name, col)
    assert after["ZN"]["n_tokens"] == 20  # c9 written: the file had one
    assert after["C1"]["n_tokens"] == 19 and after["COG"]["n_tokens"] == 19  # no c9 column


def test_mpmc_records_without_those_columns_are_written_as_before():
    mol = read_file(str(HKUST), remove_overlaps=False)
    row = text_of(write_mpmc_pdb, mol).splitlines()[4]
    assert row.split()[14:16] == ["0.0", "0.0"] and len(row.split()) == 19


# ----------------------------------------------------------------------
# 14. XYZ trajectories
# ----------------------------------------------------------------------

XYZ_FRAME = "2\n\nC 0 0 0\nO %s 0 0\n"


def _xyz_frames(text):
    mols, _ = read_xyz_trajectory(io.StringIO(text))
    return [[tuple(a.x) for a in m.atoms] for m in mols]


def test_xyz_trajectory_with_an_incomplete_last_frame_keeps_the_complete_ones():
    text = "".join(XYZ_FRAME % k for k in (1, 2, 3)) + "2\n\nC 0 0 0\n"
    frames = _xyz_frames(text)
    assert [f[1][0] for f in frames] == [1.0, 2.0, 3.0]


def test_xyz_trajectory_with_a_whitespace_line_at_the_end():
    assert len(_xyz_frames("".join(XYZ_FRAME % k for k in (1, 2, 3)) + "  \n")) == 3


def test_xyz_trajectory_with_blank_lines_between_frames():
    frames = _xyz_frames((XYZ_FRAME % 1) + "\n" + (XYZ_FRAME % 2) + "\n\n" + (XYZ_FRAME % 3))
    assert [f[1][0] for f in frames] == [1.0, 2.0, 3.0]


def test_xyz_trajectory_that_is_not_one_is_a_value_error_not_an_exit():
    with pytest.raises(ValueError, match="atom count"):
        read_xyz_trajectory(io.StringIO("not an xyz file\n"))


# ----------------------------------------------------------------------
# 15, 21. DCD
# ----------------------------------------------------------------------

DCD_FRAMES = [np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5], [-7.25, 8.0, 9.0]]) + k
              for k in range(3)]
DCD_BOX = [(10.0, 11.0, 12.0, 90.0, 90.0, 90.0)] * 3
WATER_PDB = ("ATOM      1  O   HOH A   1       1.000   2.000   3.000  1.00  0.00           O\n"
             "ATOM      2  H1  HOH A   1       4.500   5.500   6.500  1.00  0.00           H\n"
             "ATOM      3  H2  HOH A   1      -7.250   8.000   9.000  1.00  0.00           H\n")


def _dcd_first_x(path):
    from pdb_wizard.trajectory import read_trajectory
    return [round(float(m.atoms[0].x[0]), 4) for m in read_trajectory(str(path))]


def test_dcd_cut_short_keeps_the_complete_frames(tmp_path):
    p = tmp_path / "t.dcd"
    write_dcd(p, DCD_FRAMES, DCD_BOX)
    p.write_bytes(p.read_bytes()[:-10])
    assert _dcd_first_x(p) == [1.0, 2.0]


def test_dcd_with_fewer_frames_than_its_header_says(tmp_path):
    p = tmp_path / "t.dcd"
    write_dcd(p, DCD_FRAMES, DCD_BOX, nset=10)
    assert _dcd_first_x(p) == [1.0, 2.0, 3.0]


def test_dcd_with_a_frame_count_of_zero_in_the_header(tmp_path):
    from pdb_wizard.trajectory import is_trajectory_file
    p = tmp_path / "t.dcd"
    write_dcd(p, DCD_FRAMES, DCD_BOX, nset=0)
    assert is_trajectory_file(str(p))
    assert _dcd_first_x(p) == [1.0, 2.0, 3.0]


def test_dcd_takes_the_first_sibling_with_the_right_atom_count(tmp_path):
    from pdb_wizard.dcd import read_dcd_trajectory
    write_dcd(tmp_path / "traj.dcd", DCD_FRAMES, DCD_BOX)
    (tmp_path / "a_ligand.pdb").write_text("".join(WATER_PDB.splitlines(True)[:2]) + "END\n")
    (tmp_path / "system.pdb").write_text(WATER_PDB + "END\n")
    frames, _ = read_dcd_trajectory(str(tmp_path / "traj.dcd"))
    assert [a.element.symbol for a in frames[0].atoms] == ["O", "H", "H"]
    assert not getattr(frames[0], "_dcd_topology_missing", False)


# ----------------------------------------------------------------------
# 16. energy.dat
# ----------------------------------------------------------------------

def test_energy_dat_with_an_incomplete_last_line(tmp_path):
    from pdb_wizard.energy import read_energy_dat
    p = tmp_path / "energy.dat"
    p.write_text("#step #energy #coulombic #rd\n100 -1.5 0.1 -1.6\n200 -1.6 0.1 -1.7\n300 -1.7 0.")
    data = read_energy_dat(str(p))
    assert list(data) == ["step", "energy", "coulombic", "rd"]
    assert data["step"].tolist() == [100.0, 200.0] and data["rd"].tolist() == [-1.6, -1.7]


def test_energy_dat_comment_after_the_header_does_not_rename_columns(tmp_path):
    from pdb_wizard.energy import read_energy_dat
    p = tmp_path / "energy.dat"
    p.write_text("#step #energy #N\n100 -1.5 5\n# restarted here\n200 -1.6 6\n")
    data = read_energy_dat(str(p))
    assert list(data) == ["step", "energy", "N"] and data["N"].tolist() == [5.0, 6.0]


# ----------------------------------------------------------------------
# 17, 27. mol2, Gaussian input, SDF, XYZ
# ----------------------------------------------------------------------

def test_mol2_with_gaff_atom_types():
    rows = ("1 C1 0.0 0.0 0.0 ca 1 PHE 0.0\n2 O1 1.4 0.0 0.0 oh 1 PHE 0.0\n"
            "3 H1 1.9 0.8 0.0 ho 1 PHE 0.0\n4 N1 3 0 0 na 1 PHE 0\n5 O2 5 0 0 os 1 PHE 0\n"
            "6 C2 7 0 0 cd 1 PHE 0\n7 CL 9 0 0 cl 1 PHE 0\n8 BR 11 0 0 br 1 PHE 0\n")
    atoms, _ = read_mol2(io.StringIO("@<TRIPOS>MOLECULE\nx\n@<TRIPOS>ATOM\n" + rows
                                     + "@<TRIPOS>BOND\n"))
    assert [a.element.symbol for a in atoms] == ["C", "O", "H", "N", "O", "C", "Cl", "Br"]


def test_mol2_with_sybyl_atom_types_reads_as_before():
    rows = ("1 C1 0 0 0 C.ar 1 X 0\n2 O1 1.4 0 0 O.3 1 X 0\n3 H1 2 0 0 H 1 X 0\n"
            "4 N1 3 0 0 N.am 1 X 0\n5 CL1 5 0 0 Cl 1 X 0\n6 CA 7 0 0 Ca 1 X 0\n")
    atoms, _ = read_mol2(io.StringIO("@<TRIPOS>ATOM\n" + rows))
    assert [a.element.symbol for a in atoms] == ["C", "O", "H", "N", "Cl", "Ca"]


GAUSSIAN_HEAD = "%chk=x.chk\n#p b3lyp/6-31g(d) opt\n\ntitle\n\n0 1\n"


def _gaussian(body):
    atoms, pbc = read_gaussian_com(io.StringIO(GAUSSIAN_HEAD + body + "\n"))
    return [(a.element.symbol, tuple(a.x)) for a in atoms], pbc


def test_gaussian_input_with_freeze_codes():
    atoms, _ = _gaussian("C -1 0.5 0.6 0.7\nO 0 1.2 0.1 0.3\n")
    assert atoms == [("C", (0.5, 0.6, 0.7)), ("O", (1.2, 0.1, 0.3))]


def test_gaussian_input_with_atomic_numbers():
    atoms, _ = _gaussian("6 0.0 0.0 0.0\n8 1.2 0.0 0.0\n")
    assert atoms == [("C", (0.0, 0.0, 0.0)), ("O", (1.2, 0.0, 0.0))]


def test_gaussian_translation_vectors_are_the_cell_not_atoms():
    atoms, pbc = _gaussian("C 0.5 0.5 0.5\nTv 2.5 0.0 0.0\nTv 0.0 3.5 0.0\nTv 0.0 0.0 4.5\n")
    assert atoms == [("C", (0.5, 0.5, 0.5))]
    assert cell_of(pbc) == pytest.approx((2.5, 3.5, 4.5, 90, 90, 90))


def test_gaussian_plain_input_reads_as_before():
    atoms, pbc = _gaussian("C 0.0 0.0 0.0\nO 1.2 0.0 0.0\nH(Fragment=1) 0 1 0\n")
    assert atoms == [("C", (0.0, 0.0, 0.0)), ("O", (1.2, 0.0, 0.0)), ("H", (0.0, 1.0, 0.0))]
    assert pbc is None


def test_sdf_v3000_is_refused_with_a_clear_message():
    text = ("water\n  prog\n\n  0  0  0     0  0            999 V3000\nM  V30 BEGIN CTAB\n"
            "M  V30 COUNTS 3 2 0 0 0\nM  V30 BEGIN ATOM\nM  V30 1 O 0 0 0 0\nM  V30 END ATOM\n")
    with pytest.raises(ValueError, match="V3000"):
        read_sdf(io.StringIO(text))


@pytest.mark.parametrize("text, n_read, words", [
    ("2\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", 2, "more atom lines follow"),
    ("5\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", 3, "file ends after 3"),
    ("3\n\nO 0 0 0\nH 0.96 0\nH -0.24 0.93 0\n", 1, "could not be read"),
])
def test_xyz_count_mismatch_gives_a_load_note_and_a_warning(text, n_read, words, tmp_path, caplog):
    p = tmp_path / "bad.xyz"
    p.write_text(text)
    with caplog.at_level(logging.WARNING, logger="pdb_wizard.io"):
        mol = read_file(str(p))
    from pdb_wizard.checks import load_notes
    assert len(mol.atoms) == n_read  # the atoms are still returned
    assert len(load_notes(mol)) == 1 and words in load_notes(mol)[0]
    assert any(words in r.getMessage() for r in caplog.records)


def test_well_formed_xyz_files_get_no_note(tmp_path):
    from pdb_wizard.checks import load_notes
    for text in ("2\n\nC 0 0 0\nO 1.2 0 0\n", "2\n\nC 0 0 0\nO 1.2 0 0\n\n\n",
                 "1\n\nC 0 0 0\n1\n\nC 1 0 0\n"):
        p = tmp_path / "ok.xyz"
        p.write_text(text)
        assert load_notes(read_file(str(p))) == []


# ----------------------------------------------------------------------
# 19. Structures without atoms
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cell", [(10.0, 10.0, 10.0, 90, 90, 90), None])
def test_writers_accept_a_structure_without_atoms(cell):
    mol = mk([], [], cell)
    for kw in ({"skip_mols_step": True}, {}):
        text = text_of(write_standard_pdb, mol, **kw)
        assert "HETATM" not in text and text.endswith("END\n")
        assert ("CRYST1" in text) == (cell is not None)
    assert text_of(write_lammps_data, mol).splitlines()[2] == "0 atoms"
    if cell:
        cif = text_of(write_cif, mol)
        assert "_cell_length_a    10.000000" in cif and cif.rstrip().endswith("_atom_site_fract_z")


# ----------------------------------------------------------------------
# 22. Charge files
# ----------------------------------------------------------------------

def test_skip_last_does_not_count_trailing_blank_lines(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("0.5\n-0.5\nTotal 0.0\n\n\n")
    assert read_charges_file(str(p), skip_last=1) == [0.5, -0.5]
    p.write_text("0.5\n-0.5\nTotal 0.0\n")
    assert read_charges_file(str(p), skip_last=1) == [0.5, -0.5]
    p.write_text("0.5\n-0.5\n\n")
    assert read_charges_file(str(p)) == [0.5, -0.5]


# ----------------------------------------------------------------------
# 24. CRYSTAL input and atoms of unknown element
# ----------------------------------------------------------------------

def test_crystal_writer_names_the_atoms_it_cannot_write():
    from pdb_wizard.crystal import crystal_geometry, write_crystal_d12
    mol = mk(["C", "X", "X", "O"], [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
             (10.0, 10.0, 10.0, 90, 90, 90), names=["C", "D", "Q1", "O"])
    with pytest.raises(ValueError, match="D, Q1"):
        crystal_geometry(mol)
    with pytest.raises(ValueError, match="D, Q1"):
        write_crystal_d12(mol, io.StringIO())


def test_crystal_writer_still_leaves_massless_model_sites_out(co2_mol):
    from pdb_wizard.crystal import crystal_geometry
    lines = crystal_geometry(co2_mol).splitlines()
    assert lines[4] == "5" and [ln.split()[0] for ln in lines[5:]] == ["30", "6", "6", "8", "8"]


# ----------------------------------------------------------------------
# 26. Alternate locations
# ----------------------------------------------------------------------

ALTLOC = ("ATOM      1  CA  SER A   1       1.000   1.000   1.000  1.00  0.00           C\n"
          "ATOM      2  CB ASER A   1       2.000   2.000   1.000  0.60  0.00           C\n"
          "ATOM      3  OG ASER A   1       3.000   2.000   1.000  0.60  0.00           O\n"
          "ATOM      4  CB BSER A   1       2.100   1.900   1.200  0.40  0.00           C\n"
          "ATOM      5  OG BSER A   1       2.500   0.900   2.000  0.40  0.00           O\n")


def test_alternate_locations_are_kept_and_noted(tmp_path):
    from pdb_wizard.checks import load_notes
    p = tmp_path / "alt.pdb"
    p.write_text(ALTLOC + "END\n")
    mol = read_file(str(p))
    assert len(mol.atoms) == 5  # unchanged: every location is kept
    assert len(load_notes(mol)) == 1 and "alternate locations (A, B)" in load_notes(mol)[0]


def test_files_without_alternate_locations_get_no_note():
    from pdb_wizard.checks import load_notes
    for path in (HKUST, HKUST_MPMC, NI_MOF, ROOT / "tests" / "fixtures" / "openmm_water.pdb"):
        assert load_notes(read_file(str(path), remove_overlaps=False)) == []


# ----------------------------------------------------------------------
# 28, 29. Database parsing (no network: _http_get is replaced)
# ----------------------------------------------------------------------

@pytest.fixture
def db(monkeypatch):
    import urllib.error
    import urllib.request

    import pdb_wizard.database as database

    def no_network(*a, **k):
        raise AssertionError("a real network request was attempted")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    def serve(payload):
        def _get(url, **kw):
            if isinstance(payload, Exception):
                raise payload
            return json.dumps(payload).encode()
        monkeypatch.setattr(database, "_http_get", _get)

    database.serve = serve
    database.URLError = urllib.error.URLError
    return database


def test_cod_hit_without_a_title_does_not_empty_the_results(db):
    db.serve([{"file": 1000041, "formula": "- Cu2 O -", "title": "Cuprite"},
              {"file": 1000063, "formula": "- Cu O -", "title": None}])
    assert [(r["id"], r["title"]) for r in db.search_cod(text="copper oxide")] == [
        ("1000041", "Cuprite"), ("1000063", "")]


def test_pubchem_hit_with_a_null_name_does_not_empty_the_results(db):
    db.serve({"PropertyTable": {"Properties": [
        {"CID": 2244, "MolecularFormula": "C9H8O4"}, {"CID": 1, "IUPACName": None}]}})
    assert [r["id"] for r in db.search_pubchem("aspirin")] == ["2244", "1"]


def test_search_backends_reports_a_network_failure(db):
    db.serve(db.URLError("no route to host"))
    results, message = db.search_backends("cod", "text", "cuprite")
    assert results == [] and "errors: COD (URLError)" in message
    assert db.search_cod(text="cuprite") == []  # the plain function still returns a list


@pytest.mark.parametrize("name, expected", [("acetic acid", "acetic_acid.sdf"),
                                            ("N/A", "N_A.sdf"), ("../escape", "_escape.sdf")])
def test_pubchem_download_stays_in_the_output_folder(db, name, expected, tmp_path):
    db.serve({"x": 1})
    out = tmp_path / "out"
    out.mkdir()
    path = Path(db.fetch_pubchem_sdf(name, str(out)))
    assert path.parent == out and path.name == expected and path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_materials_project_shared_site_is_noted_on_load(db, tmp_path):
    db.serve({"data": [{"structure": {
        "lattice": {"a": 3.0, "b": 3.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 120},
        "sites": [{"species": [{"element": "Zn", "occu": 0.5}, {"element": "Mg", "occu": 0.5}],
                   "abc": [1 / 3, 2 / 3, 0]},
                  {"species": [{"element": "O", "occu": 1}], "abc": [1 / 3, 2 / 3, 0.38]}]}}]})
    path = db.fetch_materials_project_cif("mp-1", api_key="k", output_dir=str(tmp_path))
    from pdb_wizard.checks import load_notes
    mol = read_file(path)
    assert [a.element.symbol for a in mol.atoms] == ["Zn", "O"]
    assert any("occupancy below 1" in n for n in load_notes(mol))


# ----------------------------------------------------------------------
# What worked before still gives the same files
# ----------------------------------------------------------------------

def _sig(atoms, pbc):
    parts = [("%s %.5f %.5f %.5f %.5f" % (a.element.symbol, a.x[0], a.x[1], a.x[2], a.charge))
             .replace("-0.00000", "0.00000") for a in atoms]
    cell = "none" if pbc is None else " ".join("%.5f" % v for v in cell_of(pbc))
    return hashlib.sha256(("\n".join(parts) + "|" + cell).encode()).hexdigest()[:16]


ROUND_TRIP = {
    "xyz": (lambda m, b: write_xyz(m, b), lambda b: read_xyz(b)),
    "pdb": (lambda m, b: write_standard_pdb(m, b, skip_mols_step=True),
            lambda b: read_pdb(b, "x.pdb")),
    "pdbsort": (lambda m, b: write_standard_pdb(m, b), lambda b: read_pdb(b, "x.pdb")),
    "pqr": (lambda m, b: write_mpmc_pdb(m, b), lambda b: read_pdb(b, "x.pqr")),
    "cif": (lambda m, b: write_cif(m, b), lambda b: read_cif(b)),
    "vasp": (lambda m, b: write_poscar(m, b), lambda b: read_poscar(b)),
    "lmp": (lambda m, b: write_lammps_data(m, b), lambda b: read_lammps_data(b)),
}
# sha256[:16] of (the file written, the atoms/order/charges/cell read back from it), taken
# from head 35d43b8 before any fix (hunt/io/baseline_sigs.py).
BEFORE = {
    ("HKUST-1.pdb", "read"): "5937a01fdc4c0d4c",
    ("HKUST-1.pdb", "xyz"): ("53e74ac9e5972bf8", "670bc909a5c170ab"),
    ("HKUST-1.pdb", "pdb"): ("6d16e467583587f4", "79808bac2cbbda93"),
    ("HKUST-1.pdb", "pdbsort"): ("248c29f7c8a891b4", "c4c01f660ddc790b"),
    ("HKUST-1.pdb", "pqr"): ("99c6bb87dab45595", "69b159a98e77a45a"),
    ("HKUST-1.pdb", "cif"): ("95618da2cc32c6e4", "c7133acabf064b63"),
    ("HKUST-1.pdb", "vasp"): ("1d7161b67a56365e", "670bc909a5c170ab"),
    ("HKUST-1.pdb", "lmp"): ("8ea1172bee5a818e", "5937a01fdc4c0d4c"),
    ("Ni-MOF-74.pqr", "read"): "f180c768d3ed57fc",
    ("Ni-MOF-74.pqr", "xyz"): ("dbbaa232d3270e68", "ca0e308917b85fdf"),
    ("Ni-MOF-74.pqr", "pdb"): ("3eaf2e2c56a1a1e0", "354049e1d339f45c"),
    ("Ni-MOF-74.pqr", "pdbsort"): ("30d8d7009c237563", "5c3b74f977d609e5"),
    ("Ni-MOF-74.pqr", "pqr"): ("d847c1b900bc7787", "3f54d6bde611cb82"),
    ("Ni-MOF-74.pqr", "cif"): ("343353474116329a", "1929153a8ce0e0dd"),
}


@pytest.fixture(scope="module")
def samples():
    return {p.name: read_file(str(p), remove_overlaps=False) for p in (HKUST, NI_MOF)}


@pytest.mark.parametrize("name", ["HKUST-1.pdb", "Ni-MOF-74.pqr"])
def test_sample_files_read_as_before(samples, name):
    assert _sig(samples[name].atoms, samples[name].pbc) == BEFORE[(name, "read")]


@pytest.mark.parametrize("name, fmt", [k for k in sorted(BEFORE) if k[1] != "read"])
def test_sample_round_trip_is_unchanged(samples, name, fmt):
    write, read = ROUND_TRIP[fmt]
    buf = io.StringIO()
    write(samples[name], buf)
    text = buf.getvalue()
    atoms, pbc = read(io.StringIO(text))
    assert (hashlib.sha256(text.encode()).hexdigest()[:16], _sig(atoms, pbc)) == BEFORE[(name, fmt)]


def test_ni_mof_74_poscar_changes_only_in_the_order_of_the_positions(samples):
    """Ni-MOF-74.pqr is not sorted by element (O1, O2, C1..C4, O3), so its POSCAR was one
    of the wrong ones: the fix regroups the positions. Header, cell and the set of
    (element, position) pairs are the ones of the structure."""
    mol = samples["Ni-MOF-74.pqr"]
    lines = text_of(write_poscar, mol).splitlines()
    assert lines[5:8] == ["Ni O C H", "72 216 288 72", "Cartesian"]
    lat, atoms = vasp_parse("\n".join(lines))
    assert np.abs(lat - ref_basis(*cell_of(mol.pbc))).max() < 1e-8
    want = sorted((a.element.symbol,) + tuple(np.round(a.x, 6)) for a in mol.atoms)
    assert sorted((sp,) + tuple(np.round(x, 6)) for sp, x in atoms) == want
    back, pbc = read_poscar(io.StringIO("\n".join(lines) + "\n"))
    assert cell_of(pbc) == pytest.approx(cell_of(mol.pbc), abs=1e-8)


def test_ni_mof_74_lammps_changes_only_in_the_box(samples):
    """The hexagonal box was written as orthogonal: only the ylo/yhi line and the new tilt
    line differ from the file written before; atoms, order and charges read back as the
    structure itself, now with the right cell."""
    mol = samples["Ni-MOF-74.pqr"]
    lines = text_of(write_lammps_data, mol).splitlines()
    assert lines[7:11] == ["0.000000 25.785600 xlo xhi", "0.000000 22.330985 ylo yhi",
                           "0.000000 27.080400 zlo zhi", "-12.892800 0.000000 0.000000 xy xz yz"]
    old = lines[:8] + ["0.000000 25.785600 ylo yhi"] + lines[9:10] + lines[11:]
    assert hashlib.sha256(("\n".join(old) + "\n").encode()).hexdigest()[:16] == "f82f380931cf65b1"
    atoms, pbc = read_lammps_data(io.StringIO("\n".join(lines) + "\n"))
    assert _sig(atoms, pbc) == BEFORE[("Ni-MOF-74.pqr", "read")]
