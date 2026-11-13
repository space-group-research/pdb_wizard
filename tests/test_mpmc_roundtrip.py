"""Tests for reading MPMC files with movable molecules and writing them back.

Run from the repo root:  python -m pytest tests/
"""

import importlib.util
import io
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "dist" / "pdb_wizard.py"  # built by conftest.py
spec = importlib.util.spec_from_file_location("pdb_wizard_dist", SRC)
pw = importlib.util.module_from_spec(spec)
sys.modules["pdb_wizard_dist"] = (
    pw  # dataclasses look the module up while it loads; not "pdb_wizard", which is the package
)
spec.loader.exec_module(pw)

# Two framework atoms, then two DC97 waters (molecules 2 and 3; M sites massless)
MPMC = """CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1 Cu   MOF F    1   1.000 1.000 1.000  63.546 0.8 2.196 2.516 3.114 0.0 0.0 0.0 0.0 0.0
ATOM      2 O    MOF F    1   2.900 1.000 1.000  15.999 -0.6 0.852 30.19 3.118 0.0 0.0 0.0 0.0 0.0
ATOM      3 OXY  H2O M    2  10.000 10.000 10.000 15.9994 0.0 0.0 91.84 3.234 0.0 0.0 0.0 0.0 0.0
ATOM      4 HYD  H2O M    2  10.757  9.414 10.000 1.008 0.519 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
ATOM      5 HYD  H2O M    2   9.243  9.414 10.000 1.008 0.519 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
ATOM      6 M    H2O M    2  10.000  9.785 10.000 0.0 -1.038 1.444 0.0 0.0 0.0 0.0 0.0 0.0 0.0
ATOM      7 OXY  H2O M    3   5.000 15.000  5.000 15.9994 0.0 0.0 91.84 3.234 0.0 0.0 0.0 0.0 0.0
ATOM      8 HYD  H2O M    3   5.757 14.414  5.000 1.008 0.519 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
ATOM      9 HYD  H2O M    3   4.243 14.414  5.000 1.008 0.519 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
ATOM     10 M    H2O M    3   5.000 14.785  5.000 0.0 -1.038 1.444 0.0 0.0 0.0 0.0 0.0 0.0 0.0
REMARK BOX BASIS[0] =  20.0 0.0 0.0
REMARK BOX BASIS[1] =  0.0 20.0 0.0
REMARK BOX BASIS[2] =  0.0 0.0 20.0
END
"""


def read(text=MPMC, name="in.pqr"):
    atoms, pbc = pw.read_pdb(io.StringIO(text), name)
    return pw.Molecule(atoms=atoms, pbc=pbc)


def written_records(mol, **kw):
    buf = io.StringIO()
    pw.write_mpmc_pdb(mol, buf, **kw)
    return [
        ln.split()
        for ln in buf.getvalue().splitlines()
        if ln.startswith("ATOM") and " BOX " not in ln
    ]


def test_reader_keeps_mpmc_metadata():
    mol = read()
    flags = [a.mpmc_flag for a in mol.atoms]
    assert flags == ["F", "F"] + ["M"] * 8
    assert [a.mpmc_mol for a in mol.atoms[2:]] == [2] * 4 + [3] * 4
    assert mol.atoms[5].file_mass == 0.0 and mol.atoms[5].mpmc_res == "H2O"
    assert pw.mpmc_movable_count(mol) == 2


def test_no_warning_for_dummy_sites(caplog):
    with caplog.at_level(logging.WARNING):
        read()
    assert "Unrecognised element" not in caplog.text
    # a real unknown element with mass still warns
    with caplog.at_level(logging.WARNING):
        read(
            MPMC.replace(
                "Cu   MOF F    1   1.000 1.000 1.000  63.546",
                "Qq   MOF F    1   1.000 1.000 1.000  63.546",
            )
        )
    assert "Qq x1" in caplog.text


def test_round_trip_keeps_movable_molecules():
    recs = written_records(read())
    fw = [r for r in recs if r[4] == "F"]
    mv = [r for r in recs if r[4] == "M"]
    assert len(fw) == 2 and all(r[5] == "1" for r in fw)
    assert len(mv) == 8 and [r[5] for r in mv] == ["2"] * 4 + ["3"] * 4
    assert [r[3] for r in mv] == ["H2O"] * 8 and mv[3][2] == "M"
    assert float(mv[3][9]) == 0.0  # massless M site keeps its mass, not X's
    assert float(mv[3][10]) == pytest.approx(-1.038) and float(mv[3][11]) == pytest.approx(1.444)
    # writing what we read back again gives the same records
    buf = io.StringIO()
    pw.write_mpmc_pdb(read(), buf)
    again = read(buf.getvalue())
    assert [a.mpmc_flag for a in again.atoms].count("M") == 8


def test_split_molecule_is_made_whole():
    mol = read()
    w = mol.atoms[6:10]  # water 3: move it to the a face, then wrap atom by atom
    shift = np.array([14.6, 0.0, 0.0])
    for a in w:
        a.x = a.x + shift
    pw.wrap_atoms(mol, forward=True)
    assert max(a.x[0] for a in w) - min(a.x[0] for a in w) > 10  # really split
    mv = [r for r in written_records(mol) if r[4] == "M" and r[5] == "3"]
    xs = [float(r[6]) for r in mv]
    assert max(xs) - min(xs) < 2.0  # whole again
    o, h = np.array([float(v) for v in mv[0][6:9]]), np.array([float(v) for v in mv[1][6:9]])
    assert np.linalg.norm(o - h) == pytest.approx(0.9572, abs=2e-3)


def test_extend_axis_gives_copies_new_molecule_ids():
    mol = pw.extend_axis(read(), axis=0, times=1)
    assert pw.mpmc_movable_count(mol) == 4
    recs = written_records(mol)
    ids = sorted({r[5] for r in recs if r[4] == "M"})
    assert ids == ["2", "3", "4", "5"]


def test_new_sorbates_continue_after_existing_ones():
    mol = read()
    placed = pw.insert_sorbates(mol, "h2o_dc97", 2, seed=1)
    lines = pw.format_inserted_sorbates(
        "h2o_dc97", placed, len(mol.atoms) + 1, start_mol_id=2 + pw.mpmc_movable_count(mol)
    )
    recs = written_records(mol, sorbate_lines=lines)
    assert [r[5] for r in recs if r[4] == "M"][::4] == ["2", "3", "4", "5"]


def test_plain_pdb_atoms_stay_framework():
    atoms = [pw.Atom(1, 1, 1, "C"), pw.Atom(2.5, 1, 1, "O")]
    mol = pw.Molecule(atoms=atoms, pbc=pw.PBC(10, 10, 10, 90, 90, 90))
    recs = written_records(mol)
    assert [r[4] for r in recs] == ["F", "F"]
    assert float(recs[0][9]) == pytest.approx(pw.get_element("C").mass, abs=1e-4)


# ======================================================================
# Thorough checks
# ======================================================================


CELL = "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"


def mpmc_line(i, name, res, flag, mol, xyz, mass, q=0.0, alpha=0.0, eps=0.0, sig=0.0):
    x, y, z = xyz
    return (
        f"ATOM {i} {name} {res} {flag} {mol} {x} {y} {z} {mass} {q} {alpha} {eps} {sig}"
        " 0.0 0.0 0.0 0.0 0.0\n"
    )


BOX = (
    "REMARK BOX BASIS[0] = 20.0 0.0 0.0\nREMARK BOX BASIS[1] = 0.0 20.0 0.0\n"
    "REMARK BOX BASIS[2] = 0.0 0.0 20.0\nEND\n"
)


@pytest.mark.parametrize("flag", ["A", "S", "T"])
def test_other_mpmc_flags_are_read_and_stay_framework(flag):
    text = (
        CELL
        + mpmc_line(1, "C", "MOF", "F", 1, (1, 1, 1), 12.011)
        + mpmc_line(2, "Ar", "AR", flag, 2, (5, 5, 5), 39.948)
        + BOX
    )
    mol = read(text)
    assert mol.atoms[1].mpmc_flag == flag
    assert [r[4] for r in written_records(mol)] == ["F", "F"]


def test_non_numeric_molecule_id_and_plain_pdb():
    mol = read(CELL + mpmc_line(1, "OXY", "H2O", "M", "x", (1, 1, 1), 15.9994) + BOX)
    assert mol.atoms[0].mpmc_flag == "M" and mol.atoms[0].mpmc_mol is None
    plain = (
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  C   LIG A   1       1.000   1.000   1.000  1.00  0.00           C\n"
    )
    atom = read(plain, "x.pdb").atoms[0]
    assert (atom.mpmc_flag, atom.mpmc_mol, atom.mpmc_res, atom.file_mass) == (
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "name,mass,warns",
    [("LP", 1e-6, False), ("M", 0.0, False), ("Qz", 0.6, True), ("Qz", 40.0, True)],
)
def test_dummy_site_threshold(caplog, name, mass, warns):
    text = (
        CELL
        + mpmc_line(1, "C", "MOF", "F", 1, (1, 1, 1), 12.011)
        + mpmc_line(2, name, "MOL", "M", 2, (5, 5, 5), mass)
        + BOX
    )
    with caplog.at_level(logging.WARNING):
        read(text)
    assert ("Unrecognised element" in caplog.text) == warns


def test_framework_masses_stay_element_masses_movable_keep_file_masses():
    """Framework atoms are written with element masses, as before (MPMC uses them for
    wt%); movable sites keep the file's masses (e.g. a deuterated sorbate)."""
    text = (
        CELL
        + mpmc_line(1, "H", "MOF", "F", 1, (1, 1, 1), 2.014)
        + mpmc_line(2, "D", "D2O", "M", 2, (5, 5, 5), 2.014)
        + BOX
    )
    recs = written_records(read(text))
    assert float(recs[0][9]) == pytest.approx(pw.get_element("H").mass, abs=1e-4)
    assert float(recs[1][9]) == pytest.approx(2.014)


def test_writer_does_not_modify_its_input():
    mol = read()
    before = [(a.name, a.x.copy(), a.mpmc_flag, a.mpmc_mol) for a in mol.atoms]
    written_records(mol)
    after = [(a.name, a.x.copy(), a.mpmc_flag, a.mpmc_mol) for a in mol.atoms]
    assert len(before) == len(after)
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert b[0] == a[0] and np.array_equal(b[1], a[1]) and b[2:] == a[2:]


@pytest.mark.parametrize("forward", [True, False])
def test_both_wraps_are_undone_per_molecule(forward):
    mol = read()
    for a in mol.atoms[2:6]:  # water 2 onto the c face
        a.x = a.x + np.array([0.0, 0.0, 9.6 if forward else -0.4])
    pw.wrap_atoms(mol, forward=forward)
    mv = [r for r in written_records(mol) if r[4] == "M" and r[5] == "2"]
    pos = np.array([[float(v) for v in r[6:9]] for r in mv])
    assert np.ptp(pos[:, 2]) < 1.5
    assert np.linalg.norm(pos[0] - pos[3]) == pytest.approx(0.215, abs=2e-3)  # O to M


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("times", [1, 2])
def test_extend_axis_all_axes(axis, times):
    mol = pw.extend_axis(read(), axis=axis, times=times)
    n = times + 1
    assert pw.mpmc_movable_count(mol) == 2 * n
    recs = written_records(mol)
    assert sum(r[4] == "F" for r in recs) == 2 * n
    mv = [r for r in recs if r[4] == "M"]
    assert len(mv) == 8 * n and len({r[5] for r in mv}) == 2 * n
    oxy = sorted(float(r[6 + axis]) for r in mv if r[2] == "OXY")
    assert oxy[-1] - oxy[0] > 20 * times - 15  # copies really moved along the axis


def test_extend_twice_keeps_ids_unique():
    mol = pw.extend_axis(pw.extend_axis(read(), 0, 1), 1, 1)
    assert pw.mpmc_movable_count(mol) == 8
    ids = [r[5] for r in written_records(mol) if r[4] == "M"]
    assert len(set(ids)) == 8 and len(ids) == 32


def test_extra_frozen_molecules_become_framework():
    text = (
        CELL
        + mpmc_line(1, "C", "MOF", "F", 1, (1, 1, 1), 12.011)
        + mpmc_line(2, "C", "LNK", "F", 7, (9, 9, 9), 12.011)
        + mpmc_line(3, "OXY", "H2O", "M", 2, (5, 5, 5), 15.9994)
        + BOX
    )
    recs = written_records(read(text))
    assert [(r[4], r[5]) for r in recs] == [("F", "1"), ("F", "1"), ("M", "2")]


def test_write_read_write_is_stable():
    first = io.StringIO()
    pw.write_mpmc_pdb(read(), first)
    second = io.StringIO()
    pw.write_mpmc_pdb(read(first.getvalue()), second)
    assert first.getvalue() == second.getvalue()


def test_residue_fallback_and_interleaved_records():
    text = (
        CELL
        + mpmc_line(1, "C", "MOF", "F", 1, (1, 1, 1), 12.011)
        + mpmc_line(2, "OXY", "H2O", "M", 3, (5, 5, 5), 15.9994)
        + mpmc_line(3, "OXY", "H2O", "M", 2, (9, 9, 9), 15.9994)
        + mpmc_line(4, "HYD", "H2O", "M", 3, (5.757, 4.414, 5), 1.008)
        + mpmc_line(5, "HYD", "H2O", "M", 2, (9.757, 8.414, 9), 1.008)
        + BOX
    )
    mol = read(text)
    mol.atoms[2].mpmc_res = None
    mv = [r for r in written_records(mol) if r[4] == "M"]
    # grouped by molecule in order of first appearance, renumbered 2, 3
    assert [(r[2], r[5]) for r in mv] == [("OXY", "2"), ("HYD", "2"), ("OXY", "3"), ("HYD", "3")]
    assert mv[2][3] == "MOL"


def test_movable_count_tracks_deletions():
    mol = read()
    mol.atoms = mol.atoms[:6]  # drop water 3
    assert pw.mpmc_movable_count(mol) == 1
    mol.atoms = mol.atoms[:2]
    assert pw.mpmc_movable_count(mol) == 0


def test_isotherm_sorbate_continues_after_existing_waters(tmp_path):
    mol = read()
    dirs = pw.generate_isotherm(
        mol, "h2o_tip3p", 298.0, [0.1], str(tmp_path), sorbate_pos=(15, 5, 15)
    )
    text = (Path(dirs[0]) / "input.pqr").read_text()
    m_ids = [
        ln.split()[5] for ln in text.splitlines() if ln.startswith("ATOM") and ln.split()[4] == "M"
    ]
    assert m_ids == ["2"] * 4 + ["3"] * 4 + ["4"] * 3


def test_menu_insert_into_a_file_with_waters(monkeypatch, tmp_path):
    out = tmp_path / "more.pqr"
    feed = iter(["h2o_tip3p", "2", str(out)])
    monkeypatch.setattr("builtins.input", lambda *_: next(feed))
    pw._menu_insert_sorbates(read())
    ids = [
        ln.split()[5]
        for ln in out.read_text().splitlines()
        if ln.startswith("ATOM") and ln.split()[4] == "M"
    ]
    assert ids == ["2"] * 4 + ["3"] * 4 + ["4"] * 3 + ["5"] * 3


def test_tui_default_mpmc_name_is_pqr(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pw, "FileSaveModal", lambda *a, **k: seen.append(k.get("default")) or "modal"
    )

    class App:
        filepath = "/data/mof.cif"

        def push_screen(self, screen, callback=None):
            pass

        def _mpmc_step_write(self, value):
            pass

    pw.PdbWizardApp.__dict__["_mpmc_step_filename"](App())
    assert seen == ["/data/mof_out_mpmc.pqr"]


# ---- optional: real MPMC (pytest --mpmc /path/to/mpmc) --------------------------------


@pytest.fixture
def mpmc_bin(request):
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("pass --mpmc /path/to/mpmc to run MPMC itself")
    return path


def mpmc_energy(mpmc_bin, workdir, pqr_text):
    workdir.mkdir()
    (workdir / "in.pqr").write_text(pqr_text)
    (workdir / "sp.inp").write_text(
        "job_name sp\nensemble total_energy\ntemperature 298\npqr_input in.pqr\nread_pqr_box on\n"
    )
    out = subprocess.run(
        [mpmc_bin, "sp.inp"], cwd=workdir, capture_output=True, text=True, timeout=300
    ).stdout
    return float(out.split("potential energy =")[1].split()[0])


def test_resaved_file_gives_the_same_mpmc_energy(tmp_path, mpmc_bin):
    """Frozen-frozen pairs are skipped by MPMC, so the energy only stays the same
    if the waters came back movable."""
    pbc = pw.PBC(20.88, 13.3, 17.1, 90, 99.9, 90)
    atoms = []
    for f in np.array(np.meshgrid(*[np.arange(0, 1, 0.3)] * 3)).reshape(3, -1).T:
        a = pw.Atom(*(f @ pbc.basis_matrix), "C")
        a.charge, a.sigma, a.epsilon = 0.0, 3.4, 50.0
        atoms.append(a)
    mof = pw.Molecule(atoms=atoms, pbc=pbc)
    placed = pw.insert_sorbates(mof, "h2o_spce", 8, seed=40)
    lines = pw.format_inserted_sorbates("h2o_spce", placed, start_atom_id=len(atoms) + 1)
    original = io.StringIO()
    pw.write_mpmc_pdb(mof, original, sorbate_lines=lines)
    resaved = io.StringIO()
    pw.write_mpmc_pdb(read(original.getvalue()), resaved)
    e1 = mpmc_energy(mpmc_bin, tmp_path / "a", original.getvalue())
    e2 = mpmc_energy(mpmc_bin, tmp_path / "b", resaved.getvalue())
    assert -1e6 < e1 < 1e4 and e2 == pytest.approx(e1, abs=1e-6)


def test_short_mpmc_record_has_no_file_mass():
    """MPMC tolerates records that stop after the coordinates; the mass is then unknown
    and the element mass is written instead."""
    text = CELL + "ATOM 1 C MOF F 1 1.0 1.0 1.0\n" + BOX
    atom = read(text).atoms[0]
    assert atom.file_mass is None and atom.mpmc_flag == "F"
    assert float(written_records(read(text))[0][9]) == pytest.approx(
        pw.get_element("C").mass, abs=1e-4
    )


# ---- framework-only edits leave the movable molecules' model alone ---------------------


def water_params(mol):
    return [
        (a.name, a.charge, a.alpha, a.epsilon, a.sigma) for a in mol.atoms if a.mpmc_flag == "M"
    ]


@pytest.mark.parametrize("ff_idx", [0, 1])
def test_force_field_skips_movable_molecules(ff_idx):
    mol = read()
    before = water_params(mol)
    _, missing = pw.apply_ff_to_system_report(mol.atoms, pw.get_forcefield(ff_idx))
    assert water_params(mol) == before  # DC97 O eps 91.84, M alpha 1.444 untouched
    assert "X" not in missing  # the massless M sites are not "missing" framework atoms
    cu = mol.atoms[0]
    ff = pw.get_forcefield(ff_idx)["Cu"]
    assert (cu.alpha, cu.epsilon, cu.sigma) == (ff["alpha"], ff["epsilon"], ff["sigma"])


def test_force_field_still_applies_to_plain_structures():
    atoms = [pw.Atom(1, 1, 1, "C"), pw.Atom(2.5, 1, 1, "O")]
    _, missing = pw.apply_ff_to_system_report(atoms, pw.get_forcefield(0))
    assert not missing and all(a.sigma > 0 for a in atoms)


def test_charges_go_to_the_framework_only():
    mol = read()
    before = water_params(mol)
    assert pw.assign_framework_charges(mol, [0.5, -0.5]) is None
    assert [a.charge for a in mol.atoms[:2]] == [0.5, -0.5]
    assert water_params(mol) == before


def test_charges_repeat_over_a_supercell_and_reject_bad_counts():
    mol = pw.extend_axis(read(), 0, 2)  # 6 framework atoms
    assert pw.assign_framework_charges(mol, [0.3, -0.3]) is None
    assert [a.charge for a in mol.atoms if a.mpmc_flag != "M"] == [0.3, -0.3] * 3
    assert "framework atom count (6)" in pw.assign_framework_charges(mol, [0.1] * 4)
    assert pw.assign_framework_charges(mol, []) is not None


def test_classic_charges_prompt_covers_framework_only(monkeypatch, tmp_path):
    (tmp_path / "q.txt").write_text("0.7\n-0.7\n")
    mol = read()
    out = tmp_path / "o.pqr"
    feed = iter(["y", str(tmp_path / "q.txt"), "n", str(out)])
    monkeypatch.setattr("builtins.input", lambda *_: next(feed))
    pw._write_mpmc_options(mol)
    recs = [
        ln.split()
        for ln in out.read_text().splitlines()
        if ln.startswith("ATOM") and " BOX " not in ln
    ]
    assert sorted(float(r[10]) for r in recs if r[4] == "F") == [-0.7, 0.7]
    assert [float(r[10]) for r in recs if r[4] == "M" and r[2] == "M"] == [-1.038, -1.038]


def test_tui_charges_step_covers_framework_only(tmp_path):
    (tmp_path / "q.txt").write_text("0.7\n-0.7\n")
    notes, steps = [], []

    class App:
        molecule = read()

        def notify(self, msg, **kw):
            notes.append(msg)

        def _mpmc_step_ask_ff(self):
            steps.append("ff")

    app = App()
    pw.PdbWizardApp.__dict__["_mpmc_step_charges_file"](app, {"path": str(tmp_path / "q.txt")})
    assert steps == ["ff"] and [a.charge for a in app.molecule.atoms[:2]] == [0.7, -0.7]
    assert app.molecule.atoms[5].charge == -1.038
    (tmp_path / "bad.txt").write_text("0.1\n0.2\n0.3\n")
    steps.clear()
    pw.PdbWizardApp.__dict__["_mpmc_step_charges_file"](app, {"path": str(tmp_path / "bad.txt")})
    assert not steps and any("framework atom count" in n for n in notes)


def test_qeq_leaves_movable_charges_alone():
    mol = read()
    before = water_params(mol)
    pw.apply_qeq_charges(mol)
    assert water_params(mol) == before
    assert sum(a.charge for a in mol.atoms[:2]) == pytest.approx(0.0, abs=1e-6)
