"""Differential regression tests: this pdb_wizard.py against a baseline version.

Both modules are loaded side by side and run over the same structures. Results must
match exactly, except for the changes the insert/round-trip work makes on purpose (and
the three writer bugs listed in test_other_writers_are_unchanged):

  * MPMC box remarks are written ``REMARK BOX BASIS[i] = ...`` (MPMC needs the '=')
  * the BOX pseudo-molecule gets the ID after the last movable molecule
  * movable (M) molecules read from an MPMC file stay movable and keep their model
    parameters; massless MPMC dummy sites no longer trigger the unknown-element warning

Options (see conftest.py):
  --baseline REF_OR_PATH   baseline: a git ref (its sources are built with build.py) or the
                           path of a built pdb_wizard.py (default: main)
  --corpus DIR             extra structures to run (every *.pqr/*.cif/*.xyz/*.pdb under DIR),
                           e.g. MPMC's tutorials_and_examples/pqrs
  --mpmc PATH              also compare MPMC single-point energies

Without a usable baseline (e.g. after merging, when main *is* this code) the tests skip.
Run from the repo root:  python -m pytest tests/test_pdb_wizard_regression.py
"""

import contextlib
import copy
import importlib.util
import io
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dist" / "pdb_wizard.py"  # built by conftest.py
MAX_QEQ_ATOMS = 400


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while it loads
    spec.loader.exec_module(mod)
    return mod


pw = load("pdb_wizard_under_test", SRC)


@pytest.fixture(scope="session")
def bw(request, tmp_path_factory):
    """The baseline module."""
    ref = request.config.getoption("--baseline")
    path = Path(ref)
    if not path.is_file():
        # build the single file from that git ref's sources, as build.py would
        tree = tmp_path_factory.mktemp("baseline")
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", ref], capture_output=True)
        if archive.returncode != 0:
            pytest.skip(f"no baseline: {ref!r} is neither a file nor a git ref here")
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
        subprocess.run([sys.executable, "build.py"], cwd=tree, check=True, capture_output=True)
        path = tree / "dist" / "pdb_wizard.py"
    if path.read_text() == SRC.read_text():
        pytest.skip("baseline is identical to the code under test")
    return load("pdb_wizard_baseline", path)


# ---- the structures ---------------------------------------------------------------------


def _synthetic(tmp):
    """Small structures in every input dialect, made without pdb_wizard."""
    b = np.radians(99.0)
    basis = np.array([[12.0, 0, 0], [0, 10.0, 0], [11.0 * np.cos(b), 0, 11.0 * np.sin(b)]])
    frac = np.array(np.meshgrid(*[np.arange(0, 1, 1 / 3)] * 3)).reshape(3, -1).T
    xyz = frac @ basis
    els = ["C", "O", "H", "N", "Cu", "Zn", "F", "Cl", "S"]
    files = {}
    cryst = "CRYST1   12.000   10.000   11.000  90.00  99.00  90.00 P 1           1\n"
    remark = "".join(
        f"REMARK BOX BASIS[{i}] = {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n" for i, r in enumerate(basis)
    )
    lines = [
        f"ATOM {i + 1} {els[i % 9]} MOF F 1 {x:.4f} {y:.4f} {z:.4f} 12.0 {0.1 * (-1) ** i:.2f}"
        f" 0.5 30.0 3.2 0.0 0.0 0.0 0.0 0.0\n"
        for i, (x, y, z) in enumerate(xyz)
    ]
    files["framework_mpmc.pqr"] = cryst + "".join(lines) + remark + "END\n"
    water = (
        "ATOM 900 OXY H2O M 2 6.0 5.0 2.0 15.9994 0.0 0.0 91.84 3.234 0 0 0 0 0\n"
        "ATOM 901 HYD H2O M 2 6.757 4.414 2.0 1.008 0.519 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 902 HYD H2O M 2 5.243 4.414 2.0 1.008 0.519 0.0 0.0 0.0 0 0 0 0 0\n"
        "ATOM 903 M H2O M 2 6.0 4.785 2.0 0.0 -1.038 1.444 0.0 0.0 0 0 0 0 0\n"
    )
    files["framework_water_mpmc.pqr"] = cryst + "".join(lines) + water + remark + "END\n"
    files["plain.pdb"] = (
        cryst
        + "".join(
            f"ATOM  {i + 1:5d} {els[i % 9]:>2}   LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {els[i % 9]:>2}\n"
            for i, (x, y, z) in enumerate(xyz)
        )
        + "END\n"
    )
    files["pdb2pqr.pqr"] = (
        "".join(
            f"ATOM  {i + 1:5d}  {els[i % 9]:<3} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
            f" {0.1 * (-1) ** i:7.4f} 1.7000\n"
            for i, (x, y, z) in enumerate(xyz)
        )
        + "END\n"
    )
    files["mol.xyz"] = f"{len(xyz)}\ncomment\n" + "".join(
        f"{els[i % 9]} {x:.5f} {y:.5f} {z:.5f}\n" for i, (x, y, z) in enumerate(xyz)
    )
    cif = [
        "data_test",
        "_cell_length_a 12.0",
        "_cell_length_b 10.0",
        "_cell_length_c 11.0",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 99",
        "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    cif += [
        f"{els[i % 9]}{i} {els[i % 9]} {f[0]:.5f} {f[1]:.5f} {f[2]:.5f}" for i, f in enumerate(frac)
    ]
    files["cell.cif"] = "\n".join(cif) + "\n"
    out = []
    for name, text in files.items():
        (tmp / name).write_text(text)
        out.append(tmp / name)
    return out


def pytest_generate_tests(metafunc):  # noqa: D103 (pytest hook, kept local to this module)
    if "structure" not in metafunc.fixturenames:
        return
    ids = [
        "synthetic:" + n
        for n in (
            "framework_mpmc.pqr",
            "framework_water_mpmc.pqr",
            "plain.pdb",
            "pdb2pqr.pqr",
            "mol.xyz",
            "cell.cif",
        )
    ]
    corpus = metafunc.config.getoption("--corpus")
    if corpus:
        for f in sorted(Path(corpus).rglob("*")):
            if f.suffix.lower() in (".pqr", ".pdb", ".cif", ".xyz") and f.is_file():
                ids.append(str(f))
    metafunc.parametrize("structure", ids, indirect=True)


@pytest.fixture
def structure(request, tmp_path_factory):
    if request.param.startswith("synthetic:"):
        tmp = tmp_path_factory.getbasetemp() / "synthetic"
        if not tmp.exists():
            tmp.mkdir()
            _synthetic(tmp)
        return tmp / request.param.split(":", 1)[1]
    return Path(request.param)


# ---- helpers ----------------------------------------------------------------------------

_READ_CACHE: dict = {}


def read_both(bw, path):
    """Read with both modules (once per session, then copies), with each one's log warnings."""
    key = str(path)
    if key not in _READ_CACHE:
        _READ_CACHE[key] = _read_both_uncached(bw, path)
    return copy.deepcopy(_READ_CACHE[key])


def _read_both_uncached(bw, path):
    out = []
    for mod in (bw, pw):
        logs = io.StringIO()
        handler = logging.StreamHandler(logs)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            try:
                mol = mod.read_file(str(path))
            except Exception as e:  # both must fail the same way
                mol = e
        finally:
            root.removeHandler(handler)
        out.append((mol, logs.getvalue()))
    return out


def has_movable(path):
    """MPMC records with the movable flag (MPMC files may also be named .pdb)."""
    if path.suffix.lower() not in (".pqr", ".pdb"):
        return False
    for ln in path.read_text(errors="replace").splitlines():
        t = ln.split()
        if len(t) > 5 and t[0] in ("ATOM", "HETATM") and t[4] == "M":
            return True
    return False


def atom_rows(mol):
    return [
        (
            a.name,
            a.element.symbol,
            *np.round(a.x, 10),
            a.charge,
            a.alpha,
            a.epsilon,
            a.sigma,
            a.c6,
            a.c8,
            a.c10,
            a.mass,
        )
        for a in mol.atoms
    ]


def cell(mol):
    p = mol.pbc
    return None if p is None else (p.a, p.b, p.c, p.alpha, p.beta, p.gamma)


def normalize_mpmc(text):
    """Remove the two intended format changes: the '=' and the BOX molecule ID."""
    text = re.sub(r"^(REMARK BOX BASIS\[\d\]) = ", r"\1  ", text, flags=re.M)
    return re.sub(r"^(ATOM\s+\d+ X\s+BOX F)\s+\d+", r"\1 B", text, flags=re.M)


def same_failure(a, b):
    """Both raised, the same kind of error (each module has its own exception classes)."""
    return (
        isinstance(a, Exception)
        and isinstance(b, Exception)
        and type(a).__name__ == type(b).__name__
    )


def raises_like(e):
    """pytest.raises for the under-test version of the baseline's exception ``e``."""
    return pytest.raises(getattr(pw, type(e).__name__, type(e)))


def mpmc_text(mod, mol, **kw):
    buf = io.StringIO()
    mod.write_mpmc_pdb(mol, buf, **kw)
    return buf.getvalue()


# ---- reading ----------------------------------------------------------------------------


def test_read_is_unchanged(bw, structure):
    (b, blog), (p, plog) = read_both(bw, structure)
    if isinstance(b, Exception) or isinstance(p, Exception):
        assert same_failure(b, p), (b, p)
        return
    assert cell(p) == cell(b)
    assert atom_rows(p) == atom_rows(b)
    # warnings: identical, except that the dummy-site report may disappear
    strip = re.compile(r"Unrecognised element names.*\n?")
    if strip.sub("", plog) != strip.sub("", blog):
        assert plog == blog
    if "Unrecognised" in plog:
        assert plog == blog  # anything still reported is reported the same way


# ---- writing ----------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["xyz", "pdb", "cif", "poscar", "lammps"])
def test_other_writers_are_unchanged(bw, structure, fmt):
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception):
        return
    writer = {
        "xyz": "write_xyz",
        "pdb": "write_standard_pdb",
        "cif": "write_cif",
        "poscar": "write_poscar",
        "lammps": "write_lammps_data",
    }[fmt]
    outs = []
    for mod, mol in ((bw, b), (pw, p)):
        buf = io.StringIO()
        try:
            getattr(mod, writer)(mol, buf)
            outs.append(buf.getvalue())
        except Exception as e:
            outs.append(type(e).__name__)
    old, new = outs
    if new == old:
        return
    # Three bugs fixed on purpose (io bug hunt, 2026-11) change what these writers give for
    # some structures. Each case below asserts the corrected relationship to the baseline;
    # everything else must still be equal.
    massless = _massless_sites(p)
    if fmt in ("xyz", "pdb", "cif"):
        # Bug: massless MPMC model sites (water M, H2 centre) were written as atoms of
        # element X. They are left out now. Only structures that have such sites may differ,
        # and only by those sites.
        assert massless, "no massless model sites: the output must not change"
        expected = {"xyz": _xyz_without, "pdb": _pdb_without, "cif": _cif_without}[fmt]
        assert new == expected(old, new, b, massless)
    elif fmt == "poscar":
        # Bug: the positions were written in structure order under species counts grouped
        # by element, so VASP gave atoms the wrong element unless the structure happened to
        # be sorted. They are grouped by species now (and massless sites are left out).
        symbols = [a.element.symbol for a in b.atoms]
        assert massless or not _grouped(symbols), "already grouped: the output must not change"
        _assert_same_poscar_content(old, new, symbols, massless)
    else:
        # Bug: every cell was written as an orthogonal box a x b x c. A non-orthogonal cell
        # now gets LAMMPS's triclinic box and tilt line (and massless sites are left out).
        angles = cell(p)[3:] if p.pbc is not None else (90.0, 90.0, 90.0)
        skewed = any(abs(v - 90.0) > 1e-9 for v in angles)
        assert skewed or massless, "orthogonal, no model sites: the output must not change"
        _assert_same_lammps_content(old, new, cell(p), skewed, massless)


def _massless_sites(mol):
    """Indices of the massless model sites of movable MPMC molecules."""
    return [
        i
        for i, a in enumerate(mol.atoms)
        if getattr(a, "mpmc_flag", None) == "M"
        and getattr(a, "file_mass", None) is not None
        and a.file_mass < 0.5
    ]


def _grouped(symbols):
    """True if every element's atoms are consecutive (the only case the old POSCAR was right)."""
    seen, last = set(), None
    for s in symbols:
        if s != last and s in seen:
            return False
        seen.add(s)
        last = s
    return True


def _xyz_without(old, new, mol, massless):
    """The baseline XYZ with the massless sites' lines removed and the count lowered."""
    lines = old.splitlines(True)
    keep = [ln for i, ln in enumerate(lines[2:]) if i not in set(massless)]
    return f"{len(keep)}{lines[0][len(lines[0].rstrip()):]}" + lines[1] + "".join(keep)


def _cif_without(old, new, mol, massless):
    """The baseline CIF with the massless sites' rows (one row per atom, in order) removed."""
    lines = old.splitlines(True)
    first = max(i for i, ln in enumerate(lines) if ln.startswith("_atom_site_")) + 1
    return "".join(lines[:first]) + "".join(
        ln for i, ln in enumerate(lines[first:]) if i not in set(massless)
    )


def _pdb_without(old, new, mol, massless):
    """The baseline PDB with the massless sites' records removed and the serials closed up.

    A residue that lost its massless site may also change its name: the writer names a
    molecule by its elements, and a four-site water (H, H, O plus the M site) only matches
    its H/H/O entry, HOH, once the M site is gone. Before, it was UNK. That one change of
    name is allowed; any other residue name must stay.
    """
    wanted = {tuple(f"{v:8.3f}" for v in mol.atoms[i].x) for i in massless}
    out, serial, touched = [], 0, set()
    for ln in old.splitlines(True):
        if not ln.startswith("HETATM"):
            out.append(ln)
            continue
        if ln[76:78].strip() == "X" and (ln[30:38], ln[38:46], ln[46:54]) in wanted:
            touched.add(ln[22:26])
            continue
        serial += 1
        out.append(f"HETATM{serial:>5}" + ln[11:])
    assert len(touched) and serial == len(mol.atoms) - len(massless)
    new_lines = new.splitlines(True)
    for k, ln in enumerate(out):
        if ln.startswith("HETATM") and ln[22:26] in touched:
            elements = sorted(
                o[76:78].strip() for o in out if o.startswith("HETATM") and o[22:26] == ln[22:26]
            )
            name = "HOH" if elements == ["H", "H", "O"] else ln[17:20]
            assert new_lines[k][17:20] == name
            out[k] = ln[:17] + name + ln[20:]
    return "".join(out)


def _poscar_parts(text):
    lines = text.splitlines()
    species, counts = lines[5].split(), [int(v) for v in lines[6].split()]
    return lines[:5], dict(zip(species, counts)), lines[7], lines[8:]


def _assert_same_poscar_content(old, new, symbols, massless):
    """Same lattice, same species counts and the same (element, position) pairs. The
    element of a baseline position is the one of its atom in the structure (that is what
    the baseline meant); in the new file it is the one VASP assigns: the first N1 positions
    to the first species, the next N2 to the second."""
    head_old, counts_old, mode_old, pos_old = _poscar_parts(old)
    head_new, counts_new, mode_new, pos_new = _poscar_parts(new)
    assert head_new == head_old and mode_new == mode_old
    gone = set(massless)
    kept = [(s, ln) for i, (s, ln) in enumerate(zip(symbols, pos_old)) if i not in gone]
    for i in gone:
        counts_old[symbols[i]] -= 1
    assert counts_new == {s: n for s, n in counts_old.items() if n}
    assigned = [s for s, n in counts_new.items() for _ in range(n)]
    assert len(assigned) == len(pos_new)
    assert sorted(zip(assigned, pos_new)) == sorted(kept)
    assert _grouped(assigned)


def _lammps_parts(text):
    head, rest = text.split("\nMasses\n", 1)
    masses, rest = rest.split("\nAtoms\n", 1)
    atoms, _, bonds = rest.partition("\nBonds\n")
    types = {ln.split()[0]: ln.split()[-1] for ln in masses.strip().splitlines()}
    rows = [ln.split() for ln in atoms.strip().splitlines()]
    box = [ln for ln in head.splitlines() if ln.endswith(("lo xhi", "lo yhi", "lo zhi", "yz"))]
    other = [ln for ln in head.splitlines() if ln not in box]
    return other, box, masses, rows, types, bonds


def _assert_same_lammps_content(old, new, cellpar, skewed, massless):
    """The box is the one LAMMPS defines for this cell (computed here from the cell
    parameters: a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz)); the atom lines are the
    baseline's, without the massless sites."""
    o_head, o_box, o_masses, o_rows, o_types, o_bonds = _lammps_parts(old)
    n_head, n_box, n_masses, n_rows, n_types, n_bonds = _lammps_parts(new)
    a, b, c = cellpar[:3]
    al, be, ga = np.radians(cellpar[3:])
    lx, xy, ly = a, b * np.cos(ga), b * np.sin(ga)
    xz = c * np.cos(be)
    yz = (b * c * np.cos(al) - xy * xz) / ly
    lz = np.sqrt(c * c - xz * xz - yz * yz)
    want = [f"0.000000 {lx:.6f} xlo xhi", f"0.000000 {ly:.6f} ylo yhi",
            f"0.000000 {lz:.6f} zlo zhi"]
    if skewed:
        want.append(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz".replace("-0.000000", "0.000000"))
    assert [ln.replace("-0.000000", "0.000000") for ln in n_box] == want
    if not skewed:
        assert n_box == o_box
    if not massless:  # only the box differs
        assert (n_head, n_masses, n_rows, n_bonds) == (o_head, o_masses, o_rows, o_bonds)
        return
    # ids, molecule ids and type ids close up when sites go: compare what an atom line says
    gone = set(massless)
    kept = [r for i, r in enumerate(o_rows) if i not in gone]
    assert len(n_rows) == len(kept)
    assert [(n_types[r[2]], r[3:]) for r in n_rows] == [(o_types[r[2]], r[3:]) for r in kept]
    assert [r[0] for r in n_rows] == [str(i + 1) for i in range(len(kept))]


def test_mpmc_export_is_unchanged_for_frameworks(bw, structure):
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception):
        return
    try:
        old = normalize_mpmc(mpmc_text(bw, b))
    except Exception as e:
        with raises_like(e):
            mpmc_text(pw, p)
        return
    new = normalize_mpmc(mpmc_text(pw, p))
    if not has_movable(structure):
        assert new == old
        return

    # with movable molecules the old writer froze them into the framework (the bug):
    # the framework records must still match, as a set, and the waters must be M now
    def rows(text, flag):
        return sorted(
            tuple(ln.split()[2:3] + ln.split()[6:14])
            for ln in text.splitlines()
            if ln.startswith("ATOM") and " BOX " not in ln and ln.split()[4] == flag
        )

    if rows(old, "M"):  # a baseline from after that fix: nothing may differ
        assert new == old
        return
    moved = [a for a in p.atoms if a.mpmc_flag == "M"]
    assert len(rows(new, "M")) == len(moved) > 0
    assert set(rows(new, "F")) <= set(rows(old, "F"))
    assert len(rows(new, "F")) + len(moved) == len(rows(old, "F"))


def test_mpmc_export_with_a_sorbate_line_is_unchanged(bw, structure):
    """The single-sorbate path (old TUI / isotherm generator) on framework-only input."""
    if has_movable(structure):
        return
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception) or b.pbc is None:
        return
    line = bw.format_sorbate_pqr(
        "h2o_tip3p", 1.0, 1.0, 1.0, mol_id=2, start_atom_id=len(b.atoms) + 1
    )
    assert (
        pw.format_sorbate_pqr("h2o_tip3p", 1.0, 1.0, 1.0, mol_id=2, start_atom_id=len(p.atoms) + 1)
        == line
    )
    assert normalize_mpmc(mpmc_text(pw, p, sorbate_lines=line)) == normalize_mpmc(
        mpmc_text(bw, b, sorbate_lines=line)
    )


# ---- editing ----------------------------------------------------------------------------


@pytest.mark.parametrize("ff_idx", [0, 1])
def test_force_fields_are_unchanged_on_frameworks(bw, structure, ff_idx):
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception):
        return
    _, bmiss = bw.apply_ff_to_system_report(b.atoms, bw.get_forcefield(ff_idx))
    _, pmiss = pw.apply_ff_to_system_report(p.atoms, pw.get_forcefield(ff_idx))
    if not has_movable(structure):
        assert atom_rows(p) == atom_rows(b) and pmiss == bmiss
    else:  # framework atoms identical; movable ones are deliberately left alone now
        fw = [i for i, a in enumerate(p.atoms) if a.mpmc_flag != "M"]
        assert [atom_rows(p)[i] for i in fw] == [atom_rows(b)[i] for i in fw]


def test_extend_axis_is_unchanged(bw, structure):
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception) or b.pbc is None:
        return
    for axis, times in ((0, 1), (2, 2)):
        b = bw.extend_axis(b, axis, times)
        p = pw.extend_axis(p, axis, times)
    assert cell(p) == cell(b) and atom_rows(p) == atom_rows(b)


def test_qeq_runs_on_the_same_small_frameworks(bw, structure):
    """QEq was rewritten on purpose (2026-11-10: the old version used a bare 14.4/R and
    inverted the signs), so the values differ from older baselines. What must not change
    is which structures it runs on; tests/test_qeq_reference.py checks the values."""
    if has_movable(structure):
        return
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception) or not 0 < len(b.atoms) <= MAX_QEQ_ATOMS:
        return
    try:
        bw.apply_qeq_charges(b)
    except Exception as e:
        with raises_like(e):
            pw.apply_qeq_charges(p)
        return
    pw.apply_qeq_charges(p)
    charges = np.array([a.charge for a in p.atoms])
    assert np.all(np.isfinite(charges)) and abs(charges.sum()) < 1e-6


def test_charges_file_prompt_is_unchanged_on_frameworks(bw, structure, tmp_path, monkeypatch):
    if has_movable(structure):
        return
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception) or b.pbc is None:
        return
    q = tmp_path / "q.txt"
    q.write_text("".join(f"{0.01 * ((i % 7) - 3):.4f}\n" for i in range(len(b.atoms))))
    texts = []
    for mod, mol, name in ((bw, b, "old.pqr"), (pw, p, "new.pqr")):
        feed = iter(["y", str(q), "n", str(tmp_path / name)])
        monkeypatch.setattr("builtins.input", lambda *_, f=feed: next(f))
        with contextlib.redirect_stdout(io.StringIO()):
            mod._write_mpmc_options(mol)
        texts.append(normalize_mpmc((tmp_path / name).read_text()))
    assert texts[1] == texts[0]


def test_info_is_unchanged(bw, structure):
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception):
        return
    outs = []
    for mod, mol in ((bw, b), (pw, p)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.print_info(mol, "x")
        outs.append(buf.getvalue())
    assert outs[1] == outs[0]


# ---- through MPMC -----------------------------------------------------------------------


@pytest.fixture
def mpmc_bin(request):
    path = request.config.getoption("--mpmc") or shutil.which("mpmc")
    if not path:
        pytest.skip("pass --mpmc /path/to/mpmc to run MPMC itself")
    return path


def mpmc_energy(mpmc_bin, workdir, text, basis):
    workdir.mkdir()
    (workdir / "in.pqr").write_text(text)
    (workdir / "sp.inp").write_text(
        "job_name sp\nensemble total_energy\ntemperature 298\npqr_input in.pqr\n"
        + "".join(
            f"basis{i + 1} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n" for i, r in enumerate(basis)
        )
    )
    out = subprocess.run(
        [mpmc_bin, "sp.inp"], cwd=workdir, capture_output=True, text=True, timeout=600
    ).stdout
    if "potential energy =" not in out:
        return out.strip().splitlines()[-1:]  # compare the failure instead
    return float(out.split("potential energy =")[1].split()[0])


def test_mpmc_energy_is_unchanged_for_frameworks(bw, structure, tmp_path, mpmc_bin):
    """Framework plus one test water, written by both versions, same MPMC energy."""
    if has_movable(structure):
        return
    (b, _), (p, _) = read_both(bw, structure)
    if isinstance(b, Exception) or b.pbc is None or not any(a.sigma > 0 for a in b.atoms):
        return
    line = bw.format_sorbate_pqr(
        "h2o_tip3p",
        *map(float, b.pbc.basis_matrix.sum(axis=0) / 2),
        mol_id=2,
        start_atom_id=len(b.atoms) + 1,
    )
    e_old = mpmc_energy(
        mpmc_bin, tmp_path / "old", mpmc_text(bw, b, sorbate_lines=line), b.pbc.basis_matrix
    )
    e_new = mpmc_energy(
        mpmc_bin, tmp_path / "new", mpmc_text(pw, p, sorbate_lines=line), p.pbc.basis_matrix
    )
    both_nan = (
        isinstance(e_old, float)
        and isinstance(e_new, float)
        and np.isnan(e_old)
        and np.isnan(e_new)
    )
    assert both_nan or e_new == e_old  # some example frameworks give nan in MPMC; then both must
