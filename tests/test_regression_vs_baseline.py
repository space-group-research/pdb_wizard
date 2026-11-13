"""Differential regression tests: this pdb_wizard.py against a baseline version.

Both modules are loaded side by side and run over the same structures. Results must
match exactly, except for the changes the insert/round-trip work makes on purpose:

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
    assert outs[1] == outs[0]


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
