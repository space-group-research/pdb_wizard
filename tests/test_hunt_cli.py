"""Bugs found by the classic-menu / command-line / single-file bug hunt.

Each test reproduces one finding and checks the corrected behaviour against something
independent of the code under test: the file a plain ``--classic`` session writes, the
bytes of the input file, the value the user typed, a full copy of the structure taken
before an edit.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import pdb_wizard.__main__ as entry
import pdb_wizard.menu as menu
import pdb_wizard.menu_tools as mt
from pdb_wizard.atom import Atom
from pdb_wizard.checks import load_notes
from pdb_wizard.formats import write_cif
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis
from pdb_wizard.pbc import PBC
from pdb_wizard.sorbates import SORBATE_MODELS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
HKUST = ROOT / "HKUST-1.pdb"
DIST = ROOT / "dist" / "pdb_wizard.py"


# ---- helpers --------------------------------------------------------------------------


def run_cli(args, stdin="", cwd=None, blocked=None, dist=False, timeout=120):
    """``python -m pdb_wizard`` (or the single file) as a real process with piped stdin.
    ``blocked`` is a folder holding a ``textual.py`` that raises ImportError."""
    path = [str(blocked)] if blocked else []
    if not dist:
        path.append(str(SRC))
    env = {"PATH": "/usr/bin:/bin", "TERM": "dumb"}
    if path:
        env["PYTHONPATH"] = ":".join(path)
    cmd = [sys.executable, str(DIST)] if dist else [sys.executable, "-m", "pdb_wizard"]
    return subprocess.run(cmd + list(args), input=stdin, capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None, env=env, timeout=timeout)


@pytest.fixture
def no_textual(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "textual.py").write_text('raise ImportError("textual blocked for test")\n')
    return blocker


@pytest.fixture
def work(tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    shutil.copy(HKUST, folder)
    return folder


def drive(func, responses, *args):
    """Run ``func`` with scripted input(); returns (result, printed text)."""
    printed: list = []
    it = iter(responses)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"menu asked for more input than scripted: {prompt!r}")

    def fake_print(*a, **k):
        printed.append(" ".join(str(x) for x in a))

    with mock.patch("builtins.input", fake_input), mock.patch("builtins.print", fake_print):
        result = func(*args)
    assert next(it, None) is None, "scripted input left over"
    return result, "\n".join(printed)


def small_mol() -> Molecule:
    """32 atoms with distinct charges and parameters, some outside the cell."""
    rng = np.random.default_rng(42)
    atoms = [Atom(*rng.uniform(-3, 17, 3), el) for el in ("C", "O", "N", "H") for _ in range(8)]
    for i, a in enumerate(atoms):
        a.charge, a.alpha, a.epsilon, a.sigma = 0.01 * i, 1.0 + i, 2.0 + i, 3.0 + i
        a.c6, a.c8, a.c10 = 4.0 + i, 5.0 + i, 6.0 + i
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()
    return mol


def water_frames(n_frames: int = 4) -> list:
    rng = np.random.default_rng(7)
    centres = rng.uniform(2, 13, (5, 3))
    frames = []
    for f in range(n_frames):
        atoms = []
        for c in centres:
            o = c + np.array([0.25 * f, 0, 0])
            atoms.append(Atom(*o, "O"))
            for d in ((0.96, 0, 0), (-0.24, 0.93, 0)):
                atoms.append(Atom(*(o + np.array(d)), "H"))
        m = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
        m.detect_bonds()
        frames.append(m)
    return frames


def full_state(mol: Molecule) -> tuple:
    """Everything an edit can touch (more than the menu's own change signature)."""
    p = mol.pbc
    cell = None if p is None else (p.a, p.b, p.c, p.alpha, p.beta, p.gamma)
    atoms = tuple((a.name, a.element.symbol, tuple(float(v) for v in a.x), a.charge, a.alpha,
                   a.epsilon, a.sigma, a.c6, a.c8, a.c10, a.id, a.mpmc_flag, a.mpmc_mol,
                   a.mpmc_res, a.file_mass) for a in mol.atoms)
    return (cell, atoms, tuple(sorted(tuple(sorted(b)) for b in mol.bonds)),
            tuple(load_notes(mol)))


def snapshots_of_single_menu(mol, answers):
    """Drive the single-structure menu; returns the structure's full state at every redraw
    of the main menu (print_info is called with the current structure each time)."""
    states = []
    with mock.patch.object(menu, "print_info", lambda m, f: states.append(full_state(m))):
        drive(menu._main_loop_single, answers, mol, "t.pdb")
    return states


# ---- 1: a script piped in without --classic -------------------------------------------

WRAP_SCRIPT = "3\n6\nout.xyz\n9\n"


def _reference_wrap(work) -> str:
    ref = run_cli(["--classic", "HKUST-1.pdb"], WRAP_SCRIPT, cwd=work)
    assert ref.returncode == 0
    text = (work / "out.xyz").read_text()
    (work / "out.xyz").unlink()
    return text


def test_piped_script_without_classic_and_without_textual(work, no_textual):
    expected = _reference_wrap(work)
    result = run_cli(["HKUST-1.pdb"], WRAP_SCRIPT, cwd=work, blocked=no_textual)
    assert result.returncode == 0, result.stderr
    assert "Install it now" not in result.stdout + result.stderr  # nothing ate the "3"
    assert "falling back to the classic text menu" in result.stderr
    assert (work / "out.xyz").read_text() == expected  # the wrap really happened


def test_piped_script_without_classic_with_textual(work):
    pytest.importorskip("textual")
    expected = _reference_wrap(work)
    result = run_cli(["HKUST-1.pdb"], WRAP_SCRIPT, cwd=work, timeout=60)  # used to hang
    assert result.returncode == 0, result.stderr
    assert (work / "out.xyz").read_text() == expected
    assert "\x1b[?1049h" not in result.stdout + result.stderr  # the viewer never started


def test_piped_cell_answers_are_not_shifted(tmp_path, no_textual):
    (tmp_path / "w.xyz").write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    script = "20\n21\n22\n90\n90\n90\n8\nout.pdb\n9\n"
    result = run_cli(["w.xyz"], script, cwd=tmp_path, blocked=no_textual)
    assert result.returncode == 0, result.stderr
    cryst = [ln for ln in (tmp_path / "out.pdb").read_text().splitlines()
             if ln.startswith("CRYST1")][0].split()
    assert [float(v) for v in cryst[1:7]] == [20.0, 21.0, 22.0, 90.0, 90.0, 90.0]


def test_single_file_piped_script_without_classic(work, no_textual):
    assert DIST.exists(), "run build.py first"
    expected = _reference_wrap(work)
    result = run_cli(["HKUST-1.pdb"], WRAP_SCRIPT, cwd=work, blocked=no_textual, dist=True)
    assert result.returncode == 0, result.stderr
    assert "Install it now" not in result.stdout + result.stderr
    assert (work / "out.xyz").read_text() == expected


def test_install_prompt_only_on_a_terminal(monkeypatch):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "n")
    monkeypatch.setattr(entry, "_stdin_is_terminal", lambda: False)
    assert entry._ensure_installed("no_such_package_for_this_test") is False
    assert asked == []  # a script is never asked
    monkeypatch.setattr(entry, "_stdin_is_terminal", lambda: True)
    assert entry._ensure_installed("no_such_package_for_this_test") is False
    assert len(asked) == 1 and "Install it now" in asked[0]  # a person still is


def test_terminal_launch_still_takes_the_viewer_branch(monkeypatch):
    pytest.importorskip("textual")
    import pdb_wizard.viewer as viewer

    class Reached(Exception):
        pass

    def boom(*a, **k):
        raise Reached

    monkeypatch.setattr(entry, "_stdin_is_terminal", lambda: True)
    monkeypatch.setattr(viewer, "PdbWizardApp", boom)
    monkeypatch.setattr(menu, "run_classic", lambda fp: pytest.fail("classic menu on a tty"))
    monkeypatch.setattr(sys, "argv", ["pdb_wizard", str(HKUST)])
    with pytest.raises(Reached):
        entry.main()


# ---- 3: the single file uses only its bundled templates -------------------------------


def test_single_file_ignores_a_same_named_file_beside_it(tmp_path):
    assert DIST.exists(), "run build.py first"
    shutil.copy(DIST, tmp_path / "pdb_wizard.py")
    shutil.copy(HKUST, tmp_path)
    (tmp_path / "cp2k_cell_opt.inp").write_text("&GLOBAL\n  PROJECT some_old_job\n&END GLOBAL\n")
    script = "14\n2\n\n\n\n\n\n\n\nout_cp2k\n9\n"  # CP2K cell optimisation, all defaults
    result = subprocess.run([sys.executable, str(tmp_path / "pdb_wizard.py"), "--classic",
                             "HKUST-1.pdb"], input=script, capture_output=True, text=True,
                            cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert result.returncode == 0, result.stderr
    written = (tmp_path / "out_cp2k" / "input.inp").read_text()
    # reference: the package's generator, same structure and defaults
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    shutil.copy(HKUST, ref_dir)
    ref = run_cli(["--classic", "HKUST-1.pdb"], script, cwd=ref_dir)
    assert ref.returncode == 0, ref.stderr
    assert written == (ref_dir / "out_cp2k" / "input.inp").read_text()
    assert "some_old_job" not in written


# ---- 5: --convert on a trajectory -----------------------------------------------------


def _write_xyz_trajectory(path: Path, n_frames: int = 4) -> None:
    with open(path, "w") as f:
        for k in range(n_frames):
            f.write(f"3\nframe {k}\nO {k}.0 0 0\nH {k}.96 0 0\nH {k}.0 0.93 0\n")


def test_convert_warns_that_only_the_first_frame_is_written(tmp_path):
    _write_xyz_trajectory(tmp_path / "t.xyz")
    result = run_cli(["--convert", "t.xyz", "-o", "one.xyz"], cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout == "Converted t.xyz -> one.xyz\n"  # stdout as before
    assert "4 frames" in result.stderr and "first" in result.stderr
    out = (tmp_path / "one.xyz").read_text().split()
    assert out[0] == "3" and "0.96" in out and "1.96" not in out  # still frame 1 only


def test_convert_refuses_to_overwrite_a_trajectory_with_its_first_frame(tmp_path):
    _write_xyz_trajectory(tmp_path / "t.xyz")
    before = (tmp_path / "t.xyz").read_bytes()
    for target in ("t.xyz", "./t.xyz"):
        result = run_cli(["--convert", "t.xyz", "-o", target], cwd=tmp_path)
        assert result.returncode != 0
        assert "4 frames" in result.stderr
        assert (tmp_path / "t.xyz").read_bytes() == before
    # a single structure may still be converted onto itself, as before
    (tmp_path / "s.xyz").write_text("1\nx\nAr 0.0 0.0 0.0\n")
    assert run_cli(["--convert", "s.xyz", "-o", "s.xyz"], cwd=tmp_path).returncode == 0


# ---- 6: batch mode shows what the reader had to assume --------------------------------


def test_batch_mode_prints_cif_load_notes_on_stderr(tmp_path):
    import io
    buf = io.StringIO()
    write_cif(read_file(str(HKUST)), buf)
    p1 = buf.getvalue()
    assert "_symmetry_equiv_pos_as_xyz" in p1
    head, rest = p1.split("_symmetry_space_group_name_H-M", 1)
    atoms = rest[rest.index("loop_\n_atom_site_label"):]
    noted = (head + "_symmetry_space_group_name_H-M   'F m -3 m'\n"
             "_symmetry_Int_Tables_number       225\n\n" + atoms)
    (tmp_path / "p1.cif").write_text(p1)
    (tmp_path / "noted.cif").write_text(noted)
    assert load_notes(read_file(str(tmp_path / "noted.cif")))  # the reader does note it
    plain = run_cli(["--info", "p1.cif"], cwd=tmp_path)
    flagged = run_cli(["--info", "noted.cif"], cwd=tmp_path)
    assert plain.returncode == flagged.returncode == 0
    assert "asymmetric unit" in flagged.stderr and "asymmetric unit" not in plain.stderr
    assert flagged.stdout.replace("noted.cif", "p1.cif") == plain.stdout  # stdout untouched
    converted = run_cli(["--convert", "noted.cif", "-o", "o.pqr"], cwd=tmp_path)
    assert converted.returncode == 0 and "asymmetric unit" in converted.stderr
    assert converted.stdout == "Converted noted.cif -> o.pqr\n"


# ---- 7: "open another file" on something unreadable -----------------------------------


@pytest.mark.parametrize("name, content", [
    ("garbage.out", b"nothing here\n"),
    ("binary.pdb", bytes(range(128, 256)) * 3),
])
def test_open_another_file_keeps_the_session_when_it_cannot_be_read(work, name, content):
    expected = _reference_wrap(work)
    (work / name).write_bytes(content)
    result = run_cli(["--classic", "HKUST-1.pdb"], f"3\n18\n{name}\n6\nout.xyz\n9\n", cwd=work)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "!!! Error" in result.stdout
    assert (work / "out.xyz").read_text() == expected  # the wrapped structure survived


def test_open_another_file_still_switches_to_a_readable_file(work):
    (work / "ar.pdb").write_text(
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  Ar  MOL A   1       1.000   1.000   1.000  1.00  0.00          Ar\nEND\n")
    result = run_cli(["--classic", "HKUST-1.pdb"], "18\nar.pdb\n6\nout.xyz\n9\n", cwd=work)
    assert result.returncode == 0, result.stderr
    assert (work / "out.xyz").read_text().split()[0] == "1"


# ---- 8: the minified build must run on the oldest supported Python --------------------


def _build_module():
    spec = importlib.util.spec_from_file_location("pdbw_build_for_test", ROOT / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minified_check_spots_syntax_older_pythons_cannot_read():
    check = _build_module().minified_problem_for_old_python
    assert check('x = {"a": 1}\ny = f"{x[\'a\']:>4}"\n') == ""
    assert check('x = {"a": 1}\ny = f"""{x["a"]}"""\n') == ""
    assert "quote" in check('x = {"a": 1}\ny = f"{x["a"]:>4}"\n') or sys.version_info < (3, 12)
    assert check('x = {"a": 1}\ny = f"{x["a"]:>4}"\n') != ""  # PEP 701 quote reuse
    assert check("match x:\n    case 1:\n        pass\n") != ""  # 3.10+ syntax


def test_build_writes_no_minified_file_that_old_pythons_cannot_read(tmp_path):
    pytest.importorskip("python_minifier")
    shutil.copy(ROOT / "build.py", tmp_path)
    shutil.copytree(SRC, tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run([sys.executable, "build.py"], capture_output=True, text=True,
                            cwd=str(tmp_path), timeout=300)
    assert result.returncode == 0, result.stderr
    minified = tmp_path / "dist" / "pdb_wizard.min.py"
    if minified.exists():
        assert _build_module().minified_problem_for_old_python(minified.read_text()) == ""
    else:
        assert "Minified build skipped" in result.stdout
    assert (tmp_path / "dist" / "pdb_wizard.py").exists()  # the plain file is always built


# ---- 9: stdin closing mid-prompt; output paths that cannot be written -----------------


@pytest.mark.parametrize("script", [
    "3\n6\nout.xyz\n",          # an old script without its final 9
    "11\n5\nCu\n",              # inside an analysis prompt
    "12\n",                     # inside the edit menu
    "5\n30\n",                  # half-way through the cell prompt
])
def test_end_of_input_exits_with_one_line_and_code_1(work, script):
    result = run_cli(["--classic", "HKUST-1.pdb"], script, cwd=work)
    assert result.returncode == 1  # scripts may test it: unchanged
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    if "out.xyz" in script:
        assert (work / "out.xyz").exists()  # the work done before the end is kept


@pytest.mark.parametrize("failing", [
    "6\n/nonexistent_dir/x.xyz\n",
    "6\n\n",
    "8\n/nonexistent_dir/x.pdb\n",
    "7\nn\nn\n/nonexistent_dir/x.pdb\n",
    "7\ny\n.\n",                                   # the charges "file" is a folder
    "10\n0\n1\n/nonexistent_dir/x.pdb\n",
])
def test_unwritable_output_returns_to_the_main_menu(work, failing):
    expected = _reference_wrap(work)
    # the answers after the failure must be read by the main menu, not by a re-prompt
    result = run_cli(["--classic", "HKUST-1.pdb"], "3\n" + failing + "6\nout.xyz\n9\n", cwd=work)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "!!! Error" in result.stdout
    assert (work / "out.xyz").read_text() == expected


def test_classic_on_a_missing_file_gives_one_line(tmp_path):
    result = run_cli(["--classic", "missing.pdb"], "9\n", cwd=tmp_path)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert "missing.pdb" in result.stderr


def test_trajectory_menu_survives_an_unwritable_path(tmp_path):
    out = tmp_path / "ok.xyz"
    drive(menu._main_loop_movie, ["6", "/nonexistent_dir/x.xyz", "8", "", "6", str(out), "9"],
          water_frames(3), "t.xyz")
    assert out.read_text().count("\n15\n") + out.read_text().startswith("15\n") == 3


# ---- 10: the temperature written to mpmc.inp ------------------------------------------


@pytest.mark.parametrize("asked, written", [
    (298.15, "298.15"), (273.15, "273.15"), (77.35, "77.35"),
    (77.0, "77.0"), (87.3, "87.3"), (300, "300.0"), (0.5, "0.5"),  # these are as before
])
def test_isotherm_temperature_is_the_one_asked_for(tmp_path, asked, written):
    from pdb_wizard.isotherm import generate_isotherm
    dirs = generate_isotherm(read_file(str(ROOT / "Ni-MOF-74.pqr")), "h2_bss", asked, [1.0],
                             str(tmp_path / "iso"))
    line = [ln for ln in (Path(dirs[0]) / "mpmc.inp").read_text().splitlines()
            if ln.startswith("temperature")][0]
    assert line.split() == ["temperature", written]
    assert float(line.split()[1]) == float(asked)


# ---- 11: undo covers "write MPMC .pdb" changing charges and parameters ----------------


def test_undo_restores_parameters_changed_by_write_mpmc(tmp_path):
    states = snapshots_of_single_menu(
        small_mol(), ["7", "n", "y", "0", str(tmp_path / "ff.pdb"), "17", "17", "9"])
    original = states[0]  # as the menu first showed it
    assert states[1] != original  # the force field replaced the parameters (as it always did)
    assert states[2] == original  # ... and undo now brings them back
    assert states[3] == original  # nothing more to undo


def test_undo_restores_charges_read_by_write_mpmc(tmp_path):
    charges = tmp_path / "q.dat"
    charges.write_text("".join(f"{0.5 - 0.03 * i:.5f}\n" for i in range(32)))
    states = snapshots_of_single_menu(
        small_mol(), ["7", "y", str(charges), "n", str(tmp_path / "q.pdb"), "17", "9"])
    original = states[0]
    assert [a[3] for a in states[1][1]] == pytest.approx([0.5 - 0.03 * i for i in range(32)])
    assert states[2] == original


def test_signature_sees_parameter_changes():
    mol = small_mol()
    before = mt.cm_signature(mol)
    mol.atoms[5].sigma += 0.25
    assert mt.cm_signature(mol) != before


@pytest.mark.parametrize("action", [
    ["3"], ["4"], ["2", "x", "1"], ["5", "30", "31", "32", "90", "90", "90"],
    ["1", "5", "9"], ["12", "1", "6"], ["12", "2", "O S", "6"],
    ["12", "5", "6"],
])
def test_undo_restores_the_structure_exactly(action):
    states = snapshots_of_single_menu(small_mol(), action + ["17", "9"])
    assert len(states) == 3
    assert states[1] != states[0], "the action was meant to change something"
    assert states[2] == states[0]


# ---- 12: the undo size cutoff ---------------------------------------------------------


def test_one_undo_never_reverts_two_actions(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "CM_UNDO_MAX_ATOMS", 40)  # 32 atoms fit, 64 do not
    out = tmp_path / "o.xyz"
    _, text = drive(menu._main_loop_single,
                    ["2", "x", "1", "3", "17", "6", str(out), "9"], small_mol(), "t.pdb")
    # the wrap of 64 atoms was not recorded, so undo must not jump back to 32 atoms
    assert out.read_text().split()[0] == "64"
    assert "Undid the last change" not in text
    assert "Undo is not available" in text


def test_undo_within_the_cutoff_is_unaffected(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "CM_UNDO_MAX_ATOMS", 40)
    out = tmp_path / "o.xyz"
    _, text = drive(menu._main_loop_single, ["3", "17", "17", "6", str(out), "9"],
                    small_mol(), "t.pdb")
    assert text.count("Undid the last change") == 1 and text.count("Nothing to undo") == 1


# ---- 13: reducing a supercell keeps the file's load notes -----------------------------


def test_reduce_supercell_keeps_load_notes():
    mol = water_frames(1)[0]
    extend_axis(mol, 0, 1)
    mol.detect_bonds()
    mol.load_notes = ["read as the asymmetric unit only"]
    n_before = len(mol.atoms)
    answers = ["2", "1", "1"]
    try:
        result, _ = drive(mt._cm_reduce, answers, mol)
    except AssertionError:  # it lists sorbate molecules and asks which to keep
        result, _ = drive(mt._cm_reduce, answers + [""], mol)
    assert len(result.atoms) < n_before
    assert load_notes(result) == ["read as the asymmetric unit only"]


# ---- 16: batch-mode tracebacks, left-over files, element case -------------------------


@pytest.mark.parametrize("target", ["e.pdb", "e.lmp", "e.data"])
def test_convert_of_an_empty_structure_leaves_no_file(tmp_path, target):
    (tmp_path / "empty.pdb").write_text("")
    result = run_cli(["--convert", "empty.pdb", "-o", target], cwd=tmp_path)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr and "Error" in result.stderr
    assert not (tmp_path / target).exists()


def test_batch_errors_are_one_line_not_tracebacks(work):
    (work / "what.foo").write_text("hello\n")
    cases = [
        ["--convert", "HKUST-1.pdb", "-o", "/nonexistent_dir/x.xyz"],
        ["--rdf", "HKUST-1.pdb", "-e1", "Cu", "-e2", "O", "--csv", "/nonexistent_dir/r.csv"],
        ["--rdf", "what.foo", "-e1", "Cu", "-e2", "O"],
        ["--msd", "what.foo", "-e", "O"],
    ]
    for args in cases:
        result = run_cli(args, cwd=work)
        assert result.returncode == 1, args
        assert "Traceback" not in result.stderr, args
        assert result.stderr.startswith("Error"), args
        assert result.stdout == "", args


def test_msd_csv_into_a_missing_folder_is_one_line(tmp_path):
    _write_xyz_trajectory(tmp_path / "t.xyz", 6)
    result = run_cli(["--msd", "t.xyz", "-e", "O", "--csv", "/nonexistent_dir/m.csv"],
                     cwd=tmp_path)
    assert result.returncode == 1 and "Traceback" not in result.stderr


def test_batch_elements_are_accepted_in_any_case(work, tmp_path):
    proper = run_cli(["--rdf", "HKUST-1.pdb", "-e1", "Cu", "-e2", "O"], cwd=work)
    assert proper.returncode == 0 and len(proper.stdout.splitlines()) > 10
    for e1, e2 in (("cu", "o"), ("CU", "O")):
        other = run_cli(["--rdf", "HKUST-1.pdb", "-e1", e1, "-e2", e2], cwd=work)
        assert other.returncode == 0 and other.stdout == proper.stdout
    _write_xyz_trajectory(tmp_path / "t.xyz", 6)
    upper = run_cli(["--msd", "t.xyz", "-e", "O"], cwd=tmp_path)
    lower = run_cli(["--msd", "t.xyz", "-e", "o"], cwd=tmp_path)
    assert upper.returncode == 0 and lower.stdout == upper.stdout and upper.stdout


# ---- 17: undo snapshots only what an option can change --------------------------------


def _count_structure_copies():
    """Record every snapshot the undo history takes (a Molecule, or a list of them)."""
    copied = []
    real = mt.CmHistory.before

    def spy(self, *a, **k):
        had = self._pending
        real(self, *a, **k)
        if self._pending is not None and self._pending is not had:
            copied.append([x for x in self._pending if isinstance(x, (Molecule, list))][0])

    return copied, mock.patch.object(mt.CmHistory, "before", spy)


def test_undo_snapshot_copies_everything_and_shares_nothing():
    import copy
    mol = read_file(str(ROOT / "HKUST-1_out_mpmc.pdb"))  # MPMC flags, masses, charges
    mol.load_notes = ["a note"]
    reference = full_state(copy.deepcopy(mol))
    snap = mt._cm_snapshot(mol)
    assert full_state(snap) == reference and type(snap) is Molecule
    frames = mt._cm_snapshot([mol, mol])
    assert [full_state(m) for m in frames] == [reference, reference]
    # changing the original afterwards must not reach the snapshot
    for atom in mol.atoms:
        atom.x += 1.0
        atom.charge += 1.0
        atom.name = "Zz"
    mol.atoms.pop()
    mol.bonds.clear()
    mol.pbc.update(1, 2, 3, 90, 90, 90)
    mol.load_notes.append("another")
    assert full_state(snap) == reference


def test_options_that_cannot_change_the_structure_do_not_copy_it(tmp_path):
    copied, patch = _count_structure_copies()
    answers = ["1", "1", "3", "4", "6", "10", "9",                    # geometry listings
               "6", str(tmp_path / "a.xyz"), "8", str(tmp_path / "a.pdb"),
               "7", "n", "n", str(tmp_path / "m.pdb"),               # MPMC write, no changes
               "11", "1", "16", "17", "9"]
    with patch:
        drive(menu._main_loop_single, answers, small_mol(), "t.pdb")
    assert copied == []


def test_trajectory_frame_edit_copies_only_that_frame(tmp_path):
    frames = water_frames(4)
    originals = [full_state(m) for m in frames]
    others = [frames[0], frames[2], frames[3]]
    copied, patch = _count_structure_copies()
    seen = {}

    def capture(n, filename):  # called at every redraw of the trajectory menu
        import inspect
        seen["frames"] = inspect.currentframe().f_back.f_locals["molecules"]

    with patch, mock.patch.object(menu, "print_info_movie", capture):
        # list bonds in frame 1 (no copy), set O-H to 1.1 A in frame 2 (copies frame 2 only)
        drive(menu._main_loop_movie, ["1", "1", "1", "9", "1", "2", "7", "O", "1.1", "9", "9"],
              frames, "t.xyz")
        assert len(copied) == 1 and isinstance(copied[0], Molecule)
        edited = seen["frames"]
        assert full_state(edited[1]) != originals[1]
        assert [edited[0], edited[2], edited[3]] == others  # the very same objects
        drive(menu._main_loop_movie, ["1", "2", "7", "O", "1.1", "9", "17", "9"],
              water_frames(4), "t.xyz")
    restored = seen["frames"]
    assert [full_state(m) for m in restored] == originals  # undo restored frame 2 exactly


def test_trajectory_undo_of_whole_and_single_frame_edits(tmp_path):
    seen = {}

    def capture(n, filename):
        import inspect
        seen["frames"] = inspect.currentframe().f_back.f_locals["molecules"]

    frames = water_frames(3)
    originals = [full_state(m) for m in frames]
    with mock.patch.object(menu, "print_info_movie", capture):
        _, text = drive(menu._main_loop_movie,
                        ["5", "20", "21", "22", "90", "90", "90",      # all frames
                         "12", "3", "2", "O S", "6",                   # frame 3 only
                         "7", "1", "n", "y", "0", str(tmp_path / "f.pdb"),  # frame 1 parameters
                         "17", "17", "17", "17", "9"], frames, "t.xyz")
    assert text.count("Undid the last change") == 3 and text.count("Nothing to undo") == 1
    assert [full_state(m) for m in seen["frames"]] == originals


# ---- 18: the new analysis prompts reject numbers that are not numbers -----------------


def test_analysis_number_prompt_rejects_nan_and_inf():
    value, text = drive(mt._cm_ask_number, ["nan", "inf", "-inf", "2.5"], "Maximum r (A)", 13.0,
                        float, 0.1)
    assert value == 2.5 and text.count("!!! Error") == 3
    value, _ = drive(mt._cm_ask_number, [""], "Maximum r (A)", 13.0, float, 0.1)
    assert value == 13.0  # the default still works


# ---- option 10 numbers the built-in sorbate models by position ------------------------

SORBATE_MODELS_AS_NUMBERED = [
    "h2_bss", "h2_buch", "h2_dl", "co2_trappe", "co2_epm2", "co2_phast", "ch4_trappe",
    "ch4_9site", "n2_trappe", "n2_mcquarrie", "h2o_tip3p", "h2o_tip4p", "h2o_spce",
    "h2o_tip5p", "he", "o2_trappe", "h2_bssp", "co2_phastp", "co2_becker", "ch4_9sitep",
    "n2_polar", "h2o_pol3", "h2o_dc97", "ar_pol", "ne_pol", "kr_pol", "xe_pol", "h2_phahst",
    "n2_phahst", "he_phahst", "ne_phahst", "ar_phahst", "kr_phahst", "xe_phahst",
]


def test_sorbate_model_numbers_only_ever_grow():
    """Old scripts answer option 10's "Model number" prompt with these positions, so a new
    model may only be appended: never inserted, renamed, removed or reordered."""
    names = list(SORBATE_MODELS)
    assert names[:len(SORBATE_MODELS_AS_NUMBERED)] == SORBATE_MODELS_AS_NUMBERED
    for number in (0, 5, 33):
        model, _ = drive(mt.cm_prompt_sorbate_model, [str(number)])
        assert model is SORBATE_MODELS[SORBATE_MODELS_AS_NUMBERED[number]]
