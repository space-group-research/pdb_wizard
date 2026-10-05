"""Regression tests for the 'silent no-op / lost feedback' bug class.

Each test pins a behaviour where the tool used to change or skip data without
telling the user: writers that produced empty files without a cell, operations
that reported success while doing nothing, messages computed but never shown,
and classic-menu prompts that could not accept their documented answers.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from unittest import mock

import pytest

from pdb_wizard.atom import Atom
from pdb_wizard.forcefields import (
    apply_ff_to_system,
    apply_ff_to_system_report,
    get_forcefield,
    missing_ff_messages,
)
from pdb_wizard.formats import write_cif, write_poscar
from pdb_wizard.geometry import (
    adjust_h_distances,
    delete_lone_atoms,
    edit_h_dist,
    remove_lone_atoms,
)
from pdb_wizard.io import (
    CellRequiredError,
    check_pdb_trajectory,
    fallback_cell,
    is_fallback_cell,
    read_file,
    read_file_trajectory,
    read_pdb_trajectory,
    read_xyz_trajectory,
    write_mpmc_pdb,
    write_standard_pdb,
)
from pdb_wizard.menu import _menu_geom_analysis, _write_mpmc_options
from pdb_wizard.molecule import Molecule, set_atom_ids
from pdb_wizard.pbc import PBC

PBC10 = lambda: PBC(10, 10, 10, 90, 90, 90)  # noqa: E731


def _water(pbc=True) -> Molecule:
    atoms = [Atom(0, 0, 0, "O"), Atom(0.757, 0.586, 0, "H"), Atom(-0.757, 0.586, 0, "H")]
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC10() if pbc else None)
    mol.detect_bonds()
    return mol


def _with_lone_atom() -> Molecule:
    atoms = [Atom(0, 0, 0, "C"), Atom(1.4, 0, 0, "C"), Atom(5, 5, 5, "C")]
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC10())
    mol.detect_bonds()
    return mol


def _capture(func, responses, *args):
    it = iter(responses)
    with mock.patch("builtins.input", lambda prompt="": next(it)):
        return func(*args)


# ---------------------------------------------------------------------------
# geometry: results are returned, not swallowed
# ---------------------------------------------------------------------------

class TestLoneAtoms:
    def test_remove_returns_deleted_atoms(self):
        mol, deleted = remove_lone_atoms(_with_lone_atom())
        assert len(mol.atoms) == 2
        assert [a.element.symbol for a in deleted] == ["C"]
        assert deleted[0].x[0] == pytest.approx(5.0)

    def test_remove_nothing(self):
        mol, deleted = remove_lone_atoms(_water())
        assert len(mol.atoms) == 3
        assert deleted == []

    def test_no_cell_is_noop(self):
        mol = _with_lone_atom()
        mol.pbc = None
        mol, deleted = remove_lone_atoms(mol)
        assert len(mol.atoms) == 3
        assert deleted == []

    def test_legacy_wrapper_still_returns_molecule(self):
        mol = delete_lone_atoms(_with_lone_atom())
        assert isinstance(mol, Molecule)
        assert len(mol.atoms) == 2


class TestAdjustHDistances:
    def test_returns_one_message_per_pair(self):
        mol, msgs = adjust_h_distances(_water(), "O", 1.1)
        assert len(msgs) == 2
        assert all(m.startswith("O-H") or m.startswith("H-O") for m in msgs)
        for h in mol.atoms[1:]:
            assert mol.pbc.min_image(h.x - mol.atoms[0].x) == pytest.approx(1.1, abs=1e-6)

    def test_no_matching_pairs(self):
        mol, msgs = adjust_h_distances(_water(), "C", 1.1)
        assert msgs == []

    def test_no_cell_is_noop(self):
        mol = _water(pbc=False)
        before = [a.x.copy() for a in mol.atoms]
        mol, msgs = adjust_h_distances(mol, "O", 1.1)
        assert msgs == []
        for a, b in zip(mol.atoms, before):
            assert a.x == pytest.approx(b)

    def test_legacy_wrapper_still_returns_molecule(self):
        mol = edit_h_dist(_water(), "O", 1.0)
        assert isinstance(mol, Molecule)


# ---------------------------------------------------------------------------
# forcefields: missing elements are reported and zeroed
# ---------------------------------------------------------------------------

class TestForceFieldReport:
    def test_missing_elements_counted_and_zeroed(self):
        atoms = [Atom(0, 0, 0, "C"), Atom(2, 0, 0, "Ar"), Atom(4, 0, 0, "Ar")]
        for a in atoms:
            a.alpha = a.sigma = a.epsilon = 9.9
        _, missing = apply_ff_to_system_report(atoms, get_forcefield(1))  # PHAHST
        assert missing == {"Ar": 2}
        assert atoms[0].sigma != 0.0
        assert atoms[1].alpha == atoms[1].sigma == atoms[1].epsilon == 0.0
        assert atoms[1].c6 == atoms[1].c8 == atoms[1].c10 == 0.0

    def test_no_missing(self):
        atoms = [Atom(0, 0, 0, "C"), Atom(2, 0, 0, "H")]
        _, missing = apply_ff_to_system_report(atoms, get_forcefield(1))
        assert missing == {}
        assert missing_ff_messages(missing) == []

    def test_messages_match_original_wording(self):
        msgs = missing_ff_messages({"Ar": 2, "Xe": 1})
        assert msgs[0] == (
            "!!! 2 atoms of element Ar not found in forcefield, parameters set to all zeros !!!"
        )
        assert "1 atom of element Xe" in msgs[1]

    def test_legacy_wrapper(self):
        atoms = [Atom(0, 0, 0, "C")]
        assert apply_ff_to_system(atoms, get_forcefield(1)) is atoms


# ---------------------------------------------------------------------------
# writers: no cell -> error, never an empty file
# ---------------------------------------------------------------------------

class TestWritersRequireCell:
    @pytest.mark.parametrize("writer", [
        lambda m, f: write_mpmc_pdb(m, f),
        lambda m, f: write_cif(m, f),
        lambda m, f: write_poscar(m, f),
    ])
    def test_raises_cell_required(self, writer):
        buf = io.StringIO()
        with pytest.raises(CellRequiredError) as ei:
            writer(_water(pbc=False), buf)
        assert "unit cell" in str(ei.value)
        assert buf.getvalue() == ""

    def test_cell_required_is_a_value_error(self):
        assert issubclass(CellRequiredError, ValueError)

    def test_standard_pdb_without_cell_is_written_without_cryst1(self):
        """OpenMM NoCutoff topologies are non-periodic PDBs; used to come out empty."""
        buf = io.StringIO()
        write_standard_pdb(_water(pbc=False), buf)
        text = buf.getvalue()
        assert "CRYST1" not in text
        assert text.count("HETATM") == 3
        assert text.rstrip().endswith("END")

    def test_writers_work_with_cell(self):
        for writer in (write_standard_pdb, write_mpmc_pdb, write_cif, write_poscar):
            buf = io.StringIO()
            writer(_water(), buf)
            assert buf.getvalue()


class TestStandardPdbSkipMolsStep:
    def test_skip_mols_step_keeps_positions(self):
        """`--convert` style writing must not re-image atoms to the last atom."""
        atoms = [Atom(0.5, 0.5, 0.5, "C"), Atom(9.5, 0.5, 0.5, "C")]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC10())
        buf = io.StringIO()
        write_standard_pdb(mol, buf, skip_mols_step=True)
        coords = [line[30:54].split() for line in buf.getvalue().splitlines()
                  if line.startswith("HETATM")]
        assert [float(c[0]) for c in coords] == [0.5, 9.5]

    def test_default_keeps_molecule_whole(self):
        """Without skip, a bonded pair split across the boundary is re-imaged together."""
        atoms = [Atom(0.5, 0.5, 0.5, "C"), Atom(9.5, 0.5, 0.5, "C")]
        set_atom_ids(atoms)
        mol = Molecule(atoms=atoms, pbc=PBC10())
        mol.detect_bonds()
        buf = io.StringIO()
        write_standard_pdb(mol, buf)
        xs = sorted(float(line[30:38]) for line in buf.getvalue().splitlines()
                    if line.startswith("HETATM"))
        assert abs(xs[1] - xs[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MPMC PDB: CONECT indices and full round trip
# ---------------------------------------------------------------------------

def _mpmc_mol() -> Molecule:
    atoms = [Atom(1, 1, 1, "Cu"), Atom(3, 1, 1, "O"), Atom(1, 3, 1, "C"), Atom(2.0, 2.0, 2.0, "H")]
    for i, a in enumerate(atoms):
        a.charge = (-1) ** i * 0.25 * (i + 1)
    apply_ff_to_system(atoms, get_forcefield(1))
    set_atom_ids(atoms)
    mol = Molecule(atoms=atoms, pbc=PBC(12.5, 11.0, 10.0, 90, 95, 90))
    mol.detect_bonds()
    return mol


def _parse_mpmc(text: str):
    atoms = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
    conect = [tuple(int(t) for t in ln.split()[1:]) for ln in text.splitlines()
              if ln.startswith("CONECT")]
    return atoms, conect


class TestMpmcPdb:
    def test_box_conect_references_box_atoms(self):
        buf = io.StringIO()
        write_mpmc_pdb(_mpmc_mol(), buf)
        atoms, conect = _parse_mpmc(buf.getvalue())
        box_ids = [int(ln.split()[1]) for ln in atoms if " BOX " in ln]
        assert box_ids == list(range(5, 13))  # 4 framework atoms, then 8 corners
        assert len(conect) == 12
        referenced = {i for pair in conect for i in pair}
        assert referenced == set(box_ids), "every corner must be wired, no framework atom"

    def test_box_conect_with_sorbate(self):
        from pdb_wizard.sorbates import format_sorbate_pqr
        mol = _mpmc_mol()
        sorb = format_sorbate_pqr(_first_sorbate_name(), 6, 6, 6, mol_id=2, start_atom_id=5)
        buf = io.StringIO()
        write_mpmc_pdb(mol, buf, sorbate_lines=sorb)
        atoms, conect = _parse_mpmc(buf.getvalue())
        box_ids = [int(ln.split()[1]) for ln in atoms if " BOX " in ln]
        assert box_ids[0] == 5 + len(sorb)
        assert {i for pair in conect for i in pair} == set(box_ids)

    def test_round_trip_preserves_charges_params_and_cell(self, tmp_path):
        mol = _mpmc_mol()
        path = tmp_path / "mof.pqr"
        write_mpmc_pdb(mol, str(path))
        back = read_file(str(path))
        # Box corner "X" atoms are not real atoms and must not come back.
        assert len(back.atoms) == len(mol.atoms)
        assert back.pbc is not None
        for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
            assert getattr(back.pbc, attr) == pytest.approx(getattr(mol.pbc, attr), abs=1e-3)
        by_el_orig = {a.element.symbol: a for a in mol.atoms}
        for a in back.atoms:
            o = by_el_orig[a.element.symbol]
            assert a.charge == pytest.approx(o.charge, abs=1e-5)
            assert a.alpha == pytest.approx(o.alpha, abs=1e-5)
            assert a.epsilon == pytest.approx(o.epsilon, abs=1e-5)
            assert a.sigma == pytest.approx(o.sigma, abs=1e-5)
            assert a.c6 == pytest.approx(o.c6, abs=1e-5)
            assert a.c8 == pytest.approx(o.c8, abs=1e-4)
            assert a.c10 == pytest.approx(o.c10, abs=1e-3)

    def test_c10_is_not_rounded_to_two_digits(self):
        """Inherited format wrote PHAHST c10 as e.g. '1.4e+04' (0.35% error)."""
        buf = io.StringIO()
        write_mpmc_pdb(_mpmc_mol(), buf)
        cu_line = next(ln for ln in buf.getvalue().splitlines() if ln.startswith("ATOM") and " Cu " in ln)
        c10 = float(cu_line.split()[18])
        assert c10 == pytest.approx(13951.49740, abs=1e-3)
        assert "e+" not in cu_line

    def test_flags_do_not_truncate_record(self):
        """MPMC needs every column; the flags must never produce placeholders."""
        buf_a, buf_b = io.StringIO(), io.StringIO()
        write_mpmc_pdb(_mpmc_mol(), buf_a, write_charges=False, write_params=False)
        write_mpmc_pdb(_mpmc_mol(), buf_b, write_charges=True, write_params=True)
        assert buf_a.getvalue() == buf_b.getvalue()
        assert "xxx" not in buf_a.getvalue()
        first_atom = next(ln for ln in buf_a.getvalue().splitlines() if ln.startswith("ATOM"))
        assert len(first_atom.split()) == 19


def _first_sorbate_name() -> str:
    from pdb_wizard.sorbates import SORBATE_MODELS
    return next(iter(SORBATE_MODELS))


# ---------------------------------------------------------------------------
# trajectories: END-separated frames and the fallback cell
# ---------------------------------------------------------------------------

def _pdb_frame(x: float, cryst: bool = True) -> str:
    head = "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n" if cryst else ""
    return (
        head
        + f"ATOM      1  C   MOL A   1    {x:8.3f}   0.000   0.000  1.00  0.00           C\n"
        + f"ATOM      2  O   MOL A   1    {x + 1.2:8.3f}   0.000   0.000  1.00  0.00           O\n"
    )


class TestEndSeparatedTrajectories:
    def test_detects_end_separated_frames(self, tmp_path):
        p = tmp_path / "vmd.pdb"
        p.write_text(_pdb_frame(0) + "END\n" + _pdb_frame(1) + "END\n" + _pdb_frame(2) + "END\n")
        assert check_pdb_trajectory(str(p)) is True
        frames = read_file_trajectory(str(p))
        assert frames is not None and len(frames) == 3
        assert [round(f.atoms[0].x[0]) for f in frames] == [0, 1, 2]

    def test_single_frame_with_end_is_not_trajectory(self, tmp_path):
        p = tmp_path / "one.pdb"
        p.write_text(_pdb_frame(0) + "END\n")
        assert check_pdb_trajectory(str(p)) is False
        assert read_file_trajectory(str(p)) is None

    def test_model_endmdl_frames_not_double_counted(self, tmp_path):
        text = "".join(
            f"MODEL     {i + 1}\n" + _pdb_frame(i) + "ENDMDL\n" for i in range(3)
        ) + "END\n"
        p = tmp_path / "models.pdb"
        p.write_text(text)
        with open(p) as f:
            mols, _ = read_pdb_trajectory(f)
        assert len(mols) == 3

    def test_remark_step_frames_still_work(self, tmp_path):
        text = "".join(f"REMARK step={i}\n" + _pdb_frame(i) for i in range(4))
        p = tmp_path / "mpmc_traj.pdb"
        p.write_text(text)
        assert check_pdb_trajectory(str(p)) is True
        with open(p) as f:
            mols, _ = read_pdb_trajectory(f)
        assert len(mols) == 4


class TestFallbackCell:
    def test_is_fallback_cell(self):
        assert is_fallback_cell(fallback_cell())
        assert not is_fallback_cell(PBC10())
        assert not is_fallback_cell(None)

    def test_pdb_trajectory_without_cell_gets_fallback_on_molecule(self, tmp_path):
        p = tmp_path / "nocell.pdb"
        p.write_text(_pdb_frame(0, cryst=False) + "END\n" + _pdb_frame(1, cryst=False) + "END\n")
        with open(p) as f:
            mols, pbcs = read_pdb_trajectory(f)
        assert len(mols) == 2
        assert all(is_fallback_cell(m.pbc) for m in mols)
        assert all(is_fallback_cell(b) for b in pbcs)

    def test_xyz_trajectory_without_cell_gets_fallback(self):
        text = "2\n\nC 0 0 0\nO 1.2 0 0\n2\n\nC 0.1 0 0\nO 1.3 0 0\n"
        mols, _ = read_xyz_trajectory(io.StringIO(text))
        assert len(mols) == 2
        assert all(is_fallback_cell(m.pbc) for m in mols)


class TestReadFileDcd:
    def test_read_file_on_dcd_explains_itself(self, tmp_path):
        p = tmp_path / "traj.dcd"
        p.write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="trajector"):
            read_file(str(p))


# ---------------------------------------------------------------------------
# classic menu: prompts and printed results
# ---------------------------------------------------------------------------

class TestClassicGeometryMenu:
    def test_delete_lone_prints_deleted_atoms(self, capsys):
        _capture(_menu_geom_analysis, ["5", "9"], _with_lone_atom())
        out = capsys.readouterr().out
        assert "Deleting lone atoms:" in out
        assert "  C     3" in out

    def test_delete_lone_prints_none_found(self, capsys):
        _capture(_menu_geom_analysis, ["5", "9"], _water())
        assert "No lone atoms found" in capsys.readouterr().out

    def test_edit_h_reprompts_on_bad_element_then_prints_pairs(self, capsys):
        mol = _water()
        _capture(_menu_geom_analysis, ["7", "Zz", "O", "1.1", "9"], mol)
        out = capsys.readouterr().out
        assert "!!! Error finding element or reading distance !!!" in out
        assert out.count("O-H") + out.count("H-O") == 2
        for h in mol.atoms[1:]:
            assert mol.pbc.min_image(h.x - mol.atoms[0].x) == pytest.approx(1.1, abs=1e-6)

    def test_edit_h_reprompts_on_bad_distance(self, capsys):
        _capture(_menu_geom_analysis, ["7", "O", "abc", "O", "1.0", "9"], _water())
        out = capsys.readouterr().out
        assert "!!! Error finding element or reading distance !!!" in out

    def test_edit_h_reports_no_pairs(self, capsys):
        _capture(_menu_geom_analysis, ["7", "C", "1.0", "9"], _water())
        assert "No C-H bonds found" in capsys.readouterr().out


class TestClassicMpmcPrompt:
    @pytest.mark.parametrize("answer", ["OPLSAA", "PHAHST", "0", "1", "phahst"])
    def test_force_field_accepts_documented_answers(self, answer, tmp_path, capsys):
        out_path = tmp_path / "out.pqr"
        _capture(_write_mpmc_options, ["n", "y", answer, str(out_path)], _mpmc_mol())
        out = capsys.readouterr().out
        assert "!!! Error reading input !!!" not in out
        assert out_path.exists() and out_path.stat().st_size > 0
        assert f"wrote {out_path}" in out

    def test_force_field_rejects_garbage_then_accepts(self, tmp_path, capsys):
        out_path = tmp_path / "out.pqr"
        _capture(_write_mpmc_options, ["n", "y", "AMBER", "PHAHST", str(out_path)], _mpmc_mol())
        out = capsys.readouterr().out
        assert out.count("!!! Error reading input !!!") == 1
        assert out_path.exists()

    def test_missing_force_field_element_is_printed(self, tmp_path, capsys):
        mol = _mpmc_mol()
        mol.atoms.append(Atom(5, 5, 5, "Ar"))
        set_atom_ids(mol.atoms)
        out_path = tmp_path / "out.pqr"
        _capture(_write_mpmc_options, ["n", "y", "PHAHST", str(out_path)], mol)
        out = capsys.readouterr().out
        assert "1 atom of element Ar not found in forcefield" in out


# ---------------------------------------------------------------------------
# batch CLI
# ---------------------------------------------------------------------------

def _run_cli(*args, timeout=120):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in ["src", env.get("PYTHONPATH", "")] if p
    )
    return subprocess.run(
        [sys.executable, "-m", "pdb_wizard", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


@pytest.fixture
def nocell_xyz(tmp_path):
    p = tmp_path / "nocell.xyz"
    p.write_text("3\n\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n")
    return p


@pytest.fixture
def cell_xyz(tmp_path):
    p = tmp_path / "cell.xyz"
    p.write_text("3\n10 10 10 90 90 90\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n")
    return p


class TestBatchConvert:
    @pytest.mark.parametrize("ext", [".pqr", ".cif", ".vasp"])
    def test_no_cell_refuses_and_leaves_no_file(self, nocell_xyz, tmp_path, ext):
        out = tmp_path / f"out{ext}"
        r = _run_cli("--convert", str(nocell_xyz), "-o", str(out))
        assert r.returncode == 1
        assert "requires a unit cell" in r.stderr
        assert not out.exists()

    def test_no_cell_to_pdb_writes_non_periodic_pdb(self, nocell_xyz, tmp_path):
        out = tmp_path / "out.pdb"
        r = _run_cli("--convert", str(nocell_xyz), "-o", str(out))
        assert r.returncode == 0, r.stderr
        text = out.read_text()
        assert "CRYST1" not in text and text.count("HETATM") == 3

    def test_no_cell_to_xyz_still_works(self, nocell_xyz, tmp_path):
        out = tmp_path / "out.xyz"
        r = _run_cli("--convert", str(nocell_xyz), "-o", str(out))
        assert r.returncode == 0, r.stderr
        assert out.exists()

    def test_pqr_output_is_mpmc_format(self, cell_xyz, tmp_path):
        out = tmp_path / "out.pqr"
        r = _run_cli("--convert", str(cell_xyz), "-o", str(out))
        assert r.returncode == 0, r.stderr
        text = out.read_text()
        assert " MOF F " in text
        assert "REMARK BOX BASIS" in text
        assert " BOX F " in text

    def test_overwrite_warns_on_stderr(self, cell_xyz, tmp_path):
        out = tmp_path / "out.xyz"
        out.write_text("stale\n")
        r = _run_cli("--convert", str(cell_xyz), "-o", str(out))
        assert r.returncode == 0, r.stderr
        assert "overwriting existing file" in r.stderr
        assert "stale" not in out.read_text()

    def test_unsupported_output_checked_before_reading(self, cell_xyz, tmp_path):
        out = tmp_path / "out.foo"
        r = _run_cli("--convert", str(cell_xyz), "-o", str(out))
        assert r.returncode == 1
        assert "Unsupported output format" in r.stderr
        assert not out.exists()


class TestBatchErrors:
    def test_info_on_dcd_is_clean_error(self, tmp_path):
        p = tmp_path / "t.dcd"
        p.write_bytes(b"\x00" * 16)
        r = _run_cli("--info", str(p))
        assert r.returncode == 1
        assert "Traceback" not in r.stderr
        assert "could not read" in r.stderr

    def test_tui_launch_with_unknown_extension_is_clean_error(self, tmp_path):
        p = tmp_path / "structure.foo"
        p.write_text("nonsense\n")
        r = _run_cli(str(p), timeout=60)
        assert r.returncode == 2
        assert "Traceback" not in r.stderr
        assert "Unsupported file format" in r.stderr

    def test_tui_launch_with_missing_file_is_clean_error(self, tmp_path):
        r = _run_cli(str(tmp_path / "nope.pdb"), timeout=60)
        assert r.returncode == 2
        assert "file not found" in r.stderr

    def test_rdf_warns_about_fallback_cell(self, tmp_path):
        p = tmp_path / "traj.xyz"
        p.write_text("2\n\nC 0 0 0\nO 1.2 0 0\n2\n\nC 0.1 0 0\nO 1.3 0 0\n")
        r = _run_cli("--rdf", str(p), "-e1", "C", "-e2", "O")
        assert "no cell information" in r.stderr
        assert "1000000 A" in r.stderr
