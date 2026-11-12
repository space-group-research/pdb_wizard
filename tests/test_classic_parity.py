"""The classic text menu must offer what the viewers offer, without moving the original
menu numbers that old input scripts rely on.

Covers: the feature map against the terminal viewer's Operations list, the pinned numbering,
every new submenu driven through input(), and agreement with the shared analysis functions
(the same ones the terminal viewer and the GUI call).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import pdb_wizard.menu as menu
import pdb_wizard.menu_tools as mt
from pdb_wizard.analyses import run_msd, run_pxrd, run_rdf, system_info_text
from pdb_wizard.atom import Atom
from pdb_wizard.io import read_file
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis
from pdb_wizard.pbc import PBC

ROOT = Path(__file__).resolve().parent.parent
HKUST = ROOT / "HKUST-1.pdb"

ORIGINAL_MAIN = {
    1: "geometry analysis", 2: "extend along axis",
    3: "wrap atoms from (0, 0, 0) to (1, 1, 1)",
    4: "wrap atoms from (-1/2, -1/2, -1/2) to (1/2, 1/2, 1/2)",
    5: "update cell dimensions", 6: "write .xyz", 7: "write MPMC .pdb",
    8: "write standardized .pdb", 9: "quit", 10: "insert sorbates and write MPMC .pdb",
}
ORIGINAL_GEOM = {
    1: "list bonds", 2: "list close vdw contacts", 3: "list angles", 4: "list lone atoms",
    5: "delete lone atoms", 6: "list coordinates", 7: "edit hydrogen bond distances",
    8: "preview with VMD", 9: "back to main menu",
}


def mixed_mol() -> Molecule:
    rng = np.random.default_rng(42)
    atoms = [Atom(*rng.uniform(0, 14, 3), el) for el in ("C", "O", "N", "H") for _ in range(8)]
    mol = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
    mol.detect_bonds()
    return mol


def water_trajectory(n_frames: int = 14) -> list:
    rng = np.random.default_rng(7)
    centres = rng.uniform(2, 13, (10, 3))
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


def drive(func, responses, *args):
    """Run ``func`` with scripted input(); returns (result, prompts, printed text)."""
    prompts: list = []
    printed: list = []
    it = iter(responses)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"menu asked for more input than scripted: {prompt!r}")

    def fake_print(*a, **k):
        printed.append(" ".join(str(x) for x in a))

    with mock.patch("builtins.input", fake_input), mock.patch("builtins.print", fake_print):
        result = func(*args)
    assert next(it, None) is None, "scripted input left over"
    return result, prompts, "\n".join(printed)


def options(prompt: str) -> dict:
    return {int(m.group(1)): m.group(2).strip()
            for m in re.finditer(r"(\d+)\s*=\s*([^\n]+)", prompt)}


# ---- numbering ------------------------------------------------------------------------


def test_original_main_numbers_are_unchanged():
    _, prompts, _ = drive(menu._main_loop_single, ["9"], mixed_mol(), "t.pdb")
    shown = options(prompts[0])
    for number, label in ORIGINAL_MAIN.items():
        assert shown[number] == label
    assert sorted(shown) == list(range(1, 19))


def test_original_geometry_numbers_are_unchanged():
    _, prompts, _ = drive(menu._menu_geom_analysis, ["9"], mixed_mol())
    shown = options(prompts[0])
    for number, label in ORIGINAL_GEOM.items():
        assert shown[number] == label
    assert sorted(shown) == list(range(1, 12))


def test_movie_menu_uses_the_same_numbers_as_the_single_menu():
    _, single, _ = drive(menu._main_loop_single, ["9"], mixed_mol(), "t.pdb")
    _, movie, _ = drive(menu._main_loop_movie, ["9"], water_trajectory(3), "t.xyz")
    one, many = options(single[0]), options(movie[0])
    assert sorted(one) == sorted(many)
    for number in one:
        assert many[number].startswith(one[number])


def test_an_old_script_still_runs(tmp_path):
    """A pre-existing input script: wrap, extend x by 1, write xyz, quit."""
    out = tmp_path / "old.xyz"
    mol = mixed_mol()
    drive(menu._main_loop_single, ["3", "2", "x", "1", "6", str(out), "9"], mol, "t.pdb")
    assert out.read_text().splitlines()[0].strip() == "64"


# ---- feature map ----------------------------------------------------------------------


def test_every_viewer_feature_is_in_the_classic_menu():
    panels = pytest.importorskip("pdb_wizard.panels")
    keys = {key for key, _, _ in panels.OPERATIONS}
    missing = keys - set(mt.CLASSIC_FEATURES) - mt.CLASSIC_NOT_APPLICABLE
    assert not missing, f"viewer features with no classic-menu entry: {sorted(missing)}"
    assert not set(mt.CLASSIC_FEATURES) - keys, "classic feature map lists unknown keys"


def test_analysis_menu_lists_every_analysis():
    keys = [k for k, _ in mt.CM_ANALYSES]
    assert all(mt.CLASSIC_FEATURES[k] == f"main 11 > {i}" for i, k in enumerate(keys, 1))
    _, prompts, _ = drive(mt.cm_menu_analysis, ["15"], mixed_mol(), None, "t.pdb")
    assert len(options(prompts[0])) == 15


# ---- analyses give the shared functions' numbers -----------------------------------------


def test_system_info_matches_shared_text():
    mol = mixed_mol()
    _, _, out = drive(mt.cm_menu_analysis, ["1", "15"], mol, None, "t.pdb")
    assert system_info_text(mol, "t.pdb") in out


def test_rdf_matches_shared_run_and_writes_csv(tmp_path):
    mol = mixed_mol()
    csv = tmp_path / "rdf.csv"
    _, _, out = drive(mt.cm_menu_analysis, ["5", "C", "O", "", "", "", "", str(csv), "15"],
                      mol, None, "t.pdb")
    ref = run_rdf(mol, "C", "O")
    assert ref.summary in out and ref.title in out
    rows = np.loadtxt(csv, delimiter=",", skiprows=1)
    np.testing.assert_allclose(rows[:, 1], ref.y, atol=1e-6)
    assert csv.read_text().splitlines()[0] == ref.csv_header


def test_trajectory_rdf_uses_the_frame_range():
    frames = water_trajectory()
    _, _, out = drive(mt.cm_menu_analysis,
                      ["5", "O", "H", "", "", "", "", "2", "9", "", "15"], frames[0], frames, "t")
    ref = run_rdf(frames[0], "O", "H", frames=frames, start=2, end=9)
    assert "frames 2-9" in ref.title and ref.title in out and ref.summary in out


def test_msd_matches_shared_run():
    frames = water_trajectory()
    _, _, out = drive(mt.cm_menu_analysis, ["8", "O", "", "", "", "", "", "15"],
                      frames[0], frames, "t.xyz")
    assert run_msd(frames, "O").summary in out


def test_pxrd_matches_shared_run():
    mol = read_file(str(HKUST))
    _, _, out = drive(mt.cm_menu_analysis, ["13", "", "", "", "", "15"], mol, None, "h.pdb")
    assert run_pxrd(mol).summary in out


@pytest.mark.parametrize("option,inputs", [
    ("2", []), ("3", []), ("4", [""]), ("6", ["C", "O", "", ""]), ("7", ["", "", ""]),
    ("11", ["C", "", "", ""]),
])
def test_single_structure_analyses_run(option, inputs):
    _, _, out = drive(mt.cm_menu_analysis, [option, *inputs, "15"], mixed_mol(), None, "t.pdb")
    assert "Error" not in out and "Invalid" not in out


@pytest.mark.parametrize("option,inputs", [
    ("9", ["", "", "", "", "", ""]), ("10", ["", "", "", "", ""]),
    ("12", ["O", "4", ""]), ("6", ["O", "H", "", "y", "", "", ""]),
])
def test_trajectory_analyses_run(option, inputs):
    frames = water_trajectory()
    _, _, out = drive(mt.cm_menu_analysis, [option, *inputs, "15"], frames[0], frames, "t.xyz")
    assert "Error" not in out and "Invalid" not in out


def test_trajectory_only_analyses_explain_themselves_on_one_structure():
    _, _, out = drive(mt.cm_menu_analysis, ["8", "10", "12", "15"], mixed_mol(), None, "t.pdb")
    assert "Need a trajectory with 3+ frames for MSD" in out
    assert "Need a trajectory for Rg" in out


def test_q_cancels_an_analysis_and_bad_numbers_are_reasked():
    _, _, out = drive(mt.cm_menu_analysis, ["5", "C", "q", "5", "Zz", "C", "O", "", "", "abc",
                                            "", "", "", "15"], mixed_mol(), None, "t.pdb")
    assert "Cancelled" in out and "No Zz atoms" in out and "enter a number" in out


def test_energy_plot_and_csv(tmp_path):
    energy = tmp_path / "energy.dat"
    energy.write_text("#step energy N\n" + "".join(f"{i} {-i * 1.5} {i % 7}\n"
                                                   for i in range(1, 41)))
    csv = tmp_path / "e.csv"
    _, _, out = drive(mt.cm_menu_analysis, ["14", "", "energy", "5", "30", str(csv), "15"],
                      mixed_mol(), None, str(tmp_path / "s.pdb"))
    assert "energy (steps 5-30)" in out
    assert len(csv.read_text().splitlines()) == 27


def test_isotherm_reports_an_empty_folder(tmp_path):
    _, _, out = drive(mt.cm_menu_isotherm, [str(tmp_path)])
    assert "No completed pressure points found" in out


def test_ascii_plot_marks_the_extremes():
    text = mt.cm_ascii_plot([0, 1, 2, 3], [0, 1, 4, 9])
    lines = text.splitlines()
    assert lines[0].strip().startswith("9") and lines[0].rstrip().endswith("*")
    assert mt.cm_ascii_plot([], []) == ""
    assert "*" in mt.cm_ascii_plot([1, 1], [2, 2])  # zero span must not divide by zero


# ---- edit -----------------------------------------------------------------------------


def test_substitute_and_undo():
    mol = mixed_mol()
    result, _, out = drive(menu._main_loop_single,
                           ["12", "2", "N Zn", "6", "6", "undo.xyz", "17", "9"], mol, "t.pdb")
    assert "Replaced 8 N atoms with Zn" in out and "Undid the last change" in out
    Path("undo.xyz").unlink()


def test_undo_restores_the_structure(tmp_path):
    mol = mixed_mol()
    before = [a.element.symbol for a in mol.atoms]
    a, b = tmp_path / "a.xyz", tmp_path / "b.xyz"
    drive(menu._main_loop_single,
          ["12", "2", "N Zn", "6", "6", str(a), "17", "6", str(b), "17", "9"], mol, "t.pdb")
    assert "Zn" in a.read_text() and "Zn" not in b.read_text()
    assert [ln.split()[0] for ln in b.read_text().splitlines()[2:]] == before


def test_undo_with_nothing_to_undo_and_after_a_no_op():
    _, _, out = drive(menu._main_loop_single, ["17", "1", "9", "17", "9"], mixed_mol(), "t.pdb")
    assert out.count("Nothing to undo") == 2


def test_undo_covers_the_original_options(tmp_path):
    mol = mixed_mol()
    out = tmp_path / "o.xyz"
    drive(menu._main_loop_single, ["2", "x", "1", "17", "6", str(out), "9"], mol, "t.pdb")
    assert out.read_text().splitlines()[0].strip() == "32"


def test_history_skips_huge_states(monkeypatch):
    monkeypatch.setattr(mt, "CM_UNDO_MAX_ATOMS", 10)
    h = mt.CmHistory()
    mol = mixed_mol()
    h.before(mol)
    mol.atoms.pop()
    h.after(mol)
    assert h.undo() is None


def test_sort_groups_molecules():
    frames = water_trajectory(1)
    mol = frames[0]
    mol.atoms.reverse()
    mol.bonds = []
    mol.detect_bonds()
    import copy

    from pdb_wizard.operations import sort_system
    expected, _ = sort_system(copy.deepcopy(mol), False)
    unsorted = [a.element.symbol for a in mol.atoms]
    result, _, out = drive(mt.cm_menu_edit, ["1", "6"], mol)
    assert "Sorted atoms" in out
    symbols = [a.element.symbol for a in result.atoms]
    assert symbols == [a.element.symbol for a in expected.atoms] and symbols != unsorted
    np.testing.assert_allclose([a.x for a in result.atoms], [a.x for a in expected.atoms])


def test_reduce_supercell_keeps_chosen_sorbates():
    base = Molecule(atoms=[Atom(1, 1, 1, "Cu"), Atom(3, 3, 3, "O")], pbc=PBC(6, 6, 6, 90, 90, 90))
    sup = extend_axis(extend_axis(base, 0, 1), 1, 1)
    sup.atoms.append(Atom(8.0, 2.0, 4.5, "Ar"))
    sup.detect_bonds()
    result, _, out = drive(mt.cm_menu_edit, ["3", "2", "2", "1", "all", "6"], sup)
    assert "kept 1/1 sorbate molecules" in out
    assert sorted(a.element.symbol for a in result.atoms) == ["Ar", "Cu", "O"]
    assert result.pbc.a == pytest.approx(6.0)
    dropped, _, _ = drive(mt.cm_menu_edit, ["3", "2", "2", "1", "9", "none", "6"], sup)
    assert sorted(a.element.symbol for a in dropped.atoms) == ["Cu", "O"]


def test_pack_sorbates_adds_atoms_to_the_structure():
    mol = Molecule(atoms=[Atom(0, 0, 0, "Cu")], pbc=PBC(20, 20, 20, 90, 90, 90))
    result, _, out = drive(mt.cm_menu_edit, ["4", "h2_bss", "3", "6"], mol)
    n_sites = len(mt.SORBATE_MODELS["h2_bss"].sites)
    assert len(result.atoms) == 1 + 3 * n_sites
    assert "Placed 3" in out


def test_pack_sorbates_needs_a_cell_and_can_be_cancelled():
    _, _, out = drive(mt.cm_menu_edit, ["4", "6"], Molecule(atoms=[Atom(0, 0, 0, "C")]))
    assert "needs a unit cell" in out
    mol = mixed_mol()
    result, _, _ = drive(mt.cm_menu_edit, ["4", "q", "6"], mol)
    assert len(result.atoms) == 32


def test_qeq_reports_charges():
    atoms = [Atom(0, 0, 0, "O"), Atom(0.96, 0, 0, "H"), Atom(-0.24, 0.93, 0, "H")]
    mol = Molecule(atoms=atoms, pbc=None)
    result, _, out = drive(mt.cm_menu_edit, ["5", "6"], mol)
    assert "QEq charges computed for 3 atoms" in out
    assert result.atoms[0].charge < 0 < result.atoms[1].charge


def test_edit_menu_rejects_bad_input():
    _, _, out = drive(mt.cm_menu_edit, ["x", "42", "2", "Cu", "2", "C Qq", "2", "Cu Zn", "6"],
                      mixed_mol())
    assert "Error converting input to int" in out and "Invalid option" in out
    assert "old_element new_element" in out and "Unknown element: Qq" in out
    assert "No Cu atoms found" in out


# ---- geometry additions ---------------------------------------------------------------------


def test_measure_distance_angle_dihedral():
    atoms = [Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C"), Atom(1.5, 1.5, 0, "C"),
             Atom(1.5, 1.5, 1.5, "C")]
    mol = Molecule(atoms=atoms, pbc=PBC(30, 30, 30, 90, 90, 90))
    assert "r = 1.5000 A" in mt.cm_measure(mol, [1, 2])
    assert "angle = 90.00 deg" in mt.cm_measure(mol, [1, 2, 3])
    assert "dihedral = -90.00 deg" in mt.cm_measure(mol, [1, 2, 3, 4]) \
        or "dihedral = 90.00 deg" in mt.cm_measure(mol, [1, 2, 3, 4])
    for bad in ([1], [1, 1], [1, 9]):
        with pytest.raises(ValueError):
            mt.cm_measure(mol, bad)


def test_measure_uses_the_minimum_image():
    mol = Molecule(atoms=[Atom(0.5, 0, 0, "C"), Atom(9.5, 0, 0, "O")],
                   pbc=PBC(10, 10, 10, 90, 90, 90))
    assert "r = 1.0000 A" in mt.cm_measure(mol, [1, 2])


def test_geometry_menu_new_options():
    frames = water_trajectory(1)
    _, _, out = drive(menu._menu_geom_analysis, ["11", "1 2", "1,2,3", "x", "7 7", "q", "10", "9"],
                      frames[0])
    assert "O1-H2   r = 0.9600 A" in out and "angle =" in out
    assert "Enter atom numbers" in out and "different atom numbers" in out
    assert "Dihedrals:" in out
    butane = Molecule(atoms=[Atom(0, 0, 0, "C"), Atom(1.5, 0, 0, "C"), Atom(2.0, 1.4, 0, "C"),
                             Atom(3.5, 1.4, 0.3, "C")], pbc=PBC(30, 30, 30, 90, 90, 90))
    butane.detect_bonds()
    assert len(mt.cm_dihedrals_list(butane)) == 1


# ---- writing ----------------------------------------------------------------------------


def test_write_other_formats(tmp_path):
    mol = mixed_mol()
    cif, vasp, lmp = tmp_path / "o.cif", tmp_path / "POSCAR", tmp_path / "o.data"
    _, _, out = drive(mt.cm_menu_write_other,
                      ["1", str(cif), "2", str(vasp), "3", str(lmp), "0", "6"], mol, "t.pdb")
    assert out.count("wrote") == 3 and "Invalid option" not in out
    for path in (cif, vasp):
        back = read_file(str(path))
        assert len(back.atoms) == len(mol.atoms)
        assert back.pbc.a == pytest.approx(15.0)
    assert "atoms" in lmp.read_text()


def test_write_reports_an_unwritable_path(tmp_path):
    _, _, out = drive(mt.cm_menu_write_other, ["1", str(tmp_path / "no" / "x.cif"), "6"],
                      mixed_mol(), "t.pdb")
    assert "Error" in out


def test_pictures(tmp_path):
    image = pytest.importorskip("PIL.Image")
    png, gif = tmp_path / "p.png", tmp_path / "g.gif"
    _, _, out = drive(mt.cm_menu_write_other, ["4", str(png), "5", str(gif), "6"],
                      mixed_mol(), "t.pdb")
    with image.open(png) as im:
        assert im.size == (1600, 1200)
        assert np.asarray(im).max() > 0  # something was drawn
    with image.open(gif) as im:
        assert im.n_frames == 36


def test_pictures_without_pillow(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pil(name, *a, **k):
        if name == "PIL":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    _, _, out = drive(mt.cm_menu_write_other, ["4", str(tmp_path / "p.png"), "6"],
                      mixed_mol(), "t.pdb")
    assert "Pillow required" in out


# ---- inputs -----------------------------------------------------------------------------


def test_cp2k_inputs_from_the_menu(tmp_path):
    mol = read_file(str(HKUST))
    n_vars = len(mt.ENGINES["cp2k_cell_opt"].user_vars)
    number = str(list(mt.ENGINES).index("cp2k_cell_opt") + 1)
    _, _, out = drive(mt.cm_menu_inputs, [number, *[""] * n_vars, str(tmp_path / "cp2k")],
                      mol, "h.pdb")
    assert "Wrote" in out
    written = list((tmp_path / "cp2k").glob("*.inp"))
    assert written and "&CELL" in written[0].read_text()


def test_inputs_match_the_shared_generator(tmp_path):
    from pdb_wizard.inputgen import generate_engine_inputs
    from pdb_wizard.templates import engine_defaults
    mol = read_file(str(HKUST))
    number = str(list(mt.ENGINES).index("cp2k_dft_md") + 1)
    n_vars = len(mt.ENGINES["cp2k_dft_md"].user_vars)
    drive(mt.cm_menu_inputs, [number, *[""] * n_vars, str(tmp_path / "a")], mol, "h.pdb")
    generate_engine_inputs(mol, "cp2k_dft_md", engine_defaults("cp2k_dft_md"), str(tmp_path / "b"))
    files_a = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert files_a == sorted(p.name for p in (tmp_path / "b").iterdir())
    for name in files_a:
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_inputs_refuse_when_validation_fails(tmp_path):
    mol = Molecule(atoms=[Atom(0, 0, 0, "C")], pbc=None)
    number = str(list(mt.ENGINES).index("cp2k_cell_opt") + 1)
    n_vars = len(mt.ENGINES["cp2k_cell_opt"].user_vars)
    _, _, out = drive(mt.cm_menu_inputs, [number, *[""] * n_vars], mol, "c.xyz")
    assert "CP2K requires a periodic box" in out


def test_inputs_bad_value_reasked_cancel_and_back():
    mol = mixed_mol()
    back = str(len(mt.ENGINES) + 1)
    drive(mt.cm_menu_inputs, [back], mol, "t.pdb")
    _, _, out = drive(mt.cm_menu_inputs, ["99"], mol, "t.pdb")
    assert "Invalid option" in out
    number = str(list(mt.ENGINES).index("cp2k_cell_opt") + 1)
    _, _, out = drive(mt.cm_menu_inputs, [number, "", "abc", "q"], mol, "t.pdb")
    assert "Cancelled" in out


# ---- databases and opening files --------------------------------------------------------


def test_database_search_download_and_open(tmp_path, monkeypatch):
    hits = [{"source": "cod", "id": "1000041", "formula": "Cu O", "title": "cuprite"}]
    monkeypatch.setattr(mt, "search_backends", lambda s, m, q: (hits, "1 results in COD"))
    fetched = []

    def fetch(rid, outdir="."):
        fetched.append((rid, outdir))
        return str(Path(outdir) / f"{rid}.cif")

    monkeypatch.setitem(mt.BACKENDS["cod"], "fetch", fetch)
    path, _, out = drive(mt.cm_menu_database,
                         ["cod", "text", "cuprite", "1", str(tmp_path), "y"])
    assert fetched == [("1000041", str(tmp_path))]
    assert path == str(tmp_path / "1000041.cif") and "cuprite" in out
    path, _, _ = drive(mt.cm_menu_database, ["cod", "id", "1000041", str(tmp_path), "n"])
    assert path is None and len(fetched) == 2


def test_database_failures_are_reported(monkeypatch):
    monkeypatch.setattr(mt, "search_backends", lambda s, m, q: ([], "0 results in COD"))
    path, _, out = drive(mt.cm_menu_database, ["cod", "text", "nothing"])
    assert path is None and "0 results" in out

    def boom(s, m, q):
        raise OSError("network down")

    monkeypatch.setattr(mt, "search_backends", boom)
    path, _, out = drive(mt.cm_menu_database, ["all", "text", "x"])
    assert path is None and "network down" in out
    path, _, out = drive(mt.cm_menu_database, ["q"])
    assert path is None and "Cancelled" in out


def test_open_another_file_switches_structure(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(menu, "print_info", lambda mol, name: seen.append((name, len(mol.atoms))))
    first = tmp_path / "one.xyz"
    first.write_text("1\n\nC 0 0 0\n")
    drive(menu.run_classic,
          ["10", "10", "10", "90", "90", "90",           # cell for the .xyz
           "18", str(tmp_path / "missing.pdb"),          # not found: stay
           "18", str(tmp_path / "x.foo"),                # missing too
           "18", str(HKUST), "9"],
          str(first))
    assert seen[0] == (str(first), 1)
    assert seen[-1] == (str(HKUST), 624)


def test_classic_now_reads_cif_and_poscar(tmp_path, monkeypatch):
    mol = mixed_mol()
    cif = tmp_path / "s.cif"
    drive(mt.cm_menu_write_other, ["1", str(cif), "6"], mol, "t.pdb")
    seen = []
    monkeypatch.setattr(menu, "print_info", lambda m, name: seen.append(len(m.atoms)))
    drive(menu.run_classic, ["9"], str(cif))
    assert seen == [32]


def test_unknown_extension_still_exits(tmp_path):
    bad = tmp_path / "s.foo"
    bad.write_text("x")
    with pytest.raises(SystemExit) as e:
        menu.run_classic(str(bad))
    assert "please rename" in str(e.value)


def test_movie_menu_new_options(tmp_path):
    frames = water_trajectory(4)
    out_cif = tmp_path / "f.cif"
    _, _, out = drive(menu._main_loop_movie,
                      ["11", "1", "15", "12", "2", "2", "O S", "6", "17", "17",
                       "13", "1", "1", str(out_cif), "6", "15", str(tmp_path), "9"],
                      frames, "t.xyz")
    assert "Replaced 10 O atoms with S" in out and "Undid the last change" in out
    assert "Nothing to undo" in out and "No completed pressure points" in out
    assert out_cif.exists()
