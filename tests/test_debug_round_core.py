"""Regression tests for bugs found in the 2026-11-10 debugging round (shared core and the
terminal viewer). Each test names the bug it guards against."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
import pytest

from pdb_wizard import analyses, inputgen
from pdb_wizard.atom import Atom
from pdb_wizard.geometry import get_close_contacts, get_lone_atoms
from pdb_wizard.io import next_mpmc_mol_id, read_file, write_mpmc_pdb
from pdb_wizard.molecule import Molecule
from pdb_wizard.operations import extend_axis
from pdb_wizard.pbc import PBC
from pdb_wizard.rdf import compute_rdf_trajectory
from pdb_wizard.sorbates import (
    SORBATE_MODELS,
    insert_sorbates,
    parse_sorbate_count,
    sorbate_atoms,
)
from pdb_wizard.templates import ENGINES
from pdb_wizard.trajectory import read_trajectory

ROOT = Path(__file__).resolve().parent.parent


def movable_molecules(text: str) -> dict:
    mols: dict = {}
    for ln in text.splitlines():
        if ln.startswith("ATOM") and ln.split()[4] == "M":
            mols.setdefault(ln.split()[5], []).append(ln.split()[2])
    return mols


# ---- molecule ids and reloading sorbates -----------------------------------------------


def test_new_sorbates_never_reuse_a_molecule_id():
    """Packing after a supercell used 2 + count, which collided with the copied ids (2, 4)
    and merged two sorbates into one MPMC molecule."""
    mol = read_file(str(ROOT / "HKUST-1_out_mpmc.pdb"))
    mol = extend_axis(mol, 0, 1) or mol
    assert sorted({a.mpmc_mol for a in mol.atoms if a.mpmc_flag == "M"}) == [2, 4]
    assert next_mpmc_mol_id(mol) == 5
    model = SORBATE_MODELS["h2o_tip3p"]
    mol.atoms.extend(sorbate_atoms(model, insert_sorbates(mol, model, 2, seed=0),
                                   next_mpmc_mol_id(mol)))
    out = io.StringIO()
    write_mpmc_pdb(mol, out)
    sizes = sorted(len(v) for v in movable_molecules(out.getvalue()).values())
    assert sizes == [3, 3, 3, 3]  # two CO2 and two waters, none merged
    assert next_mpmc_mol_id(Molecule(atoms=[Atom(0, 0, 0, "C")])) == 2


@pytest.mark.parametrize("key", ["h2_bss", "h2_bssp", "co2_phastp"])
def test_reloading_packed_models_keeps_every_site(tmp_path, key):
    """The reader's duplicate-atom check deleted model sites 0.04 Å apart (BSS H2, PHAST
    CO2): a 5-site model came back with 3."""
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    model = SORBATE_MODELS[key]
    mol.atoms.extend(sorbate_atoms(model, insert_sorbates(mol, model, 3, seed=2),
                                   next_mpmc_mol_id(mol)))
    path = tmp_path / "packed.pqr"
    with open(path, "w") as f:
        write_mpmc_pdb(mol, f)
    back = read_file(str(path))
    assert sum(a.mpmc_flag == "M" for a in back.atoms) == 3 * len(model.sites)


def test_duplicate_framework_atoms_are_still_removed(tmp_path):
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    copy_of_first = Atom(*mol.atoms[0].x, mol.atoms[0].element.symbol)
    mol.atoms.append(copy_of_first)
    path = tmp_path / "dup.pqr"
    with open(path, "w") as f:
        write_mpmc_pdb(mol, f)
    messages = []
    assert len(read_file(str(path), on_message=messages.append).atoms) == 624 and messages


@pytest.mark.parametrize("text, expect", [("fill", None), (" 3 ", 3), ("12", 12)])
def test_parse_sorbate_count(text, expect):
    assert parse_sorbate_count(text) == expect


@pytest.mark.parametrize("text", ["²", "0", "-1", "two", "", "1.5"])
def test_parse_sorbate_count_rejects(text):
    """'²'.isdigit() is True but int('²') raises: the prompt crashed."""
    with pytest.raises(ValueError, match="positive number or 'fill'"):
        parse_sorbate_count(text)


# ---- analyses ---------------------------------------------------------------------------


def frames_without_cell(tmp_path, n=6):
    path = tmp_path / "t.xyz"
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (12, 3))
    with open(path, "w") as f:
        for k in range(n):
            f.write("12\nframe\n")
            for i, p in enumerate(pts):
                f.write(f"{'O' if i % 3 == 0 else 'H'} {p[0] + 0.1 * k:.4f} {p[1]:.4f} "
                        f"{p[2]:.4f}\n")
    return read_trajectory(str(path))


def test_stand_in_cell_is_not_a_cell(tmp_path):
    """Trajectories without a cell get a 10^6 Å stand-in box; RDF and density against it
    gave g(r) ~ 1e17 and a peak at 5000 Å."""
    frames = frames_without_cell(tmp_path)
    assert analyses.run_rdf(frames[0], "O", "H", frames=frames).summary == (
        "RDF needs a periodic cell")
    assert analyses.run_density_profile(frames[0], "O", frames=frames).summary == (
        "A density profile needs a periodic cell")
    assert analyses.unavailable("pxrd", frames[0], frames).startswith("Need a crystal")
    assert analyses.unavailable("void_vol", frames[0], frames) == "Need PBC for void volume"


def water_frames(n=10):
    rng = np.random.default_rng(3)
    centres = rng.uniform(2, 13, (6, 3))
    out = []
    for f in range(n):
        atoms = []
        for c in centres:
            o = Atom(*(c + [0.2 * f, 0, 0]), "O")
            atoms += [o, Atom(*(o.x + [0.96, 0, 0]), "H"), Atom(*(o.x + [-0.24, 0.93, 0]), "H")]
        m = Molecule(atoms=atoms, pbc=PBC(15, 15, 15, 90, 90, 90))
        m.detect_bonds()
        out.append(m)
    return out


def test_empty_frame_window_says_so():
    """Start after End used to report 'No O atoms found' or 'No data'."""
    frames = water_frames()
    mol = frames[0]
    runs = [analyses.run_msd(frames, "O", start=5, end=2),
            analyses.run_rmsd(frames, start=9, end=3),
            analyses.run_gyration(frames, start=11),
            analyses.run_hbonds(mol, frames=frames, start=7, end=6),
            analyses.run_rdf(mol, "O", "H", frames=frames, start=8, end=2),
            analyses.run_coordination(mol, "O", "H", frames=frames, start=8, end=2,
                                      use_traj=True),
            analyses.run_density_profile(mol, "O", frames=frames, start=8, end=2)]
    for res in runs:
        assert res.empty and res.summary.startswith("No frames between"), res.summary
    assert "(the trajectory has 10)" in runs[0].summary


def test_frame_numbers_count_from_the_start_of_the_trajectory():
    frames = water_frames()
    hb = analyses.run_hbonds(frames[0], frames=frames, start=5, end=8)
    assert hb.x.tolist() == [5.0, 6.0, 7.0, 8.0]
    rmsd = analyses.run_rmsd(frames, start=4, end=6, dt=2.0)
    assert rmsd.csv_arrays()[0].tolist() == [4.0, 5.0, 6.0]
    assert rmsd.x.tolist() == [0.0, 2.0, 4.0]  # time from the window's first frame
    rg = analyses.run_gyration(frames, start=3, end=4)
    assert rg.csv_arrays()[0].tolist() == [3.0, 4.0]


def test_energy_ignores_nan_rows():
    data = {"step": np.arange(1.0, 6.0), "energy": np.array([1.0, np.nan, 3.0, 4.0, 5.0])}
    res = analyses.run_energy(data, "energy")
    assert np.isfinite(res.y_min) and np.isfinite(res.y_max) and "nan" not in res.summary
    allnan = analyses.run_energy({"step": np.arange(3.0), "e": np.full(3, np.nan)}, "e")
    assert allnan.empty and "no finite values" in allnan.summary


def test_rdf_trajectory_reports_progress_within_frames():
    frames = water_frames(3)
    seen = []
    compute_rdf_trajectory(frames, "O", "H", 50, 6.0, seen.append)
    assert len(seen) > 3 and seen == sorted(seen) and seen[-1] == 1.0
    assert any(0 < v < 1 / 3 for v in seen)  # inside the first frame


@pytest.mark.parametrize("text", [
    "two_theta intensity\n5 10\n6 40\n7 20\n",       # whitespace with a header
    "2theta\tI\n5\t10\n6\t40\n7\t20\n",              # tabs with a header
    "# Cu Ka\ntt,i\n5,10\n6,40\n7,20\n",              # comment then a header
    "5;10\n6;40\n7;20\n",                             # semicolons
    "5, 10, 99\n6, 40, 99\n7, 20, 99\n",              # extra column
])
def test_experimental_pxrd_headers_and_separators(tmp_path, text):
    path = tmp_path / "exp.txt"
    path.write_text(text)
    tt, inten = analyses.read_experimental_pxrd(str(path))
    assert tt.tolist() == [5, 6, 7] and inten.tolist() == pytest.approx([25, 100, 50])


def test_experimental_pxrd_single_row(tmp_path):
    path = tmp_path / "one.xy"
    path.write_text("7.5 40\n")
    tt, inten = analyses.read_experimental_pxrd(str(path))
    assert tt.tolist() == [7.5] and inten.tolist() == [100.0]


def test_contacts_and_lone_atoms_match_the_all_pairs_definition():
    """The all-pairs distance matrix took ~3.6 GB for 7.5k atoms; the chunked search must
    give exactly the same contacts and lone atoms."""
    rng = np.random.default_rng(4)
    pbc = PBC(13, 15, 12, 80, 95, 110)
    atoms = [Atom(*(f @ pbc.basis_matrix), el)
             for f, el in zip(rng.random((700, 3)), rng.choice(["C", "O", "H", "N", "Cu"], 700))]
    mol = Molecule(atoms=atoms, pbc=pbc)
    i, j, r = pbc.get_all_rs_min_image(mol.atoms)
    vdw = np.array([a.vdw for a in mol.atoms])
    bond = np.array([a.bond_r for a in mol.atoms])
    mask = (0.5 * (vdw[i] + vdw[j]) > r) & (r > 0.5 * (bond[i] + bond[j]))
    contacts = get_close_contacts(mol)
    assert len(contacts) == int(mask.sum()) > 0
    assert [tuple(map(int, c.split()[1:3])) for c in contacts] == list(
        zip((i[mask] + 1).tolist(), (j[mask] + 1).tolist()))
    near = 0.5 * (vdw[i] + vdw[j]) > r
    has = np.zeros(len(mol.atoms), bool)
    has[i[near]] = has[j[near]] = True
    assert [mol.atoms.index(a) for a in get_lone_atoms(mol)] == np.where(~has)[0].tolist()


# ---- input generator ---------------------------------------------------------------------


def test_form_values_respect_ranges():
    var = {v.name: v for v in ENGINES["mpmc_isotherm"].user_vars}
    cp2k = {v.name: v for v in ENGINES["cp2k_cell_opt"].user_vars}
    with pytest.raises(ValueError, match="at least 1"):
        inputgen.parse_form_value(var["temperature"], "-5")
    with pytest.raises(ValueError, match="finite"):
        inputgen.parse_form_value(cp2k["cutoff_ry"], "nan")
    with pytest.raises(ValueError, match="at most 1"):
        inputgen.parse_form_value(var["insert_probability"], "1.5")
    with pytest.raises(ValueError, match="at least 10"):
        inputgen.parse_form_value(var["corrtime"], 5)
    assert inputgen.parse_form_value(cp2k["cutoff_ry"], "1e3") == 1000.0


def test_unknown_engines_are_reported_everywhere():
    mol = read_file(str(ROOT / "HKUST-1.pdb"))
    for out in (inputgen.check_engine(mol, "zzz"), inputgen.engine_template_text("zzz"),
                inputgen.generate_engine_inputs(mol, "zzz", {}, "d")):
        assert not out.ok and out.status == "Unknown engine: zzz"


# ---- terminal viewer ---------------------------------------------------------------------

textual = pytest.importorskip("textual")


def test_tui_input_generator_keeps_a_chosen_output_folder():
    from textual.widgets import Input, Select

    from pdb_wizard.viewer import InputGeneratorScreen, PdbWizardApp

    mol = read_file(str(ROOT / "HKUST-1.pdb"))

    async def run():
        app = PdbWizardApp(molecule=mol, filepath=str(ROOT / "HKUST-1.pdb"))
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            screen = InputGeneratorScreen(mol, str(ROOT / "HKUST-1.pdb"))
            app.push_screen(screen)
            await pilot.pause()
            outdir = screen.query_one("#gen-outdir", Input)
            engine = screen.query_one("#gen-engine", Select)
            engine.value = "cp2k_cell_opt"
            await pilot.pause()
            default = outdir.value
            outdir.value = "/tmp/my_runs"
            engine.value = "openmm_nvt"
            await pilot.pause()
            return default, outdir.value

    default, after = asyncio.run(run())
    assert default.endswith("cp2k_cell_opt") and after == "/tmp/my_runs"


def test_tui_database_rows_tolerate_missing_fields(monkeypatch):
    from textual.widgets import DataTable, Input

    from pdb_wizard import database
    from pdb_wizard.viewer import DatabaseSearchScreen, PdbWizardApp

    monkeypatch.setitem(database.BACKENDS["cod"], "search", lambda q, mode="text": [
        {"source": "cod", "id": "1", "formula": None, "title": None}])
    for code in ("rcsb", "pubchem"):
        monkeypatch.setitem(database.BACKENDS[code], "search", lambda q, mode="text": [])

    async def run():
        app = PdbWizardApp(molecule=read_file(str(ROOT / "HKUST-1.pdb")), filepath="x.pdb")
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            screen = DatabaseSearchScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#db-query", Input).value = "cuprite"
            await screen._do_search()
            await pilot.pause()
            return screen.query_one("#db-results", DataTable).row_count

    assert asyncio.run(run()) == 1


@pytest.mark.parametrize("layout", ["every_frame", "once_at_top"])
def test_multi_model_pdb_keeps_its_cell(tmp_path, layout):
    """A CRYST1 line before MODEL was dropped, so the trajectory became non-periodic
    (10^6 Å stand-in box): RDFs and densities were meaningless."""
    lines = []
    for k in range(3):
        if layout == "every_frame" or k == 0:
            lines.append("CRYST1   15.000   16.000   17.000  90.00  90.00  90.00 P 1           1")
        lines.append(f"MODEL     {k + 1:4d}")
        lines.append(f"ATOM      1 O    HOH A   1      {1 + k:6.3f}   2.000   3.000  1.00  0.00"
                     "           O")
        lines.append("ENDMDL")
    path = tmp_path / "t.pdb"
    path.write_text("\n".join(lines) + "\n")
    frames = read_trajectory(str(path))
    assert len(frames) == 3
    assert [(f.pbc.a, f.pbc.b, f.pbc.c) for f in frames] == [(15.0, 16.0, 17.0)] * 3
